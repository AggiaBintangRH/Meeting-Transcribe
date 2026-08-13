import Foundation

// Nested value types of `AudioRecorder`, moved verbatim out of the core file.
// Declaration-only: no behaviour, no stored state of the recorder itself.
extension AudioRecorder {

    /// Raw ASR unit: Nemotron text appears instantly (unconfirmed), then is
    /// replaced in place by the accurate chunked text (confirmed). Diarization
    /// then splits each confirmed chunk into per-speaker rows for display.
    struct TranscriptSegment: Identifiable, Equatable {
        let id = UUID()
        var text: String
        let confirmed: Bool               // true = chunked ASR (accurate), false = realtime
        var window: ClosedRange<Double>? = nil  // recording-time span (confirmed only)
        // Set only by overlap repair: this segment's text belongs entirely to one
        // separated speaker, so display rows use it directly (skip re-attribution).
        // Two fields (not a tuple) to keep Equatable synthesis working.
        var pinnedSpeakerID: Int? = nil
        var pinnedSpeakerName: String? = nil
        // Debug/inspection: a raw MossFormer2 separated-track ASR result, shown as
        // its own "MossFormer2 Index N" row (name in pinnedSpeakerName). No
        // attribution, no merge — never replaces speaker text.
        var isSeparationDebug: Bool = false
        // Word timestamps from the optional forced aligner (nil when alignment
        // is off or failed). Times are CHUNK-RELATIVE seconds exactly as the
        // sidecar sent them; `alignedChunkDuration` is that chunk buffer's
        // length, kept so the conversion to recording time can be sanity-checked
        // against `window`. Consumed by `WordAttribution` in `derivedRows`.
        var words: [ChunkedASRService.AlignedWord]? = nil
        var alignedChunkDuration: Double? = nil
        // The chunked model's confidence in THIS chunk's text (0…1, pooled
        // per-token probability). nil when the model reports none — Whisper is
        // the only runtime that does — and nil on repaired/debug segments, whose
        // text came from a separated track rather than this chunk's audio.
        var asrConf: Double? = nil
    }

    /// One transcribed Remote (conferencing) chunk. Deliberately NOT a
    /// `TranscriptSegment`: remote text never goes through the office display
    /// pipeline (`derivedRows` → speaker ranges → position gap-fill → word
    /// attribution). It IS speaker-split (phase 4), but against `remoteLiveTurns`
    /// in the remote identity space, and it must never reach the ATND/position
    /// path — so it is a separate collection that only meets office rows at the
    /// final sort in `rebuildDisplayRows`.
    ///
    /// Text and its ASR confidence, by design — no word timings: the transcript
    /// comes from the sidecar's `-2` file-transcribe frame, which calls
    /// `transcribe_path()` directly and never runs the forced aligner, so there
    /// are no `words`/`dur` and remote attribution stays sentence-level.
    /// Acceptable: word-exact timing matters where ATND and pyannote boundaries
    /// compete, which is Office-only. The confidence, unlike the timings, DOES
    /// come back on that frame (same model, same pooling), so remote rows show
    /// the same `asr` number office rows do.
    struct RemoteSegment: Identifiable, Equatable {
        let id = UUID()
        var text: String
        var window: ClosedRange<Double>   // recording-time span (shared clock)
        var conf: Double? = nil           // chunked-ASR confidence (Whisper only)

        /// False while this is REALTIME text waiting to be replaced by the chunked
        /// pass — the office twin of `TranscriptSegment.confirmed`.
        ///
        /// Added 2026-08-13. Before it, a remote realtime final had nowhere to go
        /// but the caption card, so every remote utterance piled into ONE growing
        /// card while every office utterance became its own row placed by time:
        /// *"row remote panjang banget ketika realtime"*. The cards render below
        /// the row list, so the room also appeared to jump above the far end. One
        /// asymmetry, both symptoms.
        var confirmed = true

        /// Word times from the aligner, or nil until (and unless) they arrive.
        ///
        /// The office twin of these two fields lives on `TranscriptSegment`, and
        /// they exist here for the same reason: without them a remote row's
        /// speaker boundary is placed by each sentence's CHARACTER POSITION in the
        /// chunk, which is a guess, while the office side places every word by the
        /// turn that covers it IN TIME. Owner, 2026-08-13, on feeding one source
        /// into both channels and getting two different splits: *"saya test nya
        /// sama audio sama dll sama tapi kenapa remote hasilnya kurang bagus"*.
        /// The audio and the model were the same; only this was missing.
        var words: [ChunkedASRService.AlignedWord]? = nil

        /// The aligner's view of how long the audio it was given was — checked
        /// against `window`'s span before any word time is trusted.
        var alignedChunkDuration: Double? = nil

        /// Set by overlap repair: this segment's text belongs to exactly ONE
        /// remote speaker and must NOT be re-split by the turns at display time.
        ///
        /// The office twins on `TranscriptSegment` exist for the same reason. A
        /// repaired segment's text came from a SEPARATED track, so the diarization
        /// turns that describe the MIXED audio no longer describe it — running it
        /// back through `speakerRanges` would hand one speaker's recovered words
        /// to the other, which is the failure repair exists to fix.
        var pinnedSpeakerID: Int? = nil
        var pinnedSpeakerName: String? = nil

        /// A raw separated-track row, kept for inspection and rendered after the
        /// real transcript. Never folded into anything and never repaired again.
        var isSeparationDebug = false
    }

    /// The live Remote caption — what the Remote realtime engine has produced
    /// since the last remote chunk boundary, shown as its own provisional card
    /// exactly like the office `partialTranscript`.
    ///
    /// A tiny type rather than a bare String because the *rule* is the whole
    /// point and it differs from the office one: a Remote FINAL is KEPT on
    /// screen, not cleared. Office can clear on a final because it immediately
    /// turns that final into an unconfirmed segment; Remote has no unconfirmed
    /// segment (see `RemoteSegment` — remote text never enters the office
    /// pipeline), so clearing on the final would blank the caption for the
    /// several seconds the confirmed remote chunk takes to come back. The
    /// caption instead survives until `commit()`, which every terminal outcome
    /// of that chunk calls — transcribed, empty, skipped as silence, or failed.
    struct RemoteCaption: Equatable {
        /// What the view draws; empty means no caption card at all.
        private(set) var text = ""

        /// Recording time at which the CURRENT provisional utterance began —
        /// the instant this caption went from empty to non-empty, not the last
        /// keystroke of it. That is what decides whether this card is drawn above
        /// or below the office one (owner, 2026-08-13: the two cards used to be
        /// pinned office-then-remote, so the far end appeared under the room even
        /// when the room had not spoken).
        ///
        /// Kept in RECORDING time rather than `Date`, so the rule is the same
        /// clock as every row on screen and is testable without a wall clock.
        private(set) var startedAt: Double?

        /// A realtime result arrived (partial or final — both are just "the best
        /// text so far" for audio no confirmed row covers yet).
        mutating func update(to incoming: String, at elapsed: Double) {
            let trimmed = incoming.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { startedAt = nil }
            else if text.isEmpty { startedAt = elapsed }
            text = trimmed
        }

        /// This caption's audio is now represented (or deliberately not) by a
        /// confirmed remote row — drop it so nothing stale lingers underneath.
        mutating func commit() {
            text = ""
            startedAt = nil
        }
    }

    /// One rendered row: a single speaker's turn with its time span and text.
    struct SpeakerUtterance: Identifiable, Equatable {
        let id: String                    // stable per (segment, turn index)
        var speaker: String?              // nil = not diarized yet
        var speakerID: Int?               // profile id (for rename)
        var start: Double?                // recording-time seconds (confirmed only)
        var end: Double?
        var text: String
        let confirmed: Bool
        var overlapped: Bool = false      // spoke over another speaker in this window
        /// Row came from the Remote stream, not the room. A display flag: it keeps
        /// the two label spaces visually distinct (amber vs teal) and tells the
        /// rename dialog to edit the remote PROFILE name rather than the composed
        /// "Remote Speaker - …" label. The id range, not this flag, is what routes
        /// a rename to the right store.
        var isRemote: Bool = false
        /// How confident the identification of THIS speaker is: the cosine
        /// similarity that matched the voice to its saved profile, taken as the
        /// MINIMUM over the turns this row spans (the weakest evidence behind the
        /// label is what the label is worth). Filled by
        /// `annotateSpeakerConfidence` as the last step of the row build.
        ///
        /// nil = no measurement, which happens legitimately: a speaker's first
        /// appearance (a brand-new profile matched nothing), an ATND position
        /// row (positions come from the beam, not from a voice embedding), an
        /// unconfirmed realtime row, or a row whose label came from
        /// `absorbShortSpeakerIslands` (a heuristic relabel, not a cosine).
        var speakerConf: Double? = nil
        /// The chunked model's confidence in this row's TEXT, carried down from
        /// the segment it came from. When rows from different chunks coalesce it
        /// is the MINIMUM of the non-nil values — overstating confidence is the
        /// invisible, harmful direction, the same asymmetry the hallucination
        /// gates are tuned around. nil for models that report none (everything
        /// but Whisper) and for repaired/debug rows.
        var asrConf: Double? = nil
    }

    /// One line in the processing overlay — a leg of the post-stop work.
    /// Reuses `ModelLoader.ItemState` for the icons; the loader itself doesn't
    /// fit here (its `loadAll` runs one item at a time, these run concurrently).
    struct StopStep: Identifiable, Equatable {
        let id: String            // "chunk" | "diarize" | "repair"
        /// `var` since the stop-time full re-transcription: `ItemState` has no
        /// progress payload, so a leg that runs for minutes counts its windows in
        /// its name (`setStopStepName`). The id, not the name, identifies a step.
        var name: String
        var state: ModelLoader.ItemState
    }
}
