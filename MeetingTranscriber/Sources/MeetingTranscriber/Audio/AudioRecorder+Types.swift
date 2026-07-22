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
    }

    /// One transcribed Remote (conferencing) chunk. Deliberately NOT a
    /// `TranscriptSegment`: remote text never goes through the office display
    /// pipeline (`derivedRows` → speaker ranges → position gap-fill → word
    /// attribution). It IS speaker-split (phase 4), but against `remoteLiveTurns`
    /// in the remote identity space, and it must never reach the ATND/position
    /// path — so it is a separate collection that only meets office rows at the
    /// final sort in `rebuildDisplayRows`.
    ///
    /// Text only, by design: the transcript comes from the sidecar's `-2`
    /// file-transcribe frame, which calls `transcribe_path()` directly and never
    /// runs the forced aligner — so there are no `words`/`dur` and remote
    /// attribution stays sentence-level. Acceptable: word-exact timing matters
    /// where ATND and pyannote boundaries compete, which is Office-only.
    struct RemoteSegment: Identifiable, Equatable {
        let id = UUID()
        var text: String
        var window: ClosedRange<Double>   // recording-time span (shared clock)
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

        /// A realtime result arrived (partial or final — both are just "the best
        /// text so far" for audio no confirmed row covers yet).
        mutating func update(to incoming: String) {
            text = incoming.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        /// This caption's audio is now represented (or deliberately not) by a
        /// confirmed remote row — drop it so nothing stale lingers underneath.
        mutating func commit() {
            text = ""
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
    }

    /// One line in the processing overlay — a leg of the post-stop work.
    /// Reuses `ModelLoader.ItemState` for the icons; the loader itself doesn't
    /// fit here (its `loadAll` runs one item at a time, these run concurrently).
    struct StopStep: Identifiable, Equatable {
        let id: String            // "chunk" | "diarize" | "repair"
        let name: String
        var state: ModelLoader.ItemState
    }
}
