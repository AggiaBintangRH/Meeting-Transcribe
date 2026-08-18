import Foundation

// MOSS-Transcribe-Diarize as the DIARIZATION engine (`diarization.engine` ==
// "moss"): one 0.9B model returns the words, the speaker labels and the times
// together, instead of an ASR pass plus a separate pyannote pass.
//
// WHAT THESE LABELS ARE, AND ARE NOT
// ----------------------------------
// A MOSS label distinguishes speakers WITHIN ONE CHUNK and nothing more. The
// model is called once per chunk and its `S01`/`S02` are anonymous per call —
// S01 in chunk 7 is not S01 in chunk 8, and the model makes no claim that they
// are. The STRUCTURE here is built so that fact survives into the transcript:
//
//   * the wire id embeds the chunk index, so ids from two chunks can never be
//     equal and `coalesceAdjacentSameSpeaker` cannot silently weld two chunks'
//     rows into one person;
//   * the ids sit in their own topmost range (`mossIDBase`), above position,
//     so they reach neither profile store nor the position diarizer, and are
//     not renameable (there is nothing persistent to rename);
//   * the turns live in `mossTurns`, never `liveTurns`, so every Office-only
//     invariant (`officeTurnsOnly`, `overlapRegions`, `applyFinalSpeakers`,
//     `speakerCount`) stays pure pyannote without a single assert relaxed.
//
// THE DISPLAY NAME IS THE ONE EXCEPTION, by owner decision (2026-07-31): it used
// to be "S1 · #7" and is now plainly "Speaker 1", so the label no longer admits
// its per-chunk scope. See `mossDisplayName` for the full cost. The three points
// above are what keep the ROWS correct while the LABELS overstate, and they are
// now load-bearing in a way they were not before — nothing else is left saying
// a MOSS name is chunk-local.
//
// Linking the same voice across chunks is still NOT attempted here — the owner
// deferred it again when choosing the plain label. `WeSpeakerService.identify`
// takes any `label → spans` set, so this is wiring, not research.
//
// TWO MODES, ONE PROCESS WHEREVER POSSIBLE
// ----------------------------------------
//   MOSS + MOSS   the chunked model IS MOSS. One sidecar, one 3.6 GB load, one
//                 forward pass: the `final` carries both the text and the
//                 segments, and each segment becomes its own pinned confirmed
//                 row — no `assignSentences` re-guessing text the model already
//                 attributed. (`ModelLoader.needsSecondMossProcess` == false.)
//   other + MOSS  Qwen3/Whisper/Granite does the ASR; a SECOND MOSS process
//                 (`ModelLoader.mossDiarization`) is fed the same samples and
//                 flushed on the SAME boundary — identical windows are what let
//                 its turns split the ASR text exactly. Its text is discarded.
//
// A pyannote session never enters any of this: `mossDiarizationActive` stays
// false, every callback returns immediately, and the display path reads
// `liveTurns` exactly as it always has.
extension AudioRecorder {

    // MARK: - Pure id / name / turn construction

    /// Wire id for one speaker of one chunk.
    ///
    /// `mossIDBase + chunkIndex * 100 + localIndex` — globally unique per chunk
    /// ON PURPOSE. Two chunks that both name a speaker `S01` get different ids,
    /// so nothing downstream can merge them: continuity across chunks is unknown,
    /// and an id that pretended otherwise would be a claim the model never made.
    ///
    /// `localIndex` is clamped to 99 so one chunk's speakers can never run into
    /// the next chunk's block. A real meeting chunk has two or three speakers;
    /// a hundred would already be a malformed reply.
    nonisolated static func mossWireID(chunkIndex: Int, localIndex: Int) -> Int {
        mossIDBase + chunkIndex * 100 + min(max(localIndex, 0), 99)
    }

    /// Display name for one speaker of one chunk: `"Speaker 1"`.
    ///
    /// OWNER DECISION, 2026-07-31, made with the cost stated. This used to read
    /// `"S1 · #7"` — speaker 1 of chunk 7 — precisely because a MOSS label means
    /// nothing outside its own chunk, and the old name said so on its face.
    ///
    /// **The name now claims more than the model does, and that is deliberate.**
    /// `chunkIndex` is still taken (it is what `mossWireID` needs) but no longer
    /// appears: chunk 7's `Speaker 1` and chunk 8's `Speaker 1` READ as one
    /// person while being, as far as anything here has measured, unrelated. The
    /// owner was shown this and chose the plain label anyway.
    ///
    /// What did NOT change, and must not: the wire id still embeds the chunk
    /// index, so `coalesceAdjacentSameSpeaker` still cannot weld two chunks'
    /// rows into one. The transcript therefore keeps the rows honest even though
    /// the labels no longer are — two adjacent `Speaker 1` rows from different
    /// chunks stay two rows. Do NOT "simplify" the id to match the name.
    ///
    /// The fix that would make this label TRUE is cross-chunk stitching through
    /// `WeSpeakerService` — structurally possible since the pyannote/wespeaker
    /// split (`identify` accepts any `label → spans` set), deferred by the owner.
    /// Until then a MOSS session still has no saved profiles, no renaming and no
    /// `spk` confidence, exactly as the DiarizationTab states.
    nonisolated static func mossDisplayName(chunkIndex: Int, localIndex: Int) -> String {
        "Speaker \(min(max(localIndex, 0), 99))"
    }

    /// `"S01"` → 1. Zero when the label carries no digits at all.
    ///
    /// The sidecar's parser only ever produces `S` + digits, so the fallback is
    /// unreachable on the normal path; it exists so a malformed label degrades to
    /// one shared id for that chunk rather than throwing away the chunk.
    nonisolated static func mossSpeakerIndex(_ label: String) -> Int {
        let digits = label.filter(\.isNumber)
        return Int(digits) ?? 0
    }

    /// MOSS segments (CHUNK-LOCAL seconds) → turns in RECORDING time.
    ///
    /// Pure and static so the id scheme, the offsetting and the per-chunk
    /// uniqueness are testable without a recorder, a sidecar or a model.
    ///
    /// Times are offset by the window start (the `offsetTurns` precedent) and
    /// clamped INTO the window. The clamp is arithmetic hygiene rather than a
    /// quality judgement on the text: a segment that claimed to end after its own
    /// chunk would overlap the next chunk's rows and corrupt the ordering of a
    /// transcript that is otherwise strictly increasing in time.
    ///
    /// `conf` is left nil throughout — MOSS scores no speaker match against
    /// anything, and nil means "not measured" here exactly as it does everywhere
    /// else in this codebase, never "low".
    nonisolated static func mossTurns(from segments: [ChunkedASRService.MossSegment],
                                      window: ClosedRange<Double>,
                                      chunkIndex: Int) -> [SpeakerTurn] {
        segments.compactMap { seg in
            guard !seg.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }
            let local = mossSpeakerIndex(seg.speaker)
            let start = min(max(window.lowerBound + seg.start, window.lowerBound),
                            window.upperBound)
            let end = min(max(window.lowerBound + seg.end, start), window.upperBound)
            return SpeakerTurn(
                start: start, end: end,
                id: mossWireID(chunkIndex: chunkIndex, localIndex: local),
                name: mossDisplayName(chunkIndex: chunkIndex, localIndex: local))
        }
    }

    /// MOSS segments → one PINNED confirmed transcript segment each.
    ///
    /// Pinned because the model already decided who said these words. The pinned
    /// path in `derivedRows` renders such a segment as exactly one row with that
    /// speaker and skips re-attribution entirely, so `assignSentences` never gets
    /// to re-guess, from character offsets, an attribution the model made from the
    /// audio itself.
    ///
    /// Only used in the MOSS+MOSS mode. With another model doing the ASR the text
    /// is that model's, and it is split by `mossTurns` through the ordinary
    /// display path instead.
    nonisolated static func mossPinnedSegments(from segments: [ChunkedASRService.MossSegment],
                                               window: ClosedRange<Double>,
                                               chunkIndex: Int) -> [TranscriptSegment] {
        segments.compactMap { seg in
            let text = seg.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            let local = mossSpeakerIndex(seg.speaker)
            let start = min(max(window.lowerBound + seg.start, window.lowerBound),
                            window.upperBound)
            let end = min(max(window.lowerBound + seg.end, start), window.upperBound)
            return TranscriptSegment(
                text: text, confirmed: true, window: start...end,
                pinnedSpeakerID: mossWireID(chunkIndex: chunkIndex, localIndex: local),
                pinnedSpeakerName: mossDisplayName(chunkIndex: chunkIndex, localIndex: local))
            // No `asrConf`: MOSS reports no confidence of any kind, and absent
            // stays absent rather than becoming a number nobody measured.
        }
    }

    // MARK: - Session configuration

    /// The second MOSS process for THIS session, or nil when there isn't one
    /// (pyannote engine, or MOSS filling both roles from one sidecar).
    var mossDiarService: ChunkedASRService? {
        mossDiarizationActive ? modelLoader.mossDiarization : nil
    }

    /// Read the engine choice for this session and reset all MOSS state.
    /// Called from `beginCapture` alongside `configureDiarization`, so the engine
    /// is fixed for the whole recording like every other setting.
    func configureMoss() {
        let d = UserDefaults.standard
        let engine = d.string(forKey: "diarization.engine") ?? ModelLoader.pyannoteEngineID
        let diarOn = d.object(forKey: "diarization.enabled") as? Bool ?? true
        let chunkedID = d.string(forKey: "chunked.model") ?? "qwen3"

        mossDiarizationActive = diarOn && engine == ModelLoader.mossEngineID
        mossIsChunkedModel = chunkedID == "moss"
        mossIdentifyStarted = false
        mossTurns = []
        mossChunkIndex = 0
        mossIncomingSegments = nil
        mossPendingWindows = []
        mossStaleReplies = 0
        mossChunkWatchdog?.cancel()
        mossChunkWatchdog = nil
        // A stop-time full pass from the PREVIOUS session must not survive into
        // this one. It appends to `mossTurns`, which is cleared a few lines
        // above, so a survivor would drip a finished meeting's speaker turns into
        // a live transcript — and it holds the recorder weakly, so nothing else
        // would stop it. Same reasoning as the watchdog on the line before.
        mossFullPassTask?.cancel()
        mossFullPassTask = nil
        // A session with no second process has this leg of the stop gate already
        // complete before it is ever consulted — the same rule `remoteLastChunkDone`
        // follows.
        mossLastChunkDone = mossDiarService == nil

        guard mossDiarizationActive else { return }
        mossLog("engine=moss chunked=\(chunkedID) "
                + "mode=\(mossIsChunkedModel ? "one process (ASR + diarization)" : "second MOSS process")")
        // Stated once per session rather than on every row rebuild, because a
        // display plan that is silently NOT the configured one has no other place
        // to be seen. Beam collection itself is untouched either way — only the
        // display plan is decided here.
        //
        // Both outcomes are logged, not just the suppressed one: "the beam is
        // labelling this meeting" is the line someone reads to confirm `Live only`
        // is doing its job, and its absence was read as failure on 2026-08-18 when
        // the truth was simply that nothing had been written.
        if positionDiarizer != nil, positionSource != .pyannote {
            if positionSource == .atndLiveOnly {
                mossLog("position source 'atndLive' IS applied while recording under the MOSS "
                        + "engine — beam direction labels the meeting as it happens, and the "
                        + "layer drops out entirely at Stop, so the finished transcript is "
                        + "MOSS only. Expect FILL lines in logs/position-diarization.log.")
            } else {
                mossLog("position source '\(positionSource.rawValue)' is not applied under the MOSS "
                        + "engine — the position layer fills gaps in PYANNOTE turns, and there are "
                        + "none this session. Rows use MOSS labels only. Pick 'Live only' if you "
                        + "want beam labels while recording; it is the one mode that survives "
                        + "this engine, because it removes itself at Stop.")
            }
        }
        installMossDiarizationCallbacks()
    }

    /// Wire the SECOND MOSS process (other ASR + MOSS diarization). No-op in the
    /// one-process mode, where the chunked service's own callbacks carry the
    /// segments and `AudioRecorder.beginCapture` installs them.
    private func installMossDiarizationCallbacks() {
        guard let service = mossDiarService else { return }
        service.onChunkSegments = { [weak self] segments in
            Task { @MainActor in self?.mossIncomingSegments = segments }
        }
        // The text of this process is DISCARDED — another model is doing the ASR.
        // The callback still matters: it is the one place that knows the chunk
        // settled, so it owns popping the window FIFO and settling the stop leg.
        service.onChunkTranscript = { [weak self] _, _ in
            Task { @MainActor in
                guard let self else { return }
                self.mossChunkWatchdog?.cancel()
                let window = self.mossPendingWindows.isEmpty
                    ? nil : self.mossPendingWindows.removeFirst()
                let segments = self.mossIncomingSegments ?? []
                self.mossIncomingSegments = nil
                // A reply queued BEFORE the stop-time full pass began describes
                // audio the pass is re-labelling right now. Applying it would
                // duplicate every turn in that span — see `mossStaleReplies`.
                if self.mossStaleReplies > 0 {
                    self.mossStaleReplies -= 1
                    self.mossLog("chunk from before the FULL PASS discarded"
                                 + (window.map { " [\(self.fmt($0.lowerBound))-"
                                                 + "\(self.fmt($0.upperBound))]" } ?? "")
                                 + " — \(segments.count) segments dropped")
                    self.checkMossChunkDone()
                    return
                }
                if let window { self.applyMossTurns(segments, window: window) }
                self.checkMossChunkDone()
            }
        }
        service.onChunkError = { [weak self] message in
            Task { @MainActor in
                guard let self else { return }
                self.mossChunkWatchdog?.cancel()
                self.mossIncomingSegments = nil
                // Drain the queued window even on failure, exactly as the chunked
                // path does: a stuck entry would block the stop gate and misalign
                // every window after it.
                if !self.mossPendingWindows.isEmpty {
                    self.mossPendingWindows.removeFirst()
                }
                // A failed reply consumes its stale slot too, or the counter would
                // never drain and the pass's OWN first reply would be discarded.
                if self.mossStaleReplies > 0 { self.mossStaleReplies -= 1 }
                self.mossLog("chunk FAILED — \(message)")
                self.checkMossChunkDone()
            }
        }
    }

    // MARK: - Applying a chunk's segments

    /// Turns only (other ASR + MOSS diarization): the ASR model's text is split
    /// by these spans through the ordinary display path.
    func applyMossTurns(_ segments: [ChunkedASRService.MossSegment],
                        window: ClosedRange<Double>) {
        guard mossDiarizationActive else { return }
        let index = mossChunkIndex
        mossChunkIndex += 1
        let turns = Self.mossTurns(from: segments, window: window, chunkIndex: index)
        mossTurns.append(contentsOf: turns)
        mossLog("chunk #\(index) [\(fmt(window.lowerBound))-\(fmt(window.upperBound))] "
                + "\(turns.count) turns, speakers \(Set(turns.map(\.name)).sorted())")
        rebuildDisplayRows()
    }

    /// Text AND turns (MOSS in both roles): every segment becomes its own pinned
    /// confirmed row, and the same segments also become turns.
    ///
    /// The turns are kept even though the rows are pinned, because they are what
    /// `speakerRanges` hands to any segment that is NOT pinned — a repaired or
    /// preserved row, or the short tail at Stop.
    ///
    /// Returns true when it handled the chunk, so the chunked-ASR callback knows
    /// not to also append the joined text as one unattributed row.
    @discardableResult
    func applyMossChunk(window: ClosedRange<Double>?) -> Bool {
        guard mossDiarizationActive, mossIsChunkedModel,
              let segments = mossIncomingSegments else { return false }
        mossIncomingSegments = nil
        guard let window else { return false }
        let index = mossChunkIndex
        mossChunkIndex += 1
        let turns = Self.mossTurns(from: segments, window: window, chunkIndex: index)
        mossTurns.append(contentsOf: turns)
        let pinned = Self.mossPinnedSegments(from: segments, window: window, chunkIndex: index)
        segmentsAppendPinned(pinned)
        mossLog("chunk #\(index) [\(fmt(window.lowerBound))-\(fmt(window.upperBound))] "
                + "\(pinned.count) rows, speakers \(Set(turns.map(\.name)).sorted())")
        return true
    }

    /// Append MOSS rows keeping the transcript sorted by start time, with any
    /// separation-debug segments left pinned at the end (the same ordering rule
    /// `insertPinnedSorted` applies for overlap repair).
    private func segmentsAppendPinned(_ newSegs: [TranscriptSegment]) {
        guard !newSegs.isEmpty else { return }
        var real = segments.filter { !$0.isSeparationDebug }
        let debug = segments.filter { $0.isSeparationDebug }
        real.append(contentsOf: newSegs)
        real.sort { ($0.window?.lowerBound ?? .greatestFiniteMagnitude)
                  < ($1.window?.lowerBound ?? .greatestFiniteMagnitude) }
        segments = real + debug
    }

    // MARK: - Flushing the second process

    /// Flush the second MOSS process on the OFFICE chunk boundary.
    ///
    /// Deliberately the ASR boundary and not `diarization.intervalSec`: identical
    /// windows are the whole reason its turns can split the ASR model's text
    /// exactly. A diarization cadence of its own would produce spans that cut
    /// across chunk windows, and every row would then be assembled from two
    /// different clocks' idea of where a chunk ends.
    func flushMossDiarChunk(window: ClosedRange<Double>) {
        guard let service = mossDiarService else { return }
        mossPendingWindows.append(window)
        service.flush()
        mossChunkWatchdog?.cancel()
        mossChunkWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(180))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.mossPendingWindows.isEmpty else { return }
                self.mossPendingWindows.removeFirst()
                self.mossIncomingSegments = nil
                self.mossLog("chunk TIMED OUT — see logs/moss-diar.log")
                self.checkMossChunkDone()
            }
        }
    }

    // MARK: - Stop gate leg

    /// The MOSS leg of the stop gate, mirroring `checkLastChunkDone`.
    /// Idempotent; every terminal path of the second process calls it.
    ///
    /// There is no separate "busy" flag to consult: every flush appends a window
    /// and every terminal callback removes one, so an empty FIFO already means
    /// nothing is in flight.
    func checkMossChunkDone() {
        guard stopped, mossPendingWindows.isEmpty, !mossLastChunkDone else { return }
        mossLastChunkDone = true
        mossChunkWatchdog?.cancel()
        mossChunkWatchdog = nil
        setStopStep("moss-diarize", .done)
        checkStopProcessingDone()
    }

    // MARK: - Log

    /// Append a line to logs/moss-diarization.log.
    ///
    /// A DIFFERENT file from the sidecar's own `logs/moss-diar.log`, which the Python
    /// process writes on its own stderr. Two writers on one file is precisely the
    /// collision the overlap-repair logs were split to stop (2026-07-15); one
    /// process per file, no exceptions.
    func mossLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "moss-diarization", message: message)
    }
}
