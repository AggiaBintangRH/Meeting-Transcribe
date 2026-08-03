import Foundation

// Word alignment, which became ASYNCHRONOUS when the Qwen3-ForcedAligner moved
// out of the ASR sidecars into `AlignerService` (owner, 2026-07-29).
//
// The contract, and the reason every guard below exists:
//   * the chunk's TEXT is shown immediately, split by the estimated
//     character-proportional rule, exactly as it is without an aligner;
//   * the words arrive later (measured ~0.2 s per 30 s chunk on this M4) and
//     only ever change WHERE a speaker boundary falls inside that text;
//   * anything that goes wrong — a throw, a timeout, a rejected alignment, a
//     segment that has since been rewritten — simply leaves the estimate in
//     place. Alignment can never delay, alter or delete transcript text.
//
// Nothing here holds the stop gate. Words add and remove no content, the five
// existing legs already guard content, and a wedged aligner must not keep the
// processing overlay on screen. The accepted, stated consequence: the LAST
// chunk's rows may re-split about half a second after the overlay drops.
extension AudioRecorder {

    /// Safety cap on `alignChunkAudio`, mirroring the sidecars' `MAX_BUFFER_SEC`
    /// (300 s at 16 kHz). A chunk is normally 30 s; this only ever bites if the
    /// boundary logic stops firing, and then it bounds the memory rather than
    /// letting a whole meeting accumulate.
    nonisolated static let alignMaxBufferSamples = 300 * 16_000

    /// Park the audio accumulated since the last ASR chunk boundary under that
    /// window, and start the next one. Called from the live boundary in the tap
    /// AND from `stop()`'s tail flush — one helper so the two cannot diverge.
    ///
    /// No-op without an aligner: nothing ever appended, so there is nothing to
    /// park and no empty entry is created.
    func stashAlignAudio(windowStart: Double) {
        guard !alignChunkAudio.isEmpty else { return }
        alignAudioByWindow[windowStart] = alignChunkAudio
        alignChunkAudio = []
    }

    /// Ask the aligner for this chunk's word times, off the main actor, and hand
    /// the reply to `applyAlignedWords`.
    ///
    /// The WAV is written in the same task and deleted on EVERY exit path — the
    /// placement `flushRemoteChunk` already uses. A crash leaves `align-chunk-*`
    /// files in the temp dir exactly as it leaves `diar-chunk-*`/`remote-chunk-*`
    /// ones; there is deliberately no new sweeper.
    func requestAlignment(aligner: AlignerService, samples: [Float],
                          segmentID: UUID, text: String) {
        Task { [weak self] in
            guard let self else { return }
            let url = await Task.detached(priority: .utility) {
                Self.writeTempWAV(samples: samples, prefix: "align-chunk")
            }.value
            guard let url else {
                self.positionLog("ALIGN FAIL — could not write a temp WAV")
                return
            }
            do {
                let result = try await aligner.align(path: url.path, text: text)
                if let words = result.words {
                    self.applyAlignedWords(segmentID: segmentID, requestText: text,
                                           words: words, dur: result.dur)
                } else {
                    // A rejected alignment is a NORMAL outcome (the sidecar's own
                    // gates), and its reason is already in logs/aligner.log.
                    self.positionLog("ALIGN none — the sidecar's gates rejected "
                                     + "this chunk; keeping the estimated split")
                }
            } catch {
                self.positionLog("ALIGN FAIL — \(error.localizedDescription)")
            }
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Which segment a late alignment reply belongs to, or nil if it must be
    /// dropped. Pure and static so the two ways it can go wrong are testable
    /// without audio, a sidecar or a session.
    ///
    /// Two rejections, and the second is the subtle one:
    ///  * the segment is GONE — the session was restarted, or the unconfirmed
    ///    sweep at `checkLastChunkDone` removed it;
    ///  * the segment's TEXT CHANGED since the request. `applyRepair` /
    ///    `TranscriptMerge` COMBINE can rewrite a segment's text after Stop, and
    ///    `src` indices are only valid against the exact `text.split()` they were
    ///    computed from. An insertion shifts every later index WITHIN range, so
    ///    no gate inside `WordAttribution` would catch it — the words would land
    ///    silently on the wrong parts of the sentence. Byte-equality here is the
    ///    only check that can see it.
    ///
    /// Matching by UUID (not by window or arrival order) is what makes the whole
    /// path order-independent: two chunks can be in flight and each reply finds
    /// its own segment, or finds that it no longer exists.
    nonisolated static func indexForAlignedWords(segments: [TranscriptSegment],
                                                 id: UUID,
                                                 requestText: String) -> Int? {
        guard let index = segments.firstIndex(where: { $0.id == id }) else { return nil }
        guard segments[index].text == requestText else { return nil }
        return index
    }

    /// Attach a late alignment to its segment and re-render.
    ///
    /// No new rendering code: `derivedRows` already takes the word-exact branch
    /// whenever `seg.words` is non-nil, so this is the same path a synchronous
    /// alignment took — it simply happens a moment later.
    func applyAlignedWords(segmentID: UUID, requestText: String,
                           words: [ChunkedASRService.AlignedWord], dur: Double?) {
        guard let index = Self.indexForAlignedWords(segments: segments, id: segmentID,
                                                    requestText: requestText) else {
            let vanished = !segments.contains { $0.id == segmentID }
            positionLog("ALIGN LATE drop — "
                        + (vanished ? "the segment is gone"
                                    : "the segment's text changed after the request "
                                      + "(src indices no longer valid)"))
            return
        }
        segments[index].words = words
        segments[index].alignedChunkDuration = dur
        rebuildDisplayRows()
    }
}
