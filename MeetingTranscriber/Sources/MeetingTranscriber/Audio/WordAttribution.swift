import Foundation

/// Word-exact speaker attribution: put every word of a chunk in the speaker turn
/// that covers it in TIME, instead of guessing its position from its character
/// offset (what `AudioRecorder.assignSentences` does when no aligner is running).
///
/// Pure value types, no app types beyond the aligner's own payload struct, no
/// actor — unit-testable in isolation, like `PositionClusterer`. The caller keeps
/// ownership of ids/overlap flags: this returns plain per-speaker pieces.
///
/// THE invariant: the returned pieces' text, concatenated and split on
/// whitespace, is EXACTLY the source text split on whitespace — same words, same
/// order. A dropped word loses meeting content, which is worse than a word
/// attributed to the wrong speaker. Everything below is written around that.
enum WordAttribution {

    /// One speaker's contiguous run of words, ready to become a display row.
    struct Piece: Equatable {
        var id: Int
        var name: String
        var start: Double
        var end: Double
        var text: String
    }

    /// A speaker turn clipped to the chunk window, as `derivedRows` builds them
    /// (pyannote turns plus ATND position gap-fills). Same tuple shape as
    /// `assignSentences` takes, so the two paths stay interchangeable.
    typealias SpeakerRange = (start: Double, end: Double, id: Int, name: String)

    /// Returns nil whenever the alignment cannot be trusted — the caller then
    /// falls back to `assignSentences` and behaviour is exactly as before.
    /// Every rejection is reported through `log` so a bad recording is
    /// diagnosable from logs/position-diarization.log.
    ///
    /// - Parameters:
    ///   - text: the chunk's transcript, verbatim.
    ///   - words: the aligner's items. `start`/`end` are CHUNK-relative seconds;
    ///     `text` is the aligner's normalized token and is diagnostic only.
    ///   - chunkDuration: the sidecar's buffer length for that chunk (`dur`).
    ///   - window: the chunk's span on the `recordingElapsed` clock.
    ///   - ranges: speaker turns covering the window, sorted by start.
    static func attribute(text: String,
                          words: [ChunkedASRService.AlignedWord],
                          chunkDuration: Double?,
                          window: ClosedRange<Double>,
                          ranges: [SpeakerRange],
                          log: (String) -> Void = { _ in }) -> [Piece]? {

        func reject(_ reason: String) -> [Piece]? {
            log("WORDS SKIP \(reason) [\(fmt(window.lowerBound))..\(fmt(window.upperBound))]")
            return nil
        }

        // The source words are the ONLY text ever emitted. `src` indexes this
        // array — splitting on runs of whitespace, exactly like Python's
        // `text.split()`. `components(separatedBy: " ")` would yield empty
        // strings on double spaces and desync every later index.
        let sourceWords = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !sourceWords.isEmpty else { return reject("empty text") }
        guard !words.isEmpty else { return reject("no aligned words") }
        guard !ranges.isEmpty else { return reject("no speaker ranges") }

        // The sidecar trims its buffer at MAX_BUFFER_SEC, so a mismatch here
        // means its audio and our window describe different spans — word times
        // would land in the wrong place on the recording clock.
        let span = window.upperBound - window.lowerBound
        if let dur = chunkDuration, abs(dur - span) > 1.0 {
            return reject("dur mismatch dur=\(fmt(dur)) window=\(fmt(span))")
        }

        // Gates on the payload itself: in-bounds, in text order, time-ordered.
        var previousSrc = -1
        var previousStart = -Double.greatestFiniteMagnitude
        for w in words {
            guard w.src >= 0, w.src < sourceWords.count else {
                return reject("src \(w.src) out of range (\(sourceWords.count) words)")
            }
            guard w.src >= previousSrc else {
                return reject("src decreasing (\(previousSrc) -> \(w.src))")
            }
            guard w.end >= w.start else {
                return reject("word end < start (\(fmt(w.start))..\(fmt(w.end)))")
            }
            guard w.start >= previousStart - 1e-6 else {
                return reject("word times non-monotonic at src \(w.src)")
            }
            previousSrc = w.src
            previousStart = w.start
        }

        // 1. Place each ALIGNED source word in the turn it overlaps most.
        var assigned = [Int?](repeating: nil, count: sourceWords.count)   // index into ranges
        var bounds = [(start: Double, end: Double)?](repeating: nil, count: sourceWords.count)
        for w in words {
            let absStart = window.lowerBound + w.start
            let absEnd = window.lowerBound + w.end
            if let existing = bounds[w.src] {
                // Two aligner items on one source word (normalization split it):
                // keep the first speaker, widen the span.
                bounds[w.src] = (min(existing.start, absStart), max(existing.end, absEnd))
                continue
            }
            bounds[w.src] = (absStart, absEnd)
            assigned[w.src] = pick(start: absStart, end: absEnd, ranges: ranges)
        }
        guard assigned.contains(where: { $0 != nil }) else {
            return reject("no source word aligned")
        }

        // 2. Unaligned source words (punctuation-only tokens, and the tail the
        //    sidecar dropped for running past the audio) inherit the PRECEDING
        //    word's speaker; leading ones inherit the first assigned speaker.
        let firstAssigned = assigned.first { $0 != nil }!!
        var carried = firstAssigned
        for i in assigned.indices {
            if let a = assigned[i] { carried = a } else { assigned[i] = carried }
        }

        // 3. Merge consecutive same-speaker words into rows. Walking the source
        //    array in index order is what preserves the text exactly.
        struct Run { var rangeIndex: Int; var words: [String]; var start: Double?; var end: Double? }
        var runs: [Run] = []
        for i in sourceWords.indices {
            let rangeIndex = assigned[i]!
            let id = ranges[rangeIndex].id
            if var last = runs.last, ranges[last.rangeIndex].id == id {
                last.words.append(sourceWords[i])
                if let b = bounds[i] {
                    last.start = last.start.map { min($0, b.start) } ?? b.start
                    last.end = max(last.end ?? b.end, b.end)
                }
                runs[runs.count - 1] = last
            } else {
                let b = bounds[i]
                runs.append(Run(rangeIndex: rangeIndex, words: [sourceWords[i]],
                                start: b?.start, end: b?.end))
            }
        }

        // 4. Times: a row's span comes from its first/last aligned word. A row
        //    with no aligned word at all takes what its neighbours leave it,
        //    falling back to the window — never NaN, never inverted.
        var pieces: [Piece] = []
        for (i, run) in runs.enumerated() {
            let previousEnd = pieces.last?.end ?? window.lowerBound
            let nextStart = runs[(i + 1)...].compactMap(\.start).first ?? window.upperBound
            var start = run.start ?? previousEnd
            var end = run.end ?? max(previousEnd, nextStart)
            start = min(max(start, window.lowerBound), window.upperBound)
            end = min(max(end, start), window.upperBound)
            let r = ranges[run.rangeIndex]
            pieces.append(Piece(id: r.id, name: r.name, start: start, end: end,
                                text: run.words.joined(separator: " ")))
        }
        // Success is logged too, not just the SKIP gates: without this, an
        // absent SKIP line is ambiguous between "the word path ran" and "the
        // aligner never sent words at all", which makes an on-device run
        // impossible to judge from the log.
        log("WORDS OK \(words.count)/\(sourceWords.count) aligned → \(pieces.count) row(s) "
            + "[\(fmt(window.lowerBound))..\(fmt(window.upperBound))] "
            + pieces.map { "\($0.name)@\(fmt($0.start))" }.joined(separator: " "))
        return pieces
    }

    // MARK: - Helpers

    /// The turn `[start, end]` overlaps most. Ties go to the turn containing the
    /// word's midpoint, then to the earlier turn. No overlap with any turn →
    /// nearest by midpoint distance, identical to `assignSentences`.
    private static func pick(start: Double, end: Double, ranges: [SpeakerRange]) -> Int {
        var best: [Int] = []
        var bestOverlap = -1.0
        for i in ranges.indices {
            let ov = max(0, min(ranges[i].end, end) - max(ranges[i].start, start))
            if ov > bestOverlap + 1e-9 {
                bestOverlap = ov
                best = [i]
            } else if abs(ov - bestOverlap) <= 1e-9 {
                best.append(i)
            }
        }

        let mid = (start + end) / 2
        if bestOverlap <= 0 {
            return ranges.indices.min {
                abs((ranges[$0].start + ranges[$0].end) / 2 - mid)
                    < abs((ranges[$1].start + ranges[$1].end) / 2 - mid)
            } ?? 0
        }
        if best.count == 1 { return best[0] }
        if let containing = best.first(where: { ranges[$0].start <= mid && mid <= ranges[$0].end }) {
            return containing
        }
        return best.min { ranges[$0].start < ranges[$1].start } ?? best[0]
    }

    private static func fmt(_ v: Double) -> String { String(format: "%.3f", v) }
}
