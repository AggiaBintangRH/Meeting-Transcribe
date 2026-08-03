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

    // MARK: - Boundary snapping constants
    //
    // ALL FOUR ARE DERIVED FROM ONE 75-SECOND RECORDING
    // (`recordings/meeting-2026-07-28T03-13-37Z.wav`, 3 speakers, Whisper + the
    // Qwen3 aligner, 237 aligned words, real pyannote turns). Measured there:
    // per speaker-change boundary, |offset| to the nearest inter-word gap had
    // median 0.263s, p90 0.472s, max 0.472s, and 3 of 4 boundaries were LATE;
    // inter-word gaps overall had median 0.160s, max 0.480s.
    //
    // REVISIT AFTER MULTIPLE MEETINGS. One recording is exactly the sample size
    // that produces a constant which fits it and nothing else.

    /// How far a speaker-change boundary may be from a pause and still snap to
    /// it, measured from the boundary to the pause's NEAREST EDGE, inclusive.
    /// 0.5 just clears the measured max offset of 0.472s; at 0.501 the boundary
    /// is left alone and the near miss is logged.
    static let snapToleranceSec = 0.5

    /// A silence must be at least this long to count as a genuine pause. The
    /// median inter-word gap is 0.160s, so anything under ~0.2s is ordinary word
    /// spacing; 200ms is also the conventional bar. It must stay ≤ 0.240s or the
    /// owner's own symptom case (the 39.080–39.320 pause) stops being fixable.
    static let snapMinPauseSec = 0.2

    /// Either adjacent turn shorter than this → no snap. Protects the 0.237s and
    /// 0.287s slivers measured at 18.863/19.100: those are absorbed by
    /// `absorbShortSpeakerIslands` downstream, not repaired by moving times.
    static let snapMinTurnSec = 0.5

    /// Two turns overlapping by at least this much genuinely both spoke, so the
    /// boundary between them is not a mis-timed switch and must not be collapsed.
    /// Deliberately the SAME 0.4s the codebase already uses to define a genuine
    /// overlap (`overlapRegions()`, `repairWindows`). The measured 0.371s overlap
    /// at 39.552 sits below it, which is precisely why that case is snappable.
    static let snapGenuineOverlapSec = 0.4

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

        // 1a. Absolute span of each ALIGNED source word. Two aligner items on one
        //     source word (normalization split it) widen the span; `firstSpan`
        //     keeps the FIRST item's span, which is what decides the speaker —
        //     unchanged from before the split into 1a/1b/1c.
        var bounds = [(start: Double, end: Double)?](repeating: nil, count: sourceWords.count)
        var firstSpan = [(start: Double, end: Double)?](repeating: nil, count: sourceWords.count)
        for w in words {
            let absStart = window.lowerBound + w.start
            let absEnd = window.lowerBound + w.end
            if let existing = bounds[w.src] {
                bounds[w.src] = (min(existing.start, absStart), max(existing.end, absEnd))
                continue
            }
            bounds[w.src] = (absStart, absEnd)
            firstSpan[w.src] = (absStart, absEnd)
        }
        guard bounds.contains(where: { $0 != nil }) else {
            return reject("no source word aligned")
        }

        // 1b. Interior inter-word pauses, from the DEDUPED spans in src order —
        //     a normalization-split word must not be able to fabricate a gap
        //     between its own two halves.
        var gaps: [(start: Double, end: Double)] = []
        var runningEnd: Double? = nil
        for b in bounds.compactMap({ $0 }) {
            if let prev = runningEnd, b.start - prev > 0 { gaps.append((prev, b.start)) }
            runningEnd = max(runningEnd ?? b.end, b.end)
        }

        // 1c. Snap speaker-change boundaries onto those pauses, then place every
        //     word against the SNAPPED ranges. Diarization boundaries land mid-word
        //     (measured: 3 of 4 late, up to 0.472s), which hands the first word of
        //     a new turn to the previous speaker.
        let snapped = snapRanges(ranges, gaps: gaps, window: window, log: log)

        var assigned = [Int?](repeating: nil, count: sourceWords.count)   // index into `snapped`
        for i in sourceWords.indices {
            guard let s = firstSpan[i] else { continue }
            assigned[i] = pick(start: s.start, end: s.end, ranges: snapped)
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
            let id = snapped[rangeIndex].id
            if var last = runs.last, snapped[last.rangeIndex].id == id {
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
            let r = snapped[run.rangeIndex]
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

    // MARK: - Boundary snapping

    /// Move each speaker-change boundary onto a nearby inter-word pause, so a
    /// diarization boundary that landed mid-word stops handing the first word of
    /// a new turn to the previous speaker.
    ///
    /// PURE, and strictly local to word attribution: it returns a COPY. The
    /// snapped times never reach `liveTurns`, so `overlapRegions()`, the repair
    /// windows and the rows' overlap tagging are all unaffected — this can only
    /// change which speaker a word is printed under, never whether a region is
    /// flagged as overlapped.
    ///
    /// A snap is `Y.start = P` AND `X.end = min(X.end, P)`. Trimming `X` is
    /// REQUIRED, not cosmetic: `pick()` breaks a full-containment tie by "earlier
    /// turn", so moving only `Y.start` would leave the word tied between the two
    /// and it would still resolve to `X`.
    ///
    /// `P` is the TEMPORAL MIDPOINT of the chosen pause — equidistant from the
    /// words on either side, so a small aligner timing error cannot flip a word
    /// across it.
    ///
    /// The dangerous direction here is a wrong snap, which silently moves correct
    /// words onto another speaker. So the rule is inert unless the evidence is
    /// strong, and EVERY decision — snap or skip — is logged with its reason.
    ///
    /// - Parameters:
    ///   - ranges: speaker turns clipped to the chunk window, sorted by start.
    ///   - gaps: interior inter-word pauses on the recording clock, in order.
    ///   - window: the chunk's span, used only for the window-edge guard.
    static func snapRanges(_ ranges: [SpeakerRange],
                           gaps: [(start: Double, end: Double)],
                           window: ClosedRange<Double>,
                           log: (String) -> Void = { _ in }) -> [SpeakerRange] {
        guard ranges.count > 1 else { return ranges }
        var out = ranges

        /// Distance from `b` to a pause's nearest edge; 0 when `b` is inside it.
        func edgeDistance(_ g: (start: Double, end: Double), _ b: Double) -> Double {
            b < g.start ? g.start - b : (b > g.end ? b - g.end : 0)
        }

        let qualifying = gaps.filter { $0.end - $0.start >= snapMinPauseSec }
        var previousSnapped = -Double.greatestFiniteMagnitude

        for i in 0..<(out.count - 1) {
            // Read through `out`, not `ranges`: pair i-1 may already have moved
            // this turn's start, and the guards below must see the real values.
            let x = out[i], y = out[i + 1]
            guard x.id != y.id else { continue }
            let b = y.start
            let tag = "\(fmt(b)) id\(x.id)→id\(y.id)"

            // Guard 8: the ranges are clipped per chunk, so within a tolerance of
            // either edge the pause picture is truncated and cannot be trusted.
            if b - window.lowerBound < snapToleranceSec
                || window.upperBound - b < snapToleranceSec {
                log("SNAP SKIP \(tag) near window edge")
                continue
            }

            // Guard 3: slivers. A 0.237s turn is a diarization flicker, not a
            // boundary worth aligning — `absorbShortSpeakerIslands` handles those.
            let xDur = x.end - x.start, yDur = y.end - y.start
            if xDur < snapMinTurnSec || yDur < snapMinTurnSec {
                log("SNAP SKIP \(tag) adjacent turn \(fmt(min(xDur, yDur)))s < \(snapMinTurnSec)s")
                continue
            }

            // Guard 5: nesting — `y` sits inside `x`, so `y.start` is not the
            // boundary between two consecutive speakers at all.
            if x.end > y.end {
                log("SNAP SKIP \(tag) nested turn (\(fmt(x.start))..\(fmt(x.end)) contains "
                    + "\(fmt(y.start))..\(fmt(y.end)))")
                continue
            }

            // Guard 4: they really did both speak. Collapsing that to a point
            // would be inventing a clean handover that did not happen.
            let overlap = x.end - y.start
            if overlap >= snapGenuineOverlapSec {
                log("SNAP SKIP \(tag) genuine overlap \(fmt(overlap))s >= \(snapGenuineOverlapSec)s")
                continue
            }

            // Guard 2: the boundary is ALREADY inside a genuine pause — no word
            // straddles it, so moving it could only move correct words. This wins
            // even when a bigger pause exists within tolerance.
            if let inside = qualifying.first(where: { $0.start <= b && b <= $0.end }) {
                log("SNAP SKIP \(tag) already inside pause \(fmt(inside.start))..\(fmt(inside.end))")
                continue
            }

            // Guard 1: no genuine pause close enough. The common case by design.
            let candidates = qualifying.filter { edgeDistance($0, b) <= snapToleranceSec + 1e-9 }
            guard !candidates.isEmpty else {
                let near = gaps.filter { edgeDistance($0, b) <= snapToleranceSec + 1e-9 }
                    .max { ($0.end - $0.start) < ($1.end - $1.start) }
                    ?? gaps.min { edgeDistance($0, b) < edgeDistance($1, b) }
                let best = near.map { g in
                    "(best \(fmt(g.end - g.start))s@\(offsetString((g.start + g.end) / 2 - b)))"
                } ?? "(no inter-word gaps)"
                log("SNAP SKIP \(tag) no pause >=\(snapMinPauseSec)s within \(snapToleranceSec)s \(best)")
                continue
            }

            // LARGEST pause within tolerance, ties broken by nearest to `b`. The
            // median gap is 0.160s, so "nearest qualifying" is often just ordinary
            // word spacing that scraped past 0.2s; a speaker switch coincides with
            // the most prominent nearby silence, not merely the closest one.
            let chosen = candidates.min { a, c in
                let da = a.end - a.start, dc = c.end - c.start
                if abs(da - dc) > 1e-9 { return da > dc }
                return edgeDistance(a, b) < edgeDistance(c, b)
            }!
            let p = (chosen.start + chosen.end) / 2

            // Guard 6: range sanity — never invert a turn.
            guard p > x.start, p < y.end else {
                log("SNAP SKIP \(tag) pause midpoint \(fmt(p)) outside "
                    + "(\(fmt(x.start))..\(fmt(y.end)))")
                continue
            }
            // Guard 7: pairs are processed left to right; a boundary may never
            // cross the one already snapped before it.
            //
            // NOT redundant with guard 6, and do not delete it as such. Guard 6
            // is the stronger test only while the PREVIOUS pair snapped, because
            // then `x.start` IS `previousSnapped`. A same-id pair takes the early
            // `continue` above without touching anything, and that leaves an `x`
            // whose `.start` still sits BEHIND an earlier snap — guard 6 then
            // passes with real margin while this one correctly refuses.
            // `BoundarySnapTests.testBoundaryMayNotCrossTheOneSnappedBeforeIt`
            // pins exactly that shape; without this guard it produces
            // `out[i+1].start < out[i].start`, i.e. two ranges out of temporal
            // order, which every later step assumes cannot happen.
            guard p > previousSnapped else {
                log("SNAP SKIP \(tag) pause midpoint \(fmt(p)) not past previous boundary "
                    + "\(fmt(previousSnapped))")
                continue
            }

            let previousEnd = x.end
            out[i].end = min(x.end, p)
            out[i + 1].start = p
            previousSnapped = p
            log("SNAP OK \(tag) → \(fmt(p)) (gap \(fmt(chosen.start))..\(fmt(chosen.end)) "
                + "\(fmt(chosen.end - chosen.start))s, offset \(offsetString(p - b)), "
                + "prevEnd \(fmt(previousEnd))→\(fmt(out[i].end)))")
        }
        return out
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

    /// Signed offset, always with its sign — the log's whole point is showing
    /// whether boundaries run late or early.
    private static func offsetString(_ v: Double) -> String { String(format: "%+.3f", v) }
}
