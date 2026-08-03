import XCTest
@testable import MeetingTranscriber

/// `WordAttribution.snapRanges` — moving a speaker-change boundary onto a nearby
/// inter-word pause, and (mostly) refusing to.
///
/// Every number below that is called "measured" comes from ONE 75-second
/// recording (`recordings/meeting-2026-07-28T03-13-37Z.wav`, 3 speakers, Whisper
/// + the Qwen3 aligner, 237 aligned words, real pyannote turns). The negative
/// tests outnumber the positive one on purpose: a wrong snap silently moves
/// correct words onto another speaker, which is the failure nobody sees.
final class BoundarySnapTests: XCTestCase {

    // MARK: - Helpers

    private func ranges(_ items: [(Double, Double, Int, String)])
        -> [WordAttribution.SpeakerRange] {
        items.map { (start: $0.0, end: $0.1, id: $0.2, name: $0.3) }
    }

    private func gaps(_ items: [(Double, Double)]) -> [(start: Double, end: Double)] {
        items.map { (start: $0.0, end: $0.1) }
    }

    /// Runs the snap and captures the log lines it emitted.
    private func snap(_ r: [WordAttribution.SpeakerRange],
                      _ g: [(start: Double, end: Double)],
                      window: ClosedRange<Double>)
        -> (out: [WordAttribution.SpeakerRange], log: String) {
        var lines: [String] = []
        let out = WordAttribution.snapRanges(r, gaps: g, window: window,
                                             log: { lines.append($0) })
        return (out, lines.joined(separator: "\n"))
    }

    private func assertUnchanged(_ before: [WordAttribution.SpeakerRange],
                                 _ after: [WordAttribution.SpeakerRange],
                                 file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(after.count, before.count, file: file, line: line)
        for (b, a) in zip(before, after) {
            XCTAssertEqual(a.start, b.start, accuracy: 1e-9, file: file, line: line)
            XCTAssertEqual(a.end, b.end, accuracy: 1e-9, file: file, line: line)
            XCTAssertEqual(a.id, b.id, file: file, line: line)
        }
    }

    // MARK: - 1. The owner's symptom, with the real measured numbers

    /// pyannote put Speaker 3's turn at 39.552, which is 0.472s LATE and lands in
    /// the middle of a word; the real pause is 39.080–39.320. The word at 39.32
    /// and the whole 39.552–39.923 stripe were printed under Speaker 2.
    func testMeasuredLateBoundarySnapsBackOntoThePause() {
        let before = ranges([(19.387, 39.923, 2, "Speaker 2"),
                             (39.552, 73.707, 3, "Speaker 3")])
        let r = snap(before, gaps([(39.080, 39.320),   // the real pause, 0.240s
                                   (39.500, 39.600),   // ordinary word spacing
                                   (39.800, 39.900)]),
                     window: 19.0...74.0)

        XCTAssertEqual(r.out[0].end, 39.200, accuracy: 1e-9, "Speaker 2 trimmed to the pause midpoint")
        XCTAssertEqual(r.out[1].start, 39.200, accuracy: 1e-9)
        XCTAssertEqual(r.out[0].start, 19.387, accuracy: 1e-9, "starts are untouched")
        XCTAssertEqual(r.out[1].end, 73.707, accuracy: 1e-9)
        XCTAssertLessThan(r.out[0].start, r.out[0].end, "no inverted span")
        XCTAssertLessThan(r.out[1].start, r.out[1].end)
        // The decision log is the only place a snap is visible after the fact,
        // so its exact shape is pinned, not just its presence.
        XCTAssertEqual(r.log, "SNAP OK 39.552 id2→id3 → 39.200 "
            + "(gap 39.080..39.320 0.240s, offset -0.352, prevEnd 39.923→39.200)")
    }

    /// End to end through `attribute`: the words that used to be printed under
    /// Speaker 2 come out under Speaker 3, and no text is lost.
    func testOwnerSymptomThroughAttribute() {
        // Window 19.0–74.0; aligner times are chunk-relative, so subtract 19.0.
        func word(_ text: String, _ absStart: Double, _ absEnd: Double, _ src: Int)
            -> ChunkedASRService.AlignedWord {
            ChunkedASRService.AlignedWord(text: text, start: absStart - 19.0,
                                          end: absEnd - 19.0, src: src)
        }
        let text = "alpha bravo charlie delta"
        let words = [word("alpha",   38.500, 39.080, 0),   // clearly Speaker 2
                     word("bravo",   39.320, 39.500, 1),   // after the pause, before 39.552
                     word("charlie", 39.600, 39.800, 2),   // inside the old overlap stripe
                     word("delta",   39.900, 40.600, 3)]
        let turns = ranges([(19.387, 39.923, 2, "Speaker 2"),
                            (39.552, 73.707, 3, "Speaker 3")])

        let pieces = WordAttribution.attribute(text: text, words: words, chunkDuration: 55.0,
                                               window: 19.0...74.0, ranges: turns)
        XCTAssertEqual(pieces?.map(\.text), ["alpha", "bravo charlie delta"])
        XCTAssertEqual(pieces?.map(\.id), [2, 3])
        // THE invariant still holds: same words, same order.
        XCTAssertEqual(pieces?.map(\.text).joined(separator: " "), text)
        for p in pieces ?? [] {
            XCTAssertFalse(p.start.isNaN || p.end.isNaN)
            XCTAssertLessThanOrEqual(p.start, p.end)
        }
    }

    /// Without the snap the same input puts "bravo" and "charlie" under Speaker 2
    /// — this is the control that proves the test above measures the fix.
    func testUnsnappedRangesWouldMisattributeTheSameWords() {
        func word(_ text: String, _ absStart: Double, _ absEnd: Double, _ src: Int)
            -> ChunkedASRService.AlignedWord {
            ChunkedASRService.AlignedWord(text: text, start: absStart - 19.0,
                                          end: absEnd - 19.0, src: src)
        }
        // Same words, but a pause too short to qualify → no snap happens.
        let words = [word("alpha",   38.500, 39.080, 0),
                     word("bravo",   39.180, 39.500, 1),
                     word("charlie", 39.600, 39.800, 2),
                     word("delta",   39.900, 40.600, 3)]
        let turns = ranges([(19.387, 39.923, 2, "Speaker 2"),
                            (39.552, 73.707, 3, "Speaker 3")])
        let pieces = WordAttribution.attribute(text: "alpha bravo charlie delta", words: words,
                                               chunkDuration: 55.0, window: 19.0...74.0,
                                               ranges: turns)
        XCTAssertEqual(pieces?.map(\.id), [2, 3])
        XCTAssertEqual(pieces?.map(\.text), ["alpha bravo charlie", "delta"])
    }

    // MARK: - 2. Guard 1 — no qualifying pause

    func testNoGapsAtAllLeavesRangesAlone() {
        let before = ranges([(5.0, 20.0, 1, "S1"), (20.0, 35.0, 2, "S2")])
        let r = snap(before, gaps([]), window: 0...40)
        assertUnchanged(before, r.out)
        XCTAssertTrue(r.log.contains("no pause"), r.log)
        XCTAssertTrue(r.log.contains("no inter-word gaps"), r.log)
    }

    /// 0.15s is ordinary word spacing (measured median 0.160s), not a pause.
    func testShortGapWithinToleranceIsRefused() {
        let before = ranges([(5.0, 20.0, 1, "S1"), (20.0, 35.0, 2, "S2")])
        let r = snap(before, gaps([(20.150, 20.300)]), window: 0...40)
        assertUnchanged(before, r.out)
        XCTAssertTrue(r.log.contains("no pause >=0.2s within 0.5s"), r.log)
        XCTAssertTrue(r.log.contains("best 0.150s"), r.log)
    }

    /// A genuine pause that sits past the tolerance is a logged near miss, not a
    /// snap: 0.501s away is exactly the case the constant exists to refuse.
    func testQualifyingPauseBeyondToleranceIsRefused() {
        let before = ranges([(5.0, 20.0, 1, "S1"), (20.0, 35.0, 2, "S2")])
        let r = snap(before, gaps([(20.6, 21.2)]), window: 0...40)
        assertUnchanged(before, r.out)
        XCTAssertTrue(r.log.contains("no pause"), r.log)
    }

    // MARK: - 3. Guard 2 — the most important negative case

    /// The boundary already sits inside a genuine pause, so no word straddles it
    /// and moving it could only move CORRECT words. A larger pause elsewhere in
    /// tolerance must not tempt it. (The measured 19.387 boundary is this case.)
    func testBoundaryAlreadyInsideAPauseIsANoopEvenWithABiggerPauseNearby() {
        let before = ranges([(5.0, 19.500, 1, "S1"), (19.387, 30.0, 2, "S2")])
        let r = snap(before, gaps([(19.280, 19.760),    // 0.480s, contains 19.387
                                   (19.800, 20.450)]),  // 0.650s, only 0.413s away
                     window: 0...40)
        assertUnchanged(before, r.out)
        XCTAssertTrue(r.log.contains("already inside pause 19.280..19.760"), r.log)
    }

    // MARK: - 4. Which pause wins

    func testLargestPauseWinsNotTheNearest() {
        let before = ranges([(5.0, 20.2, 1, "S1"), (20.0, 35.0, 2, "S2")])
        let r = snap(before, gaps([(19.700, 19.950),    // 0.250s, 0.050s away
                                   (20.100, 20.500)]),  // 0.400s, 0.100s away
                     window: 0...40)
        XCTAssertEqual(r.out[1].start, 20.300, accuracy: 1e-9, "midpoint of the LARGER pause")
        XCTAssertEqual(r.out[0].end, 20.200, accuracy: 1e-9, "min(X.end, P) — never extended")
        XCTAssertTrue(r.log.contains("SNAP OK"), r.log)
    }

    // MARK: - 5. Guard 3 — slivers are absorbed downstream, never snapped

    /// The measured A-B-A-B slivers: 18.863..19.100 (0.237s) and
    /// 19.100..19.387 (0.287s). All three boundaries around them are refused,
    /// even with a fat pause sitting right there.
    func testSliverAdjacentBoundariesAreAllRefused() {
        let before = ranges([(1.735, 18.863, 1, "S1"),
                             (18.863, 19.100, 2, "S2"),
                             (19.100, 19.387, 1, "S1"),
                             (19.387, 39.923, 2, "S2")])
        let r = snap(before, gaps([(18.500, 19.000),
                                   (19.150, 19.700)]),
                     window: 0...45)
        assertUnchanged(before, r.out)
        XCTAssertEqual(r.log.components(separatedBy: "adjacent turn").count - 1, 3, r.log)
        XCTAssertTrue(r.log.contains("0.237s < 0.5s"), r.log)
        XCTAssertTrue(r.log.contains("0.287s < 0.5s"), r.log)
    }

    // MARK: - 6. Guard 4 — genuine overlap

    /// 0.5s of overlap means they really did both speak; collapsing that to a
    /// point would invent a clean handover. (The measured 0.371s overlap sits
    /// under the bar, which is why the case in test 1 IS snappable.)
    func testGenuineOverlapIsRefused() {
        let before = ranges([(5.0, 20.5, 1, "S1"), (20.0, 35.0, 2, "S2")])
        let r = snap(before, gaps([(19.800, 20.300)]), window: 0...40)
        assertUnchanged(before, r.out)
        XCTAssertTrue(r.log.contains("genuine overlap 0.500s >= 0.4s"), r.log)
    }

    func testSubThresholdOverlapIsStillSnappable() {
        // 0.371s of overlap — exactly the measured figure at 39.552.
        let before = ranges([(5.0, 20.371, 1, "S1"), (20.0, 35.0, 2, "S2")])
        let r = snap(before, gaps([(19.400, 19.800)]), window: 0...40)
        XCTAssertEqual(r.out[1].start, 19.600, accuracy: 1e-9)
        XCTAssertEqual(r.out[0].end, 19.600, accuracy: 1e-9, "the overlap collapses to the point P")
        XCTAssertLessThan(r.out[0].start, r.out[0].end)
        // Trim is bounded by overlap (< 0.4) + tolerance (0.5).
        XCTAssertLessThan(before[0].end - r.out[0].end, 0.9)
    }

    // MARK: - 7. Guard 8 — chunk window edges

    /// Ranges are clipped per chunk, so near an edge the pause picture is
    /// truncated and the "nearest pause" reading is not trustworthy.
    func testBoundaryNearTheWindowEdgeIsRefused() {
        let before = ranges([(19.2, 19.500, 1, "S1"), (19.387, 45.0, 2, "S2")])
        let r = snap(before, gaps([(19.100, 19.600)]), window: 19.2...50.0)
        assertUnchanged(before, r.out)
        XCTAssertTrue(r.log.contains("near window edge"), r.log)

        let atEnd = ranges([(5.0, 49.9, 1, "S1"), (49.8, 50.0, 2, "S2")])
        let r2 = snap(atEnd, gaps([(49.500, 50.000)]), window: 0...50.0)
        assertUnchanged(atEnd, r2.out)
        XCTAssertTrue(r2.log.contains("near window edge"), r2.log)
    }

    // MARK: - 8. Structural refusals

    func testNestedTurnIsRefused() {
        let before = ranges([(5.0, 35.0, 1, "S1"), (20.0, 30.0, 2, "S2")])
        let r = snap(before, gaps([(19.700, 20.100)]), window: 0...40)
        assertUnchanged(before, r.out)
        XCTAssertTrue(r.log.contains("nested turn"), r.log)
    }

    func testSameSpeakerBoundaryIsNotASpeakerChange() {
        let before = ranges([(5.0, 20.0, 1, "S1"), (20.0, 35.0, 1, "S1")])
        let r = snap(before, gaps([(19.700, 20.300)]), window: 0...40)
        assertUnchanged(before, r.out)
        XCTAssertTrue(r.log.isEmpty, "nothing to decide, so nothing to log")
    }

    /// Three speakers, one fat pause sitting near BOTH boundaries. The first pair
    /// takes it; the second must not also take it and fold S2 down to nothing.
    /// Pairs are evaluated against already-snapped values, so the shortened S2 is
    /// what the second pair's guards see.
    func testASnappedBoundaryNeverCrossesTheOneBeforeIt() {
        let before = ranges([(5.0, 20.0, 1, "S1"),
                             (20.0, 20.9, 2, "S2"),
                             (20.9, 35.0, 3, "S3")])
        let r = snap(before, gaps([(20.300, 20.800)]), window: 0...40)
        XCTAssertEqual(r.out[1].start, 20.550, accuracy: 1e-9, "first boundary snapped")
        XCTAssertEqual(r.out[2].start, 20.900, accuracy: 1e-9, "second boundary refused")
        XCTAssertLessThan(r.out[1].start, r.out[1].end, "S2 still has a span")
        // Boundaries stay strictly ordered — nothing crossed, nothing inverted.
        for i in 0..<(r.out.count - 1) {
            XCTAssertLessThanOrEqual(r.out[i].start, r.out[i + 1].start)
            XCTAssertLessThanOrEqual(r.out[i].start, r.out[i].end)
        }
        XCTAssertTrue(r.log.contains("SNAP SKIP"), r.log)
    }

    /// Guard 7 on its own, because it was once believed to be unreachable behind
    /// guard 6 — and it is not. Guard 6 is the stronger test only while the pair
    /// before this one snapped, since then `x.start` IS the previous boundary. A
    /// SAME-ID pair takes the early `continue` without touching anything, and the
    /// `x` that reaches the next pair still starts BEHIND the earlier snap.
    ///
    /// Here the second transition's own best pause is the pause the first
    /// transition already consumed. Guard 6 passes with 0.065s to spare
    /// (5.865 > 5.800); only guard 7 refuses. Delete it and the boundary walks
    /// backwards onto a pause that is already spoken for.
    func testBoundaryMayNotCrossTheOneSnappedBeforeIt() {
        let before = ranges([(2.0, 6.0, 1, "S1"),
                             (5.7, 6.3, 2, "S2"),
                             (5.8, 6.5, 2, "S2"),    // same id → skipped outright
                             (6.4, 21.0, 3, "S3")])
        let r = snap(before, gaps([(5.750, 5.980)]), window: 0...25)

        XCTAssertEqual(r.out[1].start, 5.865, accuracy: 1e-9, "first boundary took the pause")
        XCTAssertEqual(r.out[3].start, 6.400, accuracy: 1e-9, "second boundary refused, unmoved")
        XCTAssertEqual(r.out[2].end, 6.500, accuracy: 1e-9, "and its turn was not trimmed")

        // The refusal is guard 7's, not guard 3's or guard 6's — assert the
        // reason, or the test would still pass if the guard were removed and
        // some other guard happened to fire.
        XCTAssertTrue(r.log.contains("not past previous boundary"), r.log)
        // Guard 6's condition genuinely held: 5.865 > out[2].start (5.800).
        XCTAssertGreaterThan(5.865, r.out[2].start, "guard 6 alone would have allowed it")
    }

    // MARK: - 9. Nothing to do

    func testSingleRangeIsReturnedUntouched() {
        let before = ranges([(5.0, 20.0, 1, "S1")])
        let r = snap(before, gaps([(10.0, 11.0)]), window: 0...40)
        assertUnchanged(before, r.out)
    }
}
