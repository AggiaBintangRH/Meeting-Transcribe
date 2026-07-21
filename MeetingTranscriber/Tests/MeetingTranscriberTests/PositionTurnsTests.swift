import XCTest
import simd
@testable import MeetingTranscriber

/// Unit tests for `PositionClusterer.turns(in:)` and `ClusterChangeDetector`
/// (real-time speaker-split building blocks). Pure value types — no app state.
final class PositionTurnsTests: XCTestCase {

    // MARK: - Helpers

    private func vec(rotate: Double, angle: Double = 0) -> SIMD3<Double> {
        PositionMath.unitVector(rotateDeg: rotate, angleDeg: angle)
    }

    /// Feed one direction per 0.1s (10 Hz) over `[start, end)` into the clusterer.
    private func feed(_ c: inout PositionClusterer, rotate: Double,
                      start: Double, end: Double) {
        var t = start
        while t < end - 1e-9 {
            _ = c.assign(DirectionSample(t: t, vector: vec(rotate: rotate)))
            t += 0.1
        }
    }

    // MARK: - 1. Two clusters split at the boundary

    func testTwoClustersSplitAtBoundary() {
        var c = PositionClusterer(tauDeg: 15)
        feed(&c, rotate: 0, start: 0.0, end: 3.0)    // cluster A (id 0)
        feed(&c, rotate: 90, start: 3.0, end: 6.0)   // cluster B (id 1)

        let turns = c.turns(in: 0.0...6.0)
        XCTAssertEqual(turns.count, 2)
        XCTAssertEqual(turns[0].clusterID, 0)
        XCTAssertEqual(turns[1].clusterID, 1)
        // Split lands at ~3s: A ends just before 3.0, B starts at 3.0.
        XCTAssertEqual(turns[0].end, 2.9, accuracy: 0.05)
        XCTAssertEqual(turns[1].start, 3.0, accuracy: 0.05)
    }

    // MARK: - 2. Flicker inside a stable turn is dropped and flanks merge

    func testFlickerIsAbsorbedIntoOneTurn() {
        var c = PositionClusterer(tauDeg: 15)
        // A from 0..3s, but two stray B samples at 2.0 and 2.1.
        var t = 0.0
        while t < 3.0 - 1e-9 {
            let isStray = (abs(t - 2.0) < 1e-9 || abs(t - 2.1) < 1e-9)
            _ = c.assign(DirectionSample(t: t, vector: vec(rotate: isStray ? 90 : 0)))
            t += 0.1
        }

        let turns = c.turns(in: 0.0...3.0)
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].clusterID, 0)
        XCTAssertEqual(turns[0].start, 0.0, accuracy: 0.05)
        XCTAssertEqual(turns[0].end, 2.9, accuracy: 0.05)
    }

    // MARK: - 3. A silence hole larger than maxGap is never bridged

    func testGapBreaksIntoTwoTurns() {
        var c = PositionClusterer(tauDeg: 15)
        feed(&c, rotate: 0, start: 0.0, end: 2.0)    // A
        // nothing 2..4s (ATND silence)
        feed(&c, rotate: 0, start: 4.0, end: 6.0)    // A again, same direction

        let turns = c.turns(in: 0.0...6.0)
        XCTAssertEqual(turns.count, 2)               // same cluster, but the hole splits it
        XCTAssertEqual(turns[0].clusterID, 0)
        XCTAssertEqual(turns[1].clusterID, 0)
        XCTAssertLessThan(turns[0].end, 2.0)
        XCTAssertGreaterThanOrEqual(turns[1].start, 4.0)
    }

    // MARK: - 4. Sub-min run at a genuine boundary is unlabeled slop

    func testSubMinRunBetweenClustersIsDropped() {
        var c = PositionClusterer(tauDeg: 15)
        feed(&c, rotate: 0, start: 0.0, end: 3.0)    // A (3s)
        feed(&c, rotate: 90, start: 3.0, end: 3.3)   // B (0.3s < minDuration)
        feed(&c, rotate: 180, start: 3.3, end: 6.3)  // C (3s)

        let turns = c.turns(in: 0.0...6.3)
        XCTAssertEqual(turns.count, 2)               // B dropped, A and C differ → no merge
        XCTAssertEqual(turns[0].clusterID, 0)        // A
        XCTAssertEqual(turns[1].clusterID, 2)        // C
    }

    // MARK: - 5. ClusterChangeDetector confirmation, rate-limit, first-cluster

    func testClusterChangeDetector() {
        var d = ClusterChangeDetector()   // consecutiveRequired 3, minIntervalSec 0.6

        // First stable cluster never fires.
        XCTAssertNil(d.push(t: 0.0, clusterID: 0))
        // Two consecutive of a new cluster: not yet.
        XCTAssertNil(d.push(t: 0.1, clusterID: 1))
        XCTAssertNil(d.push(t: 0.2, clusterID: 1))
        // Third consecutive → fire.
        XCTAssertNotNil(d.push(t: 0.3, clusterID: 1))

        // A rapid second change is suppressed by minIntervalSec (t - 0.3 < 0.6).
        XCTAssertNil(d.push(t: 0.4, clusterID: 2))
        XCTAssertNil(d.push(t: 0.5, clusterID: 2))
        XCTAssertNil(d.push(t: 0.6, clusterID: 2))     // confirmed but rate-limited

        // Once the interval passes, the still-building candidate fires.
        XCTAssertNotNil(d.push(t: 1.0, clusterID: 2))
    }

    func testForceChangeFiresImmediatelyExceptFirst() {
        var d = ClusterChangeDetector()

        // The very first speaker establishes silently — nothing to split from.
        XCTAssertNil(d.forceChange(to: 0, at: 0.0))
        // A brand-new direction fires immediately, no confirmation, no rate limit.
        XCTAssertNotNil(d.forceChange(to: 1, at: 1.0))
        // Another new direction ~1 s later still fires immediately — this is the
        // case that the 0.6 s push() rate-limit alone would have swallowed.
        XCTAssertNotNil(d.forceChange(to: 2, at: 2.0))
        // Even back-to-back (no debounce on a genuinely new cluster).
        XCTAssertNotNil(d.forceChange(to: 3, at: 2.1))
        // After a force to cluster 2, staying on 2 via push() does not re-fire.
        _ = d.forceChange(to: 2, at: 3.0)
        XCTAssertNil(d.push(t: 3.1, clusterID: 2))
    }

    func testDetectorResetClearsStableCluster() {
        var d = ClusterChangeDetector()
        XCTAssertNil(d.push(t: 0.0, clusterID: 5))     // establishes stable = 5
        d.reset()
        // After reset the next cluster is a fresh first-stable → never fires.
        XCTAssertNil(d.push(t: 1.0, clusterID: 7))
        XCTAssertNil(d.push(t: 1.1, clusterID: 7))
    }

    // MARK: - 6. Assignments outside the range are ignored

    func testRangeClipping() {
        var c = PositionClusterer(tauDeg: 15)
        feed(&c, rotate: 0, start: 0.0, end: 2.0)    // A, before the queried range
        feed(&c, rotate: 90, start: 2.0, end: 5.0)   // B, inside

        let turns = c.turns(in: 2.0...5.0)
        XCTAssertEqual(turns.count, 1)               // only B is in range
        XCTAssertEqual(turns[0].clusterID, 1)
        XCTAssertGreaterThanOrEqual(turns[0].start, 2.0)
    }

    // MARK: - 7. PositionTimeline — spans always tile the queried window

    /// The property that the whole event-driven design exists for: from the first
    /// boundary onward the spans cover the query window EXACTLY — no gaps (a gap
    /// is what strands words as SPEAKER UNKNOWN) and no overlaps.
    func testTimelineSpansCompletelyPartitionTheWindow() {
        var rng = SystemRandomNumberGenerator()

        for iteration in 0..<5_000 {
            // Random boundary list: times deliberately NOT pre-sorted, so the
            // monotonic clamp in append() is exercised too.
            let count = Int.random(in: 1...8, using: &rng)
            var raw: [(t: Double, clusterID: Int)] = []
            var timeline = PositionTimeline()
            for _ in 0..<count {
                let t = (Double.random(in: 0...40, using: &rng) * 10).rounded() / 10
                let id = Int.random(in: 0...3, using: &rng)
                raw.append((t: t, clusterID: id))
                timeline.append(t: t, clusterID: id)
            }
            let lo = (Double.random(in: -5...40, using: &rng) * 10).rounded() / 10
            let hi = max(lo, ((lo + Double.random(in: 0...40, using: &rng)) * 10).rounded() / 10)
            let window = lo...hi

            let spans = timeline.spans(in: window)
            let context = "iteration \(iteration) boundaries=\(raw) window=\(window)"

            // The covered region is [max(firstBoundary, lo) .. hi], clipped at 0.
            let firstBoundary = timeline.boundaries[0].t
            let coveredStart = max(firstBoundary, lo)
            let expected = max(0, hi - coveredStart)

            let total = spans.reduce(0.0) { $0 + ($1.end - $1.start) }
            XCTAssertEqual(total, expected, accuracy: 1e-9,
                           "span durations must sum to the covered window — \(context) spans=\(spans)")

            if hi > coveredStart {
                XCTAssertFalse(spans.isEmpty, "covered window is non-empty — \(context)")
                XCTAssertEqual(spans.first?.start ?? .nan, coveredStart, accuracy: 1e-9,
                               "first span must start at the covered start — \(context) spans=\(spans)")
                XCTAssertEqual(spans.last?.end ?? .nan, hi, accuracy: 1e-9,
                               "last span must reach the window end — \(context) spans=\(spans)")
            } else {
                XCTAssertTrue(spans.isEmpty, "nothing to cover — \(context) spans=\(spans)")
            }

            // Contiguous, ordered, non-overlapping, and never two identical
            // neighbours.
            for i in 1..<max(spans.count, 1) where spans.count > 1 {
                XCTAssertEqual(spans[i].start, spans[i - 1].end, accuracy: 1e-9,
                               "spans must be contiguous at index \(i) — \(context) spans=\(spans)")
                XCTAssertNotEqual(spans[i].clusterID, spans[i - 1].clusterID,
                                  "adjacent spans must differ in cluster at index \(i) — \(context) spans=\(spans)")
            }
            for s in spans {
                XCTAssertGreaterThan(s.end, s.start, "empty span — \(context) spans=\(spans)")
            }
        }
    }

    /// The exact bug: Speaker 1 → a ~0.3 s Speaker 2 → Speaker 3. `turns(in:)`
    /// drops the 0.3 s run as flicker and leaves 14.6..16.2 uncovered; the
    /// timeline keeps it as its own span, so the words there have a row.
    func testShortNewDirectionSurvivesAsItsOwnSpan() {
        var timeline = PositionTimeline()
        timeline.append(t: 0.0, clusterID: 0)
        timeline.append(t: 14.6, clusterID: 1)   // ~0.3 s turn
        timeline.append(t: 16.2, clusterID: 2)

        let spans = timeline.spans(in: 0.0...32.5)
        XCTAssertEqual(spans.count, 3)
        XCTAssertEqual(spans[1].clusterID, 1)
        XCTAssertEqual(spans[1].start, 14.6, accuracy: 1e-9)
        XCTAssertEqual(spans[1].end, 16.2, accuracy: 1e-9)
        // No hole anywhere: 0 → 32.5 is fully tiled.
        XCTAssertEqual(spans[0].start, 0.0, accuracy: 1e-9)
        XCTAssertEqual(spans[2].end, 32.5, accuracy: 1e-9)
    }

    /// Even a sub-0.1 s span survives — debounce lives upstream, not here.
    func testVeryShortSpanIsNotDropped() {
        var timeline = PositionTimeline()
        timeline.append(t: 1.0, clusterID: 0)
        timeline.append(t: 2.0, clusterID: 1)
        timeline.append(t: 2.05, clusterID: 0)

        let spans = timeline.spans(in: 0.0...5.0)
        XCTAssertEqual(spans.count, 3)
        XCTAssertEqual(spans[1].clusterID, 1)
        XCTAssertEqual(spans[1].end - spans[1].start, 0.05, accuracy: 1e-9)
    }

    func testAdjacentSameClusterBoundariesCollapse() {
        var timeline = PositionTimeline()
        timeline.append(t: 0.0, clusterID: 7)
        timeline.append(t: 1.0, clusterID: 7)
        timeline.append(t: 2.0, clusterID: 7)
        timeline.append(t: 3.0, clusterID: 8)

        let spans = timeline.spans(in: 0.0...4.0)
        XCTAssertEqual(spans.count, 2)
        XCTAssertEqual(spans[0].clusterID, 7)
        XCTAssertEqual(spans[0].start, 0.0, accuracy: 1e-9)
        XCTAssertEqual(spans[0].end, 3.0, accuracy: 1e-9)
        XCTAssertEqual(spans[1].clusterID, 8)
    }

    /// Time before the first boundary is pre-speech silence and stays uncovered
    /// by design — attributing it would put words on a talker who hadn't spoken.
    func testWindowEntirelyBeforeFirstBoundaryIsEmpty() {
        var timeline = PositionTimeline()
        timeline.append(t: 10.0, clusterID: 0)

        XCTAssertTrue(timeline.spans(in: 0.0...5.0).isEmpty)
        XCTAssertTrue(timeline.spans(in: 0.0...10.0).isEmpty)   // touches, covers nothing
        XCTAssertTrue(PositionTimeline().spans(in: 0.0...5.0).isEmpty)
    }

    func testWindowStartingMidSpanClips() {
        var timeline = PositionTimeline()
        timeline.append(t: 0.0, clusterID: 0)
        timeline.append(t: 5.0, clusterID: 1)
        timeline.append(t: 10.0, clusterID: 2)

        let spans = timeline.spans(in: 2.5...7.5)
        XCTAssertEqual(spans.count, 2)
        XCTAssertEqual(spans[0].clusterID, 0)
        XCTAssertEqual(spans[0].start, 2.5, accuracy: 1e-9)     // clipped, not 0.0
        XCTAssertEqual(spans[0].end, 5.0, accuracy: 1e-9)
        XCTAssertEqual(spans[1].clusterID, 1)
        XCTAssertEqual(spans[1].end, 7.5, accuracy: 1e-9)       // clipped, not 10.0
    }

    /// A back-dated boundary that lands before its predecessor is clamped, not
    /// inserted out of order.
    func testAppendClampsMonotonically() {
        var timeline = PositionTimeline()
        timeline.append(t: 5.0, clusterID: 0)
        timeline.append(t: 3.0, clusterID: 1)    // back-dated past its predecessor
        timeline.append(t: 8.0, clusterID: 2)

        XCTAssertEqual(timeline.boundaries.map { $0.t }, [5.0, 5.0, 8.0])
        let spans = timeline.spans(in: 0.0...10.0)
        // Cluster 0's span collapses to zero length and drops out — two events at
        // the same instant means the later one wins — but no reordering happens
        // and the window is still fully tiled from 5.0.
        XCTAssertEqual(spans.map { $0.clusterID }, [1, 2])
        XCTAssertEqual(spans[0].start, 5.0, accuracy: 1e-9)
        XCTAssertEqual(spans[1].end, 10.0, accuracy: 1e-9)
    }

    // MARK: - 8. Back-dating helper

    func testBackdatingNormalEmissionSubtractsHalfWindow() {
        let t = BoundaryBackdating.correctedBoundary(reportedT: 10.0,
                                                     windowSec: 0.4,
                                                     isFirstEmissionAfterReset: false,
                                                     windowStart: 3.0,
                                                     previousBoundaryT: nil)
        XCTAssertEqual(t, 9.8, accuracy: 1e-9)
    }

    func testBackdatingFirstEmissionAfterResetUsesWindowStart() {
        let t = BoundaryBackdating.correctedBoundary(reportedT: 10.0,
                                                     windowSec: 0.4,
                                                     isFirstEmissionAfterReset: true,
                                                     windowStart: 9.6,
                                                     previousBoundaryT: nil)
        XCTAssertEqual(t, 9.6, accuracy: 1e-9)

        // Missing windowStart → fall back to the normal half-window estimate.
        let fallback = BoundaryBackdating.correctedBoundary(reportedT: 10.0,
                                                           windowSec: 0.4,
                                                           isFirstEmissionAfterReset: true,
                                                           windowStart: nil,
                                                           previousBoundaryT: nil)
        XCTAssertEqual(fallback, 9.8, accuracy: 1e-9)
    }

    func testBackdatingClampsAtPreviousBoundary() {
        // Normal case clamped.
        let a = BoundaryBackdating.correctedBoundary(reportedT: 10.0,
                                                     windowSec: 0.4,
                                                     isFirstEmissionAfterReset: false,
                                                     windowStart: nil,
                                                     previousBoundaryT: 9.9)
        XCTAssertEqual(a, 9.9, accuracy: 1e-9)

        // Post-reset case clamped.
        let b = BoundaryBackdating.correctedBoundary(reportedT: 10.0,
                                                     windowSec: 0.4,
                                                     isFirstEmissionAfterReset: true,
                                                     windowStart: 2.0,
                                                     previousBoundaryT: 8.0)
        XCTAssertEqual(b, 8.0, accuracy: 1e-9)

        // And the corrected value never reorders a real timeline.
        var timeline = PositionTimeline()
        timeline.append(t: 9.9, clusterID: 0)
        timeline.append(t: a, clusterID: 1)
        timeline.append(t: b, clusterID: 2)
        XCTAssertEqual(timeline.boundaries.map { $0.t }, [9.9, 9.9, 9.9])
    }

    // MARK: - 9. Detector reports the candidate's FIRST sample time

    func testDetectorReportsCandidateFirstSampleTime() {
        var d = ClusterChangeDetector()
        XCTAssertNil(d.push(t: 0.0, clusterID: 0))     // stable = 0

        XCTAssertNil(d.push(t: 0.1, clusterID: 1))     // candidate starts HERE
        XCTAssertNil(d.push(t: 0.2, clusterID: 1))
        let fired = d.push(t: 0.3, clusterID: 1)       // confirmed at 0.3
        XCTAssertEqual(fired ?? .nan, 0.1, accuracy: 1e-9)   // ...reported as 0.1

        // Under the rate limit the candidate keeps building; the reported time is
        // still the candidate's first sample, not the (much later) fire time.
        XCTAssertNil(d.push(t: 0.4, clusterID: 2))     // candidate 2 starts HERE
        XCTAssertNil(d.push(t: 0.5, clusterID: 2))
        XCTAssertNil(d.push(t: 0.6, clusterID: 2))     // confirmed but rate-limited
        let fired2 = d.push(t: 1.0, clusterID: 2)
        XCTAssertEqual(fired2 ?? .nan, 0.4, accuracy: 1e-9)

        // An interrupted candidate restarts its clock.
        XCTAssertNil(d.push(t: 1.1, clusterID: 3))
        XCTAssertNil(d.push(t: 1.2, clusterID: 2))     // back to stable → candidate cleared
        XCTAssertNil(d.push(t: 1.9, clusterID: 3))     // fresh candidate starts HERE
        XCTAssertNil(d.push(t: 2.0, clusterID: 3))
        let fired3 = d.push(t: 2.1, clusterID: 3)
        XCTAssertEqual(fired3 ?? .nan, 1.9, accuracy: 1e-9)
    }

    func testForceChangeReportsItsOwnTime() {
        var d = ClusterChangeDetector()
        XCTAssertNil(d.forceChange(to: 0, at: 0.0))
        // No confirmation lag to undo — the boundary is the sample's own time.
        XCTAssertEqual(d.forceChange(to: 1, at: 4.25) ?? .nan, 4.25, accuracy: 1e-9)
    }

    // MARK: - 10. DirectionSmoother exposes what back-dating needs

    func testSmootherExposesWindowStartAndFirstEmissionFlag() {
        var s = DirectionSmoother(windowSec: 0.4)
        XCTAssertNil(s.windowStart)

        var emissions = 0
        var firstEmissionT: Double?
        // Index-derived times, not an accumulator: `10.0 + 4 * 0.1` clears the
        // warm-up threshold exactly, whereas repeated `t += 0.1` lands just under
        // it — the very drift the windowStart comment in the smoother is about.
        for i in 0..<10 {
            let t = 10.0 + Double(i) * 0.1
            if let out = s.push(t: t, vector: vec(rotate: 0)) {
                emissions += 1
                if emissions == 1 {
                    firstEmissionT = out.t
                    XCTAssertTrue(s.lastEmissionWasFirstSinceReset)
                } else {
                    XCTAssertFalse(s.lastEmissionWasFirstSinceReset)
                }
            }
        }

        XCTAssertGreaterThan(emissions, 0)
        XCTAssertEqual(s.windowStart ?? .nan, 10.0, accuracy: 1e-9)
        // The first emission is ~a full window late; windowStart is the exact fix.
        XCTAssertEqual(firstEmissionT ?? .nan, 10.4, accuracy: 1e-9)
        let corrected = BoundaryBackdating.correctedBoundary(
            reportedT: firstEmissionT ?? 0,
            windowSec: 0.4,
            isFirstEmissionAfterReset: true,
            windowStart: s.windowStart,
            previousBoundaryT: nil)
        XCTAssertEqual(corrected, 10.0, accuracy: 1e-9)

        s.reset()
        XCTAssertNil(s.windowStart)
        XCTAssertFalse(s.lastEmissionWasFirstSinceReset)
    }
}
