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

    /// The correction is bounded by a FRACTION of the gap to the previous
    /// boundary, never by the previous boundary itself — see the next test for
    /// why that difference matters.
    func testBackdatingIsBoundedByAFractionOfTheGap() {
        // Normal case: gap 0.1 → at most 0.05 may be given back, so the raw
        // -0.2 correction is bounded at 9.95 (NOT at the previous boundary 9.9).
        let a = BoundaryBackdating.correctedBoundary(reportedT: 10.0,
                                                     windowSec: 0.4,
                                                     isFirstEmissionAfterReset: false,
                                                     windowStart: nil,
                                                     previousBoundaryT: 9.9)
        XCTAssertEqual(a, 9.95, accuracy: 1e-9)

        // Post-reset case: gap 2.0 → bounded at 9.0, not at windowStart 2.0.
        let b = BoundaryBackdating.correctedBoundary(reportedT: 10.0,
                                                     windowSec: 0.4,
                                                     isFirstEmissionAfterReset: true,
                                                     windowStart: 2.0,
                                                     previousBoundaryT: 8.0)
        XCTAssertEqual(b, 9.0, accuracy: 1e-9)

        // A wide gap leaves the full correction untouched — the bound only bites
        // when the previous boundary is close.
        let c = BoundaryBackdating.correctedBoundary(reportedT: 10.0,
                                                     windowSec: 0.4,
                                                     isFirstEmissionAfterReset: false,
                                                     windowStart: nil,
                                                     previousBoundaryT: 5.0)
        XCTAssertEqual(c, 9.8, accuracy: 1e-9)

        // A report at/before its predecessor has no time to distribute: it
        // degrades to the previous boundary rather than reordering.
        let d = BoundaryBackdating.correctedBoundary(reportedT: 7.0,
                                                     windowSec: 0.4,
                                                     isFirstEmissionAfterReset: false,
                                                     windowStart: nil,
                                                     previousBoundaryT: 9.0)
        XCTAssertEqual(d, 9.0, accuracy: 1e-9)

        // And the corrected values never reorder a real timeline.
        var timeline = PositionTimeline()
        timeline.append(t: 9.9, clusterID: 0)
        timeline.append(t: a, clusterID: 1)
        XCTAssertEqual(timeline.boundaries.map { $0.t }, [9.9, 9.95])
    }

    /// MANDATORY: back-dating must never erase the span it dates away from.
    /// Clamping at `max(previous, corrected)` would collapse cluster 0 to
    /// `[5.0, 5.0]`, `spans(in:)` would drop it, and that speaker would VANISH —
    /// the exact failure this redesign exists to eliminate.
    func testBackdatingCannotEraseAShortSpan() {
        var timeline = PositionTimeline()
        timeline.append(t: 5.0, clusterID: 0)

        // Absurdly aggressive: a 0.1 s turn whose correction (0.2 s, plus a
        // windowStart way back at 0.0) is larger than the turn itself.
        for (reportedT, first, start) in [(5.1, false, nil as Double?),
                                          (5.1, true, 0.0 as Double?),
                                          (5.001, true, 0.0 as Double?)] {
            var t = timeline
            let corrected = BoundaryBackdating.correctedBoundary(
                reportedT: reportedT,
                windowSec: 0.4,
                isFirstEmissionAfterReset: first,
                windowStart: start,
                previousBoundaryT: t.boundaries.last?.t)
            XCTAssertGreaterThan(corrected, 5.0,
                                 "boundary must land strictly after its predecessor (\(reportedT), \(first))")
            t.append(t: corrected, clusterID: 1)

            let spans = t.spans(in: 0.0...10.0)
            XCTAssertEqual(spans.count, 2, "cluster 0's span was erased (\(reportedT), \(first))")
            XCTAssertEqual(spans[0].clusterID, 0)
            XCTAssertGreaterThan(spans[0].end - spans[0].start, 0,
                                 "cluster 0 kept no duration (\(reportedT), \(first))")
        }

        // A legitimately short (~1 s) turn still keeps a USABLE span, not a sliver:
        // at most half the gap is ever given back.
        let corrected = BoundaryBackdating.correctedBoundary(
            reportedT: 6.0,
            windowSec: 0.4,
            isFirstEmissionAfterReset: true,
            windowStart: 0.0,
            previousBoundaryT: 5.0)
        XCTAssertEqual(corrected, 5.5, accuracy: 1e-9)
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

    // MARK: - 11. PositionBoundaryRule — the live timeline, end to end

    /// The whole `PositionDiarizer.ingest` pipeline minus the `@MainActor` object
    /// and its singleton subscription: smoother → clusterer → boundary rule.
    /// Same code path the app runs, so these are not a re-implementation.
    private struct Pipeline {
        var smoother = DirectionSmoother(windowSec: 0.4)
        var clusterer = PositionClusterer(tauDeg: 15)
        var detector = ClusterChangeDetector()
        var timeline = PositionTimeline()
        /// Sample times at which `onClusterChange` would have fired.
        var fireTimes: [Double] = []

        mutating func push(t: Double, rotate: Double) {
            guard let sample = smoother.push(t: t, vector: PositionMath.unitVector(rotateDeg: rotate,
                                                                                   angleDeg: 0)) else { return }
            let result = clusterer.assign(sample)
            if PositionBoundaryRule.apply(sample: sample,
                                          clusterID: result.clusterID,
                                          isNewCluster: result.isNew,
                                          smoother: smoother,
                                          detector: &detector,
                                          timeline: &timeline) {
                fireTimes.append(sample.t)
            }
        }

        /// 10 Hz over `[start, end)`, times derived from the index so they don't
        /// drift under the smoother's warm-up threshold (see the smoother's
        /// `windowStart` comment).
        mutating func feed(rotate: Double, start: Double, end: Double) {
            let n = Int(((end - start) / 0.1).rounded())
            for i in 0..<n { push(t: start + Double(i) * 0.1, rotate: rotate) }
        }
    }

    /// THE BUG CASE. Speaker 1 → a ~1 s Speaker 2 → Speaker 3. The old
    /// `turns(in:)` path drops speaker 2 (too short for the density gate) and
    /// leaves that stretch uncovered → a `SPEAKER UNKNOWN` row. The timeline
    /// yields THREE spans that tile the window with no gaps.
    func testShortMiddleTurnYieldsThreeTilingSpans() {
        /// S1 for 5 s → S2 for `middle` s → S3 for 5 s, and the spans over the
        /// whole window.
        func run(middle: Double) -> (spans: [(start: Double, end: Double, clusterID: Int)],
                                     legacy: [(start: Double, end: Double, clusterID: Int)]) {
            var p = Pipeline()
            p.feed(rotate: 0, start: 0.0, end: 5.0)                    // Speaker 1
            p.feed(rotate: 90, start: 5.0, end: 5.0 + middle)          // Speaker 2
            p.feed(rotate: 180, start: 5.0 + middle, end: 10.0 + middle)  // Speaker 3
            let window = 0.0...(10.0 + middle)
            return (p.timeline.spans(in: window), p.clusterer.turns(in: window))
        }

        for middle in [1.0, 0.5] {
            let (spans, _) = run(middle: middle)
            XCTAssertEqual(spans.map { $0.clusterID }, [0, 1, 2],
                           "middle=\(middle) spans=\(spans)")

            // No holes and no overlaps across the whole covered window.
            XCTAssertEqual(spans[0].start, 0.0, accuracy: 1e-9)
            XCTAssertEqual(spans[1].start, spans[0].end, accuracy: 1e-9)
            XCTAssertEqual(spans[2].start, spans[1].end, accuracy: 1e-9)
            XCTAssertEqual(spans[2].end, 10.0 + middle, accuracy: 1e-9)
            for s in spans {
                XCTAssertGreaterThan(s.end - s.start, 0, "empty span — middle=\(middle) \(spans)")
            }

            // The middle speaker keeps a span near their real turn.
            XCTAssertEqual(spans[1].start, 5.0, accuracy: 0.35)
            XCTAssertGreaterThan(spans[1].end - spans[1].start, middle / 2)
        }

        // And the legacy path really does lose the short one: at 0.5 s the
        // density gate drops it and 5.x..6.x is left uncovered — the stretch that
        // renders as SPEAKER UNKNOWN today.
        XCTAssertEqual(run(middle: 0.5).legacy.count, 2,
                       "turns(in:) is expected to drop the 0.5 s turn — got \(run(middle: 0.5).legacy)")
    }

    /// The first stable cluster of a session opens the timeline but must NOT fire
    /// a row split — there is no previous row to split from.
    func testFirstStableClusterRecordsABoundaryButFiresNothing() {
        var p = Pipeline()
        p.feed(rotate: 0, start: 2.0, end: 5.0)

        XCTAssertTrue(p.fireTimes.isEmpty, "first speaker must not fire onClusterChange")
        XCTAssertEqual(p.timeline.boundaries.count, 1)
        XCTAssertEqual(p.timeline.boundaries[0].clusterID, 0)
        // Back-dated to the raw stream's own start, not to the first emission
        // (which is a full smoothing window late).
        XCTAssertEqual(p.timeline.boundaries[0].t, 2.0, accuracy: 1e-9)
        // And speech before the first boundary stays uncovered.
        XCTAssertTrue(p.timeline.spans(in: 0.0...1.9).isEmpty)
    }

    /// A return to an ALREADY-STORED speaker starts a NEW span, not a
    /// continuation of their earlier one — every direction change is a new row.
    func testReturnToKnownSpeakerStartsANewSpan() {
        var p = Pipeline()
        p.feed(rotate: 0, start: 0.0, end: 4.0)     // S1
        p.feed(rotate: 90, start: 4.0, end: 8.0)    // S2 (new direction)
        p.feed(rotate: 0, start: 8.0, end: 12.0)    // back to S1 — known cluster

        XCTAssertEqual(p.clusterer.clusters.count, 2, "the return must reuse cluster 0")
        let spans = p.timeline.spans(in: 0.0...12.0)
        XCTAssertEqual(spans.map { $0.clusterID }, [0, 1, 0], "spans=\(spans)")
        XCTAssertEqual(spans[2].start, 8.0, accuracy: 0.4)
        // Three separate spans, not two merged ones.
        XCTAssertEqual(p.fireTimes.count, 2)
    }

    /// Boundary times come from the ATND sample's own audio clock. Two identical
    /// direction sequences differing only by a clock offset produce boundaries
    /// offset by exactly that much — nothing depends on call order or wall time.
    func testBoundaryTimesTrackTheSampleClock() {
        func boundaries(offsetBy offset: Double) -> [Double] {
            var p = Pipeline()
            p.feed(rotate: 0, start: offset + 0.0, end: offset + 4.0)
            p.feed(rotate: 90, start: offset + 4.0, end: offset + 8.0)
            return p.timeline.boundaries.map { $0.t }
        }

        let base = boundaries(offsetBy: 0)
        let shifted = boundaries(offsetBy: 137.5)
        XCTAssertEqual(base.count, 2)
        XCTAssertEqual(shifted.count, base.count)
        for (b, s) in zip(base, shifted) {
            XCTAssertEqual(s - b, 137.5, accuracy: 1e-6, "base=\(base) shifted=\(shifted)")
        }

        // And a rate-limited `push` fire is stamped with the CANDIDATE's first
        // sample time, not with the (later) time it was allowed to fire.
        var detector = ClusterChangeDetector()
        var timeline = PositionTimeline()
        var smoother = DirectionSmoother(windowSec: 0.4)
        for i in 0..<8 { _ = smoother.push(t: Double(i) * 0.1, vector: vec(rotate: 0)) }
        _ = detector.forceChange(to: 0, at: 0.0)          // establish a stable cluster
        for t in [20.0, 20.1] {
            PositionBoundaryRule.apply(sample: DirectionSample(t: t, vector: vec(rotate: 90)),
                                       clusterID: 1, isNewCluster: false,
                                       smoother: smoother, detector: &detector, timeline: &timeline)
        }
        XCTAssertTrue(timeline.boundaries.isEmpty, "not yet confirmed")
        PositionBoundaryRule.apply(sample: DirectionSample(t: 20.2, vector: vec(rotate: 90)),
                                   clusterID: 1, isNewCluster: false,
                                   smoother: smoother, detector: &detector, timeline: &timeline)
        XCTAssertEqual(timeline.boundaries.count, 1)
        // 20.0 (candidate's first sample) − 0.2 (half window), not 20.2.
        XCTAssertEqual(timeline.boundaries[0].t, 19.8, accuracy: 1e-9)
    }
}
