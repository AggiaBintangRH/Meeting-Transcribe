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
        var d = ClusterChangeDetector()   // consecutiveRequired 3, minIntervalSec 2.0

        // First stable cluster never fires.
        XCTAssertFalse(d.push(t: 0.0, clusterID: 0))
        // Two consecutive of a new cluster: not yet.
        XCTAssertFalse(d.push(t: 0.1, clusterID: 1))
        XCTAssertFalse(d.push(t: 0.2, clusterID: 1))
        // Third consecutive → fire.
        XCTAssertTrue(d.push(t: 0.3, clusterID: 1))

        // A rapid second change is suppressed by minIntervalSec (t - 0.3 < 2.0).
        XCTAssertFalse(d.push(t: 0.4, clusterID: 2))
        XCTAssertFalse(d.push(t: 0.5, clusterID: 2))
        XCTAssertFalse(d.push(t: 0.6, clusterID: 2))   // confirmed but rate-limited

        // Once the interval passes, the still-building candidate fires.
        XCTAssertTrue(d.push(t: 2.4, clusterID: 2))
    }

    func testDetectorResetClearsStableCluster() {
        var d = ClusterChangeDetector()
        XCTAssertFalse(d.push(t: 0.0, clusterID: 5))   // establishes stable = 5
        d.reset()
        // After reset the next cluster is a fresh first-stable → never fires.
        XCTAssertFalse(d.push(t: 1.0, clusterID: 7))
        XCTAssertFalse(d.push(t: 1.1, clusterID: 7))
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
}
