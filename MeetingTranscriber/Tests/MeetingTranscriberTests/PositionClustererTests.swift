import XCTest
import simd
@testable import MeetingTranscriber

final class PositionClustererTests: XCTestCase {

    // MARK: - Helpers

    private func vec(rotate: Double, angle: Double) -> SIMD3<Double> {
        PositionMath.unitVector(rotateDeg: rotate, angleDeg: angle)
    }

    private func azimuthDeg(_ v: SIMD3<Double>) -> Double {
        atan2(v.y, v.x) * 180 / .pi
    }

    /// Smallest absolute angular difference between two azimuths, in [0, 180].
    private func azimuthDelta(_ a: Double, _ b: Double) -> Double {
        var d = (a - b).truncatingRemainder(dividingBy: 360)
        if d > 180 { d -= 360 }
        if d < -180 { d += 360 }
        return abs(d)
    }

    // MARK: - 1. Wrap-around

    func testWrapAroundIsOneClusterNear359And1() {
        let a = vec(rotate: 359, angle: 0)
        let b = vec(rotate: 1, angle: 0)

        // The two directions really are ~2° apart, not ~358°.
        XCTAssertEqual(PositionMath.angularDistanceDeg(a, b), 2, accuracy: 0.01)

        var c = PositionClusterer(tauDeg: 15)
        _ = c.assign(DirectionSample(t: 0.0, vector: a))
        _ = c.assign(DirectionSample(t: 0.1, vector: b))

        XCTAssertEqual(c.clusters.count, 1)
        // Centroid azimuth must be near 0°, NOT the 180° a naive linear mean gives.
        let az = azimuthDeg(c.clusters[0].centroid)
        XCTAssertLessThan(azimuthDelta(az, 0), 5)
    }

    // MARK: - 2. τ boundary

    func testTauBoundary() {
        // 14° apart, τ=15 → one cluster.
        do {
            var c = PositionClusterer(tauDeg: 15)
            _ = c.assign(DirectionSample(t: 0, vector: vec(rotate: 0, angle: 0)))
            _ = c.assign(DirectionSample(t: 1, vector: vec(rotate: 14, angle: 0)))
            XCTAssertEqual(c.clusters.count, 1)
        }
        // 16° apart, τ=15 → two clusters.
        do {
            var c = PositionClusterer(tauDeg: 15)
            _ = c.assign(DirectionSample(t: 0, vector: vec(rotate: 0, angle: 0)))
            _ = c.assign(DirectionSample(t: 1, vector: vec(rotate: 16, angle: 0)))
            XCTAssertEqual(c.clusters.count, 2)
        }
        // Tight τ=10 → 14° apart splits into two.
        do {
            var c = PositionClusterer(tauDeg: 10)
            _ = c.assign(DirectionSample(t: 0, vector: vec(rotate: 0, angle: 0)))
            _ = c.assign(DirectionSample(t: 1, vector: vec(rotate: 14, angle: 0)))
            XCTAssertEqual(c.clusters.count, 2)
        }
        // Loose τ=25 → 16° apart stays one.
        do {
            var c = PositionClusterer(tauDeg: 25)
            _ = c.assign(DirectionSample(t: 0, vector: vec(rotate: 0, angle: 0)))
            _ = c.assign(DirectionSample(t: 1, vector: vec(rotate: 16, angle: 0)))
            XCTAssertEqual(c.clusters.count, 1)
        }
    }

    // MARK: - 3. Moved speaker

    func testMovedSpeakerBirthsNewClusterAndLeavesOldUnmoved() {
        var c = PositionClusterer(tauDeg: 15)
        for i in 0..<5 {
            _ = c.assign(DirectionSample(t: Double(i) * 0.1, vector: vec(rotate: 100, angle: 0)))
        }
        XCTAssertEqual(c.clusters.count, 1)
        let oldCentroid = c.clusters[0].centroid

        // Jump 40° away → a brand-new cluster with the next id.
        let r = c.assign(DirectionSample(t: 1.0, vector: vec(rotate: 140, angle: 0)))
        XCTAssertTrue(r.isNew)
        XCTAssertEqual(r.clusterID, 1)
        XCTAssertEqual(c.clusters.count, 2)

        // The original cluster's centroid is untouched by the new, far sample.
        XCTAssertEqual(c.clusters[0].centroid.x, oldCentroid.x, accuracy: 1e-12)
        XCTAssertEqual(c.clusters[0].centroid.y, oldCentroid.y, accuracy: 1e-12)
        XCTAssertEqual(c.clusters[0].centroid.z, oldCentroid.z, accuracy: 1e-12)
    }

    // MARK: - 4. Smoothing

    func testSmootherHoldsBackThenSuppressesOutlier() {
        var s = DirectionSmoother(windowSec: 0.4)
        s.reset()

        // Right after a reset: no emission until the window is actually spanned.
        XCTAssertNil(s.push(t: 0.0, vector: vec(rotate: 0, angle: 0)))
        XCTAssertNil(s.push(t: 0.1, vector: vec(rotate: 0, angle: 0)))
        XCTAssertNil(s.push(t: 0.2, vector: vec(rotate: 0, angle: 0)))  // span 0.2 < 0.4
        XCTAssertNil(s.push(t: 0.3, vector: vec(rotate: 0, angle: 0)))  // span 0.3 < 0.4
        XCTAssertNotNil(s.push(t: 0.4, vector: vec(rotate: 0, angle: 0)))  // span 0.4 → emit

        // Keep the stream stable, then inject a single 30° outlier.
        _ = s.push(t: 0.5, vector: vec(rotate: 0, angle: 0))
        _ = s.push(t: 0.6, vector: vec(rotate: 0, angle: 0))
        _ = s.push(t: 0.7, vector: vec(rotate: 0, angle: 0))
        _ = s.push(t: 0.8, vector: vec(rotate: 0, angle: 0))
        let emitted = s.push(t: 0.9, vector: vec(rotate: 30, angle: 0))
        let out = try! XCTUnwrap(emitted)
        // The median rejects the lone 30° jump — direction stays near 0°.
        XCTAssertLessThan(azimuthDelta(azimuthDeg(out.vector), 0), 2)
    }

    // MARK: - 5. EMA drift

    func testEMAStaysNearTrueDirectionUnderJitter() {
        var c = PositionClusterer()  // τ=15, α=0.1
        for i in 0..<200 {
            let jitter = 3 * sin(Double(i))  // deterministic, ~symmetric in [-3,3]
            _ = c.assign(DirectionSample(t: Double(i) * 0.1,
                                         vector: vec(rotate: 156 + jitter, angle: 0)))
        }
        XCTAssertEqual(c.clusters.count, 1)
        XCTAssertLessThan(azimuthDelta(azimuthDeg(c.clusters[0].centroid), 156), 1)
    }

    // MARK: - 6. dominantCluster

    func testDominantClusterModeAndMinSamples() {
        var c = PositionClusterer(tauDeg: 15)
        // 7 A-samples (rotate 0 → id 0), 3 B-samples (rotate 90 → id 1).
        let aTimes = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        for t in aTimes {
            _ = c.assign(DirectionSample(t: t, vector: vec(rotate: 0, angle: 0)))
        }
        let bTimes = [1.7, 1.8, 1.9]
        for t in bTimes {
            _ = c.assign(DirectionSample(t: t, vector: vec(rotate: 90, angle: 0)))
        }

        XCTAssertEqual(c.dominantCluster(in: 1.0...1.9, minSamples: 1), 0)
        XCTAssertEqual(c.sampleCount(in: 1.0...1.9), 10)

        // A window holding only 2 samples with minSamples:3 → nil.
        XCTAssertEqual(c.sampleCount(in: 1.0...1.1), 2)
        XCTAssertNil(c.dominantCluster(in: 1.0...1.1, minSamples: 3))
    }

    // MARK: - 7. Elevation-only difference

    func testElevationOnlyDifferenceSeparates() {
        var c = PositionClusterer(tauDeg: 15)
        _ = c.assign(DirectionSample(t: 0, vector: vec(rotate: 0, angle: 30)))
        _ = c.assign(DirectionSample(t: 1, vector: vec(rotate: 0, angle: 70)))
        // Same azimuth, 40° elevation gap → φ must contribute → two clusters.
        XCTAssertEqual(c.clusters.count, 2)
    }

    // MARK: - 8. Smoother keeps emitting under an accumulated (non-exact) clock

    func testSmootherKeepsEmittingUnderAccumulatedClock() {
        // Reproduces the real recording clock: recordingElapsed += 4096/48000
        // per audio buffer. The old buffer-span gate locked the filter silent a
        // couple of seconds in; the elapsed-time gate must not.
        var s = DirectionSmoother(windowSec: 0.4)
        let v = PositionMath.unitVector(rotateDeg: 156, angleDeg: 48)
        let step = 4096.0 / 48000.0   // ~0.0853s, a non-dyadic fraction
        var t = 0.0
        var emittedAfter2s = 0
        var totalAfter2s = 0
        for _ in 0..<400 {            // ~34s of stream
            let out = s.push(t: t, vector: v)
            if t > 2.0 {
                totalAfter2s += 1
                if out != nil { emittedAfter2s += 1 }
            }
            t += step
        }
        // A stationary talker past warm-up must emit on essentially every push.
        XCTAssertGreaterThan(Double(emittedAfter2s) / Double(totalAfter2s), 0.95)
    }
}
