import Foundation
import simd

/// A single beam-direction observation on the `recordingElapsed` clock.
struct DirectionSample: Equatable, Sendable {
    let t: Double              // seconds on the recordingElapsed clock
    let vector: SIMD3<Double>  // unit vector
}

/// Pure spherical geometry for beam directions. No app types, no actor.
enum PositionMath {
    /// x=cosφ·cosθ, y=cosφ·sinθ, z=sinφ ; θ=rotate (azimuth), φ=angle (elevation),
    /// both in degrees. The result is a unit vector by construction.
    static func unitVector(rotateDeg: Double, angleDeg: Double) -> SIMD3<Double> {
        let theta = rotateDeg * .pi / 180
        let phi = angleDeg * .pi / 180
        let cosPhi = cos(phi)
        return SIMD3<Double>(cosPhi * cos(theta),
                             cosPhi * sin(theta),
                             sin(phi))
    }

    /// acos(clamp(dot,-1,1)) → DEGREES. The clamp is MANDATORY — floating-point
    /// dot products of near-parallel unit vectors can slip just past ±1, which
    /// would make acos return NaN.
    static func angularDistanceDeg(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
        let dot = simd_dot(a, b)
        let clamped = min(1.0, max(-1.0, dot))
        return acos(clamped) * 180 / .pi
    }
}

/// Median/hold filter over a trailing time window, to kill the jitter right
/// after a talker switch. Emits nil until it has ≥3 samples spanning the window;
/// then emits the component-wise-median direction (renormalized).
struct DirectionSmoother {
    var windowSec: Double = 0.4

    private var buffer: [DirectionSample] = []

    init(windowSec: Double = 0.4) {
        self.windowSec = windowSec
    }

    mutating func push(t: Double, vector: SIMD3<Double>) -> DirectionSample? {
        // A gap larger than the window means the stream broke — start fresh.
        if let last = buffer.last, t - last.t > windowSec {
            buffer.removeAll(keepingCapacity: true)
        }
        buffer.append(DirectionSample(t: t, vector: vector))

        // Drop anything older than the trailing window.
        let cutoff = t - windowSec
        while let first = buffer.first, first.t < cutoff {
            buffer.removeFirst()
        }

        // Need at least 3 samples that actually span the full window.
        guard buffer.count >= 3,
              let first = buffer.first, let last = buffer.last,
              last.t - first.t >= windowSec else {
            return nil
        }

        let xs = buffer.map { $0.vector.x }.sorted()
        let ys = buffer.map { $0.vector.y }.sorted()
        let zs = buffer.map { $0.vector.z }.sorted()
        let med = SIMD3<Double>(Self.median(xs), Self.median(ys), Self.median(zs))
        let len = simd_length(med)
        // Degenerate (opposing vectors cancelling) → fall back to the newest.
        let unit = len > 1e-9 ? med / len : last.vector
        return DirectionSample(t: t, vector: unit)
    }

    mutating func reset() {
        buffer.removeAll(keepingCapacity: true)
    }

    private static func median(_ sorted: [Double]) -> Double {
        let n = sorted.count
        if n == 0 { return 0 }
        if n % 2 == 1 { return sorted[n / 2] }
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }
}

/// Online angular clustering of beam directions. Each new sample joins the
/// nearest centroid within `tauDeg` (updated with an EMA that IS the incremental
/// circular mean once vectors are re-normalized), or spawns a new cluster.
struct PositionClusterer {
    struct Cluster: Equatable { let id: Int; var centroid: SIMD3<Double>; var sampleCount: Int }

    var tauDeg: Double = 15
    var emaAlpha: Double = 0.1

    private(set) var clusters: [Cluster] = []
    private(set) var assignments: [(t: Double, clusterID: Int)] = []

    /// Safety valve — ~4 MB for a 2 h meeting at 10 Hz, so no ring buffer is
    /// otherwise needed; oldest assignments are dropped past this cap.
    private let maxAssignments = 500_000

    init(tauDeg: Double = 15, emaAlpha: Double = 0.1) {
        self.tauDeg = tauDeg
        self.emaAlpha = emaAlpha
    }

    /// Nearest centroid by `angularDistanceDeg`; < `tauDeg` → assign + EMA update
    /// (centroid = normalize((1-α)·c + α·v)); else a new cluster is born with
    /// id = clusters.count.
    @discardableResult
    mutating func assign(_ s: DirectionSample) -> (clusterID: Int, isNew: Bool) {
        var bestID = -1
        var bestDist = Double.greatestFiniteMagnitude
        for c in clusters {
            let d = PositionMath.angularDistanceDeg(c.centroid, s.vector)
            if d < bestDist {
                bestDist = d
                bestID = c.id
            }
        }

        let result: (clusterID: Int, isNew: Bool)
        if bestID >= 0, bestDist < tauDeg {
            // EMA on normalized vectors = incremental circular mean; wrap-around
            // (359°/1°) falls out of the vector form, no special-casing.
            let idx = clusters.firstIndex { $0.id == bestID }!
            let blended = (1 - emaAlpha) * clusters[idx].centroid + emaAlpha * s.vector
            let len = simd_length(blended)
            if len > 1e-9 { clusters[idx].centroid = blended / len }
            clusters[idx].sampleCount += 1
            result = (bestID, false)
        } else {
            let id = clusters.count
            clusters.append(Cluster(id: id, centroid: s.vector, sampleCount: 1))
            result = (id, true)
        }

        assignments.append((t: s.t, clusterID: result.clusterID))
        if assignments.count > maxAssignments {
            assignments.removeFirst(assignments.count - maxAssignments)
        }
        return result
    }

    /// Mode over assignments whose t is in `range`; nil if the in-range count is
    /// below `minSamples`.
    func dominantCluster(in range: ClosedRange<Double>, minSamples: Int) -> Int? {
        var counts: [Int: Int] = [:]
        var total = 0
        for a in assignments where range.contains(a.t) {
            counts[a.clusterID, default: 0] += 1
            total += 1
        }
        guard total >= minSamples else { return nil }
        // Tie-break deterministically on the lower cluster id.
        return counts.max { lhs, rhs in
            lhs.value != rhs.value ? lhs.value < rhs.value : lhs.key > rhs.key
        }?.key
    }

    func sampleCount(in range: ClosedRange<Double>) -> Int {
        assignments.reduce(0) { $0 + (range.contains($1.t) ? 1 : 0) }
    }
}
