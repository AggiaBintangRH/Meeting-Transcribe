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
    /// Time of the first sample since the last reset/gap. The warm-up gate
    /// measures elapsed time from here, NOT the span of the surviving buffer:
    /// that span only reaches `windowSec` when a sample lands exactly on the
    /// trailing-window cutoff, which an accumulated (non-exact) clock like
    /// `recordingElapsed` essentially never does — using it locks the filter
    /// silent a couple of seconds into a real recording. Elapsed time is
    /// monotonic, so it crosses the threshold once and stays crossed.
    private var windowStart: Double?

    init(windowSec: Double = 0.4) {
        self.windowSec = windowSec
    }

    mutating func push(t: Double, vector: SIMD3<Double>) -> DirectionSample? {
        // A gap larger than the window means the stream broke — start fresh.
        if let last = buffer.last, t - last.t > windowSec {
            buffer.removeAll(keepingCapacity: true)
            windowStart = nil
        }
        if windowStart == nil { windowStart = t }
        buffer.append(DirectionSample(t: t, vector: vector))

        // Drop anything older than the trailing window.
        let cutoff = t - windowSec
        while let first = buffer.first, first.t < cutoff {
            buffer.removeFirst()
        }

        // Warm up for a full window after a reset/gap, then emit whenever the
        // trailing window holds enough samples for a stable median.
        guard buffer.count >= 3,
              let start = windowStart, t - start >= windowSec else {
            return nil
        }

        let xs = buffer.map { $0.vector.x }.sorted()
        let ys = buffer.map { $0.vector.y }.sorted()
        let zs = buffer.map { $0.vector.z }.sorted()
        let med = SIMD3<Double>(Self.median(xs), Self.median(ys), Self.median(zs))
        let len = simd_length(med)
        // Degenerate (opposing vectors cancelling) → fall back to the newest.
        let unit = len > 1e-9 ? med / len : vector
        return DirectionSample(t: t, vector: unit)
    }

    mutating func reset() {
        buffer.removeAll(keepingCapacity: true)
        windowStart = nil
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

    /// Contiguous same-cluster runs of the assignment stream within `range`.
    /// Each returned turn is one talker's continuous span; a beam change mid-range
    /// yields two turns so the display can split rows at the switch.
    ///
    /// - `minDurationSec`/`minSamples`: density gate — runs shorter/sparser than
    ///   this are dropped as flicker (the 0.4s smoother already kills per-notice
    ///   jitter; this catches longer strays).
    /// - `maxGapSec`: an ATND-silence hole larger than this breaks a run — no turn
    ///   may span it (so real silence stays uncovered → UNKNOWN downstream).
    func turns(in range: ClosedRange<Double>,
               minDurationSec: Double = 0.6,
               maxGapSec: Double = 1.0,
               minSamples: Int = 3) -> [(start: Double, end: Double, clusterID: Int)] {
        // `assignments` is appended in time order, so in-range stays time-ordered.
        let inRange = assignments.filter { range.contains($0.t) }
        guard !inRange.isEmpty else { return [] }

        struct Run { var start: Double; var end: Double; var clusterID: Int; var count: Int }

        // 1. Raw runs: new run when the cluster changes OR the inter-sample gap
        //    exceeds maxGapSec (an ATND-silence hole — never bridged).
        var runs: [Run] = []
        for a in inRange {
            if var last = runs.last,
               last.clusterID == a.clusterID,
               a.t - last.end <= maxGapSec {
                last.end = a.t
                last.count += 1
                runs[runs.count - 1] = last
            } else {
                runs.append(Run(start: a.t, end: a.t, clusterID: a.clusterID, count: 1))
            }
        }

        // 2a. Flicker absorption: drop sub-min runs (a stray 1–3-sample flicker
        //     inside a stable turn, or unlabeled boundary slop between two clusters).
        let survivors = runs.filter { $0.end - $0.start >= minDurationSec && $0.count >= minSamples }

        // 2b. Merge adjacent SAME-cluster survivors separated by <= maxGapSec
        //     (so a dropped flicker's same-cluster flanks rejoin into one turn).
        var merged: [Run] = []
        for r in survivors {
            if var last = merged.last,
               last.clusterID == r.clusterID,
               r.start - last.end <= maxGapSec {
                last.end = r.end
                last.count += r.count
                merged[merged.count - 1] = last
            } else {
                merged.append(r)
            }
        }

        // 3. Sorted by start (already in order; explicit for clarity).
        return merged
            .map { (start: $0.start, end: $0.end, clusterID: $0.clusterID) }
            .sorted { $0.start < $1.start }
    }
}

/// Pure detector that fires exactly once when the beam settles on a genuinely
/// different talker cluster than the current stable one. Used for real-time row
/// splitting; no app types, unit-testable in isolation.
struct ClusterChangeDetector {
    /// How many consecutive same-new-cluster samples confirm a change (~0.3s at 10 Hz).
    var consecutiveRequired = 3
    /// Minimum seconds between fires — keeps a jittery existing-cluster oscillation
    /// from thrashing the display, while still allowing ~1 s speaker turns.
    var minIntervalSec = 0.6

    private var stableCluster: Int?
    private var candidate: Int?
    private var candidateCount = 0
    private var lastFired: Double?

    /// Returns true ONLY when `clusterID` differs from the current stable cluster,
    /// has been seen `consecutiveRequired` times in a row, AND `t - lastFired >=
    /// minIntervalSec`. The FIRST stable cluster of a session never fires (nothing
    /// to split). On a fire, `clusterID` becomes the new stable cluster.
    mutating func push(t: Double, clusterID: Int) -> Bool {
        // Establish the first stable cluster silently — nothing to split yet.
        guard let stable = stableCluster else {
            stableCluster = clusterID
            candidate = nil
            candidateCount = 0
            return false
        }
        // Back on the stable cluster — reset any building candidate.
        if clusterID == stable {
            candidate = nil
            candidateCount = 0
            return false
        }
        // A different cluster — accumulate confirmation.
        if clusterID == candidate {
            candidateCount += 1
        } else {
            candidate = clusterID
            candidateCount = 1
        }
        guard candidateCount >= consecutiveRequired else { return false }
        // Confirmed change, but rate-limit: suppress a fire too soon after the
        // last one (keep the candidate building so it fires once the interval passes).
        if let last = lastFired, t - last < minIntervalSec { return false }
        stableCluster = clusterID
        candidate = nil
        candidateCount = 0
        lastFired = t
        return true
    }

    /// Switch the stable cluster IMMEDIATELY, with no confirmation or rate-limit —
    /// for a brand-new direction (outside every stored speaker's range), which is
    /// an unambiguous new speaker, not jitter. Returns true to fire, except for the
    /// very first speaker of the session (nothing to split from).
    mutating func forceChange(to clusterID: Int, at t: Double) -> Bool {
        candidate = nil
        candidateCount = 0
        let hadStable = stableCluster != nil
        stableCluster = clusterID
        guard hadStable else { return false }
        lastFired = t
        return true
    }

    mutating func reset() {
        stableCluster = nil
        candidate = nil
        candidateCount = 0
        lastFired = nil
    }
}
