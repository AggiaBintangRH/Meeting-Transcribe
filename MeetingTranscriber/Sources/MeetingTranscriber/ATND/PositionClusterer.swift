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
    ///
    /// Readable from outside because it is ALSO the exact back-dating correction
    /// for the first emission after a reset: that emission is a full `windowSec`
    /// late (the warm-up), and the raw stream really started here. See
    /// `BoundaryBackdating.correctedBoundary`.
    private(set) var windowStart: Double?

    /// Emissions produced since the last reset/gap. `== 1` right after a `push`
    /// that returned non-nil means THAT emission is the first since the reset,
    /// which is the case that back-dates to `windowStart` rather than to
    /// `reportedT - windowSec/2`. Exposed as a count rather than a flag so a
    /// caller can read it after the fact without the smoother having to hand
    /// back a wrapper type (which would churn the existing call sites).
    private(set) var emissionsSinceReset = 0

    /// True when the most recent non-nil `push` was the first emission since a
    /// reset/gap. Only meaningful immediately after such a push.
    var lastEmissionWasFirstSinceReset: Bool { emissionsSinceReset == 1 }

    init(windowSec: Double = 0.4) {
        self.windowSec = windowSec
    }

    mutating func push(t: Double, vector: SIMD3<Double>) -> DirectionSample? {
        // A gap larger than the window means the stream broke — start fresh.
        if let last = buffer.last, t - last.t > windowSec {
            buffer.removeAll(keepingCapacity: true)
            windowStart = nil
            emissionsSinceReset = 0
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
        emissionsSinceReset += 1
        return DirectionSample(t: t, vector: unit)
    }

    mutating func reset() {
        buffer.removeAll(keepingCapacity: true)
        windowStart = nil
        emissionsSinceReset = 0
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

    /// Per-cluster sample counts over `range`, densest first — diagnostic only.
    /// Distinguishes "this speaker's run was too short to survive the turn
    /// filter" (their cluster IS here, with samples) from "their direction
    /// merged into a neighbour's cluster because tauDeg is too wide" (their
    /// cluster is absent entirely). Those need opposite fixes, and the FILL/SKIP
    /// lines alone cannot tell them apart.
    func clusterHistogram(in range: ClosedRange<Double>) -> [(clusterID: Int, count: Int)] {
        var counts: [Int: Int] = [:]
        for a in assignments where range.contains(a.t) {
            counts[a.clusterID, default: 0] += 1
        }
        return counts.map { (clusterID: $0.key, count: $0.value) }
            .sorted { $0.count != $1.count ? $0.count > $1.count : $0.clusterID < $1.clusterID }
    }

    /// Angular separation between every pair of cluster centroids, in degrees —
    /// diagnostic only. A pair below `tauDeg` means those two real speakers can
    /// no longer be told apart at the current threshold.
    func centroidSeparations() -> [(a: Int, b: Int, deg: Double)] {
        var out: [(a: Int, b: Int, deg: Double)] = []
        for i in 0..<clusters.count {
            for j in (i + 1)..<clusters.count {
                out.append((clusters[i].id, clusters[j].id,
                            PositionMath.angularDistanceDeg(clusters[i].centroid,
                                                            clusters[j].centroid)))
            }
        }
        return out.sorted { $0.deg < $1.deg }
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
    /// Sample time of the FIRST sample of the current candidate run. This — not
    /// the time the fire happens — is when the talker actually switched; by the
    /// time `consecutiveRequired` samples have confirmed it, ~0.3 s of the new
    /// talker has already elapsed. Reporting it removes the confirmation lag
    /// EXACTLY (we know the sample time), instead of subtracting an estimate.
    private var candidateFirstT: Double?
    private var lastFired: Double?

    /// Returns the BOUNDARY TIME (nil = no boundary) rather than a Bool: the
    /// caller needs to stamp the new span with when the switch really happened,
    /// and only the detector knows the candidate's first sample time. A plain
    /// `Double?` is enough — there is nothing else to report — so no result
    /// struct is introduced.
    ///
    /// Non-nil ONLY when `clusterID` differs from the current stable cluster,
    /// has been seen `consecutiveRequired` times in a row, AND `t - lastFired >=
    /// minIntervalSec`. The FIRST stable cluster of a session never fires (nothing
    /// to split). On a fire, `clusterID` becomes the new stable cluster.
    ///
    /// Debounce decides only WHETHER a boundary exists. While a candidate is
    /// building or is rate-limited, no boundary is emitted, so the elapsed time
    /// keeps accruing to the CURRENT span — a suppressed fire can never punch a
    /// hole in the timeline, it only means the switch is not recognised at all.
    mutating func push(t: Double, clusterID: Int) -> Double? {
        // Establish the first stable cluster silently — nothing to split yet.
        guard let stable = stableCluster else {
            stableCluster = clusterID
            clearCandidate()
            return nil
        }
        // Back on the stable cluster — reset any building candidate.
        if clusterID == stable {
            clearCandidate()
            return nil
        }
        // A different cluster — accumulate confirmation.
        if clusterID == candidate {
            candidateCount += 1
        } else {
            candidate = clusterID
            candidateCount = 1
            candidateFirstT = t
        }
        guard candidateCount >= consecutiveRequired else { return nil }
        // Confirmed change, but rate-limit: suppress a fire too soon after the
        // last one (keep the candidate building so it fires once the interval passes).
        if let last = lastFired, t - last < minIntervalSec { return nil }
        // Report when the switch STARTED, not now. Under the rate limit this can
        // predate `lastFired`; `PositionTimeline.append` clamps monotonically, so
        // an out-of-order report degrades to a zero-length gap, never a reordering.
        let boundary = candidateFirstT ?? t
        stableCluster = clusterID
        clearCandidate()
        lastFired = t
        return boundary
    }

    /// Switch the stable cluster IMMEDIATELY, with no confirmation or rate-limit —
    /// for a brand-new direction (outside every stored speaker's range), which is
    /// an unambiguous new speaker, not jitter. Returns the boundary time to fire,
    /// except for the very first speaker of the session (nothing to split from).
    /// There is no confirmation lag to undo here, so the boundary is simply `t`.
    mutating func forceChange(to clusterID: Int, at t: Double) -> Double? {
        clearCandidate()
        let hadStable = stableCluster != nil
        stableCluster = clusterID
        guard hadStable else { return nil }
        lastFired = t
        return t
    }

    mutating func reset() {
        stableCluster = nil
        clearCandidate()
        lastFired = nil
    }

    private mutating func clearCandidate() {
        candidate = nil
        candidateCount = 0
        candidateFirstT = nil
    }
}

/// Event-driven boundary timeline: an ordered list of "from time T the talker is
/// cluster C" events, from which spans are DERIVED. The point of the type is what
/// it refuses to do — it never drops a boundary for being short-lived, because
/// dropping a boundary drops a stretch of TIME, and uncovered time is what leaves
/// words with no row to land in (`SPEAKER UNKNOWN`). Debounce belongs upstream,
/// in `ClusterChangeDetector`, where it decides whether an event exists at all.
struct PositionTimeline: Equatable {
    private(set) var boundaries: [(t: Double, clusterID: Int)] = []

    init() {}

    /// Append a boundary, clamping `t` to be ≥ the previous boundary so the list
    /// can never come out of order. Back-dating (see `BoundaryBackdating`) moves
    /// boundaries EARLIER, which can otherwise push one before its predecessor;
    /// clamping degrades that to a zero-length span rather than a scrambled list.
    mutating func append(t: Double, clusterID: Int) {
        let clamped = max(t, boundaries.last?.t ?? t)
        boundaries.append((t: clamped, clusterID: clusterID))
    }

    mutating func reset() {
        boundaries.removeAll(keepingCapacity: true)
    }

    /// Spans that TILE `range` completely from the first boundary onward: no
    /// minimum duration, no minimum sample count, no gaps, no dropping of short
    /// spans. Each boundary runs until the next one; the last runs to the end of
    /// `range`.
    ///
    /// The part of `range` BEFORE the first boundary is deliberately left
    /// uncovered — that is pre-speech silence, and covering it would attribute
    /// words to a talker who had not spoken yet.
    ///
    /// Consecutive boundaries carrying the same cluster collapse into a single
    /// span (they describe one continuous turn), so the caller never sees two
    /// adjacent identical spans.
    func spans(in range: ClosedRange<Double>) -> [(start: Double, end: Double, clusterID: Int)] {
        guard !boundaries.isEmpty else { return [] }

        var out: [(start: Double, end: Double, clusterID: Int)] = []
        for (i, b) in boundaries.enumerated() {
            // Open-ended last boundary: the timeline says nothing has changed
            // since, so it owns the rest of the queried window.
            let rawEnd = i + 1 < boundaries.count ? boundaries[i + 1].t : Double.infinity
            let start = max(b.t, range.lowerBound)
            let end = min(rawEnd, range.upperBound)
            guard end > start else { continue }   // clipped away, or zero-length
            // Collapse same-cluster neighbours AFTER clipping, not before: the
            // clamp in `append` can leave a zero-length boundary between two
            // spans of the same cluster, and dropping it would otherwise emit
            // two adjacent identical spans.
            if var last = out.last, last.clusterID == b.clusterID, last.end == start {
                last.end = end
                out[out.count - 1] = last
            } else {
                out.append((start: start, end: end, clusterID: b.clusterID))
            }
        }
        return out
    }

    static func == (lhs: PositionTimeline, rhs: PositionTimeline) -> Bool {
        lhs.boundaries.count == rhs.boundaries.count
            && zip(lhs.boundaries, rhs.boundaries).allSatisfy { $0.t == $1.t && $0.clusterID == $1.clusterID }
    }
}

/// Undoes the systematic lateness of a smoothed boundary time. Pure and
/// side-effect free so it can be unit-tested on its own; Phase 2 just calls it.
enum BoundaryBackdating {
    /// `DirectionSmoother` emits, at time `t`, the median of the TRAILING
    /// `windowSec`. That median only flips to a new direction once the new
    /// direction holds a MAJORITY of the window, i.e. about `windowSec / 2`
    /// after the real switch — so a boundary stamped `t` is that much late.
    ///
    /// The first emission after a reset/gap is a different case: the smoother
    /// warms up for a FULL `windowSec` before it emits at all, so that emission
    /// is ~`windowSec` late and the exact correction is the raw stream's own
    /// `windowStart` — no estimate needed.
    ///
    /// Roughly one word of speech rides on this correction, which is exactly the
    /// word that decides whether a turn's first word lands in the right row.
    ///
    /// `previousBoundaryT` clamps the result so the timeline stays ordered: a
    /// correction may never move a boundary before the one it follows.
    static func correctedBoundary(reportedT: Double,
                                  windowSec: Double,
                                  isFirstEmissionAfterReset: Bool,
                                  windowStart: Double?,
                                  previousBoundaryT: Double?) -> Double {
        let corrected: Double
        if isFirstEmissionAfterReset, let start = windowStart {
            corrected = start
        } else {
            corrected = reportedT - windowSec / 2
        }
        guard let previous = previousBoundaryT else { return corrected }
        return max(previous, corrected)
    }
}
