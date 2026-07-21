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

    /// Whether a stable cluster has been established since the last reset.
    /// Lets a caller distinguish "this sample established the FIRST stable
    /// cluster" from "nothing happened" — both return nil, by design, because
    /// there is no previous row to split from. The timeline still needs a
    /// boundary in the first case (the first speaker's span has to start
    /// somewhere), and reading this before/after the call gets it without
    /// changing what `push`/`forceChange` fire on.
    var hasStableCluster: Bool { stableCluster != nil }

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
    /// A boundary may never be moved back by more than this fraction of the gap
    /// to the boundary before it. See `correctedBoundary` for why a plain clamp
    /// at the previous boundary is not enough. One HALF is the natural choice:
    /// the correction being applied is itself `windowSec / 2`, and half leaves the
    /// preceding span at least half of the duration it was actually observed to
    /// have — short enough to still buy most of the correction on a normal-length
    /// turn, generous enough that a legitimately ~1 s turn keeps a usable span.
    static let maxBackdateFraction = 0.5

    /// `previousBoundaryT` bounds the result so the timeline stays ordered AND so
    /// the preceding span survives.
    ///
    /// Clamping to `max(previous, corrected)` — the obvious rule — is a bug, and
    /// it is the exact bug this redesign exists to kill: a boundary at 5.0
    /// followed by one back-dated to 4.9 clamps the second to 5.0, the first
    /// span becomes `[5.0, 5.0]`, `spans(in:)` drops it, and that speaker
    /// DISAPPEARS from the transcript. Back-dating must never be able to erase
    /// the turn it is dating away from.
    ///
    /// So the correction is made proportional to the distance from the previous
    /// boundary instead: at most `maxBackdateFraction` of that gap may be given
    /// back. The result then always lands STRICTLY after the previous boundary
    /// whenever the raw report does, so the preceding span keeps a non-zero
    /// duration no matter how aggressive the correction is.
    ///
    /// The one case that still collapses is a report at or before the previous
    /// boundary (possible when the rate limit delays a fire past a later
    /// boundary): there is no time to distribute, so it degrades to the previous
    /// boundary — two events genuinely at the same instant, later one wins.
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
        let gap = reportedT - previous
        guard gap > 0 else { return previous }
        let floorT = reportedT - gap * maxBackdateFraction
        return max(floorT, corrected)
    }
}

/// The per-sample boundary decision that drives `PositionTimeline` — factored out
/// of `PositionDiarizer.ingest` so it can be exercised without a `@MainActor`
/// object subscribed to the `ATNDBeamService` singleton. Pure: everything it
/// touches arrives as a parameter.
enum PositionBoundaryRule {
    /// Advance `detector`/`timeline` for one smoothed sample, returning whether
    /// the caller should fire its real-time row split (`onClusterChange`).
    ///
    /// Three boundary sources, matching the owner's spec that EVERY direction
    /// change starts a new row:
    ///  - the FIRST stable cluster of the session — it fires nothing (there is no
    ///    previous row to split from) but must still be recorded, otherwise the
    ///    first speaker has no span at all;
    ///  - `forceChange` — a direction outside every stored speaker's range, taken
    ///    immediately;
    ///  - `push` — a return to an already-stored speaker, confirmed and
    ///    rate-limited.
    ///
    /// The two firing paths report DIFFERENT times (`forceChange` reports the
    /// sample's own `t`, `push` the candidate's first sample time), but the same
    /// back-dating applies to both: each of those times is itself a smoother
    /// emission, and every emission is late by the same amount. The first-emission
    /// -after-reset special case can only ever coincide with `forceChange`, since
    /// a `push` fire needs `consecutiveRequired` emissions behind it — so reading
    /// the smoother's CURRENT reset state is correct for both.
    @discardableResult
    static func apply(sample: DirectionSample,
                      clusterID: Int,
                      isNewCluster: Bool,
                      smoother: DirectionSmoother,
                      detector: inout ClusterChangeDetector,
                      timeline: inout PositionTimeline) -> Bool {
        let hadStable = detector.hasStableCluster

        let fired = isNewCluster
            ? detector.forceChange(to: clusterID, at: sample.t)
            : detector.push(t: sample.t, clusterID: clusterID)

        if let boundary = fired {
            record(reportedT: boundary, clusterID: clusterID,
                   smoother: smoother, timeline: &timeline)
            return true
        }
        if !hadStable, detector.hasStableCluster {
            // Speech just started: open the first speaker's span here.
            record(reportedT: sample.t, clusterID: clusterID,
                   smoother: smoother, timeline: &timeline)
        }
        return false
    }

    private static func record(reportedT: Double,
                               clusterID: Int,
                               smoother: DirectionSmoother,
                               timeline: inout PositionTimeline) {
        let t = BoundaryBackdating.correctedBoundary(
            reportedT: reportedT,
            windowSec: smoother.windowSec,
            isFirstEmissionAfterReset: smoother.lastEmissionWasFirstSinceReset,
            windowStart: smoother.windowStart,
            previousBoundaryT: timeline.boundaries.last?.t)
        timeline.append(t: t, clusterID: clusterID)
    }
}
