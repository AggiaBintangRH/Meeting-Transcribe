import Combine
import Foundation

/// Recorder-owned collector that turns the ATND beam-notice stream into
/// position-based speaker clusters on the shared `recordingElapsed` clock.
///
/// It subscribes to `ATNDBeamService.shared.rawNotices` (the honest per-notice
/// stream, fired before the status-chip hold/stale logic). Each `.talking`
/// notice is timestamped with the current `recordingElapsed`, smoothed, and
/// clustered by beam direction; `.silence` and long gaps reset the smoother.
///
/// NOT a singleton — one per recording session, created and destroyed by
/// `AudioRecorder`. Cluster ids are mapped to *position ids* by adding
/// `positionIDBase`, which keeps them permanently disjoint from pyannote's
/// profile ids so a position id can never reach the Python-owned
/// `SpeakerProfileStore`.
@MainActor
final class PositionDiarizer: ObservableObject {
    /// Position ids start here; never collide with pyannote profile ids (small ints).
    static let positionIDBase = 100_000

    enum Mode { case firstCome, enrollment }

    /// The cluster id currently awaiting a name in enrollment mode, or nil.
    /// First-come mode never sets it. Additional births while a prompt is open
    /// wait in `enrollmentQueue`; this always holds the head.
    @Published private(set) var pendingEnrollment: Int?

    /// Clusters born while an enrollment prompt was already open, waiting their
    /// turn. `pendingEnrollment` is the head; this is everything behind it.
    private var enrollmentQueue: [Int] = []

    private(set) var isActive = false

    private var mode: Mode = .firstCome
    private var nowFn: () -> Double = { 0 }

    private var smoother = DirectionSmoother()
    private var clusterer = PositionClusterer()

    /// cluster id → display name.
    private var names: [Int: String] = [:]

    /// Last notice arrival on the shared clock — a gap > this resets the smoother.
    private var lastNoticeElapsed: Double?
    private let gapResetSec: Double = 1.0

    private var cancellable: AnyCancellable?

    // MARK: - Lifecycle

    /// Subscribe to the shared beam service. `now` yields the current
    /// `recordingElapsed` each time a notice arrives — the shared audio clock.
    ///
    /// The subscription is to the SHARED `ATNDBeamService` singleton's subject,
    /// which lives on the service, not the socket — so a control-link flap that
    /// tears down and rebuilds the UDP socket does NOT require re-subscription.
    func start(tauDeg: Double, smoothingSec: Double, mode: Mode, now: @escaping () -> Double) {
        self.mode = mode
        self.nowFn = now
        self.smoother = DirectionSmoother(windowSec: smoothingSec)
        self.clusterer = PositionClusterer(tauDeg: tauDeg)
        self.names = [:]
        self.lastNoticeElapsed = nil
        self.pendingEnrollment = nil
        self.enrollmentQueue = []
        self.isActive = true

        cancellable = ATNDBeamService.shared.rawNotices
            .sink { [weak self] event in
                self?.ingest(event.notice)
            }
    }

    /// Stop ingesting new notices, but KEEP the collected data — the final
    /// diarization pass still calls `label(for:)`/`sampleCount(in:)` afterward.
    func stop() {
        cancellable?.cancel()
        cancellable = nil
        isActive = false
    }

    // MARK: - Ingest

    private func ingest(_ notice: ATNDBeamService.ParsedNotice) {
        let t = nowFn()

        // A gap between notices (dropped/stale stream) breaks continuity.
        if let last = lastNoticeElapsed, t - last > gapResetSec {
            smoother.reset()
        }
        lastNoticeElapsed = t

        switch notice {
        case .silence:
            smoother.reset()
        case .talking(let talker):
            let vector = PositionMath.unitVector(rotateDeg: Double(talker.rotation),
                                                 angleDeg: Double(talker.elevation))
            guard let sample = smoother.push(t: t, vector: vector) else { return }
            let result = clusterer.assign(sample)
            if result.isNew {
                assignName(clusterID: result.clusterID)
            }
        }
    }

    /// Label a freshly-born cluster. First-come auto-names it immediately and
    /// never raises a prompt. Enrollment does NOT name it yet — it raises
    /// `pendingEnrollment` (queueing behind any prompt already open) and lets
    /// `label(for:)` return a provisional "Speaker N" until the name arrives.
    ///
    /// Only the labeling differs by mode; the clustering that produced
    /// `clusterID` (smoother + clusterer) ran identically before this call.
    private func assignName(clusterID: Int) {
        switch mode {
        case .firstCome:
            names[clusterID] = "Speaker \(names.count + 1)"
        case .enrollment:
            if pendingEnrollment == nil {
                pendingEnrollment = clusterID
            } else {
                enrollmentQueue.append(clusterID)
            }
        }
    }

    /// Name the head pending cluster (empty/whitespace name = keep provisional),
    /// then publish the next queued birth, or nil when the queue drains.
    func resolveEnrollment(name: String) {
        guard let clusterID = pendingEnrollment else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            names[clusterID] = trimmed
        }
        advanceEnrollment()
    }

    /// Leave the head pending cluster with its provisional "Speaker N" and
    /// publish the next queued birth (or nil).
    func skipEnrollment() {
        guard pendingEnrollment != nil else { return }
        advanceEnrollment()
    }

    private func advanceEnrollment() {
        pendingEnrollment = enrollmentQueue.isEmpty ? nil : enrollmentQueue.removeFirst()
    }

    // MARK: - Query (used by the gap-fill seam, valid after stop())

    /// Dominant cluster over `range`, as (position id, name), or nil if there
    /// were too few in-range samples. The id is `positionIDBase + clusterID`.
    /// `minSamples` is the density gate — a range with fewer in-range samples
    /// returns nil (so a silent gap stays UNKNOWN rather than being force-filled).
    func label(for range: ClosedRange<Double>, minSamples: Int = 1) -> (id: Int, name: String)? {
        guard let clusterID = clusterer.dominantCluster(in: range, minSamples: minSamples) else { return nil }
        let name = names[clusterID] ?? "Speaker \(clusterID + 1)"
        return (Self.positionIDBase + clusterID, name)
    }

    func sampleCount(in range: ClosedRange<Double>) -> Int {
        clusterer.sampleCount(in: range)
    }

    /// Rename a position cluster. `clusterID` is the RAW cluster id (already
    /// stripped of `positionIDBase` by the caller).
    func rename(clusterID: Int, to name: String) {
        names[clusterID] = name
    }
}
