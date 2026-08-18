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
    /// `nonisolated` because it is a plain constant that the id-range guards
    /// (`AudioRecorder.officeTurnsOnly`, `SpeakerProfileStore.isProfileID`) read
    /// from non-MainActor code — the class itself stays MainActor-isolated.
    nonisolated static let positionIDBase = 100_000

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

    /// Fires when the beam settles on a genuinely different talker cluster than
    /// the current one — the recorder uses it to split the live transcript into a
    /// new row in real time. Reset each session in `start(...)`.
    private var changeDetector = ClusterChangeDetector()

    /// Event-driven boundary timeline — every confirmed direction change, stamped
    /// with the ATND sample's own audio-clock time (never with when the app
    /// noticed, and never with when text arrived: text is late by seconds, so
    /// finalizing a row at processing time would make boundaries drift with
    /// machine load).
    ///
    /// This is the ONLY source of position spans for the transcript — the earlier
    /// after-the-fact turn reconstruction is gone. It could drop a short run as
    /// flicker, and dropping a run dropped a stretch of TIME, which is what left
    /// words with no row (`SPEAKER UNKNOWN`) while a talker moved between seats.
    private var timeline = PositionTimeline()

    /// Called (on the MainActor, from `ingest`) when `changeDetector` confirms a
    /// talker switch. Installed by `AudioRecorder`; nil = no real-time splitting.
    var onClusterChange: (() -> Void)?

    /// cluster id → display name.
    private var names: [Int: String] = [:]

    /// Last notice arrival on the shared clock — a gap > this resets the smoother.
    private var lastNoticeElapsed: Double?
    private let gapResetSec: Double = 1.0

    /// Only collect direction while OUR OWN VAD says someone is speaking.
    ///
    /// The array's beam follows any sound, not only speech: measured on device,
    /// piano through the room speakers still produced angle/rotation notices
    /// with the ATND's own VAD (`SVAD`) switched ON — that setting does not gate
    /// camera tracking. Ungated, a slammed door or music builds direction
    /// clusters for positions where nobody ever spoke.
    ///
    /// Silero is the right gate because it is the same voice decision the rest
    /// of the pipeline already trusts for chunk boundaries. When VAD is off in
    /// Settings there is no verdict to gate on, so the gate is disabled outright
    /// rather than silently discarding every notice.
    private var gateOnSpeech = false
    private var lastSpeechElapsed: Double?
    /// Speech verdicts arrive per audio buffer and beam notices at 10 Hz, so the
    /// two are never exactly aligned. Holding the gate open briefly after the
    /// last speech keeps a normal utterance continuous instead of punching
    /// holes in it on every inter-word pause.
    private let speechHoldSec: Double = 1.0

    private var cancellable: AnyCancellable?

    // MARK: - Lifecycle

    /// Subscribe to the shared beam service. `now` yields the current
    /// `recordingElapsed` each time a notice arrives — the shared audio clock.
    ///
    /// The subscription is to the SHARED `ATNDBeamService` singleton's subject,
    /// which lives on the service, not the socket — so a control-link flap that
    /// tears down and rebuilds the UDP socket does NOT require re-subscription.
    func start(tauDeg: Double, smoothingSec: Double, mode: Mode,
               gateOnSpeech: Bool = false, now: @escaping () -> Double) {
        self.mode = mode
        self.nowFn = now
        self.gateOnSpeech = gateOnSpeech
        self.lastSpeechElapsed = nil
        self.smoother = DirectionSmoother(windowSec: smoothingSec)
        self.clusterer = PositionClusterer(tauDeg: tauDeg)
        self.changeDetector = ClusterChangeDetector()
        self.timeline = PositionTimeline()
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

    /// Our own VAD's verdict for the buffer that just ended, on the shared audio
    /// clock. Called from the capture tap; a no-op unless the gate is on.
    func noteSpeech(_ speaking: Bool, at elapsed: Double) {
        guard gateOnSpeech, speaking else { return }
        lastSpeechElapsed = elapsed
    }

    /// Whether direction may be collected at `t`.
    private func speechAllows(_ t: Double) -> Bool {
        guard gateOnSpeech else { return true }
        guard let last = lastSpeechElapsed else { return false }
        return t - last <= speechHoldSec
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
            // ⚠ THIS USED TO CALL `smoother.reset()`, AND IT WAS THE WHOLE BUG.
            //
            // The smoother emits nothing until it has held notices for a full
            // `windowSec` (0.4 s) since its last reset. At the array's 10 Hz that
            // is a run of ~5 consecutive notices. A reset here restarts that run
            // from zero — so a `.silence` arriving more often than about once a
            // second starves the clusterer PERMANENTLY, while angle and rotation
            // keep streaming and the array's own UI shows a talker the whole time.
            //
            // Measured in `PositionBlackoutTests`, 20 s streams at 10 Hz:
            //
            //     silence every 200 / 300 / 400 / 500 ms -> samples=0  spans=0
            //     silence every 1000 ms                  -> samples=88 spans=1
            //
            // That is the owner's report exactly: 186 `no-spans` lines in their
            // log, every one `samples=0`, with 1 s to 30 s blackouts.
            //
            // DOING NOTHING IS NOT "IGNORING THE SILENCE" — the smoother already
            // handles it, and better. It drops samples older than its trailing
            // window and resets itself on a gap larger than one window, so a
            // silence longer than 0.4 s still clears the buffer exactly as before,
            // while a 100 ms blip no longer destroys a nearly-complete warm-up.
            // The reason the reset existed (never average one talker's direction
            // into the next one's) is preserved by that time rule; what is gone is
            // punishing a brief blip as if it were a change of speaker.
            break
        case .talking(let talker):
            // The array is tracking SOMETHING, but only speech may become a
            // speaker position — measured 2026-07-27, piano through the room
            // speakers still moved the beam with the array's own SVAD on.
            //
            // ⚠ NO RESET HERE EITHER, for the same reason and on the same
            // evidence. The gate closing is a statement about THIS instant, not a
            // verdict on the samples already collected. Skipping the sample keeps
            // the noise out, which is all the gate was ever for; resetting also
            // threw away the warm-up, so a normal speaking rhythm could not
            // accumulate one. Measured: with the gate flickering 300 ms open in
            // every 700 ms, the reset version still reached 196 samples on its own
            // — so this one was NOT the dominant cause, and it is fixed here only
            // because it stacks with the silence case above (both together
            // measured samples=0, either alone did not).
            guard speechAllows(t) else { return }
            let vector = PositionMath.unitVector(rotateDeg: Double(talker.rotation),
                                                 angleDeg: Double(talker.elevation))
            guard let sample = smoother.push(t: t, vector: vector) else { return }
            let result = clusterer.assign(sample)
            if result.isNew {
                assignName(clusterID: result.clusterID)
            }
            // One rule for both paths (see `PositionBoundaryRule`): a brand-new
            // direction switches immediately (no debounce, so a short ~1 s turn
            // isn't swallowed by the rate limit), a return to a KNOWN speaker is
            // confirmed + rate-limited so jitter between two nearby stored
            // speakers can't thrash the rows. It also records the boundary on the
            // timeline, including the first speaker's — which fires nothing.
            let didChange = PositionBoundaryRule.apply(sample: sample,
                                                       clusterID: result.clusterID,
                                                       isNewCluster: result.isNew,
                                                       smoother: smoother,
                                                       detector: &changeDetector,
                                                       timeline: &timeline)
            if didChange { onClusterChange?() }
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

    /// The label a LIVE CAPTION should show at `now` — responsive first, then the
    /// timeline, never nothing.
    ///
    /// ⚠ WHY THIS IS NOT JUST `label(for:minSamples:)` (owner, 2026-08-18, from a
    /// screenshot: *"itunya silent mungkin karena itu gak jadi speaker unknown"* —
    /// and they were right). The caption asked `dominantCluster` over the last
    /// second with `minSamples: 3`, while the ROWS ask `timeline.spans`, whose last
    /// boundary is OPEN-ENDED and therefore always has an answer. Direction is
    /// collected only while our own VAD hears speech, so a silent second yields no
    /// fresh samples and the caption fell to nil — `SPEAKER UNKNOWN` sitting above
    /// rows that were labelled perfectly well. Two readers of one fact, disagreeing.
    ///
    /// It bites hardest on QUIET capture, which is this project's live problem: the
    /// owner's Dante chain records at ~-47 dBFS, and Silero is level-sensitive
    /// enough that the same session measured 0 % speech on a -53 dBFS file that
    /// Whisper transcribed at 362 words. So the gate closes on audio that really
    /// does contain speech, and the caption goes blank mid-sentence.
    ///
    /// The recent-samples lookup stays FIRST and unchanged, because it is what
    /// makes the caption flip quickly on a talker switch — the reason `minSamples`
    /// was tightened in the first place. Only the nil case changes: instead of
    /// claiming ignorance, fall back to what the timeline already asserts — "the
    /// beam last settled here and nothing has changed since". That is the same
    /// sentence the rows below are printing, so the two surfaces now agree by
    /// construction rather than by coincidence.
    ///
    /// Still nil before the FIRST boundary of a session, which is correct: there is
    /// genuinely no talker to name yet.
    func captionLabel(at now: Double,
                      window: Double = 1.0,
                      minSamples: Int = 3) -> (id: Int, name: String)? {
        let range = max(0, now - window)...max(0, now)
        if let fresh = label(for: range, minSamples: minSamples) { return fresh }
        // `spans` clips to the query and its final boundary owns everything after
        // it, so this is empty ONLY when the session has no boundary at all.
        return labeledSpans(in: range).last.map { (id: $0.id, name: $0.name) }
    }

    func sampleCount(in range: ClosedRange<Double>) -> Int {
        clusterer.sampleCount(in: range)
    }

    /// "S1/S2 12.4°, S1/S3 47.1°" — the angular gap between each pair of stored
    /// speakers. Any pair below the configured threshold cannot be separated.
    func separationDescription() -> String {
        let pairs = clusterer.centroidSeparations()
        guard !pairs.isEmpty else { return "single cluster" }
        return pairs.map {
            let a = names[$0.a] ?? "cluster \($0.a)"
            let b = names[$0.b] ?? "cluster \($0.b)"
            return "\(a)/\(b) \(String(format: "%.1f", $0.deg))°"
        }.joined(separator: ", ")
    }

    /// Timeline spans over `range` as `(positionIDBase + clusterID, name)` — the
    /// payload the gap-fill fills pyannote's uncovered stretches with. Valid after
    /// `stop()` (data kept), same as `label(for:)`, since the final diarization
    /// pass queries it.
    ///
    /// These TILE `range` from the first boundary on: no minimum duration, no
    /// density gate, nothing dropped, and already clipped to `range` — so a caller
    /// never has to close holes between them.
    func labeledSpans(in range: ClosedRange<Double>)
        -> [(start: Double, end: Double, id: Int, name: String)] {
        timeline.spans(in: range).map { span in
            (start: span.start,
             end: span.end,
             id: Self.positionIDBase + span.clusterID,
             name: names[span.clusterID] ?? "Speaker \(span.clusterID + 1)")
        }
    }

    /// Rename a position cluster. `clusterID` is the RAW cluster id (already
    /// stripped of `positionIDBase` by the caller).
    func rename(clusterID: Int, to name: String) {
        names[clusterID] = name
    }
}
