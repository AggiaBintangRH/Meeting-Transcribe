import Foundation

/// Loads (prepares) all models needed for a recording session, step by step,
/// publishing per-model progress for the loading overlay.
///
/// Currently each step verifies the model is downloaded and reserves the slot
/// where the real MLX / Silero initialization will run when the ASR pipeline
/// is wired in — the UI and call sites won't change.
@MainActor
final class ModelLoader: ObservableObject {

    enum ItemState: Equatable {
        case pending, loading, done
        case failed(String)
    }

    struct Item: Identifiable {
        let id: String
        let name: String
        var state: ItemState = .pending
    }

    @Published private(set) var items: [Item] = []
    @Published private(set) var isLoading = false

    /// Set when a step fails; keeps the overlay on screen so the user can
    /// read what went wrong. Cleared via `dismissFailure()`.
    @Published private(set) var failureMessage: String?

    func dismissFailure() {
        failureMessage = nil
    }

    /// Refuse to start the session BEFORE any model loads, showing the reason in
    /// the same overlay a load failure uses (one failed row + `failureMessage`,
    /// which is what keeps the overlay on screen). The only caller today is the
    /// dual-stream + Voxtral refusal in `AudioRecorder.prepareAndCapture`: the
    /// configuration is unworkable, so loading a 4B model first would waste ~30 s
    /// before saying so.
    func failStartup(step: String, message: String) {
        items = [Item(id: "startup", name: step, state: .failed(message))]
        failureMessage = message
    }

    /// Silero sidecar, started during loading; nil → heuristic VAD fallback.
    /// Kept alive across sessions so the model only loads once.
    private(set) var sileroVAD: SileroVADService?

    /// Nemotron realtime ASR sidecar; recreated only when its settings change.
    /// ONE process serving both streams: its `office` and `remote` lanes keep
    /// separate buffers inside the sidecar, so a Remote channel costs no second
    /// model load and no second process — see `NemotronASRService`.
    private(set) var nemotronASR: NemotronASRService?

    /// Chunked ASR sidecar (Qwen3/Whisper/Voxtral per settings); persistent.
    private(set) var chunkedASR: ChunkedASRService?

    /// Forced-aligner sidecar (Qwen3-ForcedAligner); persistent, kept alive
    /// across sessions like `diarization` so its 1.2 GB loads once. nil whenever
    /// this session does not use word alignment — which is what makes the whole
    /// alignment path a natural no-op rather than a set of scattered guards.
    private(set) var aligner: AlignerService?

    /// Diarization sidecar (pyannote community-1 on MPS); persistent.
    /// nil whenever `diarization.engine` is MOSS — there is no pyannote process
    /// at all then, which is what makes every pyannote-dependent path
    /// (remote diarization, overlap repair, profiles) a natural no-op.
    ///
    /// Turns only, no identity: since 2026-07-30 this sidecar answers who-spoke-when
    /// with run-local labels and `embedding` says who those people are.
    private(set) var pyannote: PyannoteService?

    /// Speaker-identity sidecar (WeSpeaker + the two profile stores); persistent.
    /// The other half of the former `diarize-service.py`.
    ///
    /// Loaded and torn down in LOCKSTEP with `pyannote`, and that is not a
    /// convenience: a pyannote turn has no name until this service supplies one,
    /// and `SpeakerTurn` has no representation for an unidentified turn. So both
    /// exist or neither does — see `buildSteps` (two steps, embedder first) and
    /// the MOSS-engine teardown below (both terminated).
    private(set) var embedding: WeSpeakerService?

    /// A SECOND MOSS process, used only for its speaker segmentation while a
    /// different model does the ASR. nil in the primary MOSS+MOSS mode, where
    /// `chunkedASR` IS the MOSS process and one 3.6 GB load serves both roles —
    /// see `needsSecondMossProcess`.
    private(set) var mossDiarization: ChunkedASRService?

    /// Overlap-repair sidecar (MossFormer2 librimix-2spk); persistent.
    /// Only started when overlap repair is enabled in Settings.
    private(set) var overlapRepair: OverlapRepairService?

    /// Overlap-repair sidecar (DiCoW v3.3 target-speaker ASR); persistent.
    /// The alternative to `overlapRepair` — only one engine loads per session,
    /// whichever is picked in Settings → Models → Overlap.
    private(set) var dicowRepair: DicowService?

    /// Whether this session should drive the realtime sidecar's REMOTE lane.
    /// Both conditions are necessary: no Remote channel means there is no second
    /// stream at all, and with realtime captions off there is nothing for either
    /// lane to draw. Pure, so the "inert without a Remote channel" guarantee is
    /// testable without audio hardware.
    nonisolated static func wantsRemoteRealtime(remoteChannel: Int?,
                                                realtimeEnabled: Bool) -> Bool {
        remoteChannel != nil && realtimeEnabled
    }

    /// The MOSS diarization engine id (`diarization.engine`).
    nonisolated static let mossEngineID = "moss"
    /// The pyannote diarization engine id — the default.
    nonisolated static let pyannoteEngineID = "pyannote"

    /// Whether this session loads the forced aligner.
    ///
    /// ONE rule, read by both `buildSteps` (does the overlay show a row?) and
    /// `loadAll` (is a leftover sidecar torn down?). They must agree: a step
    /// without a teardown leaves a 1.2 GB process alive for a session that will
    /// never ask it anything, and a teardown without a step would kill the
    /// service the session is about to use. Pure, like `wantsRemoteRealtime`.
    ///
    /// MOSS is excluded: it runs in its own sidecar and already returns its own
    /// per-segment times. The split makes aligning MOSS text POSSIBLE for the
    /// first time (the aligner no longer lives inside the MLX ASR process), but
    /// the owner deferred cross-chunk MOSS work, so it stays excluded and the
    /// Aligner tab says so.
    nonisolated static func wantsAligner(alignEnabled: Bool,
                                         chunkedID: String) -> Bool {
        alignEnabled && chunkedID != "moss"
    }

    /// Whether this session needs a SECOND MOSS process for diarization.
    ///
    /// False when the chunked model is already MOSS: that one sidecar's `final`
    /// carries both the text and the segments, so a second process would be a
    /// second 3.6 GB load and a second ~6.4 s per chunk for output we already
    /// have. It is also why the overlay must not show two rows for one process.
    ///
    /// Pure, so the "one process in the primary mode" guarantee is testable
    /// without loading anything (the `wantsRemoteRealtime` precedent).
    /// Which diarization stack this session should keep alive, or nil for none.
    ///
    /// Pure and static, and fed to BOTH the load step and the teardown — the two
    /// computing this separately is EXACTLY how the bug below happened.
    ///
    /// THE BUG (found 2026-07-31, fixed here). The load step asked
    /// `diarization.enabled`; the teardown only asked whether the engine was
    /// MOSS. So turning the master switch OFF stopped pyannote being LOADED but
    /// never STOPPED one already running, and services are kept alive across
    /// sessions. The next recording then found `modelLoader.pyannote != nil`,
    /// which is the only thing `diarizeLiveChunk` guards on besides
    /// `diarization.live` — so **diarization kept running after the user
    /// switched it off**, and 1.17 GB stayed resident with it. It only bit when
    /// the app was not restarted in between, which is why it survived so long.
    ///
    /// The MOSS branch had the memory half of the same bug: `mossDiarization`
    /// was kept whenever `needsSecondMossProcess` said so, and that function
    /// never knew about `diarization.enabled` either. Its behaviour half was
    /// already covered, because `configureMoss` reads the switch itself.
    nonisolated static func wantedDiarizationStack(diarizationEnabled: Bool,
                                                   engine: String,
                                                   chunkedID: String) -> DiarizationStack? {
        guard diarizationEnabled else { return nil }
        guard engine == mossEngineID else { return .pyannote }
        return needsSecondMossProcess(chunkedID: chunkedID, engine: engine)
            ? .mossSecondProcess : nil
    }

    /// What a session's diarization is served by. `nil` — the third case — means
    /// "nothing", which is both "switched off" and "MOSS doing both jobs in the
    /// chunked process".
    enum DiarizationStack: Equatable, Hashable {
        case pyannote           // pyannote-service + wespeaker-service
        case mossSecondProcess  // a second MOSS process, ASR done by another model
    }

    /// The overlap-repair engine id this session should keep alive, or nil for
    /// "neither" — the single source of truth for BOTH the load step and the
    /// teardown above, so the two can never disagree.
    ///
    /// Pure and static for the same reason `wantsAligner` and
    /// `needsSecondMossProcess` are: the rule is what matters, and it should be
    /// testable without a loader, a sidecar process or UserDefaults.
    ///
    /// Returns nil under the MOSS diarization engine even when repair is switched
    /// on, because both engines locate their windows from PYANNOTE turns and there
    /// are none in a MOSS session — the load step has always known this, and the
    /// teardown now agrees with it.
    nonisolated static func wantedOverlapEngine(repairEnabled: Bool,
                                                engineID: String,
                                                diarEngine: String) -> String? {
        guard repairEnabled, diarEngine != mossEngineID else { return nil }
        return engineID
    }

    nonisolated static func needsSecondMossProcess(chunkedID: String,
                                                   engine: String) -> Bool {
        engine == mossEngineID && chunkedID != "moss"
    }

    /// Load everything the session needs. Returns true if all succeeded.
    func loadAll() async -> Bool {
        isLoading = true
        failureMessage = nil
        defer { isLoading = false }

        let d = UserDefaults.standard
        let wantsRemote = Self.wantsRemoteRealtime(
            remoteChannel: MicrophoneSettings.current().remoteChannel,
            realtimeEnabled: d.object(forKey: "realtime.enabled") as? Bool ?? true)
        // A session that does not want remote captions must not keep a previous
        // session's remote wiring: there is no second process to orphan any more,
        // but the lane's callback would still hold the finished recorder, so
        // detach it here. Nothing feeds the lane afterwards, so it stays silent.
        if !wantsRemote {
            nemotronASR?.detachRemoteLane()
        }

        // Diarization engines are exclusive, and both services are kept alive
        // across sessions — so the one this session is NOT using has to be torn
        // down here, not merely left unloaded. Every pyannote-dependent path
        // (live/tail/final passes, remote diarization, overlap repair, the
        // profile store) guards on `diarization != nil`, which is exactly what
        // makes them natural no-ops under the MOSS engine; a leftover service
        // from a previous session would quietly re-enable all of them.
        let diarEngine = d.string(forKey: "diarization.engine") ?? Self.pyannoteEngineID
        let chunkedID = d.string(forKey: "chunked.model") ?? "qwen3"
        // BOTH halves of the split go together: leaving the embedder alive under
        // the MOSS engine would keep a process resident that nothing can ask
        // anything (MOSS turns never reach the profile stores), and leaving it
        // alive while pyannote was torn down is the state no code path expects.
        // The VAD was the last persistent service with no teardown at all —
        // every other one has had a rule since 2026-07-31. Turning it off is
        // already honoured at record time (`AudioRecorder` builds no
        // `VoiceActivityDetector`), so this is memory only (~0.20 GB), but a
        // service nothing can ask anything is exactly what the other rules exist
        // to stop.
        if !(d.object(forKey: "vad.enabled") as? Bool ?? true) {
            // Released rather than `terminate()`d: this service has no public
            // terminate — it kills its process in `deinit` — so dropping the last
            // reference IS the teardown. `AudioRecorder` only holds one for the
            // duration of a session (inside `VoiceActivityDetector`), and this
            // runs before capture starts, so the reference here is the last one.
            sileroVAD = nil
        }
        let wantedDiar = Self.wantedDiarizationStack(
            diarizationEnabled: d.object(forKey: "diarization.enabled") as? Bool ?? true,
            engine: diarEngine, chunkedID: chunkedID)
        // BOTH halves of the pyannote split go together: leaving the embedder
        // alive without the pipeline is a state no code path expects, and
        // leaving either alive when diarization is off is what let a switched-off
        // feature keep running. See `wantedDiarizationStack`.
        if wantedDiar != .pyannote {
            pyannote?.terminate()
            pyannote = nil
            embedding?.terminate()
            embedding = nil
        }
        if wantedDiar != .mossSecondProcess {
            mossDiarization?.terminate()
            mossDiarization = nil
        }
        // Same reasoning for the aligner, which is now a process of its own: a
        // session that does not align must not keep a previous session's sidecar
        // alive, because `AudioRecorder` decides whether to align at all by
        // asking whether this service exists.
        if !Self.wantsAligner(alignEnabled: d.object(forKey: "align.enabled") as? Bool ?? false,
                              chunkedID: chunkedID) {
            aligner?.terminate()
            aligner = nil
        }
        // The two OVERLAP engines are mutually exclusive in exactly the way the
        // two diarization engines above are, and are equally kept alive across
        // sessions — so the same rule has to apply, and until 2026-07-31 it did
        // not. `overlapRepair` and `dicowRepair` had no `terminate()` call
        // anywhere in the app, so:
        //   * turning overlap repair OFF left its sidecar resident until quit;
        //   * switching MossFormer2 ↔ DiCoW left BOTH resident at once;
        //   * selecting the MOSS diarization engine (which cannot run repair at
        //     all) left whichever engine was loaded running with nothing able to
        //     ask it anything.
        // DiCoW is a ~6 GB model in its own interpreter, so this was the largest
        // idle process the app could hold. The condition mirrors the load step's
        // exactly — `overlap.repair.enabled` AND not the MOSS engine — because a
        // teardown rule that disagrees with the load rule would either drop a
        // service this session needs or keep one it cannot use.
        let wantedRepair = Self.wantedOverlapEngine(
            repairEnabled: d.object(forKey: "overlap.repair.enabled") as? Bool ?? false,
            engineID: d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id,
            diarEngine: diarEngine)
        if wantedRepair != ModelCatalog.overlapSeparation.id {
            overlapRepair?.terminate()
            overlapRepair = nil
        }
        if wantedRepair != ModelCatalog.overlapDicow.id {
            dicowRepair?.terminate()
            dicowRepair = nil
        }

        // Built once and reused for both the overlay list and the run below, so
        // the indices into `items` can never disagree with the steps executed.
        // The remote lane adds NO step: it rides the one realtime sidecar.
        let steps = buildSteps()
        items = steps.map { Item(id: $0.model.id, name: $0.model.name) }
        var allOK = true

        for (index, step) in steps.enumerated() {
            items[index].state = .loading
            do {
                try await load(step)
                items[index].state = .done
            } catch {
                items[index].state = .failed(error.localizedDescription)
                failureMessage = error.localizedDescription
                allOK = false
                break
            }
        }
        return allOK
    }

    // MARK: - Steps

    private struct Step {
        let model: ModelInfo
        let checkInstalled: Bool
    }

    /// Which models this session needs, based on current settings.
    private func buildSteps() -> [Step] {
        let d = UserDefaults.standard
        var steps: [Step] = []
        let chunkedID = d.string(forKey: "chunked.model") ?? "qwen3"
        let diarEngine = d.string(forKey: "diarization.engine") ?? Self.pyannoteEngineID
        let mossDiar = diarEngine == Self.mossEngineID

        if d.object(forKey: "vad.enabled") as? Bool ?? true {
            steps.append(Step(model: ModelCatalog.vad, checkInstalled: false)) // energy VAD needs no files yet
        }
        if d.object(forKey: "realtime.enabled") as? Bool ?? true {
            // One step whether or not there is a Remote stream — the sidecar's
            // second lane needs no extra weights and no extra process.
            steps.append(Step(model: ModelCatalog.realtime, checkInstalled: true))
        }
        // Word aligner — its OWN sidecar since 2026-07-29, so this step both
        // verifies the weights and starts the process. Kept BEFORE the chunked
        // step so "aligner not downloaded" is reported before a multi-GB ASR
        // load, not after it. Skipped for MOSS — see `wantsAligner`.
        if Self.wantsAligner(alignEnabled: d.object(forKey: "align.enabled") as? Bool ?? false,
                             chunkedID: chunkedID) {
            steps.append(Step(model: ModelCatalog.wordAligner, checkInstalled: true))
        }
        // Chunked model (rolling accurate pass) — per settings selection
        steps.append(Step(model: ModelCatalog.chunkedModel(id: chunkedID), checkInstalled: true))
        // Diarization (runs after recording ends, but loads up front).
        // Engine-aware: pyannote gets a step only when it is the selected engine,
        // and MOSS gets one only when it needs a SECOND process — with MOSS in
        // both roles the chunked step above already IS that load, and two rows
        // for one process would misrepresent what is happening.
        // Same rule as the teardown in `prepare`, from the SAME function.
        switch Self.wantedDiarizationStack(
            diarizationEnabled: d.object(forKey: "diarization.enabled") as? Bool ?? true,
            engine: diarEngine, chunkedID: chunkedID) {
        case .none:
            break
        case .mossSecondProcess:
            steps.append(Step(model: ModelCatalog.mossDiarization, checkInstalled: true))
        case .pyannote:
            // TWO steps for the pyannote engine since the 2026-07-30 split: the
            // embedder and the pipeline are separate processes now, and the
            // overlay says so rather than hiding one inside the other. Embedder
            // FIRST on purpose — it is ~26 MB against the pipeline's minute, so a
            // missing/broken embedder is reported in seconds instead of after a
            // long load that was going to fail anyway.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
            steps.append(Step(model: ModelCatalog.diarization, checkInstalled: true))
        }
        // Overlap repair (runs at stop) — off by default. Loads whichever engine
        // is picked in Settings → Overlap (MossFormer2 or DiCoW). Both engines
        // locate their windows from pyannote turns, so neither can run under the
        // MOSS diarization engine — don't load a 6 GB model to do nothing.
        // Same rule as the teardown in `prepare`, from the SAME function — a load
        // step and a teardown that computed this separately is exactly how the
        // engines came to be started and never stopped.
        if let engineID = Self.wantedOverlapEngine(
            repairEnabled: d.object(forKey: "overlap.repair.enabled") as? Bool ?? false,
            engineID: d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id,
            diarEngine: diarEngine) {
            steps.append(Step(model: ModelCatalog.overlapEngine(id: engineID), checkInstalled: true))
        }
        return steps
    }

    private enum LoadError: LocalizedError {
        case notDownloaded(String)
        case sidecarFailed(String)
        var errorDescription: String? {
            switch self {
            case .notDownloaded(let name):
                return "\(name) is not downloaded. Run download-best-models.sh first."
            case .sidecarFailed(let name):
                return "\(name) failed to start — mlx-audio is probably missing. "
                     + "Run download-best-models.sh again (it now installs mlx-audio), then retry."
            }
        }
    }

    private func load(_ step: Step) async throws {
        if step.checkInstalled, !ModelCatalog.isInstalled(step.model) {
            throw LoadError.notDownloaded(step.model.name)
        }

        // Word aligner: start its persistent sidecar. It used to load inside the
        // chunked ASR process, so this step was a check and nothing else; the
        // aligner is now a process of its own and this is where it comes up.
        // Failure IS fatal for the session, exactly as it was before: the load
        // used to happen inside the ASR sidecar's startup, so a broken aligner
        // already stopped the meeting from starting rather than surfacing
        // mid-recording. Keeping that means the user is told at the overlay.
        if step.model.id == ModelCatalog.wordAligner.id {
            let config = AlignerService.Config.fromSettings()
            if let existing = aligner, existing.config == config {
                return
            }
            aligner?.terminate()
            aligner = nil
            aligner = try await Task.detached(priority: .userInitiated) {
                try AlignerService(config: config)
            }.value
            return
        }

        // Silero VAD: start the Python sidecar (real model load).
        // Failure is non-fatal — the heuristic VAD takes over.
        if step.model.id == ModelCatalog.vad.id {
            if sileroVAD == nil {
                sileroVAD = await Task.detached(priority: .userInitiated) {
                    SileroVADService()
                }.value
            }
            return
        }

        // Nemotron realtime ASR: start the MLX sidecar.
        // Reused across sessions; recreated only if its settings changed.
        if step.model.id == ModelCatalog.realtime.id {
            let config = NemotronASRService.Config.fromSettings()
            if let existing = nemotronASR, existing.config == config {
                return
            }
            nemotronASR?.terminate()
            nemotronASR = nil
            // Throws with the sidecar's exact error message (shown in overlay)
            nemotronASR = try await Task.detached(priority: .userInitiated) {
                try NemotronASRService(config: config)
            }.value
            return
        }

        // Chunked ASR: start the persistent sidecar for the selected model.
        if ModelCatalog.chunked.contains(where: { $0.id == step.model.id }) {
            let config = ChunkedASRService.Config.fromSettings()
            if let existing = chunkedASR, existing.config == config {
                return
            }
            chunkedASR?.terminate()
            chunkedASR = nil
            chunkedASR = try await Task.detached(priority: .userInitiated) {
                try ChunkedASRService(config: config)
            }.value
            return
        }

        // MOSS as the DIARIZATION engine while another model does the ASR: a
        // second MOSS process, driven by the same client and fed the same audio.
        // This step exists only when `needsSecondMossProcess` said so.
        if step.model.id == ModelCatalog.mossDiarization.id {
            let config = ChunkedASRService.Config.mossDiarization()
            if let existing = mossDiarization, existing.config == config {
                return
            }
            mossDiarization?.terminate()
            mossDiarization = nil
            mossDiarization = try await Task.detached(priority: .userInitiated) {
                try ChunkedASRService(config: config)
            }.value
            return
        }

        // Speaker identity: start the persistent WeSpeaker sidecar (loads on MPS).
        // Failure IS fatal for the session, deliberately, and this is the one
        // place to say why: `SpeakerTurn` has no representation for a turn whose
        // speaker is unknown, so there is nothing for a failed identify to
        // degrade TO. The house rule is refuse loudly, never substitute — the
        // same rule Voxtral + dual-stream follows.
        if step.model.id == ModelCatalog.speakerEmbedding.id {
            let config = WeSpeakerService.Config.fromSettings()
            if let existing = embedding, existing.config == config {
                return
            }
            embedding?.terminate()
            embedding = nil
            embedding = try await Task.detached(priority: .userInitiated) {
                try WeSpeakerService(config: config)
            }.value
            return
        }

        // Diarization: start the persistent pyannote sidecar (loads on MPS).
        if step.model.id == ModelCatalog.diarization.id {
            if pyannote == nil {
                pyannote = try await Task.detached(priority: .userInitiated) {
                    try PyannoteService()
                }.value
            }
            return
        }

        // Overlap repair: start the persistent DiCoW sidecar (loads once).
        // Runs in .venv-dicow, so it throws a setup error if that venv is absent.
        if step.model.id == ModelCatalog.overlapDicow.id {
            if dicowRepair == nil {
                dicowRepair = try await Task.detached(priority: .userInitiated) {
                    try DicowService()
                }.value
            }
            return
        }

        // Overlap repair: start the persistent MossFormer2 sidecar (loads once).
        if step.model.id == ModelCatalog.overlapSeparation.id {
            if overlapRepair == nil {
                overlapRepair = try await Task.detached(priority: .userInitiated) {
                    try OverlapRepairService()
                }.value
            }
            return
        }

    }
}
