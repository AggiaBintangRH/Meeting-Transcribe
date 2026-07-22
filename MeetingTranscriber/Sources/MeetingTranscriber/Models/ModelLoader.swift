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
    private(set) var nemotronASR: NemotronASRService?

    /// Second Nemotron sidecar, captioning the Remote (conferencing) stream.
    /// Nil unless this session both has a Remote channel and wants realtime
    /// captions — see `wantsRemoteRealtime`. Kept alive across sessions exactly
    /// like `nemotronASR` (a reload costs ~30 s), but torn down as soon as a
    /// session no longer wants it so a single-stream meeting is not paying for a
    /// second resident model.
    private(set) var remoteNemotronASR: NemotronASRService?

    /// Chunked ASR sidecar (Qwen3/Whisper/Voxtral per settings); persistent.
    private(set) var chunkedASR: ChunkedASRService?

    /// Diarization sidecar (pyannote community-1 on MPS); persistent.
    private(set) var diarization: DiarizationService?

    /// Overlap-repair sidecar (MossFormer2 librimix-2spk); persistent.
    /// Only started when overlap repair is enabled in Settings.
    private(set) var overlapRepair: OverlapRepairService?

    /// Overlap-repair sidecar (DiCoW v3.3 target-speaker ASR); persistent.
    /// The alternative to `overlapRepair` — only one engine loads per session,
    /// whichever is picked in Settings → Models → Overlap.
    private(set) var dicowRepair: DicowService?

    /// Whether this session should run a SECOND realtime engine for the Remote
    /// stream. Both conditions are necessary: no Remote channel means there is no
    /// second stream at all, and with realtime captions off there is nothing for
    /// either engine to draw. Pure, so the "inert without a Remote channel"
    /// guarantee is testable without audio hardware.
    nonisolated static func wantsRemoteRealtime(remoteChannel: Int?,
                                                realtimeEnabled: Bool) -> Bool {
        remoteChannel != nil && realtimeEnabled
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
        // A session that does not want remote captions must not keep paying for
        // them: the second sidecar is a whole resident model, so shut a leftover
        // one down here rather than letting it idle for the rest of the app's life.
        if !wantsRemote, remoteNemotronASR != nil {
            remoteNemotronASR?.terminate()
            remoteNemotronASR = nil
        }

        // Built once and reused for both the overlay list and the run below, so
        // the indices into `items` can never disagree with the steps executed.
        let steps = buildSteps(includeRemoteRealtime: wantsRemote)
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
    /// `includeRemoteRealtime` is decided by the caller (it needs the resolved
    /// microphone selection, which this function has no business reading).
    private func buildSteps(includeRemoteRealtime: Bool) -> [Step] {
        let d = UserDefaults.standard
        var steps: [Step] = []

        if d.object(forKey: "vad.enabled") as? Bool ?? true {
            steps.append(Step(model: ModelCatalog.vad, checkInstalled: false)) // energy VAD needs no files yet
        }
        if d.object(forKey: "realtime.enabled") as? Bool ?? true {
            steps.append(Step(model: ModelCatalog.realtime, checkInstalled: true))
            // Same weights, second process — right after the office engine so the
            // overlay reads as one realtime pair rather than an unrelated model.
            if includeRemoteRealtime {
                steps.append(Step(model: ModelCatalog.realtimeRemote, checkInstalled: true))
            }
        }
        // Word aligner — no sidecar of its own (it loads inside the chunked ASR
        // process), so this step only verifies the weights are there. Checked
        // BEFORE the chunked step: without them the chunked sidecar itself fails
        // to start, and "aligner not downloaded" is the useful message.
        if d.object(forKey: "align.enabled") as? Bool ?? false {
            steps.append(Step(model: ModelCatalog.wordAligner, checkInstalled: true))
        }
        // Chunked model (rolling accurate pass) — per settings selection
        let chunkedID = d.string(forKey: "chunked.model") ?? "qwen3"
        steps.append(Step(model: ModelCatalog.chunkedModel(id: chunkedID), checkInstalled: true))
        // Diarization (runs after recording ends, but loads up front)
        if d.object(forKey: "diarization.enabled") as? Bool ?? true {
            steps.append(Step(model: ModelCatalog.diarization, checkInstalled: true))
        }
        // Overlap repair (runs at stop) — off by default. Loads whichever engine
        // is picked in Settings → Overlap (MossFormer2 or DiCoW).
        if d.object(forKey: "overlap.repair.enabled") as? Bool ?? false {
            let engineID = d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id
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

        // Word aligner: nothing to start — the chunked ASR sidecar loads it
        // itself when Config.alignRepoID is set. The check above is the step.
        if step.model.id == ModelCatalog.wordAligner.id {
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

        // Nemotron realtime ASR for the Remote stream: a SECOND sidecar process
        // with the same config as the office one. Its own branch (not a shared
        // one keyed on a flag) because the two must never be confused: the office
        // engine drives the clock, VAD and both cadences, this one only captions.
        // On a failed start `NemotronASRService.init` terminates its own process
        // and throws, so nothing is left running and the property stays nil.
        if step.model.id == ModelCatalog.realtimeRemote.id {
            let config = NemotronASRService.Config.fromSettings()
            if let existing = remoteNemotronASR, existing.config == config {
                return
            }
            remoteNemotronASR?.terminate()
            remoteNemotronASR = nil
            remoteNemotronASR = try await Task.detached(priority: .userInitiated) {
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

        // Diarization: start the persistent pyannote sidecar (loads on MPS).
        if step.model.id == ModelCatalog.diarization.id {
            if diarization == nil {
                diarization = try await Task.detached(priority: .userInitiated) {
                    try DiarizationService()
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
