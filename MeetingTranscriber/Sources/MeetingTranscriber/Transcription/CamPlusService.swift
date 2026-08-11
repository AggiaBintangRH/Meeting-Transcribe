import Foundation

/// Client for the CAM++ diarization sidecar (`scripts/campplus/campplus-service.py`).
///
/// The SIXTH diarizer in the project — after pyannote community-1, MOSS,
/// spectral, NeMo and DiariZen.
///
/// The engine is a pipeline BUILT AROUND an embedding model rather than an
/// off-the-shelf diarizer, because CAM++ is an embedder and nothing more:
/// Silero VAD → CAM++ embeddings over 2 s sliding sub-windows → spectral
/// clustering with an eigengap speaker count. Same three-stage SHAPE as
/// `spectral`, deliberately NOT the same implementation — that engine's
/// GMM-BIC speaker COUNTING is the stage this project has documented failing
/// (20 speakers on a 67-minute meeting, 13 on a 3-person clip), so building a
/// second engine on it would inherit the one defect worth avoiding.
///
/// Checkpoint `Wespeaker/wespeaker-voxceleb-campplus-LM` — **Apache 2.0**,
/// 66 MB, chosen on licence as much as on merit. The MLX-native CAM++ on the
/// hub declares no licence at all, and this app ships to a paying client, so
/// "unlicensed" is all-rights-reserved. Same question that ruled out five of
/// DiariZen's six checkpoints.
///
/// IT IS ON THE PIPELINE SIDE OF THE 2026-07-30 SPLIT, exactly like pyannote and
/// spectral: it answers WHO-SPOKE-WHEN with run-local labels (`SPEAKER_00`, …)
/// and has never seen a profile store. Identity is `WeSpeakerService`'s job and
/// `AudioRecorder.composeTurns` joins the two — which is why this engine gets
/// saved voice profiles, renaming and `spk` confidence for free, with no new
/// identity code and no new id space.
///
/// **FINAL ONLY — THERE IS DELIBERATELY NO LIVE / PER-CHUNK PATH**, and this
/// class enforces that STRUCTURALLY rather than by a guard: there is no
/// `diarizeChunk`, no `onChunkResult`, and no way to construct a job whose `cmd`
/// is not `final`. The sidecar refuses anything else (pinned by
/// `campplus/no-live-chunk-branch`), so a dispatched chunk would settle a gate
/// as *failed* rather than stalling — but the only safe number of ways to send
/// one is zero. The engine counts and clusters GLOBALLY, so a 30 s window's
/// labels would mean nothing across windows: the exact failure MOSS has when it
/// is called per chunk.
final class CamPlusService: @unchecked Sendable {

    /// The wire currency is SHARED with `PyannoteService`, deliberately.
    ///
    /// This sidecar's office replies are byte-identical to the pyannote and
    /// spectral sidecars' — same keys, same order, same unrounded times — and
    /// the identity stage (`AudioRecorder.identifyFinalTurns` →
    /// `WeSpeakerService.identify` → `AudioRecorder.composeTurns`) must accept
    /// every engine's turns without a second overload of every function on the
    /// path. Aliased rather than redeclared so they can never drift into two
    /// structurally identical but incompatible types.
    typealias LocalTurn = PyannoteService.LocalTurn
    typealias Stream = PyannoteService.Stream

    /// Which sidecar this service runs and which log it writes.
    ///
    /// **BOTH ARE LITERALS, and that is the point.**
    /// `ChunkedASRService.Config.mossDiarization()` once took its `logName` from
    /// a model type's property; when the ASR role was renamed, that derivation
    /// silently repointed the DIARIZATION process's stderr into the ASR log
    /// while it still ran the other script — a drifted script/log pair,
    /// invisible because both processes keep working.
    struct Config: Equatable {
        /// The CAM++ checkpoint the sidecar embeds with. Unlike spectral's,
        /// this is NOT the shared WeSpeaker ResNet34 — it is this engine's own
        /// 66 MB download, which is why it has a catalog entry of its own.
        let embedderRepoID: String

        static let scriptName = "campplus/campplus-service.py"
        static let logName = "campplus"

        static func fromSettings() -> Config {
            Config(embedderRepoID: ModelCatalog.camPlusDiarization.hfRepo)
        }
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            // NAMES THIS ENGINE'S OWN FILE. The 2026-08-10 DiariZen audit found
            // exactly this message shipped pointing at `scripts/nemo/nemo-service.py`
            // — wrong file, wrong folder, wrong engine — because the class had
            // been copied and renamed. Pinned by
            // `testNoUserFacingErrorNamesAnotherEnginesFiles`.
            case .scriptMissing:
                return "scripts/campplus/campplus-service.py not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch the CAM++ diarization sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    /// Whole-file result for one stream: (the audio file, run-local turns, stream).
    ///
    /// The AUDIO PATH IS LOAD-BEARING, not a convenience — it is what the caller
    /// hands the identity stage next, so there is no second bookkeeping map to
    /// keep in step. Same contract as `PyannoteService.onFinalResult`.
    var onFinalResult: ((String, [LocalTurn], Stream) -> Void)?
    /// Job error, tagged with the stream it belongs to so one stream's failure
    /// never settles the other's gate. Startup errors carry no stream → office.
    var onError: ((String, Stream) -> Void)?

    let config: Config

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "campplus.write", qos: .utility)

    init(config: Config) throws {
        self.config = config
        let script = PythonRuntime.scriptsDir.appendingPathComponent(Config.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: Config.logName)
        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        // torch + Silero + CAM++, all CPU-resident and small (66 MB of weights);
        // the interpreter and the torch import dominate, and the sidecar warms
        // the embedder before it says LOADED. Measured 0.9 s here — the 300 s
        // budget is the same as its siblings' and exists so a cold filesystem
        // on a client machine does not look like a failure.
        let startup = waitUntilLoaded(timeout: 300)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "The CAM++ sidecar did not become ready (timeout)."
            )
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    // MARK: - Jobs

    /// The ONLY job this engine has: one whole-file pass. `numSpeakers` 0 = auto
    /// (the eigengap decides).
    ///
    /// `exclusive` and `cluster_threshold` are pyannote's knobs and are not
    /// sent: this engine is inherently exclusive (one label per instant) and its
    /// clustering has no distance threshold. The sidecar accepts and ignores
    /// them so one caller *could* drive either engine; not sending them keeps
    /// the job line honest about what was asked for.
    func diarizeFinal(audio: URL, numSpeakers: Int = 0, stream: Stream = .office) {
        send(["cmd": "final", "audio": audio.path, "num_speakers": numSpeakers],
             stream: stream)
    }

    private func send(_ job: [String: Any], stream: Stream = .office) {
        guard process.isRunning else {
            onError?("The CAM++ diarization sidecar is not running — "
                     + "restart the recording session.", stream)
            return
        }
        // Office is sent as an ABSENT key, not as "office": the sidecar reads a
        // missing stream as office, so a single-stream session's job lines carry
        // no dual-stream vocabulary at all.
        var job = job
        if stream == .remote { job["stream"] = stream.rawValue }
        guard var data = try? JSONSerialization.data(withJSONObject: job) else { return }
        data.append(0x0A)
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: data)
        }
    }

    func terminate() {
        if process.isRunning {
            try? stdinPipe.fileHandleForWriting.close()
            process.terminate()
        }
    }

    // MARK: - Output

    private struct Message: Decodable {
        let type: String
        let text: String?
        let segments: [LocalTurn]?
        /// The file this reply is about — echoed straight back, so the caller can
        /// hand it to the identity stage without a lookup.
        let audio: String?
        /// Echoed back only for remote jobs; absent ⇒ office (see `Stream`).
        let stream: String?
    }

    private func waitUntilLoaded(timeout: TimeInterval) -> (ready: Bool, errorReason: String?) {
        let semaphore = DispatchSemaphore(value: 0)
        var ready = false
        var errorReason: String?
        var accumulated = Data()

        let reader = stdoutPipe.fileHandleForReading
        reader.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                if errorReason == nil {
                    errorReason = "The CAM++ sidecar exited during startup."
                }
                semaphore.signal()
                return
            }
            accumulated.append(data)
            guard let text = String(data: accumulated, encoding: .utf8) else { return }
            for line in text.split(separator: "\n") {
                guard let lineData = line.data(using: .utf8),
                      let message = try? JSONDecoder().decode(Message.self, from: lineData)
                else { continue }
                if message.type == "error" {
                    errorReason = message.text
                    semaphore.signal()
                    return
                }
                if message.type == "status", (message.text ?? "").contains("LOADED") {
                    ready = true
                    semaphore.signal()
                    return
                }
            }
        }
        _ = semaphore.wait(timeout: .now() + timeout)
        reader.readabilityHandler = nil
        return (ready, errorReason)
    }

    private func startResultReader() {
        var pending = ""
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self,
                  let chunk = String(data: handle.availableData, encoding: .utf8)
            else { return }
            pending += chunk
            let lines = pending.components(separatedBy: "\n")
            pending = lines.last ?? ""
            for line in lines.dropLast() where !line.isEmpty {
                guard let data = line.data(using: .utf8),
                      let message = try? JSONDecoder().decode(Message.self, from: data)
                else { continue }
                let stream = message.stream.flatMap(Stream.init(rawValue:)) ?? .office
                switch message.type {
                case "result":
                    self.onFinalResult?(message.audio ?? "", message.segments ?? [], stream)
                case "error":
                    self.onError?(message.text ?? "unknown diarization error", stream)
                // NO `chunk_result` CASE, deliberately. This engine cannot produce
                // one — see the type comment — and a case here would be the first
                // step towards someone adding the job that fills it.
                default: break
                }
            }
        }
    }
}
