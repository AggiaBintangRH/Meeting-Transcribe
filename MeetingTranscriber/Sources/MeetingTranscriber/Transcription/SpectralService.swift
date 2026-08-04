import Foundation

/// Client for the SPECTRAL diarization sidecar (`scripts/spectral/spectral-service.py`).
///
/// The THIRD diarizer in the project, alongside `PyannoteService` (pyannote
/// community-1) and MOSS (speaker-attributed ASR, driven through
/// `ChunkedASRService`). The engine is the vendored Apache-2.0 `diarize` package:
/// Silero VAD → sliding-window WeSpeaker embeddings → GMM-BIC speaker counting →
/// spectral clustering → Viterbi label smoothing.
///
/// IT IS ON THE PIPELINE SIDE OF THE 2026-07-30 SPLIT, exactly like pyannote: it
/// answers WHO-SPOKE-WHEN with run-local labels (`SPEAKER_00`, …) and has never
/// seen a profile store. Identity is `WeSpeakerService`'s job, and
/// `AudioRecorder.composeTurns` joins the two. That is the whole reason this
/// engine can keep saved voice profiles, renaming and `spk` confidence while MOSS
/// cannot — MOSS emits names of its own, this emits spans.
///
/// **FINAL ONLY — THERE IS DELIBERATELY NO LIVE / PER-CHUNK PATH IN v1**, and this
/// class enforces that STRUCTURALLY rather than by a guard: there is no
/// `diarizeChunk`, no `onChunkResult`, and no way to construct a job whose `cmd`
/// is not `final`. The sidecar REFUSES anything else (pinned by
/// `spectral/no-live-chunk-branch`), and a refused job answers with an error, so a
/// dispatched chunk would settle a gate as *failed* rather than stalling — but the
/// only safe number of ways to send one is zero. The engine counts speakers
/// globally (GMM-BIC over every embedding in the file) and clusters globally, so a
/// 30 s window's labels would mean nothing across windows: the exact failure MOSS
/// has when it is called per chunk.
final class SpectralService: @unchecked Sendable {

    /// The wire currency is SHARED with `PyannoteService`, deliberately.
    ///
    /// The sidecar's office replies are byte-identical to the pyannote sidecar's
    /// (its module docstring says so and Stage 2 built it that way), and the
    /// identity stage — `AudioRecorder.identifyFinalTurns` → `WeSpeakerService.identify`
    /// → `AudioRecorder.composeTurns` — must accept both engines' turns without a
    /// second overload of every function on the path. Aliased rather than
    /// redeclared so the two can never drift into two structurally identical but
    /// incompatible types.
    typealias LocalTurn = PyannoteService.LocalTurn
    typealias Stream = PyannoteService.Stream

    /// Which sidecar this service runs and which log it writes.
    ///
    /// **BOTH ARE LITERALS, and that is the point.** `ChunkedASRService.Config.mossDiarization()`
    /// once took its `logName` from a model type's `logName` property; when the
    /// ASR role was renamed, that derivation silently repointed the DIARIZATION
    /// process's stderr into the ASR log **while it still ran the other script** —
    /// a script/log pair that has drifted, invisible because both processes keep
    /// working. Naming both here, as strings, is what makes that impossible;
    /// `testSpectralConfigNamesItsOwnSidecarAndLogAsLiterals` is its contract.
    struct Config: Equatable {
        /// The WeSpeaker checkpoint the vendored engine embeds with
        /// (`vendor/diarize/torch_embedder.py`) — the SAME weights
        /// `wespeaker-service.py` loads, which is why the spectral stack needs no
        /// download of its own.
        let embedderRepoID: String

        static let scriptName = "spectral/spectral-service.py"
        static let logName = "spectral"

        static func fromSettings() -> Config {
            Config(embedderRepoID: ModelCatalog.speakerEmbedding.hfRepo)
        }
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "scripts/spectral/spectral-service.py not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch the spectral diarization sidecar: \(reason)"
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
    private let writeQueue = DispatchQueue(label: "spectral.write", qos: .utility)

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

        // torch + Silero + WeSpeaker, all CPU-resident and small (~28 MB of
        // weights), but the interpreter and torch import dominate. Same budget as
        // the pyannote client's for the same reason: a slow first launch on a
        // cold filesystem must not look like a failure.
        let startup = waitUntilLoaded(timeout: 300)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "The spectral sidecar did not become ready (timeout)."
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
    /// (GMM-BIC decides).
    ///
    /// `exclusive` and `cluster_threshold` are pyannote's knobs and are not sent:
    /// this engine is inherently exclusive (one speaker per instant) and its
    /// clustering has no threshold. The sidecar accepts and ignores them so one
    /// caller *could* drive either engine; not sending them keeps the job line
    /// honest about what was asked for.
    func diarizeFinal(audio: URL, numSpeakers: Int = 0, stream: Stream = .office) {
        send(["cmd": "final", "audio": audio.path, "num_speakers": numSpeakers],
             stream: stream)
    }

    private func send(_ job: [String: Any], stream: Stream = .office) {
        guard process.isRunning else {
            onError?("The spectral diarization sidecar is not running — "
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
                    errorReason = "The spectral sidecar exited during startup."
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
