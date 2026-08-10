import Foundation

/// Client for the NeMo diarization sidecar (`scripts/nemo/nemo-service.py`).
///
/// The FOURTH diarizer in the project, alongside `PyannoteService` (pyannote
/// community-1), `SpectralService` (the vendored FoxNoseTech engine) and MOSS
/// (speaker-attributed ASR, driven through `ChunkedASRService`). The engine is
/// NVIDIA NeMo's `ClusteringDiarizer`: MarbleNet VAD → multi-scale TitaNet-Large
/// embeddings → NME-SC spectral clustering (which estimates the speaker count
/// itself) → multi-scale label fusion.
///
/// IT IS ON THE PIPELINE SIDE OF THE 2026-07-30 SPLIT, exactly like pyannote and
/// spectral: it answers WHO-SPOKE-WHEN with run-local labels and has never seen a
/// profile store. Identity is `WeSpeakerService`'s job and
/// `AudioRecorder.composeTurns` joins the two, so saved voice profiles, renaming
/// and `spk` confidence all work under this engine with **no new identity code
/// and no new id space** — the labels become office profile ids (< 10 000) on the
/// same path spectral's do. That is a genuine advantage over MOSS, which names
/// its own speakers.
///
/// The labels themselves are `speaker_0…speaker_N` rather than pyannote's
/// `SPEAKER_00`. Verified before wiring: nothing in Swift parses a label — they
/// are opaque keys handed to `identify` — so they are deliberately NOT
/// normalised. A normaliser would be a second place for a label to change shape.
///
/// **FINAL ONLY — THERE IS DELIBERATELY NO LIVE / PER-CHUNK PATH**, enforced
/// STRUCTURALLY rather than by a guard, the same way `SpectralService` does it:
/// there is no `diarizeChunk`, no `onChunkResult`, and no way to construct a job
/// whose `cmd` is not `final`. NME-SC counts and clusters GLOBALLY — the eigengap
/// analysis runs over the affinity matrix of the whole file — so a 30 s window
/// would be counted and clustered on its own and its labels would mean nothing
/// across windows: the exact failure MOSS has when it is called per chunk. The
/// sidecar refuses anything else loudly, but the only safe number of ways to send
/// one is zero.
///
/// **IT RUNS IN ITS OWN INTERPRETER, `.venv-nemo`** — the second sidecar to do so
/// after DiCoW. `nemo_toolkit` drags in lightning, hydra and a large pinned
/// dependency tree that conflicts with the main `.venv`'s MLX stack.
final class NemoService: @unchecked Sendable {

    /// The wire currency is SHARED with `PyannoteService`, deliberately.
    ///
    /// The sidecar's office replies are byte-identical to the pyannote sidecar's
    /// (verified in stage 2, including the `stream` echo for remote and its
    /// ABSENCE for office), and the identity stage —
    /// `AudioRecorder.identifyFinalTurns` → `WeSpeakerService.identify` →
    /// `AudioRecorder.composeTurns` — must accept every engine's turns without a
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
    /// `testNemoConfigNamesItsOwnSidecarAndLogAsLiterals` is its contract.
    struct Config: Equatable {
        /// The interpreter this sidecar MUST run under. Not the main `.venv`: see
        /// the type doc, and `ServiceError.venvMissing` for what a missing one
        /// says.
        static let venvName = ".venv-nemo"
        static let scriptName = "nemo/nemo-service.py"
        static let logName = "nemo"
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        case venvMissing
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "scripts/nemo/nemo-service.py not found in the project folder."
            case .venvMissing:
                // NO FALLBACK to the main `.venv`, and the message says which
                // runtime is missing rather than "a model failed to load": running
                // this sidecar on the wrong interpreter fails deep inside NeMo
                // with a far more confusing error. Same rule as DiCoW's.
                return "The NeMo runtime (\(Config.venvName)) is missing. "
                     + "Run download-best-models.sh, then retry."
            case .launchFailed(let reason):
                return "Could not launch the NeMo diarization sidecar: \(reason)"
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
    private let writeQueue = DispatchQueue(label: "nemo.write", qos: .utility)

    init(config: Config = Config()) throws {
        self.config = config
        let script = PythonRuntime.scriptsDir.appendingPathComponent(Config.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }
        // NeMo's own venv only — see `ServiceError.venvMissing`. Nil means setup
        // never ran; there is deliberately no fall-through to the main `.venv`.
        guard let command = PythonRuntime.command(forScript: script,
                                                  venvName: Config.venvName) else {
            throw ServiceError.venvMissing
        }
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: Config.logName)
        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        // A DELIBERATELY GENEROUS BUDGET, and the number is measured rather than
        // copied: `import nemo.collections.asr` alone costs ~51 s cold on this M4
        // (lightning, hydra and the whole NeMo tree), before either checkpoint is
        // restored from disk. Twice the spectral client's 300 s, because a slow
        // first launch on a cold filesystem must not look like a failure — this
        // engine's startup is the slowest in the app by a wide margin.
        let startup = waitUntilLoaded(timeout: 600)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "The NeMo sidecar did not become ready (timeout)."
            )
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    // MARK: - Jobs

    /// The ONLY job this engine has: one whole-file pass. `numSpeakers` 0 = auto,
    /// which is the ONLY mode the app ever asks for — there is no Settings control
    /// for a speaker count under this engine (owner, 2026-08-07), and
    /// `max_num_speakers` is pinned at 20 inside the sidecar. The parameter stays
    /// because `AudioRecorder.diarNumSpeakers` is the one place that decision
    /// lives, shared by all four engines.
    ///
    /// `exclusive` and `cluster_threshold` are pyannote's knobs and are not sent:
    /// NME-SC assigns exactly one label per instant (so turns never intersect) and
    /// its clustering has no threshold — it searches a p-value range instead. The
    /// sidecar accepts and ignores them so one caller *could* drive either engine;
    /// not sending them keeps the job line honest about what was asked for.
    func diarizeFinal(audio: URL, numSpeakers: Int = 0, stream: Stream = .office) {
        send(["cmd": "final", "audio": audio.path, "num_speakers": numSpeakers],
             stream: stream)
    }

    private func send(_ job: [String: Any], stream: Stream = .office) {
        guard process.isRunning else {
            onError?("The NeMo diarization sidecar is not running — "
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
                    errorReason = "The NeMo sidecar exited during startup."
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
