import Foundation

/// Client for the VibeVoice diarization sidecar
/// (`scripts/vibevoice-diar/vibevoice-diar-service.py`).
///
/// The SEVENTH diarization engine (2026-09-02), owner-requested after the client
/// asked for VibeVoice by name. Same checkpoint as the chunked and realtime
/// roles — microsoft/VibeVoice-ASR-Streaming-1.5B, **MIT** — in its third.
///
/// ⚠ THE MODEL IS SPEAKER-ATTRIBUTED ASR, NOT AN EMBEDDING PIPELINE. Where
/// spectral and CAM++ run VAD → embeddings → clustering, this one transcribes
/// and attributes in a single forward pass; the `Speaker N:` runs inside its
/// transcript ARE the diarization. Nothing here embeds anything, so none of the
/// clustering knobs the other engines expose have a counterpart.
///
/// ⚠ WHOLE-FILE PASSES ONLY, and for a different reason from spectral's. Spectral
/// clusters globally, so a window's labels would not be comparable across
/// windows. Here the reason is a running KV cache: one pass keeps `Speaker 0` the
/// same decision at minute 40 as at minute 1, and that is exactly what MOSS —
/// the only other attributed-ASR engine — gets wrong by working per chunk and
/// stitching afterwards (10–11 speakers against a truth of 7 on the 43-minute
/// recording). A windowed request is REFUSED by the sidecar rather than served.
///
/// ⚠ THE SPEAKER COUNT CANNOT BE PINNED. `streaming_generate` has no
/// `num_speakers` and the package contains none anywhere. The SPK control's value
/// is still SENT — every engine is sent it — and the sidecar logs that it is
/// ignoring it, so the choice dies where the truth is rather than at two places.
/// Measured auto counts: Overlap123 3/3 ✓, Meeting5People 4 against 5 ✗ (every
/// other engine gets 5), the client's ATND recording 1 against 5.
///
/// ⚠ 10.96 GB RSS and its OWN interpreter — the only diarization engine with
/// one, because transformers <5.0 conflicts with the MLX stack's 5.x.
///
/// Turn times come from the CHUNK GRID (2.9333 s), not from the model: the
/// streaming checkpoint reports no timestamps. Boundary precision is therefore
/// ~2.9 s against pyannote's measured median error of 263 ms. The non-streaming
/// VibeVoice checkpoint reports real timestamps and is the upgrade path.
final class VibeVoiceDiarizationService: @unchecked Sendable {

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
        /// The checkpoint the sidecar loads — a HUGGING FACE REPO ID, not a
        /// path. Named `modelRepoID` rather than `embedderRepoID` (its shape in
        /// the engine this class was copied from) because nothing here embeds
        /// anything: the model transcribes and attributes in one pass, and the
        /// `Speaker N:` runs ARE the output.
        ///
        /// ⚠ "REPO ID, NOT A PATH" is the whole of a bug that reached the owner
        /// as "model loading failed". Every hand-drive during development passed
        /// an absolute snapshot path, so all three VibeVoice sidecars read
        /// `<model>/preprocessor_config.json` directly and all three were tested
        /// green — while the app sends `ModelInfo.hfRepo`. They now resolve a
        /// repo id through the offline hub cache, as upstream's own loader does.
        let modelRepoID: String

        static let scriptName = "vibevoice-diar/vibevoice-diar-service.py"
        static let logName = "vibevoice-diar"

        static func fromSettings() -> Config {
            Config(modelRepoID: ModelCatalog.vibeVoiceDiarization.hfRepo)
        }
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        /// Its own interpreter is absent.
        case runtimeMissing(String)
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
                return "scripts/vibevoice-diar/vibevoice-diar-service.py not found in the project folder."
            case .runtimeMissing(let venv):
                return "The VibeVoice diarization engine needs its own Python "
                    + "runtime (\(venv)), and it is missing. Run "
                    + "./RUN-SETUP.command — it creates every interpreter the "
                    + "app needs."
            case .launchFailed(let reason):
                return "Could not launch the VibeVoice diarization sidecar: \(reason)"
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
    private let writeQueue = DispatchQueue(label: "vibevoice-diar.write", qos: .utility)

    /// The interpreter this engine's sidecar runs under.
    static let venvName = ".venv-vibevoice"

    init(config: Config) throws {
        self.config = config
        let script = PythonRuntime.scriptsDir.appendingPathComponent(Config.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }

        // ⚠ ITS OWN INTERPRETER — the only diarization engine with one.
        // VibeVoice pins transformers <5.0 while the main `.venv` needs 5.x for
        // the MLX stack; they can never share. A missing runtime must fail HERE,
        // naming it, rather than launching the main venv and dying inside the
        // model load with a message about an attribute.
        guard let command = PythonRuntime.command(forScript: script,
                                                  venvName: Self.venvName) else {
            throw ServiceError.runtimeMissing(Self.venvName)
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

        // 5.6 GB of weights across three shards onto MPS. Measured 5.2 s here,
        // by far the slowest LOADED of any diarization engine (CAM++ 0.9 s,
        // pyannote ~2 s) — the 300 s budget is its siblings' and leaves room for
        // a cold filesystem on a client machine, which must not look like a
        // failure.
        let startup = waitUntilLoaded(timeout: 300)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "The VibeVoice sidecar did not become ready (timeout)."
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
            onError?("The VibeVoice diarization sidecar is not running — "
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
                    errorReason = "The VibeVoice sidecar exited during startup."
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
