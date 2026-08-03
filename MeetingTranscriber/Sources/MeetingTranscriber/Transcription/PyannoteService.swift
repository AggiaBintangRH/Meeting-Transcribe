import Foundation

/// Client for the pyannote DIARIZATION sidecar (`scripts/pyannote/pyannote-service.py`).
///
/// ONE SERVICE PER MODEL (owner, 2026-07-29). This used to be
/// `DiarizationService`, talking to one sidecar that both diarized and identified.
/// It now speaks only to the pipeline half: it answers WHO-SPOKE-WHEN with
/// pipeline-local labels (`SPEAKER_00`, `SPEAKER_01`, …) and knows nothing about
/// who those people are. Identity comes from `WeSpeakerService`, and the two are
/// composed by `AudioRecorder.composeTurns`.
///
/// The pipeline loads once at session start (loading overlay). During the meeting
/// each audio chunk is diarized live; at stop, a batch pass over the full
/// recording refines everything (best DER).
///
/// This client has NO `resetProfiles`. There are no profiles on this side of the
/// split — the stores belong entirely to `WeSpeakerService`, and so does their
/// reset. `clusterThreshold` stays here, because that is a pipeline parameter.
final class PyannoteService: @unchecked Sendable {

    /// One turn as the PIPELINE sees it: a span and a run-local label. No id, no
    /// name, no confidence — those are identity, and identity is the other
    /// service's job. Keeping the type this narrow is what makes it impossible
    /// for the pipeline stage to write a voice into a profile store.
    ///
    /// `start`/`end` arrive UNROUNDED, deliberately. The identity stage slices
    /// the waveform with these numbers, so rounding them on the wire would move
    /// each slice by up to ~8 samples and change the embedding. Rounding to 3 dp
    /// happens once, where it always did: in `composeTurns`, on the way to the
    /// app-visible `SpeakerTurn`.
    struct LocalTurn: Decodable, Sendable {
        let start: Double
        let end: Double
        let label: String
    }

    /// Which identity space a job (and its result) belongs to.
    ///
    /// Office and Remote are two separate identity spaces, so an office voice can
    /// never be cosine-matched onto a remote profile or vice versa — the
    /// separation lives in `WeSpeakerService`'s two stores. Here the value is
    /// PURE ROUTING: this sidecar has no per-stream state at all. `office` is the
    /// default everywhere and is sent as an ABSENT field, so a single-stream
    /// session's job lines are exactly what they were before dual-stream existed.
    enum Stream: String, Sendable {
        case office
        case remote
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "scripts/pyannote/pyannote-service.py not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch the pyannote diarization sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    /// Live chunk result: (windowStart, the audio file that was diarized,
    /// chunk-local turns, stream).
    ///
    /// The AUDIO PATH IS LOAD-BEARING, not a convenience: it is what the caller
    /// hands the identity stage next. Echoing it means there is no second
    /// bookkeeping map to keep in step, and it works identically for a chunk's
    /// temp WAV and for the recording file at stop.
    var onChunkResult: ((Double, String, [LocalTurn], Stream) -> Void)?
    /// Final batch result over the full recording, for one stream:
    /// (the audio file, turns, stream).
    var onFinalResult: ((String, [LocalTurn], Stream) -> Void)?
    /// Job error, tagged with the stream it belongs to so one stream's failure
    /// never settles the other's gate. Startup errors carry no stream → office.
    var onError: ((String, Stream) -> Void)?

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "pyannote.write", qos: .utility)

    /// The sidecar and its OWN stderr log — two writers on one file is the
    /// 2026-07-15 mistake, and splitting the services makes it live again.
    static let scriptName = "pyannote/pyannote-service.py"
    static let logName = "pyannote"

    init() throws {
        let script = PythonRuntime.scriptsDir.appendingPathComponent(Self.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: Self.logName)

        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        let startup = waitUntilLoaded(timeout: 300) // torch + pipeline
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "The pyannote sidecar did not become ready (timeout)."
            )
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    // MARK: - Jobs

    /// Diarize one live chunk; result turns are chunk-local, offset by windowStart.
    /// exclusive=false keeps overlapping speech (both speakers shown).
    func diarizeChunk(audio: URL, windowStart: Double, exclusive: Bool = false,
                      stream: Stream = .office) {
        send(["cmd": "chunk", "audio": audio.path,
              "window_start": windowStart, "exclusive": exclusive,
              "cluster_threshold": Self.clusterThreshold()], stream: stream)
    }

    /// Batch refinement over the full recording. numSpeakers 0 = auto.
    func diarizeFinal(audio: URL, numSpeakers: Int = 0, exclusive: Bool = false,
                      stream: Stream = .office) {
        send(["cmd": "final", "audio": audio.path,
              "num_speakers": numSpeakers, "exclusive": exclusive,
              "cluster_threshold": Self.clusterThreshold()], stream: stream)
    }

    /// pyannote clustering threshold — absent/0 falls back to its default 0.6.
    /// Exposed for per-room calibration; see DiarizationTab. A PIPELINE
    /// parameter, which is why it stayed on this side of the split.
    private static func clusterThreshold() -> Double {
        let v = UserDefaults.standard.double(forKey: "diarization.clusterThreshold")
        return v > 0 ? v : 0.6
    }

    private func send(_ job: [String: Any], stream: Stream = .office) {
        guard process.isRunning else {
            onError?("The pyannote diarization sidecar is not running — "
                     + "restart the recording session.", stream)
            return
        }
        // Office is sent as an absent key, not as "office": the sidecar reads a
        // missing stream as office, so single-stream job lines are byte-identical.
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
        let window_start: Double?
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
                    errorReason = "The pyannote sidecar exited during startup."
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
                case "chunk_result":
                    self.onChunkResult?(message.window_start ?? 0, message.audio ?? "",
                                        message.segments ?? [], stream)
                case "result":
                    self.onFinalResult?(message.audio ?? "", message.segments ?? [], stream)
                case "error":
                    self.onError?(message.text ?? "unknown diarization error", stream)
                default: break
                }
            }
        }
    }
}
