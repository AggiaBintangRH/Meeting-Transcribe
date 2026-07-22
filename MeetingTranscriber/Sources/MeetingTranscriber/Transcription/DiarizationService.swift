import Foundation

/// Client for the hybrid pyannote diarization sidecar (scripts/diarize-service.py).
///
/// Pipeline + WeSpeaker embedder load once at session start (loading overlay).
/// During the meeting, each audio chunk is diarized live and matched against
/// the speaker-profile store (stable names). At stop, a batch pass over the
/// full recording refines everything (best DER).
final class DiarizationService: @unchecked Sendable {

    struct Turn: Decodable, Sendable {
        let start: Double
        let end: Double
        let id: Int          // persistent profile id
        let name: String     // profile display name (renameable)
    }

    /// Which identity space a job (and its result) belongs to.
    ///
    /// The sidecar keeps ONE pyannote pipeline but TWO profile stores, so an
    /// office voice can never be cosine-matched onto a remote profile or vice
    /// versa. `office` is the default everywhere and is sent as an ABSENT field:
    /// a single-stream session's wire bytes are exactly what they were before
    /// dual-stream existed, which is the regression bar for this phase.
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
                return "scripts/diarize-service.py not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch diarization sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    /// Live chunk result: (windowStart, turns with chunk-local times, stream).
    var onChunkResult: ((Double, [Turn], Stream) -> Void)?
    /// Final batch result over the full recording, for one stream.
    var onFinalResult: (([Turn], Stream) -> Void)?
    /// Job error, tagged with the stream it belongs to so one stream's failure
    /// never settles the other's gate. Startup errors carry no stream → office.
    var onError: ((String, Stream) -> Void)?

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "diarize.write", qos: .utility)

    init() throws {
        let script = PythonRuntime.scriptsDir.appendingPathComponent("diarize-service.py")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: "diarize")

        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        let startup = waitUntilLoaded(timeout: 300) // torch + pipeline + embedder
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "Diarization sidecar did not become ready (timeout)."
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
              "window_start": windowStart, "exclusive": exclusive], stream: stream)
    }

    /// Batch refinement over the full recording. numSpeakers 0 = auto.
    func diarizeFinal(audio: URL, numSpeakers: Int = 0, exclusive: Bool = false,
                      stream: Stream = .office) {
        send(["cmd": "final", "audio": audio.path,
              "num_speakers": numSpeakers, "exclusive": exclusive], stream: stream)
    }

    /// Wipe all saved voice profiles — call to start a recording fresh.
    func resetProfiles() {
        send(["cmd": "reset"])
    }

    private func send(_ job: [String: Any], stream: Stream = .office) {
        guard process.isRunning else {
            onError?("Diarization sidecar is not running — restart the recording session.",
                     stream)
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
        let segments: [Turn]?
        let window_start: Double?
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
                if errorReason == nil { errorReason = "Diarization sidecar exited during startup." }
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
                    self.onChunkResult?(message.window_start ?? 0, message.segments ?? [], stream)
                case "result":
                    self.onFinalResult?(message.segments ?? [], stream)
                case "error":
                    self.onError?(message.text ?? "unknown diarization error", stream)
                default: break
                }
            }
        }
    }
}
