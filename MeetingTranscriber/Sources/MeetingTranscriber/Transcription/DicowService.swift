import Foundation

/// Client for the persistent DiCoW v3.3 target-speaker ASR sidecar
/// (scripts/dicow-service.py). Overlap attempt #4 (2026-07-16).
///
/// DiCoW is a diarization-conditioned Whisper: given one speaker's mask over a
/// window it transcribes only THAT speaker, so overlaps are resolved inside the
/// ASR rather than by separating the waveform (MossFormer2's approach). The
/// model loads once at session start; at stop the recorder asks for one
/// transcription per speaker per overlap window, and the results are spliced in
/// only if the recorder's quality gates pass.
///
/// Unlike every other sidecar this one runs in `.venv-dicow` — its transformers
/// pin (4.55) conflicts with the main `.venv`'s MLX stack.
final class DicowService: @unchecked Sendable {

    /// One speaker to transcribe, with its diarization turns in ABSOLUTE seconds.
    struct Target: Sendable {
        let sid: Int
        let turns: [(start: Double, end: Double)]
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        case venvMissing
        case launchFailed(String)
        case startupFailed(String)
        case requestFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "scripts/dicow-service.py not found in the project folder."
            case .venvMissing:
                return "The DiCoW runtime (.venv-dicow) is missing. "
                     + "Run download-best-models.sh, then retry."
            case .launchFailed(let reason):
                return "Could not launch DiCoW sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            case .requestFailed(let reason):
                return reason
            }
        }
    }

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "dicow.write", qos: .utility)

    private let stateLock = NSLock()
    private var nextRequestID = 0
    private var continuations: [Int: CheckedContinuation<[TargetResult], Error>] = [:]

    /// Synchronous critical section — safe from async contexts (never suspends).
    private func withStateLock<T>(_ body: () -> T) -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return body()
    }

    init() throws {
        let script = PythonRuntime.scriptsDir.appendingPathComponent("dicow-service.py")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }
        // DiCoW's own venv only — see the type doc. Nil means setup never ran.
        guard let command = PythonRuntime.command(forScript: script, venvName: ".venv-dicow") else {
            throw ServiceError.venvMissing
        }
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: "dicow-sidecar")

        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        let startup = waitUntilLoaded(timeout: 300) // torch + the ~6 GB DiCoW load
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "DiCoW sidecar did not become ready (timeout)."
            )
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    // MARK: - API

    /// Transcribe each target speaker separately across `[start,end)` of `audio`.
    /// The window must be <= 30s (the sidecar rejects longer ones so DiCoW never
    /// enters its long-form seek loop). id-correlated, 180s timeout.
    func transcribeTargets(audio: URL, start: Double, end: Double,
                           targets: [Target], language: String?) async throws -> [TargetResult] {
        guard process.isRunning else {
            throw ServiceError.launchFailed("DiCoW sidecar is not running")
        }
        let id = withStateLock { () -> Int in
            nextRequestID += 1
            return nextRequestID
        }

        let encodedTargets: [[String: Any]] = targets.map { t in
            ["sid": t.sid, "turns": t.turns.map { [$0.start, $0.end] }]
        }
        var job: [String: Any] = [
            "cmd": "transcribe_targets", "id": id, "audio": audio.path,
            "start": start, "end": end, "targets": encodedTargets,
        ]
        job["language"] = language ?? NSNull()   // null → let DiCoW auto-detect
        guard var data = try? JSONSerialization.data(withJSONObject: job) else {
            throw ServiceError.requestFailed("could not encode DiCoW request")
        }
        data.append(0x0A)
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: data)
        }

        // Timeout watchdog.
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(180))
            guard let self else { return }
            let cont = self.withStateLock { self.continuations.removeValue(forKey: id) }
            cont?.resume(throwing: ServiceError.requestFailed(
                "DiCoW transcription timed out (180s) — see logs/dicow-sidecar.log"))
        }

        return try await withCheckedThrowingContinuation { cont in
            withStateLock { continuations[id] = cont }
        }
    }

    func terminate() {
        if process.isRunning {
            let exitCmd = "{\"cmd\":\"exit\"}\n".data(using: .utf8) ?? Data()
            try? stdinPipe.fileHandleForWriting.write(contentsOf: exitCmd)
            try? stdinPipe.fileHandleForWriting.close()
            process.terminate()
        }
    }

    // MARK: - Output

    /// One speaker's transcription of the window. Text only — DiCoW's internal
    /// timestamps are deliberately never used (unreliable, see CLAUDE.md).
    struct TargetResult: Decodable, Sendable {
        let sid: Int
        let text: String
    }

    private struct Message: Decodable {
        let type: String
        let text: String?
        let id: Int?
        let targets: [TargetResult]?
    }

    private func resolve(_ message: Message) {
        guard let id = message.id else { return }
        let cont = withStateLock { continuations.removeValue(forKey: id) }
        guard let cont else { return }   // already timed out
        if message.type == "result" {
            cont.resume(returning: message.targets ?? [])
        } else {
            cont.resume(throwing: ServiceError.requestFailed(message.text ?? "unknown DiCoW error"))
        }
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
                if errorReason == nil { errorReason = "DiCoW sidecar exited during startup." }
                semaphore.signal()
                return
            }
            accumulated.append(data)
            guard let text = String(data: accumulated, encoding: .utf8) else { return }
            for line in text.split(separator: "\n") {
                guard let lineData = line.data(using: .utf8),
                      let message = try? JSONDecoder().decode(Message.self, from: lineData)
                else { continue }
                if message.type == "error", message.id == nil {
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
                switch message.type {
                case "result", "error": self.resolve(message)
                default: break
                }
            }
        }
    }
}
