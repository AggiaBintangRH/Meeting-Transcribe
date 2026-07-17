import Foundation

/// Client for the persistent MossFormer2 overlap-separation sidecar
/// (scripts/mossformer2-service.py). Overlap attempt #3 (2026-07-14).
///
/// The model (standalone `alibabasglab/MossFormer2` code + the
/// `mossformer2-librimix-2spk` 8 kHz checkpoint — NOT the earlier-removed
/// `clearvoice`/`MossFormer2_SS_16K` attempt) loads once at session start.
/// At stop, the recorder asks it to separate a short window around each
/// detected overlap into two per-speaker tracks; each track is then re-ASR'd
/// and, only if quality gates pass, spliced into the transcript.
final class OverlapRepairService: @unchecked Sendable {

    struct SeparatedTrack: Decodable, Sendable {
        let path: String
        let active: [[Double]]   // window-relative [start,end] active spans
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        case launchFailed(String)
        case startupFailed(String)
        case requestFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "scripts/mossformer2-service.py not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch overlap-repair sidecar: \(reason)"
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
    private let writeQueue = DispatchQueue(label: "overlap.write", qos: .utility)

    private let stateLock = NSLock()
    private var nextRequestID = 0
    private var continuations: [Int: CheckedContinuation<[SeparatedTrack], Error>] = [:]

    /// Synchronous critical section — safe from async contexts (never suspends).
    private func withStateLock<T>(_ body: () -> T) -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return body()
    }

    init() throws {
        let script = PythonRuntime.scriptsDir.appendingPathComponent("mossformer2-service.py")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: "overlap-repair-sidecar")

        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        let startup = waitUntilLoaded(timeout: 300) // torch + MossFormer2 load
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "Overlap-repair sidecar did not become ready (timeout)."
            )
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    // MARK: - API

    /// Separate `[start,end)` of `audio` into per-speaker tracks written under
    /// `outDir`. id-correlated, 180s timeout.
    func separate(audio: URL, start: Double, end: Double, outDir: URL) async throws -> [SeparatedTrack] {
        guard process.isRunning else {
            throw ServiceError.launchFailed("overlap-repair sidecar is not running")
        }
        let id = withStateLock { () -> Int in
            nextRequestID += 1
            return nextRequestID
        }

        let job: [String: Any] = [
            "cmd": "separate", "id": id, "audio": audio.path,
            "start": start, "end": end, "out_dir": outDir.path,
        ]
        guard var data = try? JSONSerialization.data(withJSONObject: job) else {
            throw ServiceError.requestFailed("could not encode separate request")
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
                "overlap separation timed out (180s) — see logs/overlap-repair-sidecar.log"))
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

    private struct Message: Decodable {
        let type: String
        let text: String?
        let id: Int?
        let tracks: [SeparatedTrack]?
    }

    private func resolve(_ message: Message) {
        guard let id = message.id else { return }
        let cont = withStateLock { continuations.removeValue(forKey: id) }
        guard let cont else { return }   // already timed out
        if message.type == "result" {
            cont.resume(returning: message.tracks ?? [])
        } else {
            cont.resume(throwing: ServiceError.requestFailed(message.text ?? "unknown separation error"))
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
                if errorReason == nil { errorReason = "Overlap-repair sidecar exited during startup." }
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
