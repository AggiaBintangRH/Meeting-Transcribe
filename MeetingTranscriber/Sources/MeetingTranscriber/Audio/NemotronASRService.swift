import Foundation

/// Client for the Nemotron 3.5 realtime ASR sidecar (scripts/nemotron-asr-service.py).
///
/// Feed 16 kHz mono samples continuously; call `flush()` at utterance end
/// (VAD speech→silence edge). Emits partial transcripts every ~1.5 s while
/// audio streams in, and a final transcript on flush.
///
/// Configured from Settings → Models → Realtime (language, chunk size).
final class NemotronASRService: @unchecked Sendable {

    struct Config: Equatable {
        let language: String   // "auto" or prompt key like "id-ID"
        let chunkMs: Int       // 80 / 160 / 560 / 1120

        /// Map Settings values to Nemotron language prompt keys.
        static func fromSettings() -> Config {
            let d = UserDefaults.standard
            let raw = d.string(forKey: "realtime.language") ?? "auto"
            let locale: [String: String] = [
                "id": "id-ID", "en": "en-US", "ms": "ms-MY",
                "zh": "zh-CN", "ja": "ja-JP",
            ]
            let chunk = d.object(forKey: "realtime.chunkMs") as? Int ?? 160
            return Config(language: locale[raw] ?? "auto", chunkMs: chunk)
        }
    }

    let config: Config

    enum ServiceError: LocalizedError {
        case scriptMissing
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "scripts/nemotron-asr-service.py not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch Python sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    /// Transcript callback: (text, isFinal). Called on an arbitrary thread.
    var onTranscript: ((String, Bool) -> Void)?

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()

    /// All writes off the audio thread — transcription pauses can back up the
    /// pipe, and a blocked write must never stall audio capture.
    private let writeQueue = DispatchQueue(label: "nemotron.write", qos: .userInitiated)

    init(config: Config = .fromSettings()) throws {
        self.config = config

        let script = PythonRuntime.scriptsDir.appendingPathComponent("nemotron-asr-service.py")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments + ["--language", config.language,
                                                 "--chunk-ms", String(config.chunkMs)]
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: "nemotron-asr")

        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        // First run: MLX import + weight load can take a while
        let startup = waitUntilReady(timeout: 120)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason
                ?? "Sidecar did not become ready (timeout or crash). Python: \(command.executable.path)"
            )
        }
        startTranscriptReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    // MARK: - Input (safe to call from the audio tap thread)

    /// Stream samples to the recognizer.
    func feed(_ samples: [Float]) {
        guard process.isRunning, !samples.isEmpty else { return }
        var packet = withUnsafeBytes(of: Int32(samples.count).littleEndian) { Data($0) }
        samples.withUnsafeBufferPointer { packet.append(Data(buffer: $0)) }
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: packet)
        }
    }

    /// Utterance ended — request the final transcript and reset the buffer.
    func flush() {
        guard process.isRunning else { return }
        let packet = withUnsafeBytes(of: Int32(0).littleEndian) { Data($0) }
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: packet)
        }
    }

    func terminate() {
        if process.isRunning {
            let packet = withUnsafeBytes(of: Int32(-1).littleEndian) { Data($0) }
            try? stdinPipe.fileHandleForWriting.write(contentsOf: packet)
            process.terminate()
        }
    }

    // MARK: - Output

    private func waitUntilReady(timeout: TimeInterval) -> (ready: Bool, errorReason: String?) {
        let semaphore = DispatchSemaphore(value: 0)
        var ready = false
        var errorReason: String?
        var accumulated = Data()

        let reader = stdoutPipe.fileHandleForReading
        reader.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { // EOF — process died before READY
                if errorReason == nil { errorReason = "Python sidecar exited during startup." }
                semaphore.signal()
                return
            }
            accumulated.append(data)
            guard let text = String(data: accumulated, encoding: .utf8) else { return }

            // Sidecar reports startup errors as {"type":"error","text":...}
            for line in text.split(separator: "\n") {
                guard let lineData = line.data(using: .utf8),
                      let message = try? JSONDecoder().decode(Message.self, from: lineData)
                else { continue }
                if message.type == "error" {
                    errorReason = message.text
                    semaphore.signal()
                    return
                }
                if message.type == "status", message.text.contains("READY") {
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

    private struct Message: Decodable {
        let type: String
        let text: String
    }

    private func startTranscriptReader() {
        var pending = ""
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self,
                  let chunk = String(data: handle.availableData, encoding: .utf8)
            else { return }
            pending += chunk

            // Process complete lines; keep any trailing fragment
            let lines = pending.components(separatedBy: "\n")
            pending = lines.last ?? ""
            for line in lines.dropLast() where !line.isEmpty {
                guard let data = line.data(using: .utf8),
                      let message = try? JSONDecoder().decode(Message.self, from: data)
                else { continue }
                switch message.type {
                case "partial": self.onTranscript?(message.text, false)
                case "final":   self.onTranscript?(message.text, true)
                default: break
                }
            }
        }
    }
}
