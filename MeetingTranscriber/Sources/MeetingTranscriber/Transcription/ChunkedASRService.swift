import Foundation

/// Client for the persistent chunked ASR sidecar (scripts/chunked-asr-service.py).
///
/// The model (Qwen3 / Whisper / Voxtral, per Settings) loads once at session
/// start. During the meeting, audio streams in continuously; every chunk
/// interval the recorder calls `flush()` (aligned to VAD silence) and the
/// sidecar returns an accurate transcript of that chunk.
final class ChunkedASRService: @unchecked Sendable {

    struct Config: Equatable {
        let repoID: String
        let language: String   // "auto" or ISO code
        let modelName: String
        /// Forced-aligner repo, or nil when word alignment is off. Part of Config
        /// (which is Equatable) so flipping the setting recreates the sidecar.
        let alignRepoID: String?

        /// Resolve from Settings → Models → Chunked via the model classes.
        static func fromSettings() -> Config {
            let d = UserDefaults.standard
            let model = ChunkedASRModelFactory.fromSettings()
            let code = d.string(forKey: "chunked.language") ?? "auto"
            let aligning = d.object(forKey: "align.enabled") as? Bool ?? false
            return Config(repoID: model.repoID,
                          language: model.languageArgument(for: code) ?? "auto",
                          modelName: model.info.name,
                          alignRepoID: aligning ? ModelCatalog.wordAligner.hfRepo : nil)
        }
    }

    /// One word from the forced aligner, as returned with a `final` message.
    ///
    /// `start`/`end` are seconds relative to the START OF THE CHUNK BUFFER, not
    /// the recording. `text` is the aligner's normalized token ("oneminute" for
    /// "one-minute") — display text comes from `src`, the index into the
    /// original text's whitespace split. Coverage can be incomplete: items
    /// force-fitted past the end of the audio are dropped, so trailing source
    /// words may have no entry at all.
    struct AlignedWord: Decodable, Equatable, Sendable {
        let text: String
        let start: Double
        let end: Double
        let src: Int
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "scripts/chunked-asr-service.py not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch chunked ASR sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    let config: Config

    /// Called with each chunk's transcript, plus the aligner's words and the
    /// sidecar's buffer length in seconds when word alignment is on and
    /// succeeded (both nil otherwise).
    var onChunkTranscript: ((String, [AlignedWord]?, Double?) -> Void)?

    /// Called when a chunk fails to transcribe (message from the sidecar).
    var onChunkError: ((String) -> Void)?

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "chunked.write", qos: .utility)

    // Overlap-repair file-transcribe support (additive; independent of the
    // live streaming feed/flush path). Requests are id-correlated and must run
    // one-at-a-time since the sidecar is single-threaded.
    private let fileLock = NSLock()
    private var nextRequestID = 0
    private var fileContinuations: [Int: CheckedContinuation<String, Error>] = [:]
    private let fileGate = TranscribeGate()

    /// Synchronous critical section — safe to call from async contexts because
    /// it never suspends between lock and unlock.
    private func withFileLock<T>(_ body: () -> T) -> T {
        fileLock.lock()
        defer { fileLock.unlock() }
        return body()
    }

    /// Serial async gate so only one `transcribeFile` runs at a time.
    private actor TranscribeGate {
        private var running = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        func lock() async {
            if !running { running = true; return }
            await withCheckedContinuation { waiters.append($0) }
        }
        func unlock() {
            if waiters.isEmpty { running = false }
            else { waiters.removeFirst().resume() }
        }
    }

    init(config: Config = .fromSettings()) throws {
        self.config = config

        let script = PythonRuntime.scriptsDir.appendingPathComponent("chunked-asr-service.py")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        var arguments = command.arguments + ["--model", config.repoID]
        if config.language != "auto" {
            arguments += ["--language", config.language]
        }
        // Omitted entirely when alignment is off — the sidecar then takes its
        // original, byte-identical path.
        if let alignRepoID = config.alignRepoID {
            arguments += ["--align-model", alignRepoID]
        }
        process.arguments = arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: "chunked-asr")

        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        let startup = waitUntilLoaded(timeout: 180) // Qwen/Whisper load can take a bit
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "Chunked ASR sidecar did not become ready (timeout)."
            )
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    // MARK: - Input (safe from the audio tap thread)

    func feed(_ samples: [Float]) {
        guard process.isRunning, !samples.isEmpty else { return }
        var packet = withUnsafeBytes(of: Int32(samples.count).littleEndian) { Data($0) }
        samples.withUnsafeBufferPointer { packet.append(Data(buffer: $0)) }
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: packet)
        }
    }

    /// Chunk boundary — transcribe everything buffered since the last flush.
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

    // MARK: - File transcription (overlap repair)

    /// Transcribe one WAV file with the already-loaded chunked model.
    /// Serialized (one at a time), id-correlated, 120s timeout. Used by the
    /// overlap-repair pass to re-ASR each separated track.
    func transcribeFile(path: String) async throws -> String {
        await fileGate.lock()
        do {
            let result = try await sendFileRequest(path: path)
            await fileGate.unlock()
            return result
        } catch {
            await fileGate.unlock()
            throw error
        }
    }

    private func sendFileRequest(path: String) async throws -> String {
        guard process.isRunning else {
            throw ServiceError.launchFailed("chunked ASR sidecar is not running")
        }
        let id = withFileLock { () -> Int in
            nextRequestID += 1
            return nextRequestID
        }

        let body: [String: Any] = ["id": id, "path": path]
        guard let bodyData = try? JSONSerialization.data(withJSONObject: body) else {
            throw ServiceError.launchFailed("could not encode file-transcribe request")
        }

        // Frame: [int32 -2][int32 bodyLen][body]
        var packet = withUnsafeBytes(of: Int32(-2).littleEndian) { Data($0) }
        packet.append(withUnsafeBytes(of: Int32(bodyData.count).littleEndian) { Data($0) })
        packet.append(bodyData)
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: packet)
        }

        // Timeout watchdog: if no result within 120s, fail this continuation.
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(120))
            guard let self else { return }
            let cont = self.withFileLock { self.fileContinuations.removeValue(forKey: id) }
            cont?.resume(throwing: ServiceError.startupFailed(
                "file transcription timed out (120s) — see logs/chunked-asr.log"))
        }

        return try await withCheckedThrowingContinuation { cont in
            withFileLock { fileContinuations[id] = cont }
        }
    }

    private func resolveFile(id: Int?, text: String, isError: Bool) {
        guard let id else { return }
        let cont = withFileLock { fileContinuations.removeValue(forKey: id) }
        guard let cont else { return }   // already timed out
        if isError {
            cont.resume(throwing: ServiceError.startupFailed(text))
        } else {
            cont.resume(returning: text)
        }
    }

    // MARK: - Output

    /// Internal rather than private only so the decoding contract (the aligner
    /// keys must stay optional) can be covered by a unit test.
    struct Message: Decodable {
        let type: String
        let text: String
        let id: Int?
        // Additive, alignment-only keys on `final` — absent when alignment is
        // off or failed, so every pre-alignment payload still decodes.
        let words: [AlignedWord]?
        let dur: Double?
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
                if errorReason == nil { errorReason = "Chunked ASR sidecar exited during startup." }
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
                if message.type == "status", message.text.contains("LOADED") {
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
                case "final": self.onChunkTranscript?(message.text, message.words, message.dur)
                case "error": self.onChunkError?(message.text)
                case "file_result": self.resolveFile(id: message.id, text: message.text, isError: false)
                case "file_error": self.resolveFile(id: message.id, text: message.text, isError: true)
                default: break
                }
            }
        }
    }
}
