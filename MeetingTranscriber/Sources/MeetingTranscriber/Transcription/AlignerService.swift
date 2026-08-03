import Foundation

/// Client for the persistent forced-alignment sidecar
/// (`scripts/aligner/aligner-service.py`, Qwen3-ForcedAligner).
///
/// ONE SERVICE PER MODEL (owner, 2026-07-29). The aligner used to load INSIDE
/// whichever chunked ASR sidecar was running, which made word timestamps
/// SYNCHRONOUS: the `final` message did not exist until alignment had finished.
/// It is now its own process, and the app asks it for words separately — the
/// transcript appears the moment the ASR answers, with the estimated split, and
/// rows refine when the words come back (usually well under a second).
///
/// Everything here is therefore best-effort by design. A throw, a timeout, or a
/// reply with no words all mean the same thing to the caller: keep the estimate
/// that is already on screen. Alignment can never delay, alter or remove
/// transcript TEXT — it only moves the speaker boundary inside text that is
/// already final.
final class AlignerService: @unchecked Sendable {

    struct Config: Equatable {
        let repoID: String

        /// The sidecar. Its own folder, like every other service.
        static let scriptName = "aligner/aligner-service.py"
        /// Its own stderr log — two writers on one file is the 2026-07-15
        /// mistake, and splitting the services makes it live again.
        static let logName = "aligner"

        static func fromSettings() -> Config {
            Config(repoID: ModelCatalog.wordAligner.hfRepo)
        }
    }

    /// One word from the forced aligner.
    ///
    /// `start`/`end` are seconds relative to the START OF THE CHUNK BUFFER, not
    /// the recording. `text` is the aligner's normalized token ("oneminute" for
    /// "one-minute") — display text comes from `src`, the index into the
    /// original text's whitespace split. Coverage can be incomplete: items
    /// force-fitted past the end of the audio are dropped, so trailing source
    /// words may have no entry at all.
    ///
    /// Lives here rather than on `ChunkedASRService` because the aligner is now
    /// its own service; `ChunkedASRService.AlignedWord` is kept as a typealias so
    /// `WordAttribution` and its tests are untouched by the move.
    struct AlignedWord: Decodable, Equatable, Sendable {
        let text: String
        let start: Double
        let end: Double
        let src: Int
    }

    /// One alignment reply. `words == nil` is a NORMAL outcome, not an error:
    /// the sidecar's own gates (token-pairing, past-the-end) rejected this
    /// chunk's alignment, and the honest answer is "no trustworthy times here".
    typealias Alignment = (words: [AlignedWord]?, dur: Double?)

    enum ServiceError: LocalizedError {
        case scriptMissing(String)
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing(let name):
                return "scripts/\(name) not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch the aligner sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    let config: Config

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "aligner.write", qos: .utility)

    // Id-correlated requests, one at a time (the sidecar is single-threaded).
    private let lock = NSLock()
    private var nextRequestID = 0
    private var continuations: [Int: CheckedContinuation<Alignment, Error>] = [:]
    private let gate = AlignGate()

    /// Synchronous critical section — safe to call from async contexts because
    /// it never suspends between lock and unlock.
    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    /// Serial async gate so only one `align` runs at a time.
    private actor AlignGate {
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

        let script = PythonRuntime.scriptsDir.appendingPathComponent(Config.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing(Config.scriptName)
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments + ["--model", config.repoID]
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: Config.logName)
        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        // 60 s: this is a 0.6B model on its own, and it is loaded during the
        // session's loading overlay exactly as it was when it lived inside the
        // ASR sidecar (measured there: ~2 s).
        let startup = waitUntilLoaded(timeout: 60)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "The aligner sidecar did not become ready (timeout)."
            )
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    func terminate() {
        if process.isRunning {
            let packet = withUnsafeBytes(of: Int32(-1).littleEndian) { Data($0) }
            try? stdinPipe.fileHandleForWriting.write(contentsOf: packet)
            process.terminate()
        }
    }

    // MARK: - Alignment

    /// Align `text` (the chunk transcript VERBATIM) against the WAV at `path`.
    ///
    /// Serialized, id-correlated, 30 s timeout. The 30 s is the chunk interval:
    /// alignment that has not answered within one whole chunk has lost its race
    /// with the next one, and a queue of stale requests would be worse than a
    /// dropped refinement. Measured cost on this M4 is ~0.2 s per 30 s chunk.
    func align(path: String, text: String) async throws -> Alignment {
        await gate.lock()
        do {
            let result = try await send(path: path, text: text)
            await gate.unlock()
            return result
        } catch {
            await gate.unlock()
            throw error
        }
    }

    private func send(path: String, text: String) async throws -> Alignment {
        guard process.isRunning else {
            throw ServiceError.launchFailed("the aligner sidecar is not running")
        }
        let id = withLock { () -> Int in
            nextRequestID += 1
            return nextRequestID
        }

        let body: [String: Any] = ["id": id, "path": path, "text": text]
        guard let bodyData = try? JSONSerialization.data(withJSONObject: body) else {
            throw ServiceError.launchFailed("could not encode the alignment request")
        }

        // Frame: [int32 -2][int32 bodyLen][body]
        var packet = withUnsafeBytes(of: Int32(-2).littleEndian) { Data($0) }
        packet.append(withUnsafeBytes(of: Int32(bodyData.count).littleEndian) { Data($0) })
        packet.append(bodyData)
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: packet)
        }

        Task { [weak self] in
            try? await Task.sleep(for: .seconds(30))
            guard let self else { return }
            let cont = self.withLock { self.continuations.removeValue(forKey: id) }
            cont?.resume(throwing: ServiceError.startupFailed(
                "alignment timed out (30s) — see logs/\(Config.logName).log"))
        }

        return try await withCheckedThrowingContinuation { cont in
            withLock { continuations[id] = cont }
        }
    }

    private func resolve(id: Int?, words: [AlignedWord]?, dur: Double?,
                         error: String?) {
        guard let id else { return }
        let cont = withLock { continuations.removeValue(forKey: id) }
        guard let cont else { return }   // already timed out
        if let error {
            cont.resume(throwing: ServiceError.startupFailed(error))
        } else {
            cont.resume(returning: (words: words, dur: dur))
        }
    }

    // MARK: - Output

    /// Internal rather than private only so the decoding contract (an
    /// `align_result` may legitimately carry NO words) can be unit-tested.
    ///
    /// `text` is optional because an `align_result` has none — it carries an id
    /// and, when the gates passed, the words.
    struct Message: Decodable {
        let type: String
        let text: String?
        let id: Int?
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
                if errorReason == nil { errorReason = "The aligner sidecar exited during startup." }
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
                if message.type == "status", message.text?.contains("LOADED") == true {
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
                case "align_result":
                    // words absent ⇒ the gates rejected it. Not an error.
                    self.resolve(id: message.id, words: message.words,
                                 dur: message.dur, error: nil)
                case "align_error":
                    self.resolve(id: message.id, words: nil, dur: nil,
                                 error: message.text ?? "alignment failed")
                default: break
                }
            }
        }
    }
}
