import Foundation

/// Client for the speaker-IDENTITY sidecar (`scripts/wespeaker/wespeaker-service.py`,
/// WeSpeaker + the two persistent profile stores).
///
/// ONE SERVICE PER MODEL (owner, 2026-07-29). Named for the model, not the role,
/// like every other service in the split. It is the other half of the former
/// `diarize-service.py`: `PyannoteService` finds the spans, this attaches saved
/// identities to them, and `AudioRecorder.composeTurns` joins the two into
/// `SpeakerTurn`s.
///
/// What the split buys, stated so it is not overclaimed: identity is now
/// *structurally available* to any diarizer that can produce time spans. Nothing
/// new is wired to it. MOSS still labels speakers per chunk with no cross-chunk
/// stitching (the owner deferred that — "nanti saja"), and the ATND position
/// layer still owns its own cluster ids and has no embeddings. Neither goes
/// through this service.
///
/// UNLIKE the aligner — whose whole contract is "best effort, keep the estimate"
/// — this service is NOT optional. `SpeakerTurn` has no representation for an
/// unidentified turn, so a failed identify has nothing to degrade to. A load
/// failure refuses the session, and a mid-session failure routes through the same
/// `PyannoteService.onError` path a diarization failure always has.
final class WeSpeakerService: @unchecked Sendable {

    /// Who one pipeline-local label turned out to be.
    ///
    /// `conf` is the winning cosine similarity, and it is ABSENT — not zero —
    /// when this call minted a brand-new profile: a first appearance was never
    /// scored against anything, so there is no number. A displayed 0.00 would
    /// read as "we are sure this is the wrong person", which is the opposite
    /// claim. `id` already carries `REMOTE_ID_BASE` for a remote job, so the
    /// caller never sees a store-local remote id.
    struct Identity: Decodable, Sendable {
        let id: Int
        let name: String
        let conf: Double?
    }

    /// Local label → who they are. A `nil` VALUE means the sidecar could not
    /// identify that voice at all (under its minimum speech length, or a
    /// degenerate embedding); the caller DROPS those spans, which is exactly what
    /// the single service used to do before it emitted.
    typealias IdentityMap = [String: Identity?]

    struct Config: Equatable {
        let repoID: String

        /// The sidecar. Its own folder, like every other service.
        static let scriptName = "wespeaker/wespeaker-service.py"
        /// Its own stderr log — two writers on one file is the 2026-07-15
        /// mistake, and splitting the services makes it live again.
        static let logName = "wespeaker"

        static func fromSettings() -> Config {
            Config(repoID: ModelCatalog.speakerEmbedding.hfRepo)
        }
    }

    enum ServiceError: LocalizedError {
        case scriptMissing(String)
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing(let name):
                return "scripts/\(name) not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch the speaker-identity sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    let config: Config

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "wespeaker.write", qos: .utility)

    // Id-correlated requests, one at a time (the sidecar is single-threaded).
    private let lock = NSLock()
    private var nextRequestID = 0
    private var continuations: [Int: CheckedContinuation<IdentityMap, Error>] = [:]
    private let gate = IdentifyGate()

    /// Synchronous critical section — safe to call from async contexts because
    /// it never suspends between lock and unlock.
    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    /// Serial async gate so only one `identify` runs at a time.
    ///
    /// LOAD-BEARING FOR CORRECTNESS, not just for the sidecar's comfort. Every
    /// identify MUTATES the profile stores (a new profile, a centroid nudged
    /// toward this voice), so the order in which they run decides which voice
    /// enrols first and therefore what later voices are matched against. The old
    /// single sidecar got that order for free: one stdin, drained in sequence.
    /// This gate is what preserves it — including the office/remote interleaving,
    /// since both streams share this one queue exactly as they shared one stdin.
    ///
    /// DO NOT remove it "because the requests are fast". Fast is not ordered.
    private actor IdentifyGate {
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
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: Config.logName)
        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        // 120 s: torch plus a ~26 MB embedder. Far quicker than the pipeline it
        // used to share a process with, which is why this step is ordered FIRST
        // in the overlay — a missing embedder is reported in seconds instead of
        // after the pipeline's minute.
        let startup = waitUntilLoaded(timeout: 120)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason
                    ?? "The speaker-identity sidecar did not become ready (timeout)."
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
            try? stdinPipe.fileHandleForWriting.close()
            process.terminate()
        }
    }

    // MARK: - Identity

    /// Identify every local label in ONE diarization run's turns.
    ///
    /// ONE CALL PER RUN, carrying the whole run. The sidecar's one-to-one
    /// guarantee — two distinct voices in the same run can never collapse onto
    /// one profile — is defined per call, so splitting a run across several calls
    /// would silently weaken it.
    ///
    /// Serialized (see `IdentifyGate`), id-correlated, 60 s timeout: a stop-time
    /// pass over a long recording embeds up to 10 s per speaker on MPS, so this is
    /// generous by design. The failure direction that matters is a hung request
    /// holding the stop gate, and every gate already has its own watchdog above
    /// this one.
    func identify(audio: String, turns: [PyannoteService.LocalTurn],
                  stream: PyannoteService.Stream = .office) async throws -> IdentityMap {
        await gate.lock()
        do {
            let result = try await send(audio: audio, turns: turns, stream: stream)
            await gate.unlock()
            return result
        } catch {
            await gate.unlock()
            throw error
        }
    }

    /// Wipe all saved voice profiles — call to start a recording fresh. Rides the
    /// same stdin as the identify jobs, so it is necessarily acknowledged before
    /// any job queued after it: FIFO is the ordering guarantee, not a race.
    func resetProfiles() {
        writeJob(["cmd": "reset"])
    }

    private func send(audio: String, turns: [PyannoteService.LocalTurn],
                      stream: PyannoteService.Stream) async throws -> IdentityMap {
        guard process.isRunning else {
            throw ServiceError.launchFailed("the speaker-identity sidecar is not running")
        }
        let id = withLock { () -> Int in
            nextRequestID += 1
            return nextRequestID
        }

        var job: [String: Any] = [
            "cmd": "identify", "id": id, "audio": audio,
            "turns": turns.map { ["start": $0.start, "end": $0.end, "label": $0.label] },
        ]
        // Office is sent as an ABSENT key, exactly as on the pyannote side.
        if stream == .remote { job["stream"] = stream.rawValue }
        writeJob(job)

        Task { [weak self] in
            try? await Task.sleep(for: .seconds(60))
            guard let self else { return }
            let cont = self.withLock { self.continuations.removeValue(forKey: id) }
            cont?.resume(throwing: ServiceError.startupFailed(
                "speaker identification timed out (60s) — see logs/\(Config.logName).log"))
        }

        return try await withCheckedThrowingContinuation { cont in
            withLock { continuations[id] = cont }
        }
    }

    private func writeJob(_ job: [String: Any]) {
        guard var data = try? JSONSerialization.data(withJSONObject: job) else { return }
        data.append(0x0A)
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: data)
        }
    }

    private func resolve(id: Int?, speakers: IdentityMap?, error: String?) {
        guard let id else { return }
        let cont = withLock { continuations.removeValue(forKey: id) }
        guard let cont else { return }   // already timed out
        if let error {
            cont.resume(throwing: ServiceError.startupFailed(error))
        } else {
            cont.resume(returning: speakers ?? [:])
        }
    }

    // MARK: - Output

    /// Internal rather than private only so the decoding contract can be
    /// unit-tested: an identity with NO `conf` key must decode to `conf == nil`,
    /// and a `null` identity must survive as a present key with a nil value.
    struct Message: Decodable {
        let type: String
        let text: String?
        let id: Int?
        let speakers: IdentityMap?
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
                    errorReason = "The speaker-identity sidecar exited during startup."
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
                case "identify_result":
                    self.resolve(id: message.id, speakers: message.speakers, error: nil)
                case "error":
                    // A startup `error` carries no id and resolves nothing — the
                    // startup reader above owns that case.
                    self.resolve(id: message.id, speakers: nil,
                                 error: message.text ?? "speaker identification failed")
                default: break
                }
            }
        }
    }
}
