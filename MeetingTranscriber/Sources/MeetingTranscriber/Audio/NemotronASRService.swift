import Foundation

/// Client for the Nemotron 3.5 realtime ASR sidecar (scripts/nemotron/nemotron-service.py).
///
/// ONE process, TWO independent lanes — `office` and `remote`. Feed 16 kHz mono
/// samples to a lane continuously; call `flush()` on it at utterance end (VAD
/// speech→silence edge). Each lane emits partial transcripts every ~1.5 s while
/// its audio streams in, and a final transcript on its flush.
///
/// The lanes share only the process (and therefore one copy of the ~1 GB
/// weights) — never audio. Each writes its own frame type, the sidecar keeps a
/// separate buffer per lane, and results are routed back by the `stream` tag, so
/// the two waveforms never meet on either side of the pipe. That separation is
/// the entire value of dual-stream capture; do not "optimize" it into one buffer.
///
/// A session without a Remote channel simply never touches `remote`, which means
/// the wire traffic is byte-identical to the single-stream sidecar.
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
                return "scripts/nemotron/nemotron-service.py not found in the project folder."
            case .launchFailed(let reason):
                return "Could not launch Python sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    /// One audio stream's client-side handle: the same `feed` / `flush` /
    /// `onTranscript` ergonomics the service itself used to expose, scoped to a
    /// single lane. Holding two of these (rather than two services) is what
    /// keeps the call sites unchanged while the process count drops to one.
    ///
    /// The lane owns no audio state of its own — the buffering happens in the
    /// sidecar, per lane — so there is nothing here for two lanes to share.
    final class Lane: @unchecked Sendable {

        /// Transcript callback: (text, isFinal). Called on an arbitrary thread.
        var onTranscript: ((String, Bool) -> Void)?

        /// Frame opcodes this lane writes. Office keeps the historical encoding
        /// (a bare positive count, `0` to flush); remote uses additive negative
        /// opcodes so an office-only session's byte stream is unchanged.
        private let isRemote: Bool
        private unowned let service: NemotronASRService

        init(service: NemotronASRService, isRemote: Bool) {
            self.service = service
            self.isRemote = isRemote
        }

        /// Stream samples to this lane's recognizer.
        func feed(_ samples: [Float]) {
            guard !samples.isEmpty else { return }
            var packet = Data()
            if isRemote {
                // -2 = "remote audio follows", then the sample count. Two headers
                // rather than a signed count so the office path stays literally
                // the bytes it always was.
                packet.append(Self.header(-2))
            }
            packet.append(Self.header(Int32(samples.count)))
            samples.withUnsafeBufferPointer { packet.append(Data(buffer: $0)) }
            service.write(packet)
        }

        /// Utterance ended — request this lane's final transcript and reset only
        /// this lane's buffer. The other lane keeps accumulating untouched.
        func flush() {
            service.write(Self.header(isRemote ? -3 : 0))
        }

        private static func header(_ value: Int32) -> Data {
            withUnsafeBytes(of: value.littleEndian) { Data($0) }
        }
    }

    /// The room microphone. Drives the clock, the VAD and both cadences.
    ///
    /// Assigned in `init`, never nil afterwards. Deliberately NOT `lazy`: the
    /// audio tap and the stdout reader touch the lanes from different threads,
    /// and `lazy` initialization is not thread-safe.
    private(set) var office: Lane!

    /// The conferencing (Remote) stream. Captions only — it must never reach the
    /// ATND/position path or the speaker profile store.
    private(set) var remote: Lane!

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()

    /// All writes off the audio thread — transcription pauses can back up the
    /// pipe, and a blocked write must never stall audio capture.
    private let writeQueue = DispatchQueue(label: "nemotron.write", qos: .userInitiated)

    init(config: Config = .fromSettings()) throws {
        self.config = config
        // Both lanes exist before the process does, so nothing has to check for
        // their presence later; neither writes anything until it is fed.
        self.office = Lane(service: self, isRemote: false)
        self.remote = Lane(service: self, isRemote: true)

        let script = PythonRuntime.scriptsDir.appendingPathComponent("nemotron/nemotron-service.py")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments + ["--language", config.language,
                                                 "--chunk-ms", String(config.chunkMs)]
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: "nemotron")

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

    /// The single writer both lanes go through. Serialized on `writeQueue`, so
    /// two lanes writing concurrently can never interleave halfway through a
    /// frame — which is what would let one lane's samples be read as the other's.
    fileprivate func write(_ packet: Data) {
        guard process.isRunning else { return }
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: packet)
        }
    }

    /// Detach the remote lane when a session no longer wants remote captions.
    /// There is no process to orphan any more — the lane simply goes silent —
    /// but a stale callback would still hold a finished session's recorder, so
    /// clear it. The sidecar-side buffer is already empty: `stop()` flushes the
    /// remote lane, and nothing feeds it afterwards.
    func detachRemoteLane() {
        remote.onTranscript = nil
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
        /// Which lane produced this line. ABSENT MEANS OFFICE — the same
        /// convention pyannote/pyannote-service.py and wespeaker/wespeaker-service.py
        /// use, chosen so office lines keep their exact historical bytes.
        let stream: String?
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
                // Route strictly by the tag: an untagged line is office, and a
                // remote line can only ever reach the remote lane's callback.
                let lane: Lane = message.stream == "remote" ? self.remote : self.office
                switch message.type {
                case "partial": lane.onTranscript?(message.text, false)
                case "final":   lane.onTranscript?(message.text, true)
                default: break
                }
            }
        }
    }
}
