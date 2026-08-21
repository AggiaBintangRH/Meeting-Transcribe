import Foundation

/// Client for the realtime ASR sidecars.
///
/// THREE ENGINES since 2026-08-11, ONE client: Nemotron 3.5
/// (`scripts/nemotron/nemotron-service.py`), Parakeet TDT 0.6b v3
/// (`scripts/parakeet/parakeet-service.py`) and Fun-ASR MLT Nano 2512
/// (`scripts/funasr/funasr-service.py`). Each model gets its own sidecar —
/// the one-service-per-model decision stands, and copies are the point — but
/// they speak the SAME frames, so one client drives all three. That is the
/// chunked side's shape exactly: five ASR scripts, one `ChunkedASRService`,
/// routed by `Config.scriptName`/`Config.logName`.
///
/// Why not two Swift clients: everything below is pure wire plumbing with
/// safety-critical invariants that have nothing model-specific in them — the
/// serialized write queue (two lanes must never interleave halfway through a
/// frame), routing strictly by the `stream` tag, the READY handshake. A second
/// copy would make every future lane fix a two-file edit whose omission fails
/// SILENTLY, which is the failure mode this project keeps paying for.
///
/// ONE process, TWO independent lanes — `office` and `remote`. Feed 16 kHz mono
/// samples to a lane continuously; call `flush()` on it at utterance end (VAD
/// speech→silence edge). Each lane emits partial transcripts every ~1.5 s while
/// its audio streams in, and a final transcript on its flush.
///
/// The lanes share only the process (and therefore one copy of the weights) —
/// never audio. Each writes its own frame type, the sidecar keeps a separate
/// buffer per lane, and results are routed back by the `stream` tag, so the two
/// waveforms never meet on either side of the pipe. That separation is the
/// entire value of dual-stream capture; do not "optimize" it into one buffer.
///
/// A session without a Remote channel simply never touches `remote`, which means
/// the wire traffic is byte-identical to the single-stream sidecar.
///
/// Configured from Settings → Models → Realtime (model, language, chunk size).
final class RealtimeASRService: @unchecked Sendable {

    /// The engine used when nothing has been chosen. Parakeet since 2026-08-21
    /// (owner-requested, so a fresh Mac needs no setting up) — see
    /// `ShippedDefaults.realtimeModel` for the measurement.
    ///
    /// ⚠ THIS IS NOT NEMOTRON'S ID, and it used to be. `defaultModelID` was
    /// doing double duty: the nemotron factory below named its own Config with
    /// it. Harmless while the two strings were equal, and the moment the default
    /// moved it produced a Config that said "parakeet" while running
    /// `nemotron-service.py`. Three tests caught it. Keep the two ideas apart.
    static let defaultModelID = ShippedDefaults.realtimeModel
    static let nemotronModelID = "nemotron"
    static let parakeetModelID = "parakeet"
    static let funasrModelID = "funasr"

    /// Nemotron's attention chunk size, PINNED — there is no longer a control
    /// for it (owner, 2026-08-11: *"oke hapus aja deh"*).
    ///
    /// 1120 is not a conservative pick, it is the measured best AND the value
    /// the app has effectively always run at. Seconds per partial on a 30 s
    /// buffer: **1120 → 2.08 · 560 → 3.42 · 160 → 5.49 · 80 → 18.04**, and 1120
    /// is byte-identical to running with no attention context at all. Every
    /// other option is slower, and no measurement has ever shown one of them
    /// more ACCURATE — only different. A control whose every alternative is a
    /// way to make things worse is the `ahc_threshold` case from 2026-08-10,
    /// declined for the same reason.
    ///
    /// The key `realtime.chunkMs` is deliberately NO LONGER READ. A stored value
    /// outliving its control keeps deciding behaviour with nothing in the UI able
    /// to change it back — the rule the 2026-08-06 settings pass established for
    /// six other keys. A literal here, not four scattered ones, so restoring the
    /// picker later is one edit.
    ///
    /// The flag still travels to the sidecar and `ATT_CONTEXT` still exists
    /// there: this removes a CONTROL, not a capability, and the sidecar's
    /// `mx.array` fix from the same day is what makes the pinned value actually
    /// arrive instead of being silently dropped.
    static let pinnedChunkMs = 1120

    /// The shipped caption cadence, in ms, for BOTH engines.
    ///
    /// **1500 is the value both sidecars used before the setting existed**
    /// (`PARTIAL_EVERY = 1.5 s`), so an untouched install behaves exactly as it
    /// did — the house rule that a new setting changes nothing until it is
    /// deliberately moved. Both sidecars declare the same default and both clamp
    /// to 250…10000, so a stale key cannot put a lane past 100 % duty.
    ///
    /// ONE key (`realtime.partialMs`), shared, not one per engine — the control
    /// is the same control, so a choice made under one engine survives a switch
    /// to the other. Two keys would have made "the same setting" a claim the
    /// storage quietly contradicted.
    static let defaultPartialMs = 1500

    struct Config: Equatable {
        /// Which realtime engine this process runs. Part of Config — and Config
        /// is Equatable — so switching engines RECREATES the process rather than
        /// leaving the old model answering for the rest of the session. That
        /// equality IS the engine-switch mechanism; see `ModelLoader.load`.
        let modelID: String
        /// "auto", a Nemotron locale key ("id-ID"), or a Parakeet ISO code.
        let language: String
        /// Nemotron's attention chunk size (80 / 160 / 560 / 1120), or nil for
        /// every other engine.
        ///
        /// nil-for-others is STRUCTURAL, not a convention — the same rule
        /// `ChunkedASRService.Config.whisper` follows. Parakeet's sidecar takes
        /// no `--chunk-ms` (it has no attention-context setting to point one at),
        /// so making the value unrepresentable is what guarantees two things at
        /// once: the flag can never reach the wrong script, and moving a Nemotron
        /// knob while Parakeet is selected does not recreate the Parakeet process.
        let chunkMs: Int?
        /// The caption cadence in ms — how much NEW audio a lane collects before
        /// spending a partial. **BOTH engines have it**, same control, same
        /// values (owner, 2026-08-11: *"jangan dibedakan, soalnya sama tentang
        /// waktu interval"*).
        ///
        /// It was Parakeet-only for a few hours, on the reasoning that Nemotron
        /// is too slow for the cadence to be a free choice. That was a reason to
        /// treat the value differently INSIDE the sidecar, not a reason to
        /// withhold the control — and withholding it left Nemotron's caption
        /// speed governed by a constant no one could reach.
        ///
        /// The engines differ in how the number is honoured, and the sidecars
        /// say so: for Parakeet it is the exact cadence, for Nemotron a FLOOR
        /// that `PARTIAL_DUTY` stretches on long buffers, because its partial
        /// costs 2.08 s against Parakeet's 0.235 s.
        let partialMs: Int?
        /// Which sidecar speaks the protocol for this engine.
        let scriptName: String
        /// stderr log basename for this sidecar. One log per script, never two
        /// writers on one file (the 2026-07-15 mistake).
        ///
        /// A LITERAL per engine, never derived from the model name or the script
        /// path. `Config.mossDiarization()` took its log name from a model class
        /// and a rename silently repointed one process's stderr into another
        /// service's log — while both processes kept working, which is why it
        /// was invisible. Stated once, per engine, beside the script it pairs
        /// with; `RealtimeSidecarRoutingTests` asserts the two pairs differ.
        let logName: String

        static let nemotronScriptName = "nemotron/nemotron-service.py"
        static let parakeetScriptName = "parakeet/parakeet-service.py"
        static let funasrScriptName = "funasr/funasr-service.py"

        /// Nemotron 3.5 — the original realtime engine and the default.
        static func nemotron(language: String, chunkMs: Int, partialMs: Int) -> Config {
            Config(modelID: RealtimeASRService.nemotronModelID,
                   language: language,
                   chunkMs: chunkMs,
                   partialMs: partialMs,
                   scriptName: nemotronScriptName,
                   logName: "nemotron")
        }

        /// Parakeet TDT 0.6b v3 — CC BY-4.0, 25 languages, ~130x realtime.
        static func parakeet(language: String, partialMs: Int) -> Config {
            Config(modelID: RealtimeASRService.parakeetModelID,
                   language: language,
                   chunkMs: nil,
                   partialMs: partialMs,
                   scriptName: parakeetScriptName,
                   logName: "parakeet")
        }

        /// Fun-ASR MLT Nano 2512 — Apache 2.0, 1.0B, ~49–97x realtime.
        ///
        /// The ONLY realtime engine whose `language` is actually honoured, which
        /// is why it takes the code unresolved-by-locale-table: its sidecar
        /// speaks bare ISO codes (`en`, `zh`, `ja`), not Nemotron's locale keys.
        static func funasr(language: String, partialMs: Int) -> Config {
            Config(modelID: RealtimeASRService.funasrModelID,
                   language: language,
                   chunkMs: nil,
                   partialMs: partialMs,
                   scriptName: funasrScriptName,
                   logName: "funasr")
        }

        /// Read the current settings.
        ///
        /// An ABSENT `realtime.model` must produce exactly the SHIPPED default's
        /// config, field for field — that is what stops the first launch after an
        /// update from tearing down and reloading a perfectly good sidecar
        /// (`Config` equality is the reuse test in `ModelLoader.load`). The
        /// default itself is `ShippedDefaults.realtimeModel`, Parakeet since
        /// 2026-08-21. Pinned by
        /// `RealtimeLifecycleTests.testAnAbsentModelKeyIsExactlyTheShippedDefaultConfig`.
        static func fromSettings() -> Config {
            let d = UserDefaults.standard
            // ⚠ RESOLVED THROUGH THE CATALOG, not used raw, and that is the
            // `default:` trap this project keeps recording. The chain below ends
            // in an unconditional `.nemotron(...)`, so BEFORE 2026-08-21 an id
            // this build does not have — a stored engine from a newer version, a
            // typo — silently became Nemotron, which was invisible only because
            // Nemotron was also the default. The moment the default moved, an
            // unknown id resolved to an engine the loader would never have
            // started. `ModelCatalog.realtimeModel(id:)` already implements
            // "unknown → the first entry", and the first entry is asserted equal
            // to the shipped default, so this reuses that one rule instead of
            // keeping a second list of known ids here to drift.
            let stored = d.string(forKey: "realtime.model") ?? RealtimeASRService.defaultModelID
            let id = ModelCatalog.realtimeModel(id: stored).id
            let raw = d.string(forKey: "realtime.language") ?? "auto"

            let partialMs = d.object(forKey: "realtime.partialMs") as? Int
                ?? RealtimeASRService.defaultPartialMs

            // Resolved at the READ boundary, exactly like the chunked side: a
            // code the selected engine does not have becomes "auto" rather than
            // travelling to a sidecar that has never heard of it. The user's
            // stored choice is never rewritten behind their back; it is only
            // overridden while an engine that lacks it is picked.
            //
            // For Fun-ASR this boundary is LOAD-BEARING rather than tidy: its
            // `generate()` RAISES on an unknown ISO code instead of ignoring it,
            // so an unclamped "id" would not degrade the caption, it would kill
            // the lane for the rest of the meeting. The sidecar validates the
            // code a second time for exactly that reason.
            if id == RealtimeASRService.parakeetModelID {
                return .parakeet(
                    language: Languages.resolveRealtime(language: raw, forModel: id),
                    partialMs: partialMs)
            }
            if id == RealtimeASRService.funasrModelID {
                return .funasr(
                    language: Languages.resolveRealtime(language: raw, forModel: id),
                    partialMs: partialMs)
            }

            // Nemotron speaks locale keys, not bare ISO codes.
            let locale: [String: String] = [
                "id": "id-ID", "en": "en-US", "ms": "ms-MY",
                "zh": "zh-CN", "ja": "ja-JP",
            ]
            return .nemotron(language: locale[raw] ?? "auto",
                             chunkMs: RealtimeASRService.pinnedChunkMs,
                             partialMs: partialMs)
        }

        /// The argument list for this engine's sidecar. Each optional flag is
        /// emitted only when THAT engine has it, so neither argparse ever sees a
        /// flag it does not declare — `--chunk-ms` is Nemotron's alone and
        /// `--partial-ms` is Parakeet's alone.
        var processArguments: [String] {
            var args = ["--language", language]
            if let chunkMs {
                args += ["--chunk-ms", String(chunkMs)]
            }
            if let partialMs {
                args += ["--partial-ms", String(partialMs)]
            }
            return args
        }
    }

    let config: Config

    enum ServiceError: LocalizedError {
        /// Carries the script it looked for. It used to name
        /// `scripts/nemotron/nemotron-service.py` unconditionally, which under a
        /// second engine would send the reader to the wrong file entirely — the
        /// defect the DiariZen audit found in `NemoService` and fixed for the
        /// same reason.
        case scriptMissing(String)
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing(let name):
                return "scripts/\(name) not found in the project folder."
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
        private unowned let service: RealtimeASRService

        init(service: RealtimeASRService, isRemote: Bool) {
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
    private let writeQueue = DispatchQueue(label: "realtime-asr.write", qos: .userInitiated)

    init(config: Config = .fromSettings()) throws {
        self.config = config
        // Both lanes exist before the process does, so nothing has to check for
        // their presence later; neither writes anything until it is fed.
        self.office = Lane(service: self, isRemote: false)
        self.remote = Lane(service: self, isRemote: true)

        let script = PythonRuntime.scriptsDir.appendingPathComponent(config.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing(config.scriptName)
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments + config.processArguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: config.logName)

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
