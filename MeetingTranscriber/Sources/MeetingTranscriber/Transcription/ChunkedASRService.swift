import Foundation

/// Client for the persistent chunked ASR sidecar.
///
/// The model (Qwen3 / Whisper / Voxtral / Granite / MOSS, per Settings) loads
/// once at session start. During the meeting, audio streams in continuously;
/// every chunk interval the recorder calls `flush()` (aligned to VAD silence)
/// and the sidecar returns an accurate transcript of that chunk.
///
/// ONE SIDECAR PER MODEL, finished 2026-07-30 (owner, 2026-07-29): all five models
/// now have their own standalone script and their own log, and the shared
/// `chunked/chunked-asr-service.py` is deleted. Every script speaks the SAME
/// frames, so this one client drives them all — see `Config.scriptName`, which each
/// model class declares for itself (`ChunkedASRModel.scriptName`).
final class ChunkedASRService: @unchecked Sendable {

    struct Config: Equatable {
        let repoID: String
        let language: String   // "auto" or ISO code
        let modelName: String
        /// Whisper's decoding options, or nil for every other model.
        ///
        /// nil-for-others is STRUCTURAL, not a convention: the other sidecars do
        /// not accept these flags, so making them unrepresentable is what
        /// guarantees "the other models' argument list must not change". Part of
        /// Config (which is Equatable), so moving any knob recreates the sidecar
        /// rather than leaving the previous decoder answering for the rest of
        /// the meeting.
        let whisper: WhisperOptions?
        /// Qwen3's decoding options, or nil for every other model — including
        /// Whisper. Same STRUCTURAL nil-for-others rule as `whisper` above, for
        /// the same reason: `--system-prompt` reaching whisper-service.py would
        /// be an argparse error at session start, and the two option sets must be
        /// unable to cross. Equatable, so moving a Qwen3 knob recreates the Qwen3
        /// sidecar and cannot recreate anyone else's.
        let qwen3: Qwen3Options?
        /// Which sidecar speaks the protocol for this model. Every model gets its
        /// OWN script (there is no shared one left) but they all speak the SAME
        /// frames, so one client drives them all. Part of Config — and Config is
        /// Equatable — so switching models recreates the process rather than
        /// leaving the old model answering.
        let scriptName: String
        /// stderr log basename for this sidecar. One log per script, never two
        /// writers on one file — a mistake this project already made and fixed
        /// (2026-07-15), and splitting the services makes it live again. Stored
        /// rather than switched on `scriptName`: a switch needs a `default:`, and a
        /// default is exactly how a new model would silently end up writing into
        /// another model's log. Declared next to the script by the model itself
        /// (`ChunkedASRModel.logName`), so the pair cannot drift.
        let logName: String

        // Each model's own sidecar, named here so the constants have one home;
        // WHICH one a model uses is declared by the model class, not chosen here.
        // There is deliberately no `defaultScriptName` any more — the file it
        // pointed at (`chunked/chunked-asr-service.py`) is deleted, and a fallback
        // to a deleted script fails at session start with "scripts/… not found"
        // instead of at compile time.
        /// The MOSS sidecar in its DIARIZATION role, its own service since phase 2
        /// (2026-07-31). Used ONLY by `mossDiarization()`. It is the ASR file
        /// minus the `-2` FILE-TRANSCRIBE frame and `emit_file`: `transcribeFile`
        /// is only ever reached through `modelLoader.chunkedASR`, so this process
        /// can never be sent one. Equivalence was proven before the old
        /// `moss/moss-service.py` was deleted — byte-identical stdout over the
        /// whole diar-role input set on real audio.
        static let mossDiarScriptName = "moss-diar/moss-diar-service.py"
        /// The MOSS sidecar in its CHUNKED-ASR role (PyTorch/MPS, its own model
        /// family). A verbatim extraction of the original shared file — proven by
        /// AST and by byte-identical stdout on real audio before it was wired —
        /// differing only in its docstrings and in the vendor path it puts on
        /// `sys.path`. Separate because the two roles can run AS TWO PROCESSES AT
        /// ONCE (MOSS as ASR + MOSS as diarizer), and two processes of one script
        /// means two writers on one log.
        static let mossASRScriptName = "moss-asr/moss-asr-service.py"
        /// The standalone Whisper sidecar (mlx-whisper), split out 2026-07-29.
        static let whisperScriptName = "whisper/whisper-service.py"
        /// The three mlx-audio sidecars, split out 2026-07-30 — each a VERBATIM
        /// extraction of the old shared file's mlx-audio branch, proven
        /// byte-identical on real audio before that file was removed.
        static let qwen3ScriptName = "qwen3/qwen3-service.py"
        static let graniteScriptName = "granite/granite-service.py"
        static let voxtralScriptName = "voxtral/voxtral-service.py"

        /// Settings → Models → Chunked → "Whisper options": the decoding knobs
        /// `mlx_whisper.transcribe()` accepts, exposed 2026-07-29.
        ///
        /// THE GOVERNING RULE: every default below reproduces today's behaviour
        /// exactly, and `processArguments` emits a flag ONLY for a value that
        /// differs from its default — so a default config passes the SAME
        /// argument list it passed before these options existed, and a default
        /// run stays byte-identical. The sidecar enforces the same rule a second
        /// time (`whisper_transcribe_kwargs`, pinned by `sidecar-tests.py`
        /// `whisper/option-defaults-are-todays-behaviour`).
        ///
        /// The two numeric "off" knobs use 0, which the SIDECAR converts to
        /// Whisper's `None`. Sending 0 through would be a different decoder
        /// setting, not the default one.
        ///
        /// ❌ NO `beamSize`, deliberately: mlx_whisper has no beam decoder at all
        /// (`decoding.py:437` raises `NotImplementedError` as soon as beam_size
        /// is set), so the sidecar died in its warmup when it was tried on
        /// 2026-07-29. A control that can only fail is worse than no control.
        struct WhisperOptions: Equatable {
            /// Prior context (names, jargon). RISK, stated in the UI and logged
            /// on every chunk: Whisper continues text, so it can emit words from
            /// the prompt that nobody said.
            var initialPrompt: String = ""
            var bestOf: Int = 0
            var noSpeechThreshold: Double = 0.6
            var logprobThreshold: Double = -1.0
            var compressionThreshold: Double = 2.4
            var task: String = "transcribe"
            /// true ⇒ the sidecar drops the language code entirely.
            var autoDetectLanguage: Bool = false
            /// 0 = OFF. Any value > 0 makes the sidecar force Whisper's own
            /// `word_timestamps` on (the threshold is unreachable otherwise) —
            /// which reverses a measured project decision, so the UI says so.
            var hallucinationSilenceSec: Double = 0

            static let `default` = WhisperOptions()

            /// Extra process arguments — empty at defaults, deliberately.
            var processArguments: [String] {
                func changed(_ value: Double, _ base: Double) -> Bool {
                    abs(value - base) > 1e-9   // slider/stepper values are inexact
                }
                var arguments: [String] = []
                if !initialPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    arguments += ["--initial-prompt", initialPrompt]
                }
                if bestOf > 0 { arguments += ["--best-of", String(bestOf)] }
                if changed(noSpeechThreshold, Self.default.noSpeechThreshold) {
                    arguments += ["--no-speech-threshold", String(noSpeechThreshold)]
                }
                if changed(logprobThreshold, Self.default.logprobThreshold) {
                    arguments += ["--logprob-threshold", String(logprobThreshold)]
                }
                if changed(compressionThreshold, Self.default.compressionThreshold) {
                    arguments += ["--compression-threshold", String(compressionThreshold)]
                }
                if task != Self.default.task { arguments += ["--task", task] }
                if autoDetectLanguage { arguments += ["--auto-detect-language"] }
                if hallucinationSilenceSec > 0 {
                    arguments += ["--hallucination-silence-sec",
                                  String(hallucinationSilenceSec)]
                }
                return arguments
            }

            /// Read from UserDefaults. `object(forKey:)` rather than
            /// `double(forKey:)`: the latter returns 0 for an ABSENT key, which
            /// would silently ship logprobThreshold 0 instead of -1 the first
            /// time the app runs — a real decoding change from a missing key.
            static func fromSettings() -> WhisperOptions {
                let d = UserDefaults.standard
                func number(_ key: String, _ fallback: Double) -> Double {
                    (d.object(forKey: key) as? NSNumber)?.doubleValue ?? fallback
                }
                func count(_ key: String) -> Int {
                    (d.object(forKey: key) as? NSNumber)?.intValue ?? 0
                }
                var options = WhisperOptions()
                options.initialPrompt = d.string(forKey: "whisper.initialPrompt") ?? ""
                options.bestOf = count("whisper.bestOf")
                options.noSpeechThreshold = number("whisper.noSpeechThreshold",
                                                   options.noSpeechThreshold)
                options.logprobThreshold = number("whisper.logprobThreshold",
                                                  options.logprobThreshold)
                options.compressionThreshold = number("whisper.compressionThreshold",
                                                      options.compressionThreshold)
                // `task` and `autoDetectLanguage` keep their defaults and are
                // DELIBERATELY not read. Their controls were removed on
                // 2026-08-06 (owner) and a stored value must not outlive the
                // control that set it: someone who once picked "Translate to
                // English" would otherwise keep getting a translation instead of
                // a record of the words said, with nothing in the UI able to
                // change it back. `WhisperOptions()` already holds "transcribe"
                // and false, which is what the app has always defaulted to.
                options.hallucinationSilenceSec =
                    number("whisper.hallucinationSilenceSec", 0)
                return options
            }
        }

        /// Settings → Models → Chunked → "Qwen3 options": the two decoding knobs
        /// `Qwen3ASRModel.generate()` accepts that were PROVEN to change output,
        /// exposed 2026-08-03.
        ///
        /// THE GOVERNING RULE, identical to `WhisperOptions`: every default below
        /// reproduces today's behaviour exactly, and `processArguments` emits a
        /// flag ONLY for a value that differs from its default — so a default
        /// config passes the SAME argument list it passed before these options
        /// existed, and a default run stays byte-identical (proved: pre- and
        /// post-change sidecars over real audio, md5 9955eaae…). The sidecar
        /// enforces the same rule a second time (`qwen3_generate_kwargs`, pinned
        /// by `sidecar-tests.py` `qwen3/option-defaults-are-todays-behaviour`).
        ///
        /// ONLY TWO KNOBS, and that is a measurement result rather than a partial
        /// job. `generate()` advertises fifteen; each was to be exposed only once
        /// proven to have an effect on real speech, because this project shipped a
        /// Granite language picker that did nothing on 2026-08-01. Measured on
        /// `meeting-2026-07-28T04-10-59Z.wav` from 40 s (its first 40 s are
        /// digital silence), with a same-call-twice determinism control first:
        ///   * `system_prompt` — real effect. A glossary naming "PREP framework"
        ///     turned `Prep` into `PREP`. It does NOT obey instructions and did
        ///     NOT echo invented vocabulary; what it does do is perturb wording
        ///     and punctuation on audio it says nothing about, which is the risk
        ///     the UI states and the sidecar logs per chunk.
        ///   * `repetition_penalty` — real effect, and a sharp one: 1.2 alters
        ///     wording, 2.0 degrades the text into casing/punctuation garbage.
        ///   * `repetition_context_size` — effect only WITH a penalty, so it is
        ///     never sent alone (see below).
        /// The sampling group (temperature/top_p/top_k/min_p) is unmeasured AND
        /// inert as shipped — temperature defaults to 0.0, so the sampler is
        /// greedy and the rest cannot bite. Do not add any without measuring.
        struct Qwen3Options: Equatable {
            /// Context/hotword bias (names, jargon), sent to the model's `system`
            /// role. RISK, stated in the UI and logged on every chunk: a
            /// non-empty prompt shifts wording beyond the vocabulary it names.
            var systemPrompt: String = ""
            /// 0 = OFF. mlx-audio's real default is `None`, NOT a number, so the
            /// sidecar must receive no flag at all here — sending 0 through would
            /// be penalty 0, a different decoder setting. Same sentinel discipline
            /// as WhisperOptions' two numeric "off" knobs.
            var repetitionPenalty: Double = 0
            /// mlx-audio's own default. Meaningful ONLY alongside a penalty (it is
            /// read inside `if repetition_penalty`), so `processArguments` refuses
            /// to emit it on its own — a flag that reaches the model and does
            /// nothing is exactly the Granite-picker failure.
            var repetitionContextSize: Int = 100

            static let `default` = Qwen3Options()

            /// Extra process arguments — empty at defaults, deliberately.
            var processArguments: [String] {
                var arguments: [String] = []
                if !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    arguments += ["--system-prompt", systemPrompt]
                }
                // The context size rides INSIDE this branch on purpose: with no
                // penalty there is nothing for it to size, and emitting it alone
                // would ship a control that cannot affect the transcript.
                if repetitionPenalty > 0 {
                    arguments += ["--repetition-penalty", String(repetitionPenalty)]
                    if repetitionContextSize != Self.default.repetitionContextSize {
                        arguments += ["--repetition-context-size",
                                      String(repetitionContextSize)]
                    }
                }
                return arguments
            }

            /// Read from UserDefaults. `object(forKey:)` rather than
            /// `double(forKey:)`/`integer(forKey:)`: the latter return 0 for an
            /// ABSENT key, which would silently ship repetitionContextSize 0
            /// instead of 100 the first time the app runs — a real decoding
            /// change produced by a missing key. The same trap WhisperOptions
            /// documents; it applies to the Int here just as much as the Double.
            static func fromSettings() -> Qwen3Options {
                let d = UserDefaults.standard
                func number(_ key: String, _ fallback: Double) -> Double {
                    (d.object(forKey: key) as? NSNumber)?.doubleValue ?? fallback
                }
                func count(_ key: String, _ fallback: Int) -> Int {
                    (d.object(forKey: key) as? NSNumber)?.intValue ?? fallback
                }
                var options = Qwen3Options()
                options.systemPrompt = d.string(forKey: "qwen3.systemPrompt") ?? ""
                options.repetitionPenalty = number("qwen3.repetitionPenalty",
                                                   options.repetitionPenalty)
                options.repetitionContextSize = count("qwen3.repetitionContextSize",
                                                      options.repetitionContextSize)
                return options
            }
        }

        /// Resolve from Settings → Models → Chunked via the model classes.
        static func fromSettings() -> Config {
            let d = UserDefaults.standard
            let model = ChunkedASRModelFactory.fromSettings()
            // Resolved, not read raw: `chunked.language` can hold a code picked
            // while ANOTHER model was selected (the settings value is not
            // rewritten when the model changes, by design), and passing it
            // through would send e.g. `--language id` to Voxtral, which has no
            // Indonesian. An unsupported code becomes "auto" here so a stale
            // stored value can never reach a sidecar.
            let stored = d.string(forKey: "chunked.language") ?? "auto"
            let code = Languages.resolve(language: stored, forModel: model.info.id)
            let isWhisper = model.info.id == "whisper"
            let isQwen3 = model.info.id == "qwen3"
            // `align.enabled` is deliberately NOT read here any more: the forced
            // aligner is its own sidecar (`AlignerService`), so turning alignment
            // on no longer changes this config at all — and must not recreate the
            // ASR process, which would reload a 1.7–4B model for a setting it
            // knows nothing about.
            return Config(repoID: model.repoID,
                          language: model.languageArgument(for: code) ?? "auto",
                          modelName: model.info.name,
                          // Only the Whisper sidecar understands these flags, so
                          // only the Whisper config carries them. Also means a
                          // Whisper knob moved while Qwen3 is selected does not
                          // pointlessly recreate the Qwen3 sidecar.
                          whisper: isWhisper ? WhisperOptions.fromSettings() : nil,
                          // Same rule, same reason: only the Qwen3 sidecar
                          // understands these flags, and a Qwen3 knob moved while
                          // Whisper is selected must not recreate the Whisper
                          // sidecar. The two are mutually exclusive by model id.
                          qwen3: isQwen3 ? Qwen3Options.fromSettings() : nil,
                          // Asked of the model, not decided here: the model class
                          // is where "one service per model" is enforced at
                          // compile time (see ChunkedASRModel.scriptName).
                          scriptName: model.scriptName,
                          logName: model.logName)
        }

        /// The MOSS sidecar, used purely as the DIARIZATION engine while a
        /// different model does the ASR. Same process type, same protocol; the
        /// caller keeps its `segments` and discards its `text`.
        ///
        /// Both names are LITERALS, deliberately NOT taken from
        /// `MossTranscribeDiarizeModel()` — that class declares the ASR role's
        /// script and log, and deriving either here would point this process at
        /// the other role's file. The two roles are two live processes in
        /// MOSS+MOSS mode, so a derived name is silent drift: both processes keep
        /// working while one writes into the other's log. The phase-1 catch that
        /// found this stands unchanged now that each role has its own service.
        static func mossDiarization() -> Config {
            Config(repoID: ModelCatalog.mossDiarization.hfRepo,
                   language: "auto",
                   modelName: ModelCatalog.mossDiarization.name,
                   whisper: nil,
                   qwen3: nil,
                   scriptName: mossDiarScriptName,
                   logName: "moss-diar")
        }
    }

    /// The aligner's word type, which MOVED to `AlignerService` when the aligner
    /// became its own sidecar (2026-07-29). Kept as a typealias because the word
    /// data itself did not change: `WordAttribution`, `AudioRecorder` and their
    /// tests keep naming it `ChunkedASRService.AlignedWord` and keep receiving
    /// byte-identical `{text,start,end,src}` items.
    typealias AlignedWord = AlignerService.AlignedWord

    /// One speaker-attributed segment from the MOSS sidecar, as returned on a
    /// `final` alongside the plain text.
    ///
    /// `start`/`end` are seconds relative to the START OF THE CHUNK BUFFER, not
    /// the recording — the caller offsets them by the window start. `speaker` is
    /// the model's RAW label ("S01") and is anonymous PER CALL: S01 in one chunk
    /// is not S01 in the next, and nothing downstream may assume otherwise.
    struct MossSegment: Decodable, Equatable, Sendable {
        let start: Double
        let end: Double
        let speaker: String
        let text: String
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
                return "Could not launch chunked ASR sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    let config: Config

    /// Called with each chunk's transcript and the chunk's ASR confidence.
    ///
    /// Parameters: (text, confidence). The confidence is a 0…1 pooled per-token
    /// probability and is present on the WHISPER runtime only — the mlx-audio
    /// models expose no such signal, so it is nil for them. nil means "this model
    /// reports nothing", never "low confidence".
    ///
    /// No words here any more: the forced aligner is its own sidecar
    /// (`AlignerService`) and answers on its own schedule, so the recorder asks
    /// it separately and refines the rows when the reply lands.
    var onChunkTranscript: ((String, Double?) -> Void)?

    /// Called with the MOSS sidecar's own speaker segmentation for a chunk, when
    /// the `final` carried one. Never fires for the MLX models — they send no
    /// `segments` key at all.
    ///
    /// Fired BEFORE `onChunkTranscript` for the same `final`, deliberately: the
    /// recorder reads the chunk window that `onChunkTranscript` then pops, so the
    /// order has to be fixed rather than incidental.
    var onChunkSegments: (([MossSegment]) -> Void)?

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
    private var fileContinuations: [Int: CheckedContinuation<FileTranscript, Error>] = [:]
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

        let script = PythonRuntime.scriptsDir.appendingPathComponent(config.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing(config.scriptName)
        }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        var arguments = command.arguments + ["--model", config.repoID]
        if config.language != "auto" {
            arguments += ["--language", config.language]
        }
        // Whisper-only, and empty at default settings — see WhisperOptions.
        // Nothing is appended for any other model, by construction.
        arguments += config.whisper?.processArguments ?? []
        // Qwen3-only, same discipline — see Qwen3Options. At most one of these
        // two lines can ever contribute anything: `fromSettings` populates the
        // options for the SELECTED model alone, so the other is nil.
        arguments += config.qwen3?.processArguments ?? []
        process.arguments = arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        // One log per sidecar, never two writers on one file: more than one of
        // these processes can be alive at once (e.g. "other ASR + MOSS
        // diarization"), so each script owns its own log — see Config.logName.
        process.standardError = PythonRuntime.logHandle(name: config.logName)

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

    /// One file-transcribe result: the text, and the model's confidence in it
    /// when the runtime reports one (Whisper) — nil otherwise.
    typealias FileTranscript = (text: String, conf: Double?)

    /// Transcribe one WAV file with the already-loaded chunked model.
    /// Serialized (one at a time), id-correlated, 120s timeout. Used by the
    /// overlap-repair pass to re-ASR each separated track, and by the Remote
    /// stream for its chunks.
    func transcribeFile(path: String) async throws -> FileTranscript {
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

    private func sendFileRequest(path: String) async throws -> FileTranscript {
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
                "file transcription timed out (120s) — see logs/\(self.config.logName).log"))
        }

        return try await withCheckedThrowingContinuation { cont in
            withFileLock { fileContinuations[id] = cont }
        }
    }

    private func resolveFile(id: Int?, text: String, conf: Double?, isError: Bool) {
        guard let id else { return }
        let cont = withFileLock { fileContinuations.removeValue(forKey: id) }
        guard let cont else { return }   // already timed out
        if isError {
            cont.resume(throwing: ServiceError.startupFailed(text))
        } else {
            cont.resume(returning: (text: text, conf: conf))
        }
    }

    // MARK: - Output

    /// Internal rather than private only so the decoding contract (the aligner
    /// keys must stay optional) can be covered by a unit test.
    struct Message: Decodable {
        let type: String
        let text: String
        let id: Int?
        // No `words`/`dur`: alignment moved to its own sidecar (2026-07-29) and
        // these sidecars no longer emit them. A STALE packaged sidecar still
        // might — JSONDecoder ignores unknown keys, so such a payload decodes
        // fine and the extra keys are simply not read.
        // Additive, Whisper-only: the chunk's pooled per-token probability.
        // Absent for the mlx-audio models (they report no confidence at all) and
        // for every message type other than final/file_result, so it stays
        // optional and every older payload still decodes.
        let conf: Double?
        // Additive, MOSS-only: the model's own speaker segmentation for this
        // chunk. Optional for the same reason `conf` is — the MLX sidecars never
        // send it, so every payload that predates MOSS still decodes.
        let segments: [MossSegment]?
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
                case "final":
                    // Segments FIRST: see `onChunkSegments`. Absent for every
                    // MLX model, so this line is inert on the existing path.
                    if let segments = message.segments {
                        self.onChunkSegments?(segments)
                    }
                    self.onChunkTranscript?(message.text, message.conf)
                case "error": self.onChunkError?(message.text)
                case "file_result":
                    self.resolveFile(id: message.id, text: message.text,
                                     conf: message.conf, isError: false)
                case "file_error":
                    self.resolveFile(id: message.id, text: message.text,
                                     conf: nil, isError: true)
                default: break
                }
            }
        }
    }
}
