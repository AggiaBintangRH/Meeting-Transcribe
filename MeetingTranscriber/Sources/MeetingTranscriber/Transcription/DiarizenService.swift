import Foundation

/// Client for the DiariZen diarization sidecar (`scripts/diarizen/diarizen-service.py`).
///
/// The FIFTH diarizer in the project, alongside `PyannoteService` (pyannote
/// community-1), `SpectralService` (the vendored FoxNoseTech engine),
/// `NemoService` (NVIDIA's `ClusteringDiarizer`) and MOSS (speaker-attributed
/// ASR, driven through `ChunkedASRService`).
///
/// THE ENGINE IS BUT Speech@FIT's DiariZen, checkpoint `BUT-FIT/diarizen-meeting-base`:
/// a WavLM-Base+ encoder with Conformer layers producing END-TO-END NEURAL
/// SEGMENTATION (EEND), then WeSpeaker embeddings clustered agglomeratively
/// (`ahc_threshold` 0.7, `min_speakers` 2, `max_speakers` 8, and the one tuned
/// field, `min_cluster_size` — see its declaration in the sidecar).
///
/// That makes it architecturally UNLIKE the other four, which all segment first
/// and embed second. It is also the only one TRAINED ON FAR-FIELD SINGLE-CHANNEL
/// MEETINGS (AMI, AISHELL-4, AliMeeting) — the owner's ATND array's conditions.
///
/// IT IS ON THE PIPELINE SIDE OF THE 2026-07-30 SPLIT, exactly like pyannote and
/// spectral: it answers WHO-SPOKE-WHEN with run-local labels and has never seen a
/// profile store. Identity is `WeSpeakerService`'s job and
/// `AudioRecorder.composeTurns` joins the two, so saved voice profiles, renaming
/// and `spk` confidence all work under this engine with **no new identity code
/// and no new id space** — the labels become office profile ids (< 10 000) on the
/// same path spectral's do. That is a genuine advantage over MOSS, which names
/// its own speakers.
///
/// The labels themselves are plain integer strings — `"0"`, `"1"`, `"2"` —
/// rather than pyannote's `SPEAKER_00` or NeMo's `speaker_0` (measured, not
/// assumed). Verified before wiring: nothing in Swift parses a label — they are
/// opaque keys handed to `identify` — so they are deliberately NOT normalised. A
/// normaliser would be a second place for a label to change shape.
///
/// **UNLIKE THE OTHER BATCH ENGINES, ITS TURNS CAN INTERSECT.** EEND predicts
/// per-speaker activity, so genuine simultaneous speech comes back as two turns
/// covering the same instant — measured on `recordings/Overlap123.wav`:
/// 11 intersecting pairs, 13.30 s. Spectral and NeMo cannot do this (one label
/// per instant, structurally). Overlap TAGGING therefore works under this engine
/// with no detector, since `derivedRows` reads `overlapRegions()` — and so does
/// overlap REPAIR, since `AudioRecorder.usesDetectedRegionsForRepair` moved this
/// engine onto pyannote's side of that line on 2026-08-10. Both are OFFICE-only,
/// like every other consumer of `overlapRegions()`: the remote stream is a
/// separate identity space and its turns never enter `liveTurns`.
///
/// **FINAL ONLY — THERE IS DELIBERATELY NO LIVE / PER-CHUNK PATH**, enforced
/// STRUCTURALLY rather than by a guard, the same way `SpectralService` does it:
/// there is no `diarizeChunk`, no `onChunkResult`, and no way to construct a job
/// whose `cmd` is not `final`. The agglomerative clustering runs over the WHOLE
/// file, so a 30 s window would be clustered on its own and its labels would mean
/// nothing across windows: the exact failure MOSS has when it is called per
/// chunk. The sidecar refuses anything else loudly, but the only safe number of
/// ways to send one is zero.
///
/// **IT RUNS IN ITS OWN INTERPRETER, `.venv-diarizen`** — the third sidecar to do
/// so, after DiCoW and NeMo, and the ONLY one on Python 3.11: upstream pins
/// torch 2.1.1, which has no 3.12 wheels, and its vendored pyannote-audio 3.1.1
/// imports `torchaudio.AudioMetaData`, removed in torchaudio 2.9.
final class DiarizenService: @unchecked Sendable {

    /// The wire currency is SHARED with `PyannoteService`, deliberately.
    ///
    /// The sidecar's office replies are byte-identical to the pyannote sidecar's
    /// (verified in stage 2, including the `stream` echo for remote and its
    /// ABSENCE for office), and the identity stage —
    /// `AudioRecorder.identifyFinalTurns` → `WeSpeakerService.identify` →
    /// `AudioRecorder.composeTurns` — must accept every engine's turns without a
    /// second overload of every function on the path. Aliased rather than
    /// redeclared so the two can never drift into two structurally identical but
    /// incompatible types.
    typealias LocalTurn = PyannoteService.LocalTurn
    typealias Stream = PyannoteService.Stream

    /// Which sidecar this service runs and which log it writes.
    ///
    /// **BOTH ARE LITERALS, and that is the point.** `ChunkedASRService.Config.mossDiarization()`
    /// once took its `logName` from a model type's `logName` property; when the
    /// ASR role was renamed, that derivation silently repointed the DIARIZATION
    /// process's stderr into the ASR log **while it still ran the other script** —
    /// a script/log pair that has drifted, invisible because both processes keep
    /// working. Naming both here, as strings, is what makes that impossible;
    /// `testDiarizenConfigNamesItsOwnSidecarAndLogAsLiterals` is its contract.
    struct Config: Equatable {
        /// The interpreter this sidecar MUST run under. Not the main `.venv`: see
        /// the type doc, and `ServiceError.venvMissing` for what a missing one
        /// says.
        static let venvName = ".venv-diarizen"
        static let scriptName = "diarizen/diarizen-service.py"
        static let logName = "diarizen"
    }

    enum ServiceError: LocalizedError {
        case scriptMissing
        case venvMissing
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "scripts/diarizen/diarizen-service.py not found in the project folder."
            case .venvMissing:
                // NO FALLBACK to the main `.venv`, and the message says which
                // runtime is missing rather than "a model failed to load": running
                // this sidecar on the wrong interpreter fails deep inside DiariZen
                // with a far more confusing error. Same rule as DiCoW's.
                return "The DiariZen runtime (\(Config.venvName)) is missing. "
                     + "Run download-best-models.sh, then retry."
            case .launchFailed(let reason):
                return "Could not launch the DiariZen diarization sidecar: \(reason)"
            case .startupFailed(let reason):
                return reason
            }
        }
    }

    /// Whole-file result for one stream: (the audio file, run-local turns, stream).
    ///
    /// The AUDIO PATH IS LOAD-BEARING, not a convenience — it is what the caller
    /// hands the identity stage next, so there is no second bookkeeping map to
    /// keep in step. Same contract as `PyannoteService.onFinalResult`.
    var onFinalResult: ((String, [LocalTurn], Stream) -> Void)?
    /// Job error, tagged with the stream it belongs to so one stream's failure
    /// never settles the other's gate. Startup errors carry no stream → office.
    var onError: ((String, Stream) -> Void)?

    let config: Config

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "diarizen.write", qos: .utility)

    init(config: Config = Config()) throws {
        self.config = config
        let script = PythonRuntime.scriptsDir.appendingPathComponent(Config.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }
        // DiariZen's own venv only — see `ServiceError.venvMissing`. Nil means setup
        // never ran; there is deliberately no fall-through to the main `.venv`.
        guard let command = PythonRuntime.command(forScript: script,
                                                  venvName: Config.venvName) else {
            throw ServiceError.venvMissing
        }
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: Config.logName)
        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }

        // A DELIBERATELY GENEROUS BUDGET AGAINST A FAST MEASURED STARTUP.
        // Measured on this M4: **4.8 s** warm from launch to `LOADED` — the WavLM
        // encoder and the WeSpeaker embedder are both small, so this engine is one
        // of the QUICKEST to come up, not the slowest. (The ~51 s figure in
        // `NemoService` is NeMo's `nemo.collections.asr` import and does not apply
        // here; it was copied in with this file's first draft and is corrected.)
        //
        // The budget stays at 600 s anyway, because it costs nothing when nothing
        // is wrong and the failure it guards is a COLD first launch — weights read
        // for the first time off a client Mac's disk, with Gatekeeper verifying a
        // freshly-installed bundle. That must not look like a failure.
        let startup = waitUntilLoaded(timeout: 600)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "The DiariZen sidecar did not become ready (timeout)."
            )
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    // MARK: - Jobs

    /// The ONLY job this engine has: one whole-file pass. `numSpeakers` 0 = auto,
    /// which is still the DEFAULT and the checkpoint's own `min_speakers` 2 /
    /// `max_speakers` 8 with its `min_cluster_size`.
    ///
    /// ⚠ **THE SIDECAR NOW READS IT** (2026-08-14). It did not until then — the
    /// owner's *"saya ingin speakernya itu auto / gak ditulis"* (2026-08-10) put
    /// this engine outside `ModelLoader.honoursSpeakerCount`, and this doc said in
    /// bold that the field was sent and ignored. The owner reversed that: the SPK
    /// picker must work on every engine.
    ///
    /// Nothing on the wire changed — the field was already being sent, for exactly
    /// the language-picker reason (2026-07-31: a value dies where the truth is,
    /// not at the Swift boundary). What changed is that the truth moved: see
    /// `diarizen-service.py` at the `pipeline(audio)` call for HOW it is applied
    /// (instance bounds, not a kwarg — this pipeline overrides `__call__`) and for
    /// the measurement that it obeys a smaller count and ignores a larger one.
    ///
    /// `exclusive` and `cluster_threshold` are pyannote's knobs and are not sent —
    /// this sidecar reads neither, so sending them would be inventing a knob.
    func diarizeFinal(audio: URL, numSpeakers: Int = 0, stream: Stream = .office) {
        send(["cmd": "final", "audio": audio.path, "num_speakers": numSpeakers],
             stream: stream)
    }

    private func send(_ job: [String: Any], stream: Stream = .office) {
        guard process.isRunning else {
            onError?("The DiariZen diarization sidecar is not running — "
                     + "restart the recording session.", stream)
            return
        }
        // Office is sent as an ABSENT key, not as "office": the sidecar reads a
        // missing stream as office, so a single-stream session's job lines carry
        // no dual-stream vocabulary at all.
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
        let segments: [LocalTurn]?
        /// The file this reply is about — echoed straight back, so the caller can
        /// hand it to the identity stage without a lookup.
        let audio: String?
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
                if errorReason == nil {
                    errorReason = "The DiariZen sidecar exited during startup."
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
                case "result":
                    self.onFinalResult?(message.audio ?? "", message.segments ?? [], stream)
                case "error":
                    self.onError?(message.text ?? "unknown diarization error", stream)
                // NO `chunk_result` CASE, deliberately. This engine cannot produce
                // one — see the type comment — and a case here would be the first
                // step towards someone adding the job that fills it.
                default: break
                }
            }
        }
    }
}
