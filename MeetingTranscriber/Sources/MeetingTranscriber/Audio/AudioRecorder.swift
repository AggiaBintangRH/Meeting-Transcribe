import AVFoundation
import Foundation

/// Records from the microphone + channel chosen in Settings → Microphone.
/// Publishes live RMS of the selected channel and saves a mono WAV.
/// Transcription models will hook into `onBuffer` later.
@MainActor
final class AudioRecorder: ObservableObject {

    enum State { case idle, preparing, recording, processing }

    @Published var state: State = .idle
    @Published var rms: Float = 0.0
    @Published var isSpeaking = false
    @Published var vadEnabled = true
    @Published var activeDeviceName: String?
    @Published var lastRecordingURL: URL?
    @Published var errorMessage: String?

    /// Raw ASR unit: Nemotron text appears instantly (unconfirmed), then is
    /// replaced in place by the accurate chunked text (confirmed). Diarization
    /// then splits each confirmed chunk into per-speaker rows for display.
    struct TranscriptSegment: Identifiable, Equatable {
        let id = UUID()
        var text: String
        let confirmed: Bool               // true = chunked ASR (accurate), false = realtime
        var window: ClosedRange<Double>? = nil  // recording-time span (confirmed only)
        // Set only by overlap repair: this segment's text belongs entirely to one
        // separated speaker, so display rows use it directly (skip re-attribution).
        // Two fields (not a tuple) to keep Equatable synthesis working.
        var pinnedSpeakerID: Int? = nil
        var pinnedSpeakerName: String? = nil
        // Debug/inspection: a raw MossFormer2 separated-track ASR result, shown as
        // its own "MossFormer2 Index N" row (name in pinnedSpeakerName). No
        // attribution, no merge — never replaces speaker text.
        var isSeparationDebug: Bool = false
    }

    /// One rendered row: a single speaker's turn with its time span and text.
    struct SpeakerUtterance: Identifiable, Equatable {
        let id: String                    // stable per (segment, turn index)
        var speaker: String?              // nil = not diarized yet
        var speakerID: Int?               // profile id (for rename)
        var start: Double?                // recording-time seconds (confirmed only)
        var end: Double?
        var text: String
        let confirmed: Bool
        var overlapped: Bool = false      // spoke over another speaker in this window
    }

    @Published var segments: [TranscriptSegment] = []
    @Published var displayRows: [SpeakerUtterance] = []
    @Published var partialTranscript = ""
    /// ATND position label for the live partial ("the beam active right now"),
    /// or nil when the feature is off / ATND is silent → shows SPEAKER UNKNOWN.
    @Published var partialSpeakerName: String?

    // Chunked ASR status (drives the small "refining…" indicator)
    @Published var chunkedBusy = false
    @Published var chunkedModelName = ""
    @Published var chunkedError: String?

    // Diarization (runs after the recording ends)
    @Published var diarizing = false
    /// Still written by the diarization passes below, but no view reads it
    /// since the ATND status chip was changed to show live beam data
    /// (StatusChipsView, 2026-07-17). Kept as the diarization result count.
    @Published var speakerCount: Int?
    @Published var diarizationError: String?

    // Overlap repair (MossFormer2, runs at stop after diarization + last chunk)
    @Published var overlapRepairing = false
    @Published var overlapRepairProgress: String?
    @Published var overlapRepairError: String?

    /// One line in the processing overlay — a leg of the post-stop work.
    /// Reuses `ModelLoader.ItemState` for the icons; the loader itself doesn't
    /// fit here (its `loadAll` runs one item at a time, these run concurrently).
    struct StopStep: Identifiable, Equatable {
        let id: String            // "chunk" | "diarize" | "repair"
        let name: String
        var state: ModelLoader.ItemState
    }

    @Published private(set) var stopSteps: [StopStep] = []

    // Gating for the "wait for last chunk AND diarization final" sequencing.
    private var stopped = false
    private var finalDiarDone = false
    private var lastChunkDone = false
    private var awaitingTailWindowStart: Double? = nil
    private var diarTailWatchdog: Task<Void, Never>?
    private var finalDiarWatchdog: Task<Void, Never>?
    private var stopWatchdog: Task<Void, Never>?
    private var repairTask: Task<Void, Never>?

    private var chunkElapsed: Double = 0      // seconds since last chunk flush
    private var chunkWatchdog: Task<Void, Never>?

    // Chunk time-window bookkeeping (for mapping speakers onto segments)
    private var recordingElapsed: Double = 0
    private var lastChunkBoundary: Double = 0
    private var pendingChunkWindows: [ClosedRange<Double>] = []
    // Elapsed time of the previous realtime (Nemotron) final. Each final covers
    // the audio since the last flush (the sidecar resets its buffer on flush), so
    // the not-yet-confirmed segment spans [lastRealtimeFinalElapsed, now] — a real
    // start–end range rather than a single point-in-time timestamp.
    private var lastRealtimeFinalElapsed: Double = 0

    // Live chunked diarization — runs on its OWN interval, independent of ASR
    private var chunkAudio: [Float] = []                       // 16k samples pending diarization
    private var chunkFileByWindow: [Double: URL] = [:]
    private var sessionSpeakerIDs = Set<Int>()
    private var liveTurns: [DiarizationService.Turn] = []      // absolute-time turns collected so far
    // Position-based diarization (ATND beam) — off unless atnd.position.enabled.
    // Recorder-owned, one per session; nil means the feature is off, so
    // positionGapFill returns [] and the display path is pure pyannote.
    @Published private(set) var positionDiarizer: PositionDiarizer?
    private var diarElapsed: Double = 0                        // seconds since the last diar chunk
    private var lastDiarBoundary: Double = 0                   // recording-time where this diar chunk began

    /// Publishes per-model progress for the loading overlay.
    let modelLoader = ModelLoader()

    /// Future hook: realtime ASR (Nemotron) consumes mono buffers of the selected channel.
    var onBuffer: ((AVAudioPCMBuffer) -> Void)?

    private var engine: AVAudioEngine?
    private var file: AVAudioFile?
    private var vad: VoiceActivityDetector?

    /// Recordings are stored under the data dir (project folder in dev,
    /// Application Support when bundled).
    private var recordingsDir: URL {
        let dir = PythonRuntime.dataDir.appendingPathComponent("recordings")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func toggle() {
        switch state {
        case .idle:       start()
        case .recording:  stop()
        case .preparing:  break // ignore taps while loading
        case .processing: break // ignore taps while the stop work finishes
        }
    }

    // MARK: - Start

    private func start() {
        errorMessage = nil
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            Task { @MainActor in
                guard let self else { return }
                guard granted else {
                    self.errorMessage = "Microphone access denied. Enable it in System Settings → Privacy."
                    return
                }
                await self.prepareAndCapture()
            }
        }
    }

    /// Show the loading overlay while models load, then start capturing.
    private func prepareAndCapture() async {
        state = .preparing
        let ok = await modelLoader.loadAll()
        guard ok else {
            state = .idle
            errorMessage = "Model loading failed — see the list above."
            return
        }
        beginCapture()
        if state != .recording { state = .idle } // beginCapture failed → reset
    }

    private func beginCapture() {
        // Fresh engine each session — avoids stale device/format state
        let engine = AVAudioEngine()
        self.engine = engine
        let input = engine.inputNode

        // 1. Route to the device picked in Settings (nil → system default)
        let mic = MicrophoneSettings.current()
        if let device = mic.device {
            guard AudioDeviceManager.setInputDevice(device.id, on: engine) else {
                errorMessage = "Could not open \(device.name). Try reconnecting it."
                return
            }
            activeDeviceName = device.name
        } else {
            activeDeviceName = "System default"
        }

        // 2. Clamp channel to what the device actually delivers
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            errorMessage = "Selected microphone has no usable input format."
            return
        }
        let channel = min(mic.channel, Int(format.channelCount) - 1)

        // 3. Mono output file (selected channel only)
        guard let monoFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                             sampleRate: format.sampleRate,
                                             channels: 1, interleaved: false) else {
            errorMessage = "Could not create mono recording format."
            return
        }
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let url = recordingsDir.appendingPathComponent("meeting-\(stamp).wav")
        do {
            file = try AVAudioFile(forWriting: url, settings: monoFormat.settings)
        } catch {
            errorMessage = "Could not create recording file: \(error.localizedDescription)"
            return
        }

        // 4. VAD — fresh instance per session; uses Silero sidecar if it loaded
        let vadOn = UserDefaults.standard.object(forKey: "vad.enabled") as? Bool ?? true
        vadEnabled = vadOn
        let vad = vadOn ? VoiceActivityDetector(silero: modelLoader.sileroVAD) : nil
        self.vad = vad
        let sampleRate = format.sampleRate
        let resampler = AudioResampler(inputFormat: monoFormat) // → 16 kHz for Silero/ASR

        // 5. Realtime ASR — stream audio in, receive partial/final transcripts
        let realtimeOn = UserDefaults.standard.object(forKey: "realtime.enabled") as? Bool ?? true
        let asr = realtimeOn ? modelLoader.nemotronASR : nil
        asr?.onTranscript = { [weak self] text, isFinal in
            Task { @MainActor in
                guard let self else { return }
                if isFinal {
                    // Once stopped, the chunked pass is authoritative for all
                    // remaining audio. Skip realtime finals so a trailing one
                    // (fired by the flush in stop()) can't land after the last
                    // chunk's cleanup and survive as an orphan "SPEAKER UNKNOWN"
                    // row next to the confirmed, speaker-split transcript.
                    guard !self.stopped else { self.partialTranscript = ""; self.partialSpeakerName = nil; return }
                    let end = self.recordingElapsed
                    let start = min(self.lastRealtimeFinalElapsed, end)
                    // Advance the marker on every final (incl. empty ones), since
                    // the sidecar's buffer resets on each flush.
                    self.lastRealtimeFinalElapsed = end
                    if !text.isEmpty {
                        self.segments.append(TranscriptSegment(text: text, confirmed: false, window: start...end))
                        self.rebuildDisplayRows()
                    }
                    self.partialTranscript = ""
                    self.partialSpeakerName = nil
                } else {
                    self.partialTranscript = text
                    // Trailing 1s window ≈ the beam active right now (after smoothing).
                    // Short so the live-partial label flips quickly on a talker switch.
                    self.partialSpeakerName = self.positionDiarizer?.label(
                        for: max(0, self.recordingElapsed - 1.0)...self.recordingElapsed,
                        minSamples: 3)?.name
                }
            }
        }

        // 5b. Chunked ASR — rolling accurate pass every N seconds.
        // Its result REPLACES the unconfirmed Nemotron segments in place.
        let chunked = modelLoader.chunkedASR
        chunkedModelName = chunked?.config.modelName ?? ""
        segments = []
        displayRows = []
        chunkedBusy = false
        chunkElapsed = 0
        recordingElapsed = 0
        lastChunkBoundary = 0
        pendingChunkWindows = []
        lastRealtimeFinalElapsed = 0
        diarizing = false
        speakerCount = nil
        diarizationError = nil
        // Fresh overlap-repair state for this session.
        repairTask?.cancel()
        repairTask = nil
        overlapRepairing = false
        overlapRepairProgress = nil
        overlapRepairError = nil
        stopped = false
        finalDiarDone = false
        lastChunkDone = false
        awaitingTailWindowStart = nil
        diarTailWatchdog?.cancel()
        diarTailWatchdog = nil
        finalDiarWatchdog?.cancel()
        finalDiarWatchdog = nil
        stopWatchdog?.cancel()
        stopWatchdog = nil
        stopSteps = []
        // Fresh per-session speaker state, then wire the live/final callbacks.
        chunkFileByWindow = [:]
        sessionSpeakerIDs = []
        liveTurns = []
        chunkAudio = []
        diarElapsed = 0
        lastDiarBoundary = 0
        configureDiarization()
        configurePositionDiarization()
        // Real-time speaker split: when the beam settles on a different talker,
        // end the old speaker's realtime segment now and relabel existing rows.
        // Detection lags the real switch by ~0.7s (0.4s smoother warm-up + 3
        // confirmation samples at 10 Hz), so the new speaker's first fraction of a
        // second lands in the old segment — same class of approximation as the
        // existing time→char sentence split. No-op when the feature is off
        // (positionDiarizer == nil → this optional-chain never installs anything).
        // The realtime engine is modelLoader.nemotronASR (there is no `self.asr`
        // property — `asr` is a local in beginCapture); flush() is a safe no-op
        // when idle, and empty-text finals are already dropped in onTranscript.
        positionDiarizer?.onClusterChange = { [weak self] in
            guard let self, !self.stopped else { return }
            self.modelLoader.nemotronASR?.flush()  // end the old speaker's realtime segment now
            self.rebuildDisplayRows()              // relabel existing rows' fills instantly
        }
        // Optionally start each recording with a clean speaker store.
        if UserDefaults.standard.object(forKey: "diarization.resetOnStart") as? Bool ?? true {
            modelLoader.diarization?.resetProfiles()
        }
        let chunkInterval = Double(
            UserDefaults.standard.object(forKey: "chunked.intervalSec") as? Int ?? 30
        )
        // Diarization runs on its own cadence, separate from chunked ASR.
        let diarInterval = Double(
            UserDefaults.standard.object(forKey: "diarization.intervalSec") as? Int ?? 30
        )
        chunkedError = nil
        chunked?.onChunkTranscript = { [weak self] text in
            Task { @MainActor in
                guard let self else { return }
                self.chunkWatchdog?.cancel()
                self.chunkedBusy = false
                self.chunkedError = nil
                // Replace realtime text for this chunk window with the
                // accurate version (both services flushed at the same
                // audio position, so unconfirmed segments = this chunk).
                let window = self.pendingChunkWindows.isEmpty
                    ? nil : self.pendingChunkWindows.removeFirst()
                self.segments.removeAll { !$0.confirmed }
                if !text.isEmpty {
                    self.segments.append(TranscriptSegment(text: text, confirmed: true, window: window))
                }
                // Rebuild rows: splits this chunk by any diarization turns already in.
                self.rebuildDisplayRows()
                self.checkLastChunkDone()
            }
        }
        chunked?.onChunkError = { [weak self] message in
            Task { @MainActor in
                guard let self else { return }
                self.chunkWatchdog?.cancel()
                self.chunkedBusy = false
                self.chunkedError = message
                // Drain the queued window even on failure, or a stuck entry
                // blocks checkLastChunkDone()/misaligns the next chunk's window.
                if !self.pendingChunkWindows.isEmpty {
                    self.pendingChunkWindows.removeFirst()
                }
                self.checkLastChunkDone()
            }
        }

        // 6. Tap: extract selected channel → write file → RMS + VAD → ASR
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self,
                  let mono = AudioBufferProcessor.extractChannel(buffer, channel: channel)
            else { return }

            try? self.file?.write(from: mono)
            self.onBuffer?(mono)

            let level = AudioBufferProcessor.rms(mono)
            let zcr = AudioBufferProcessor.zeroCrossingRate(mono)
            let duration = Double(mono.frameLength) / sampleRate
            let samples16k = resampler?.resample(mono) ?? []
            let speaking = vad?.process(samples16k: samples16k,
                                        rms: level, zcr: zcr,
                                        bufferDuration: duration) ?? false
            asr?.feed(samples16k)
            chunked?.feed(samples16k)

            Task { @MainActor in
                self.rms = level
                self.recordingElapsed += duration
                // VAD speech → silence edge: utterance ended, finalize it
                if self.isSpeaking && !speaking {
                    asr?.flush()
                }
                self.isSpeaking = speaking

                // Chunk boundary: interval elapsed AND we're in silence
                // (avoids cutting mid-word). Hard cap at 1.5x interval if
                // someone talks non-stop.
                self.chunkElapsed += duration
                let boundary = (self.chunkElapsed >= chunkInterval && !speaking)
                            || self.chunkElapsed >= chunkInterval * 1.5
                // Accumulate audio for live diarization (its own cadence below)
                self.chunkAudio.append(contentsOf: samples16k)

                if boundary, chunked != nil {
                    self.chunkElapsed = 0
                    // Flush Nemotron too — keeps both services aligned at the
                    // same audio position so replacement is exact.
                    asr?.flush()
                    let windowStart = self.lastChunkBoundary
                    self.pendingChunkWindows.append(windowStart...self.recordingElapsed)
                    self.lastChunkBoundary = self.recordingElapsed
                    self.startChunkFlush(chunked)
                }

                // Diarization boundary — independent of the ASR chunk interval.
                self.diarElapsed += duration
                let diarBoundary = (self.diarElapsed >= diarInterval && !speaking)
                                || self.diarElapsed >= diarInterval * 1.5
                if diarBoundary {
                    self.diarElapsed = 0
                    let diarWindowStart = self.lastDiarBoundary
                    self.lastDiarBoundary = self.recordingElapsed
                    self.diarizeLiveChunk(windowStart: diarWindowStart)
                }
            }
        }

        // 7. Go
        do {
            engine.prepare()
            try engine.start()
            lastRecordingURL = url
            state = .recording
        } catch {
            errorMessage = "Audio engine failed: \(error.localizedDescription)"
            input.removeTap(onBus: 0)
            file = nil
        }
    }

    // MARK: - Stop

    /// Flush a chunk with a watchdog: if no result within 3 minutes,
    /// clear the spinner and surface a timeout (details in logs/chunked-asr.log).
    private func startChunkFlush(_ service: ChunkedASRService?) {
        guard let service else { return }
        chunkedBusy = true
        service.flush()
        chunkWatchdog?.cancel()
        chunkWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(180))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.chunkedBusy else { return }
                self.chunkedBusy = false
                self.chunkedError = "Chunk transcription timed out — see logs/chunked-asr.log"
                // Drain the queued window on timeout too — same reasoning as onChunkError.
                if !self.pendingChunkWindows.isEmpty {
                    self.pendingChunkWindows.removeFirst()
                }
                self.checkLastChunkDone()
            }
        }
    }

    private func stop() {
        stopped = true
        // Stop ingesting beam notices, but KEEP the collected data — display-time
        // gap-fill (positionGapFill → label(for:)) still queries it afterward.
        positionDiarizer?.stop()
        modelLoader.nemotronASR?.flush() // finalize any trailing speech
        if modelLoader.chunkedASR != nil {
            pendingChunkWindows.append(lastChunkBoundary...max(recordingElapsed, lastChunkBoundary + 0.01))
            lastChunkBoundary = recordingElapsed
            startChunkFlush(modelLoader.chunkedASR) // transcribe the last partial chunk
        } else {
            // Nothing would ever drain a queued window without a chunked model,
            // so don't queue one — the gate has to complete here instead.
            lastChunkDone = true
        }
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        file = nil
        vad = nil
        rms = 0
        isSpeaking = false

        // Who spoke when — either append a tail (continue from live labels) or
        // re-diarize the whole recording (best global clustering).
        let finalOn = UserDefaults.standard.object(forKey: "diarization.finalPass") as? Bool ?? true
        let continueOnStop = UserDefaults.standard.object(forKey: "diarization.continueOnStop") as? Bool ?? true
        let willRunStopPass = finalOn && modelLoader.diarization != nil

        // Everything below lands asynchronously; block the controls until it does.
        buildStopSteps(willRunStopPass: willRunStopPass)
        state = .processing
        startStopWatchdog()

        if willRunStopPass {
            if continueOnStop {
                diarizeTailChunk()
            } else if let recording = lastRecordingURL {
                startDiarization(recording)
            } else {
                finalDiarDone = true
                setStopStep("diarize", .done)
                maybeStartOverlapRepair()
            }
        } else {
            // No stop-time pass — overlap repair can proceed once the last chunk lands.
            finalDiarDone = true
            maybeStartOverlapRepair()
        }
        checkStopProcessingDone()
    }

    // MARK: - Stop processing gate (the blocking overlay)

    /// The legs the overlay lists for this stop, in the order they finish.
    /// `repair` only appears when the feature is on AND its engine loaded —
    /// otherwise there is nothing to wait for.
    private func buildStopSteps(willRunStopPass: Bool) {
        var steps = [StopStep(id: "chunk", name: "Transcribing final audio",
                              state: lastChunkDone ? .done : .loading)]
        if willRunStopPass {
            steps.append(StopStep(id: "diarize", name: "Identifying speakers", state: .loading))
        }
        if overlapRepairWillRun {
            steps.append(StopStep(id: "repair", name: "Repairing overlapping speech",
                                  state: .pending))
        }
        stopSteps = steps
    }

    private func setStopStep(_ id: String, _ state: ModelLoader.ItemState) {
        guard let i = stopSteps.firstIndex(where: { $0.id == id }) else { return }
        stopSteps[i].state = state
    }

    /// Whether a repair leg is worth listing — mirrors `maybeStartOverlapRepair`'s
    /// feature + engine checks, without the gates that are still pending at stop.
    private var overlapRepairWillRun: Bool {
        let d = UserDefaults.standard
        guard d.object(forKey: "overlap.repair.enabled") as? Bool ?? false else { return false }
        let engine = d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id
        return engine == ModelCatalog.overlapDicow.id
            ? modelLoader.dicowRepair != nil
            : modelLoader.overlapRepair != nil
    }

    /// All post-stop work landed → drop the overlay and re-enable Start.
    /// Idempotent; every leg's completion path calls it.
    private func checkStopProcessingDone() {
        guard state == .processing, lastChunkDone, finalDiarDone,
              !overlapRepairing, repairTask == nil else { return }
        stopWatchdog?.cancel()
        stopWatchdog = nil
        state = .idle
    }

    /// Last resort: never hold the controls hostage. The background work keeps
    /// running and still updates the transcript if it lands.
    private func startStopWatchdog() {
        stopWatchdog?.cancel()
        stopWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(600))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.state == .processing else { return }
                for i in self.stopSteps.indices
                where self.stopSteps[i].state == .pending || self.stopSteps[i].state == .loading {
                    self.stopSteps[i].state = .failed("timed out")
                }
                self.stopWatchdog = nil
                self.state = .idle
            }
        }
    }

    /// Escape hatch offered by the overlay after a while: stop blocking, cancel
    /// nothing. Degrades to the pre-overlay behaviour — work continues silently.
    func continueInBackground() {
        guard state == .processing else { return }
        stopWatchdog?.cancel()
        stopWatchdog = nil
        state = .idle
    }

    // MARK: - Diarization

    /// Wire the diarization service callbacks for this session.
    private func configureDiarization() {
        guard let service = modelLoader.diarization else { return }

        service.onChunkResult = { [weak self] windowStart, turns in
            Task { @MainActor in
                guard let self else { return }
                // Clean up the temp chunk file
                if let file = self.chunkFileByWindow.removeValue(forKey: windowStart) {
                    try? FileManager.default.removeItem(at: file)
                }
                // Turn times are chunk-local — offset to absolute recording time,
                // add to the running set, then (re)label overlapping segments.
                let absolute = turns.map {
                    DiarizationService.Turn(start: $0.start + windowStart,
                                            end: $0.end + windowStart,
                                            id: $0.id, name: $0.name)
                }
                // Raw pyannote turns — pyannote is authoritative. Position labels
                // are folded in only at display time (derivedRows), never here.
                self.liveTurns.append(contentsOf: absolute)
                for turn in absolute { self.sessionSpeakerIDs.insert(turn.id) }
                self.speakerCount = self.sessionSpeakerIDs.count
                self.rebuildDisplayRows()
                // Tail-only stop mode: this chunk result IS the tail — complete the gate.
                if let expected = self.awaitingTailWindowStart, abs(windowStart - expected) < 0.001 {
                    self.completeStopDiarization()
                }
            }
        }

        service.onFinalResult = { [weak self] turns in
            Task { @MainActor in
                guard let self else { return }
                self.applyFinalSpeakers(turns)
                self.diarizing = false
                self.finalDiarDone = true
                self.finalDiarWatchdog?.cancel()
                self.finalDiarWatchdog = nil
                self.setStopStep("diarize", .done)
                self.maybeStartOverlapRepair()
                self.checkStopProcessingDone()
            }
        }

        service.onError = { [weak self] message in
            Task { @MainActor in
                guard let self else { return }
                self.diarizing = false
                self.diarizationError = message
                self.finalDiarDone = true
                self.awaitingTailWindowStart = nil
                self.diarTailWatchdog?.cancel()
                self.diarTailWatchdog = nil
                self.finalDiarWatchdog?.cancel()
                self.finalDiarWatchdog = nil
                self.setStopStep("diarize", .failed(message))
                self.maybeStartOverlapRepair()
                self.checkStopProcessingDone()
            }
        }
    }

    // MARK: - Position diarization (ATND beam) gap-fill
    //
    // Policy (owner, 2026-07-20): pyannote is AUTHORITATIVE. Wherever pyannote has
    // a turn, its own label wins. ATND position only fills the DISPLAY-time gaps
    // pyannote has not (yet) covered — freshly-committed text, the first ~1-2s
    // pyannote never turns, and the live partial. A pyannote chunk landing later
    // OVERRIDES the fill automatically (it shrinks the gap on the next rebuild).
    // Silence gaps where ATND also heard nothing STAY unknown — never force-filled.

    /// Create + start the position diarizer for this session, but ONLY when the
    /// feature is explicitly enabled AND the beam service is actually listening.
    /// Otherwise leave it nil — `positionGapFill` then returns [] and the display
    /// path is pure pyannote, byte-identical to before this feature existed.
    private func configurePositionDiarization() {
        positionDiarizer = nil
        let d = UserDefaults.standard
        guard d.bool(forKey: "atnd.position.enabled"),
              ATNDBeamService.shared.state == .listening else { return }

        let tauDeg = d.object(forKey: "atnd.position.tauDeg") as? Double ?? 15
        let smoothingMs = d.object(forKey: "atnd.position.smoothingMs") as? Double ?? 400
        let mode: PositionDiarizer.Mode =
            (d.string(forKey: "atnd.position.mode") == "enrollment") ? .enrollment : .firstCome

        let diarizer = PositionDiarizer()
        diarizer.start(tauDeg: tauDeg,
                       smoothingSec: smoothingMs / 1000,
                       mode: mode,
                       now: { [weak self] in self?.recordingElapsed ?? 0 })
        positionDiarizer = diarizer
    }

    /// Position-labeled ranges covering the sub-ranges of `window` that pyannote
    /// has NOT (yet) covered. Empty when the feature is off or ATND was silent.
    private func positionGapFill(window: ClosedRange<Double>,
                                 covered: [(start: Double, end: Double, id: Int, name: String)])
        -> [(start: Double, end: Double, id: Int, name: String)] {
        guard let pos = positionDiarizer else { return [] }
        let minGapSec = 0.75   // below this is pyannote boundary slop → filling flickers
        // `covered` is already sorted by start (from speakerRanges). Walk it and
        // emit the complement gaps of `window`.
        var fills: [(start: Double, end: Double, id: Int, name: String)] = []
        var cursor = window.lowerBound
        // Build the ordered list of gap ranges (before/between/after covered ranges).
        var gaps: [(Double, Double)] = []
        for r in covered {
            if r.start > cursor { gaps.append((cursor, r.start)) }
            cursor = max(cursor, r.end)
        }
        if window.upperBound > cursor { gaps.append((cursor, window.upperBound)) }

        for (a, b) in gaps {
            let dur = b - a
            if dur < minGapSec {
                positionLog("SKIP gap<0.75s [\(fmt3(a))..\(fmt3(b))]")
                continue
            }
            // One fill PER TURN — a beam change mid-gap splits into multiple rows.
            // The per-turn minSamples/minDuration inside `turns(...)` is the density
            // gate: a gap where ATND heard nothing (or only flicker) yields no turns
            // and STAYS unknown. (Replaces the old whole-gap `need` sample gate.)
            var turns = pos.labeledTurns(in: a...b)
            if turns.isEmpty {
                let count = pos.sampleCount(in: a...b)
                positionLog("SKIP gap=[\(fmt3(a))..\(fmt3(b))] samples=\(count) no-turns")
                continue
            }
            // Boundary snapping — the 0.4s smoother swallows ~0.4s warm-up per
            // utterance, so turns start/end a beat inside the real gap.
            // Extend the first/last turn out to the gap edges when the reach is
            // small; close small inter-turn holes at their midpoint. A hole larger
            // than 1.0s is real ATND silence → leave it (stays UNKNOWN).
            if turns[0].start - a <= 1.0 { turns[0].start = a }
            let last = turns.count - 1
            if b - turns[last].end <= 1.0 { turns[last].end = b }
            for i in 0..<(turns.count - 1) {
                let hole = turns[i + 1].start - turns[i].end
                if hole > 0, hole <= 1.0 {
                    let mid = (turns[i].end + turns[i + 1].start) / 2
                    turns[i].end = mid
                    turns[i + 1].start = mid
                }
            }
            for t in turns {
                positionLog("FILL gap=[\(fmt3(a))..\(fmt3(b))] -> \(t.id):\(t.name) [\(fmt3(t.start))..\(fmt3(t.end))]")
                fills.append((t.start, t.end, t.id, t.name))
            }
        }
        return fills
    }

    /// Live: write the current chunk's audio to a temp WAV and diarize it.
    private func diarizeLiveChunk(windowStart: Double) {
        let liveOn = UserDefaults.standard.object(forKey: "diarization.live") as? Bool ?? true
        guard liveOn, modelLoader.diarization != nil else {
            // When live labels are off but "continue from live labels (tail only)"
            // is on, DON'T clear the accumulated audio: the whole recording then
            // becomes the single tail diarized at stop (matches the UI caption).
            let continueOnStop = UserDefaults.standard.object(forKey: "diarization.continueOnStop") as? Bool ?? true
            if !(continueOnStop && !liveOn) { chunkAudio = [] }
            return
        }
        let samples = chunkAudio
        chunkAudio = []
        guard samples.count > 16_000 else { return } // skip chunks under 1s
        dispatchDiarChunk(samples: samples, windowStart: windowStart)
    }

    /// Shared: write a chunk of 16 kHz samples to a temp WAV off-thread and hand
    /// it to the diarization sidecar. Live calls pass no failure handler (silent,
    /// as before); the stop-time tail passes one so the gate can still complete.
    private func dispatchDiarChunk(samples: [Float], windowStart: Double,
                                   onDispatchFailure: (() -> Void)? = nil) {
        guard let service = modelLoader.diarization else { onDispatchFailure?(); return }
        Task.detached(priority: .utility) { [weak self] in
            guard let url = Self.writeTempWAV(samples: samples) else {
                await MainActor.run { onDispatchFailure?() }
                return
            }
            await MainActor.run { [weak self] in
                self?.chunkFileByWindow[windowStart] = url
            }
            let detectOverlap = UserDefaults.standard.object(forKey: "diarization.detectOverlap") as? Bool ?? true
            service.diarizeChunk(audio: url, windowStart: windowStart, exclusive: !detectOverlap)
        }
    }

    /// At stop (tail-only mode): diarize just the audio accumulated since the last
    /// live chunk and append it, keeping live speaker numbering stable. Routes the
    /// completion gate through `completeStopDiarization()`.
    private func diarizeTailChunk() {
        guard modelLoader.diarization != nil else { completeStopDiarization(); return }
        let samples = chunkAudio
        chunkAudio = []
        // With live labels on, the pending audio began at the last live-chunk
        // boundary. With live off (+ continueOnStop), nothing was ever cleared so
        // chunkAudio is the whole recording — it begins at 0, and lastDiarBoundary
        // (which still advances in the tap) is stale, so don't use it.
        let liveOn = UserDefaults.standard.object(forKey: "diarization.live") as? Bool ?? true
        let windowStart = liveOn ? lastDiarBoundary : 0
        guard samples.count > 16_000 else { completeStopDiarization(); return } // <1s tail
        diarizing = true
        diarizationError = nil
        awaitingTailWindowStart = windowStart
        diarTailWatchdog?.cancel()
        diarTailWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(120))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.awaitingTailWindowStart != nil else { return }
                self.diarizationError = self.diarizationError ?? "Tail diarization timed out"
                self.completeStopDiarization()
            }
        }
        dispatchDiarChunk(samples: samples, windowStart: windowStart,
                          onDispatchFailure: { [weak self] in
                              self?.diarizationError = "Could not write tail audio for diarization"
                              self?.completeStopDiarization()
                          })
    }

    /// Finish the stop-time (tail) diarization gate exactly once, then let overlap
    /// repair proceed. Safe to call from any of the tail exit paths.
    private func completeStopDiarization() {
        diarTailWatchdog?.cancel()
        diarTailWatchdog = nil
        awaitingTailWindowStart = nil
        diarizing = false
        finalDiarDone = true
        setStopStep("diarize", diarizationError.map { .failed($0) } ?? .done)
        maybeStartOverlapRepair()
        checkStopProcessingDone()
    }

    /// At stop: batch refinement over the full recording (best accuracy).
    private func startDiarization(_ recording: URL) {
        let finalOn = UserDefaults.standard.object(forKey: "diarization.finalPass") as? Bool ?? true
        guard finalOn, let service = modelLoader.diarization else { return }
        diarizing = true
        diarizationError = nil
        let numSpeakers = UserDefaults.standard.integer(forKey: "diarization.numSpeakers")
        let detectOverlap = UserDefaults.standard.object(forKey: "diarization.detectOverlap") as? Bool ?? true
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers, exclusive: !detectOverlap)
        // A final pass over a long meeting legitimately takes a while, so scale
        // the limit with the recording — never below 3 minutes.
        let limit = max(180, recordingElapsed)
        finalDiarWatchdog?.cancel()
        finalDiarWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(limit))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.finalDiarDone else { return }
                self.diarizing = false
                self.diarizationError = "Final diarization timed out"
                self.finalDiarDone = true
                self.setStopStep("diarize", .failed("Final diarization timed out"))
                self.maybeStartOverlapRepair()
                self.checkStopProcessingDone()
            }
        }
    }

    /// Final pass: the globally-clustered pyannote turns replace the running live
    /// set. Raw pyannote (authoritative); position labels are folded in only at
    /// display time (derivedRows), so liveTurns stays pure pyannote.
    private func applyFinalSpeakers(_ turns: [DiarizationService.Turn]) {
        speakerCount = Set(turns.map(\.id)).count
        liveTurns = turns
        rebuildDisplayRows()
    }

    // MARK: - Display rows (speaker · time · text)

    /// Rebuild the rendered transcript from the raw ASR segments and the
    /// diarization turns collected so far. Each confirmed chunk is split into
    /// one row per speaker turn; undiarized/realtime text stays a single row.
    private func rebuildDisplayRows() {
        var rows: [SpeakerUtterance] = []
        let regions = overlapRegions()   // genuine simultaneous-speech windows
        for seg in segments {
            rows.append(contentsOf: derivedRows(for: seg, regions: regions))
        }
        displayRows = rows
    }

    /// The display rows one raw segment expands into — the single source of truth
    /// shared by `rebuildDisplayRows` and `applyRepair` (so repair sees exactly the
    /// rows the transcript shows). `regions` are the genuine-overlap windows.
    private func derivedRows(for seg: TranscriptSegment,
                             regions: [(start: Double, end: Double)]) -> [SpeakerUtterance] {
        // Debug: raw MossFormer2 separated-track ASR, shown verbatim as its own
        // "MossFormer2 Index N" row (no rename, no attribution, never replaces).
        if seg.isSeparationDebug {
            let w = seg.window
            return [SpeakerUtterance(id: seg.id.uuidString,
                                     speaker: seg.pinnedSpeakerName, speakerID: nil,
                                     start: w?.lowerBound, end: w?.upperBound,
                                     text: seg.text, confirmed: true, overlapped: true)]
        }
        // Overlap-repair (pinned) segments belong entirely to one separated speaker —
        // a single row. Tag it orange only if it genuinely sits over an overlap
        // region, so PRESERVED bystander rows aren't all flagged.
        if let pid = seg.pinnedSpeakerID {
            let w = seg.window
            let overlapped = regions.contains {
                min($0.end, seg.window?.upperBound ?? 0) - max($0.start, seg.window?.lowerBound ?? 0) > 0
            }
            return [SpeakerUtterance(id: seg.id.uuidString,
                                     speaker: seg.pinnedSpeakerName, speakerID: pid,
                                     start: w?.lowerBound, end: w?.upperBound,
                                     text: seg.text, confirmed: true, overlapped: overlapped)]
        }
        guard let window = seg.window else {
            return [SpeakerUtterance(id: seg.id.uuidString, speaker: nil,
                                     speakerID: nil, start: nil, end: nil,
                                     text: seg.text, confirmed: seg.confirmed)]
        }
        let ranges = speakerRanges(in: window)
        // Fill only the time regions pyannote has not covered with ATND position
        // labels (pyannote wins where it has a turn). Off/silent → fills is [].
        let fills = positionGapFill(window: window, covered: ranges)
        let filled = (ranges + fills).sorted { $0.start < $1.start }

        // Unconfirmed (realtime) segments stay a single provisional row — the text
        // isn't final, so it isn't sentence-split — but it must still carry the
        // speaker the live view already showed. Label it with whoever dominates
        // the window (pyannote if it has a turn there, else the ATND position
        // fill), so it doesn't drop back to UNKNOWN the moment it commits.
        if !seg.confirmed {
            // A beam change split this window into multiple fills → split the live
            // (unconfirmed) text into per-speaker rows too, so the switch shows in
            // real time. A single fill (or none) stays one provisional row.
            if filled.count > 1 {
                return assignSentences(seg.text, window: window, ranges: filled,
                                       segID: seg.id.uuidString, regions: regions,
                                       confirmed: false)
            }
            func overlap(_ r: (start: Double, end: Double, id: Int, name: String)) -> Double {
                max(0, min(r.end, window.upperBound) - max(r.start, window.lowerBound))
            }
            let best = filled.max { overlap($0) < overlap($1) }
            return [SpeakerUtterance(id: seg.id.uuidString, speaker: best?.name,
                                     speakerID: best?.id, start: window.lowerBound,
                                     end: window.upperBound, text: seg.text,
                                     confirmed: false)]
        }

        if filled.isEmpty {
            return [SpeakerUtterance(id: seg.id.uuidString, speaker: nil,
                                     speakerID: nil, start: window.lowerBound,
                                     end: window.upperBound, text: seg.text,
                                     confirmed: true)]
        }
        return assignSentences(seg.text, window: window, ranges: filled,
                               segID: seg.id.uuidString, regions: regions)
    }

    /// Assign whole sentences to speakers. Each sentence is placed in time by its
    /// character position in the chunk, then handed to the speaker turn it most
    /// overlaps — so text is never cut mid-sentence onto the wrong speaker.
    /// Consecutive sentences by the same speaker merge into one row.
    private func assignSentences(_ text: String,
                                 window: ClosedRange<Double>,
                                 ranges: [(start: Double, end: Double, id: Int, name: String)],
                                 segID: String,
                                 regions: [(start: Double, end: Double)],
                                 confirmed: Bool = true) -> [SpeakerUtterance] {
        let sentences = splitSentences(text)
        guard !sentences.isEmpty else { return [] }
        let totalChars = max(1, sentences.reduce(0) { $0 + $1.count })
        let span = max(0, window.upperBound - window.lowerBound)

        struct Piece { var id: Int; var name: String; var start: Double; var end: Double; var text: String }
        var pieces: [Piece] = []
        var charsSoFar = 0
        for sentence in sentences {
            let sStart = window.lowerBound + span * Double(charsSoFar) / Double(totalChars)
            charsSoFar += sentence.count
            let sEnd = window.lowerBound + span * Double(charsSoFar) / Double(totalChars)

            var chosen = ranges[0]
            var bestOverlap = -1.0
            for r in ranges {
                let ov = max(0, min(r.end, sEnd) - max(r.start, sStart))
                if ov > bestOverlap { bestOverlap = ov; chosen = r }
            }
            if bestOverlap <= 0 {
                let mid = (sStart + sEnd) / 2
                chosen = ranges.min { abs(($0.start + $0.end) / 2 - mid) < abs(($1.start + $1.end) / 2 - mid) } ?? ranges[0]
            }
            pieces.append(Piece(id: chosen.id, name: chosen.name, start: sStart, end: sEnd, text: sentence))
        }

        var merged: [Piece] = []
        for p in pieces {
            if var last = merged.last, last.id == p.id {
                last.end = p.end
                last.text += " " + p.text
                merged[merged.count - 1] = last
            } else {
                merged.append(p)
            }
        }

        return merged.enumerated().map { i, p in
            // Flag only if this row's time genuinely sits over a simultaneous-
            // speech region (two different speakers active at once).
            let overlapped = regions.contains { max($0.start, p.start) < min($0.end, p.end) }
            return SpeakerUtterance(id: "\(segID)-\(i)", speaker: p.name, speakerID: p.id,
                                    start: p.start, end: p.end, text: p.text,
                                    confirmed: confirmed, overlapped: overlapped)
        }
    }

    /// Windows where two DIFFERENT speakers are active at the same time
    /// (genuine overlap), each at least 0.4s long. Empty in exclusive mode.
    private func overlapRegions() -> [(start: Double, end: Double)] {
        let turns = liveTurns
        guard turns.count > 1 else { return [] }
        var regions: [(start: Double, end: Double)] = []
        for i in 0..<turns.count {
            for j in (i + 1)..<turns.count where turns[i].id != turns[j].id {
                let s = max(turns[i].start, turns[j].start)
                let e = min(turns[i].end, turns[j].end)
                if e - s >= 0.4 { regions.append((s, e)) }
            }
        }
        return regions
    }

    /// Split text into sentences on . ? ! and line breaks, keeping punctuation.
    /// Fragments with no letters or digits (e.g. ". .") are dropped.
    private func splitSentences(_ text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        func hasContent(_ s: String) -> Bool { s.contains { $0.isLetter || $0.isNumber } }

        var result: [String] = []
        var current = ""
        for ch in trimmed {
            current.append(ch)
            if ch == "." || ch == "?" || ch == "!" || ch == "\n" {
                let s = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if hasContent(s) { result.append(s) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if hasContent(tail) { result.append(tail) }
        if result.isEmpty { return hasContent(trimmed) ? [trimmed] : [] }
        return result
    }

    /// Speaker turns overlapping a window, clipped to it and merged when the
    /// same speaker continues (small gaps bridged), sorted by start time.
    private func speakerRanges(in window: ClosedRange<Double>)
        -> [(start: Double, end: Double, id: Int, name: String)] {
        let clipped = liveTurns.compactMap { t -> (start: Double, end: Double, id: Int, name: String)? in
            let s = max(t.start, window.lowerBound)
            let e = min(t.end, window.upperBound)
            return e > s ? (s, e, t.id, t.name) : nil
        }.sorted { $0.start < $1.start }

        var merged: [(start: Double, end: Double, id: Int, name: String)] = []
        for c in clipped {
            if var last = merged.last, last.id == c.id, c.start - last.end < 1.0 {
                last.end = max(last.end, c.end)
                merged[merged.count - 1] = last
            } else {
                merged.append(c)
            }
        }
        return merged
    }

    /// Rename a speaker profile and refresh all its rows in the transcript.
    func renameSpeaker(id: Int, to name: String) {
        // Position ids are disjoint (>= positionIDBase) and must NEVER reach the
        // Python-owned SpeakerProfileStore — route them to the position diarizer.
        if id >= PositionDiarizer.positionIDBase {
            positionDiarizer?.rename(clusterID: id - PositionDiarizer.positionIDBase, to: name)
        } else {
            SpeakerProfileStore.rename(id: id, to: name)
        }
        liveTurns = liveTurns.map {
            $0.id == id
                ? DiarizationService.Turn(start: $0.start, end: $0.end, id: $0.id, name: name)
                : $0
        }
        // Repaired/preserved pinned rows carry their own name copy — keep it in sync.
        for i in segments.indices where segments[i].pinnedSpeakerID == id {
            segments[i].pinnedSpeakerName = name
        }
        rebuildDisplayRows()
    }

    // MARK: - Overlap repair (runs at stop; engine per `overlap.engine`)
    //
    // Two engines, same shape: find 2-speaker overlap windows, get one text per
    // speaker for each window, gate it, splice it in. MossFormer2 (attempt #3)
    // separates the waveform then re-ASRs each track; DiCoW (attempt #4) asks a
    // diarization-conditioned Whisper for one speaker at a time. Only one engine
    // loads per session.

    /// Track-vs-track word-set similarity above which the two separated tracks
    /// are treated as a blend/near-duplicate → skip (Gate 2). Shared by both
    /// engines: two "different speakers" producing the same words is a failure
    /// either way.
    private let nearDuplicateJaccard = 0.72

    /// DiCoW anchor cross-check: how much better a text must match the OTHER
    /// speaker's existing rows than its own before it counts as cross-speaker
    /// leakage → skip that speaker. Guards attempt #2's leakage failure.
    private let anchorLeakMargin = 0.15

    /// DiCoW word-density ceiling (words per second of that speaker's own turns
    /// inside the window). Guards attempt #2's runaway "40+ word span" failure —
    /// nobody genuinely speaks faster than this, so it means the mask leaked.
    private let maxWordsPerSecond = 6.0

    /// DiCoW's hard input limit: the sidecar rejects longer windows so generate()
    /// never enters its long-form seek loop (whose timestamps we do not trust).
    private let dicowMaxWindowSec = 30.0

    /// A merged overlap window plus the two speakers it touches.
    /// The window is centered on the overlap-region midpoint, ±windowSec.
    private struct RepairWindow {
        var start: Double
        var end: Double
        var speakerIDs: [Int]   // exactly 2
    }

    /// Last chunk finished after stop → maybe kick off overlap repair.
    private func checkLastChunkDone() {
        guard stopped, pendingChunkWindows.isEmpty, !chunkedBusy, !lastChunkDone else { return }
        lastChunkDone = true
        setStopStep("chunk", chunkedError.map { .failed($0) } ?? .done)
        // Recording is fully processed — drop any leftover realtime (unconfirmed)
        // segments so a fragment from just before Stop can't survive as an orphan
        // "SPEAKER UNKNOWN" row (e.g. if the last chunk errored/timed out before its
        // own cleanup could run). The chunked pass is authoritative for all audio.
        if segments.contains(where: { !$0.confirmed }) {
            segments.removeAll { !$0.confirmed }
            rebuildDisplayRows()
        }
        maybeStartOverlapRepair()
        checkStopProcessingDone()
    }

    /// Start overlap repair only once both the last chunk and the diarization
    /// final pass are done, the feature is on, and the selected engine loaded.
    private func maybeStartOverlapRepair() {
        let d = UserDefaults.standard
        guard d.object(forKey: "overlap.repair.enabled") as? Bool ?? false else { return }
        guard stopped, finalDiarDone, lastChunkDone else { return }
        guard repairTask == nil, !overlapRepairing else { return }
        guard let recording = lastRecordingURL else { finishRepairStep(.done); return }

        let engine = d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id
        if engine == ModelCatalog.overlapDicow.id {
            guard let service = modelLoader.dicowRepair else {
                finishRepairStep(.failed("engine unavailable")); return
            }
            let windowSec = Double(d.object(forKey: "overlap.dicow.windowSec") as? Int ?? 10)
            // DiCoW's sidecar rejects >30s windows, so drop them here (with a log)
            // rather than burning a request on a guaranteed error.
            let windows = repairWindows(windowSec: windowSec, maxDurationSec: dicowMaxWindowSec)
            guard !windows.isEmpty else {
                overlapLog("DiCoW: no 2-speaker overlap windows to repair")
                finishRepairStep(.done); return
            }
            overlapRepairing = true
            overlapRepairError = nil
            setStopStep("repair", .loading)
            overlapLog("starting overlap repair (DiCoW): \(windows.count) window(s), windowSec=\(Int(windowSec))")
            repairTask = Task { @MainActor [weak self] in
                await self?.runDicowRepair(windows: windows, service: service, recording: recording)
            }
            return
        }

        guard let service = modelLoader.overlapRepair else {
            finishRepairStep(.failed("engine unavailable")); return
        }
        let windowSec = Double(d.object(forKey: "overlap.mossformer.windowSec") as? Int ?? 10)
        let windows = repairWindows(windowSec: windowSec)
        guard !windows.isEmpty else {
            overlapLog("no 2-speaker overlap windows to repair")
            finishRepairStep(.done); return
        }
        overlapRepairing = true
        overlapRepairError = nil
        setStopStep("repair", .loading)
        overlapLog("starting overlap repair: \(windows.count) window(s), windowSec=\(Int(windowSec))")
        repairTask = Task { @MainActor [weak self] in
            await self?.runOverlapRepair(windows: windows, service: service, recording: recording)
        }
    }

    /// Repair bailed out before a driver ever started, so nothing else will settle
    /// the overlay's repair row — do it here and re-check the gate.
    private func finishRepairStep(_ state: ModelLoader.ItemState) {
        setStopStep("repair", state)
        checkStopProcessingDone()
    }

    /// Merged 2-speaker overlap windows to repair, in chronological order.
    /// Each raw window is centered on the overlap-region midpoint, ±windowSec.
    /// `maxDurationSec` (DiCoW only; nil = no limit, i.e. MossFormer2's original
    /// behaviour) drops merged windows longer than the engine can accept.
    private func repairWindows(windowSec: Double,
                               maxDurationSec: Double? = nil) -> [RepairWindow] {
        let turns = liveTurns
        guard turns.count > 1 else { return [] }

        // Raw windows from pairwise different-speaker overlaps.
        var raw: [(start: Double, end: Double)] = []
        for i in 0..<turns.count {
            for j in (i + 1)..<turns.count where turns[i].id != turns[j].id {
                let a = turns[i].start <= turns[j].start ? turns[i] : turns[j]
                let b = turns[i].start <= turns[j].start ? turns[j] : turns[i]
                let os = max(a.start, b.start)
                let oe = min(a.end, b.end)
                guard oe - os >= 0.4 else { continue }   // genuine overlap only
                // Center the window on the overlap-region midpoint.
                let mid = (os + oe) / 2
                var ws = max(0, mid - windowSec)
                var we = min(recordingElapsed, mid + windowSec)
                ws = min(ws, os)          // keep the full overlap span inside (long overlaps)
                we = max(we, oe)
                we = min(we, recordingElapsed)
                if we - ws >= 2.0 {
                    raw.append((ws, we))
                    overlapLog("window [\(fmt(ws))-\(fmt(we))] centered on overlap midpoint \(fmt(mid)) (overlap [\(fmt(os))-\(fmt(oe))])")
                }
            }
        }
        guard !raw.isEmpty else { return [] }

        // Merge intersecting ranges so a later repair never clobbers an earlier one.
        raw.sort { $0.start < $1.start }
        var merged: [(start: Double, end: Double)] = []
        for r in raw {
            if var last = merged.last, r.start <= last.end {
                last.end = max(last.end, r.end)
                merged[merged.count - 1] = last
            } else {
                merged.append(r)
            }
        }

        // Attach the distinct speakers each merged window touches; skip 3+, and
        // skip anything past the engine's window limit.
        var result: [RepairWindow] = []
        for m in merged {
            if let cap = maxDurationSec, m.end - m.start > cap {
                overlapLog("SKIP window [\(fmt(m.start))-\(fmt(m.end))] is "
                    + "\(fmt(m.end - m.start))s, over this engine's \(fmt(cap))s limit")
                continue
            }
            var ids: [Int] = []
            for t in turns where min(t.end, m.end) - max(t.start, m.start) > 0 {
                if !ids.contains(t.id) { ids.append(t.id) }
            }
            if ids.count == 2 {
                result.append(RepairWindow(start: m.start, end: m.end, speakerIDs: ids))
            } else {
                overlapLog("SKIP window [\(fmt(m.start))-\(fmt(m.end))] touches \(ids.count) speakers (only 2 supported)")
            }
        }
        return result
    }

    /// Sequential driver: one window fully completes (through the UI update)
    /// before the next begins. A per-window failure logs and continues.
    private func runOverlapRepair(windows: [RepairWindow],
                                  service: OverlapRepairService,
                                  recording: URL) async {
        let n = windows.count
        for (i, w) in windows.enumerated() {
            if Task.isCancelled { break }   // a new session owns the transcript now
            overlapRepairProgress = "\(i + 1)/\(n)"
            let tmpDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("moss-\(UUID().uuidString)")
            do {
                let tracks = try await service.separate(audio: recording,
                                                        start: w.start, end: w.end,
                                                        outDir: tmpDir)
                guard tracks.count == 2 else {
                    overlapLog("SKIP [\(fmt(w.start))-\(fmt(w.end))] separator returned \(tracks.count) tracks")
                    cleanup(tmpDir); continue
                }
                guard let chunked = modelLoader.chunkedASR else {
                    overlapLog("SKIP — chunked ASR not available for re-transcription")
                    cleanup(tmpDir); break
                }
                let text1 = (try? await chunked.transcribeFile(path: tracks[0].path)) ?? ""
                let text2 = (try? await chunked.transcribeFile(path: tracks[1].path)) ?? ""
                processRepair(window: w, tracks: tracks, text1: text1, text2: text2)
            } catch {
                overlapLog("SKIP [\(fmt(w.start))-\(fmt(w.end))] separation failed: \(error.localizedDescription)")
            }
            cleanup(tmpDir)
        }
        overlapRepairing = false
        overlapRepairProgress = nil
        repairTask = nil
        setStopStep("repair", .done)
        checkStopProcessingDone()
        overlapLog("overlap repair complete")
    }

    /// Fold each separated track's re-ASR text into its attributed speaker with
    /// COMBINE-OR-REPLACE semantics (via `TranscriptMerge`), then ALSO append the
    /// two raw "MossFormer2 Index N" debug rows. A track that adds no new words is
    /// a NO-OP (its speaker's rows are left untouched); near-duplicate tracks skip
    /// the speaker-row edit entirely. Every decision is logged.
    private func processRepair(window w: RepairWindow,
                               tracks: [OverlapRepairService.SeparatedTrack],
                               text1: String, text2: String) {
        let ws = w.start, we = w.end
        let idA = w.speakerIDs[0], idB = w.speakerIDs[1]
        let anchorA = anchorText(for: idA, ws: ws, we: we)
        let anchorB = anchorText(for: idB, ws: ws, we: we)

        // Attribution 2×2: send each track to whichever speaker it best matches.
        let j11 = jaccard(text1, anchorA), j12 = jaccard(text1, anchorB)
        let j21 = jaccard(text2, anchorA), j22 = jaccard(text2, anchorB)
        let straight = (j11 + j22) >= (j12 + j21)
        let t1Speaker = straight ? idA : idB
        let t1Anchor  = straight ? anchorA : anchorB
        let t2Speaker = straight ? idB : idA
        let t2Anchor  = straight ? anchorB : anchorA

        overlapLog("REPAIR [\(fmt(ws))-\(fmt(we))] jaccard "
            + "t1·A=\(fmt3(j11)) t1·B=\(fmt3(j12)) t2·A=\(fmt3(j21)) t2·B=\(fmt3(j22)) → "
            + "\(straight ? "straight" : "swapped"): track1→speaker \(t1Speaker), track2→speaker \(t2Speaker)")

        // Decide per-speaker merges (skipped wholesale if the tracks look duplicated).
        var decisions: [Int: String] = [:]
        let dup = jaccard(text1, text2)
        if dup > nearDuplicateJaccard {
            overlapLog("  SKIP near-duplicate tracks (jaccard=\(fmt3(dup)) > \(nearDuplicateJaccard)) — speaker rows unchanged")
        } else {
            for (trackText, speakerID, anchor) in [(text1, t1Speaker, t1Anchor),
                                                   (text2, t2Speaker, t2Anchor)] {
                if trackText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    overlapLog("  speaker \(speakerID): NO-OP (empty track)")
                    continue
                }
                let r = TranscriptMerge.merge(existing: anchor, track: trackText)
                switch r.kind {
                case .noop:
                    overlapLog("  speaker \(speakerID): NO-OP (track ⊆ existing, run=\(r.longestRun))"
                        + "\n    existing: \(anchor)\n    track: \(trackText)")
                case .combine:
                    decisions[speakerID] = r.text
                    overlapLog("  speaker \(speakerID): COMBINE (run=\(r.longestRun))"
                        + "\n    before: \(anchor)\n    track: \(trackText)\n    after: \(r.text)")
                case .replace:
                    decisions[speakerID] = r.text
                    overlapLog("  speaker \(speakerID): REPLACE (run=\(r.longestRun))"
                        + "\n    before: \(anchor)\n    track: \(trackText)\n    after: \(r.text)")
                }
            }
            if !decisions.isEmpty { applyRepair(ws: ws, we: we, decisions: decisions) }
        }

        // Debug rows: raw MossFormer2 separated-track ASR, verbatim, kept at the end
        // of `segments` so they render after the real transcript. Toggled by the
        // "Show MossFormer2 Index 1/2 rows" setting (Settings → Models → Overlap).
        let showDebug = UserDefaults.standard.object(forKey: "overlap.mossformer.showDebugRows") as? Bool ?? true
        if showDebug {
            let t1 = text1.isEmpty ? "(empty)" : text1
            let t2 = text2.isEmpty ? "(empty)" : text2
            segments.append(TranscriptSegment(text: t1, confirmed: true, window: ws...we,
                                              pinnedSpeakerName: "MossFormer2 Index1",
                                              isSeparationDebug: true))
            segments.append(TranscriptSegment(text: t2, confirmed: true, window: ws...we,
                                              pinnedSpeakerName: "MossFormer2 Index2",
                                              isSeparationDebug: true))
        }
        rebuildDisplayRows()
    }

    // MARK: - Overlap repair — DiCoW (attempt #4)

    /// Sequential driver: one window fully completes (through the UI update)
    /// before the next begins. A per-window failure logs and continues.
    /// Mirrors `runOverlapRepair`, but there is no separation step — DiCoW is
    /// asked directly for one transcription per speaker.
    private func runDicowRepair(windows: [RepairWindow],
                                service: DicowService,
                                recording: URL) async {
        // Same language the chunked pass uses; "auto" → let DiCoW detect.
        let code = UserDefaults.standard.string(forKey: "chunked.language") ?? "auto"
        let language: String? = code == "auto" ? nil : code

        let n = windows.count
        for (i, w) in windows.enumerated() {
            if Task.isCancelled { break }   // a new session owns the transcript now
            overlapRepairProgress = "\(i + 1)/\(n)"
            let ranges = speakerRanges(in: w.start...w.end)
            let targets = w.speakerIDs.map { id in
                DicowService.Target(
                    sid: id,
                    turns: ranges.filter { $0.id == id }.map { (start: $0.start, end: $0.end) })
            }
            guard targets.allSatisfy({ !$0.turns.isEmpty }) else {
                overlapLog("DiCoW SKIP [\(fmt(w.start))-\(fmt(w.end))] a speaker has no turns in-window")
                continue
            }
            do {
                let results = try await service.transcribeTargets(
                    audio: recording, start: w.start, end: w.end,
                    targets: targets, language: language)
                processDicowRepair(window: w, targets: targets, results: results)
            } catch {
                overlapLog("DiCoW SKIP [\(fmt(w.start))-\(fmt(w.end))] failed: \(error.localizedDescription)")
            }
        }
        overlapRepairing = false
        overlapRepairProgress = nil
        repairTask = nil
        setStopStep("repair", .done)
        checkStopProcessingDone()
        overlapLog("DiCoW overlap repair complete")
    }

    /// Gate DiCoW's per-speaker texts, then fold the survivors into their speaker
    /// with COMBINE-OR-REPLACE semantics (via `TranscriptMerge`), exactly like the
    /// MossFormer2 path. No 2×2 attribution is needed — DiCoW is already
    /// conditioned on each speaker's mask, so a text's speaker is known up front.
    ///
    /// The gates exist because of how attempt #2 (2026-07-14) failed; each one
    /// targets a specific observed failure:
    ///   • near-duplicate texts   → both masks picked up the same voice
    ///   • anchor cross-check     → cross-speaker leakage
    ///   • word-density ceiling   → runaway spans (the "40+ word" bug)
    /// A gated speaker keeps its original, honest text.
    private func processDicowRepair(window w: RepairWindow,
                                    targets: [DicowService.Target],
                                    results: [DicowService.TargetResult]) {
        let ws = w.start, we = w.end
        let idA = w.speakerIDs[0], idB = w.speakerIDs[1]
        func text(_ id: Int) -> String { results.first { $0.sid == id }?.text ?? "" }
        let textA = text(idA), textB = text(idB)
        let anchorA = anchorText(for: idA, ws: ws, we: we)
        let anchorB = anchorText(for: idB, ws: ws, we: we)

        overlapLog("DiCoW REPAIR [\(fmt(ws))-\(fmt(we))] speakers \(idA) & \(idB)"
            + "\n    speaker \(idA): \(textA.isEmpty ? "(empty)" : textA)"
            + "\n    speaker \(idB): \(textB.isEmpty ? "(empty)" : textB)")

        var decisions: [Int: String] = [:]
        // Gate: near-duplicate → the two masks resolved to the same voice.
        let dup = jaccard(textA, textB)
        if !textA.isEmpty, !textB.isEmpty, dup > nearDuplicateJaccard {
            overlapLog("  SKIP near-duplicate texts (jaccard=\(fmt3(dup)) > \(nearDuplicateJaccard)) — speaker rows unchanged")
        } else {
            for (id, txt, anchor, otherAnchor) in [(idA, textA, anchorA, anchorB),
                                                   (idB, textB, anchorB, anchorA)] {
                if txt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    overlapLog("  speaker \(id): NO-OP (empty text)")
                    continue
                }
                // Gate: cross-speaker leakage — the text matches the OTHER
                // speaker's existing rows clearly better than its own.
                let jOwn = jaccard(txt, anchor), jOther = jaccard(txt, otherAnchor)
                if jOther > jOwn + anchorLeakMargin {
                    overlapLog("  speaker \(id): SKIP cross-speaker leakage "
                        + "(own=\(fmt3(jOwn)) vs other=\(fmt3(jOther)), margin \(anchorLeakMargin))")
                    continue
                }
                // Gate: word density — more words than this speaker had time to say.
                let spoken = targets.first { $0.sid == id }?.turns
                    .reduce(0.0) { $0 + ($1.end - $1.start) } ?? 0
                let words = txt.split { !$0.isLetter && !$0.isNumber }.count
                let density = spoken > 0 ? Double(words) / spoken : .infinity
                if density > maxWordsPerSecond {
                    overlapLog("  speaker \(id): SKIP word density \(fmt3(density)) w/s "
                        + "(\(words) words over \(fmt(spoken))s of turns) > \(fmt(maxWordsPerSecond))")
                    continue
                }
                let r = TranscriptMerge.merge(existing: anchor, track: txt)
                switch r.kind {
                case .noop:
                    overlapLog("  speaker \(id): NO-OP (text ⊆ existing, run=\(r.longestRun))"
                        + "\n    existing: \(anchor)\n    dicow: \(txt)")
                case .combine:
                    decisions[id] = r.text
                    overlapLog("  speaker \(id): COMBINE (run=\(r.longestRun))"
                        + "\n    before: \(anchor)\n    dicow: \(txt)\n    after: \(r.text)")
                case .replace:
                    decisions[id] = r.text
                    overlapLog("  speaker \(id): REPLACE (run=\(r.longestRun))"
                        + "\n    before: \(anchor)\n    dicow: \(txt)\n    after: \(r.text)")
                }
            }
            if !decisions.isEmpty { applyRepair(ws: ws, we: we, decisions: decisions) }
        }

        // Debug rows: DiCoW's raw per-speaker output, verbatim and ungated, kept at
        // the end of `segments` so they render after the real transcript. Toggled by
        // "Show DiCoW per-speaker rows" (Settings → Models → Overlap). Shown even
        // when a gate skipped the text — seeing what was rejected is the point.
        let showDebug = UserDefaults.standard.object(forKey: "overlap.dicow.showDebugRows") as? Bool ?? true
        if showDebug {
            for (id, txt) in [(idA, textA), (idB, textB)] {
                segments.append(TranscriptSegment(
                    text: txt.isEmpty ? "(empty)" : txt, confirmed: true, window: ws...we,
                    pinnedSpeakerName: "DiCoW · \(speakerName(for: id) ?? "Speaker \(id)")",
                    isSeparationDebug: true))
            }
        }
        rebuildDisplayRows()
    }

    /// Write the repaired speakers' merged text back into `segments` WITHOUT
    /// duplicating any text. For every source segment intersecting [ws,we], its
    /// derived rows are either CONSUMED (a repaired speaker's rows in-window — their
    /// text is already folded into the merged `decisions` text) or PRESERVED verbatim
    /// as a pinned segment (every other row). The consumed rows collapse into one
    /// pinned segment per repaired speaker. Invariant: each row's text survives
    /// exactly once — folded or preserved, never split, never duplicated.
    private func applyRepair(ws: Double, we: Double, decisions: [Int: String]) {
        let regions = overlapRegions()
        let affectedIdx = segments.indices.filter { i in
            let s = segments[i]
            guard s.confirmed, !s.isSeparationDebug, let w = s.window else { return false }
            return min(w.upperBound, we) - max(w.lowerBound, ws) > 0
        }

        // No existing rows to fold into — emit repaired speakers as fresh pinned rows.
        guard !affectedIdx.isEmpty else {
            overlapLog("  applyRepair: no source segments intersect window — inserting repaired rows fresh")
            insertPinnedSorted(decisions.map { id, text in
                TranscriptSegment(text: text, confirmed: true, window: ws...we,
                                  pinnedSpeakerID: id, pinnedSpeakerName: speakerName(for: id))
            })
            return
        }

        var preserved: [TranscriptSegment] = []
        var consumedSpan: [Int: (lo: Double, hi: Double)] = [:]   // per repaired speaker

        for i in affectedIdx {
            for row in derivedRows(for: segments[i], regions: regions) {
                let rs = row.start ?? segments[i].window?.lowerBound ?? ws
                let re = row.end ?? segments[i].window?.upperBound ?? we
                let inWindow = min(re, we) - max(rs, ws) > 0
                if inWindow, let sid = row.speakerID, decisions[sid] != nil {
                    // CONSUMED — folded into this speaker's merged text.
                    let cur = consumedSpan[sid]
                    consumedSpan[sid] = (min(cur?.lo ?? rs, rs), max(cur?.hi ?? re, re))
                } else {
                    // PRESERVE verbatim (other speakers, or this speaker outside window).
                    preserved.append(TranscriptSegment(text: row.text, confirmed: true,
                                                       window: rs...max(re, rs),
                                                       pinnedSpeakerID: row.speakerID,
                                                       pinnedSpeakerName: row.speaker))
                }
            }
        }

        // Drop the affected source segments (descending, so indices stay valid).
        for i in affectedIdx.sorted(by: >) { segments.remove(at: i) }

        // One pinned segment per repaired speaker, spanning its consumed rows.
        let repaired: [TranscriptSegment] = decisions.map { id, text in
            let span = consumedSpan[id] ?? (ws, we)
            return TranscriptSegment(text: text, confirmed: true,
                                     window: span.lo...max(span.hi, span.lo),
                                     pinnedSpeakerID: id,
                                     pinnedSpeakerName: speakerName(for: id))
        }
        insertPinnedSorted(preserved + repaired)
    }

    /// Insert new (non-debug) segments and re-sort the real transcript by start time,
    /// keeping any separation-debug segments pinned at the end.
    private func insertPinnedSorted(_ newSegs: [TranscriptSegment]) {
        var real = segments.filter { !$0.isSeparationDebug }
        let debug = segments.filter { $0.isSeparationDebug }
        real.append(contentsOf: newSegs)
        real.sort { ($0.window?.lowerBound ?? .greatestFiniteMagnitude)
                  < ($1.window?.lowerBound ?? .greatestFiniteMagnitude) }
        segments = real + debug
    }

    /// Display name for a speaker id, from the diarization turns collected so far.
    private func speakerName(for id: Int) -> String? {
        liveTurns.first(where: { $0.id == id })?.name
    }

    /// Existing display-row text for one speaker intersecting [ws, we],
    /// the anchor for text-based track attribution.
    private func anchorText(for id: Int, ws: Double, we: Double) -> String {
        displayRows.filter { row in
            guard row.speakerID == id, let s = row.start, let e = row.end else { return false }
            return min(e, we) - max(s, ws) > 0
        }.map(\.text).joined(separator: " ")
    }

    /// Jaccard similarity of the two texts' lowercased alnum word sets.
    private func jaccard(_ a: String, _ b: String) -> Double {
        func tokens(_ s: String) -> Set<String> {
            Set(s.lowercased().split { !$0.isLetter && !$0.isNumber }
                .map(String.init).filter { !$0.isEmpty })
        }
        let ta = tokens(a), tb = tokens(b)
        if ta.isEmpty && tb.isEmpty { return 1.0 }
        let uni = ta.union(tb).count
        return uni == 0 ? 0.0 : Double(ta.intersection(tb).count) / Double(uni)
    }

    private func fmt(_ s: Double) -> String { String(format: "%.1f", s) }
    private func fmt3(_ x: Double) -> String { String(format: "%.3f", x) }

    private func cleanup(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Append a line to logs/overlap-repair-decisions.log (mandatory decision logging).
    /// Kept separate from the sidecar's own stderr log (overlap-repair-sidecar.log) —
    /// they used to share one file with no coordination between writers.
    private func overlapLog(_ message: String) {
        let dir = PythonRuntime.dataDir.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("overlap-repair-decisions.log")
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        let stamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let line = "[\(stamp)] \(message)\n"
        guard let handle = try? FileHandle(forWritingTo: file) else { return }
        handle.seekToEndOfFile()
        if let data = line.data(using: .utf8) { handle.write(data) }
        try? handle.close()
    }

    /// Append a line to logs/position-diarization.log (position gap-fill decisions).
    /// Mirrors overlapLog: one FILL/SKIP line per display-time gap.
    private func positionLog(_ message: String) {
        let dir = PythonRuntime.dataDir.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("position-diarization.log")
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        let stamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let line = "[\(stamp)] \(message)\n"
        guard let handle = try? FileHandle(forWritingTo: file) else { return }
        handle.seekToEndOfFile()
        if let data = line.data(using: .utf8) { handle.write(data) }
        try? handle.close()
    }

    /// Write 16 kHz mono float samples to a temp WAV (for chunk diarization).
    /// Pure/self-contained, so it runs off the main actor from the detached task.
    private nonisolated static func writeTempWAV(samples: [Float]) -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("diar-chunk-\(UUID().uuidString).wav")
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: 16_000, channels: 1,
                                         interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(samples.count))
        else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            buffer.floatChannelData?[0].update(from: source.baseAddress!, count: samples.count)
        }
        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
            return url
        } catch {
            return nil
        }
    }
}
