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

    @Published var segments: [TranscriptSegment] = []
    @Published var displayRows: [SpeakerUtterance] = []
    @Published var partialTranscript = ""
    /// Live caption for the Remote stream. Stays empty for a single-stream
    /// session, so the view draws nothing extra.
    @Published var remoteCaption = RemoteCaption()
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

    /// Track-vs-track word-set similarity above which the two separated tracks
    /// are treated as a blend/near-duplicate → skip (Gate 2). Shared by both
    /// engines: two "different speakers" producing the same words is a failure
    /// either way.
    let nearDuplicateJaccard = 0.72

    /// DiCoW anchor cross-check: how much better a text must match the OTHER
    /// speaker's existing rows than its own before it counts as cross-speaker
    /// leakage → skip that speaker. Guards attempt #2's leakage failure.
    let anchorLeakMargin = 0.15

    /// DiCoW word-density ceiling (words per second of that speaker's own turns
    /// inside the window). Guards attempt #2's runaway "40+ word span" failure —
    /// nobody genuinely speaks faster than this, so it means the mask leaked.
    let maxWordsPerSecond = 6.0

    /// DiCoW's hard input limit: the sidecar rejects longer windows so generate()
    /// never enters its long-form seek loop (whose timestamps we do not trust).
    let dicowMaxWindowSec = 30.0

    @Published var stopSteps: [StopStep] = []

    // Gating for the "wait for last chunk AND diarization final" sequencing.
    var stopped = false
    var finalDiarDone = false
    var lastChunkDone = false
    var awaitingTailWindowStart: Double? = nil
    var diarTailWatchdog: Task<Void, Never>?
    var finalDiarWatchdog: Task<Void, Never>?
    var stopWatchdog: Task<Void, Never>?
    var repairTask: Task<Void, Never>?

    private var chunkElapsed: Double = 0      // seconds since last chunk flush
    private var chunkWatchdog: Task<Void, Never>?

    // Chunk time-window bookkeeping (for mapping speakers onto segments)
    var recordingElapsed: Double = 0
    private var lastChunkBoundary: Double = 0
    var pendingChunkWindows: [ClosedRange<Double>] = []
    // Elapsed time of the previous realtime (Nemotron) final. Each final covers
    // the audio since the last flush (the sidecar resets its buffer on flush), so
    // the not-yet-confirmed segment spans [lastRealtimeFinalElapsed, now] — a real
    // start–end range rather than a single point-in-time timestamp.
    private var lastRealtimeFinalElapsed: Double = 0

    // Live chunked diarization — runs on its OWN interval, independent of ASR
    var chunkAudio: [Float] = []                       // 16k samples pending diarization
    var chunkFileByWindow: [Double: URL] = [:]
    var sessionSpeakerIDs = Set<Int>()
    var liveTurns: [DiarizationService.Turn] = []      // absolute-time turns collected so far
    // Position-based diarization (ATND beam) — off unless atnd.position.enabled.
    // Recorder-owned, one per session; nil means the feature is off, so
    // positionGapFill returns [] and the display path is pure pyannote.
    @Published var positionDiarizer: PositionDiarizer?
    /// Which layer the DISPLAY draws labels from — read once per session in
    /// `configurePositionDiarization()`, like every other setting, so it can't
    /// flip mid-recording and leave half the transcript on each policy.
    var positionSource: PositionSource = .both
    /// Last SEPARATION line written to the position log, so the per-room tau
    /// calibration diagnostic is emitted on change instead of on every rebuild.
    var lastLoggedSeparation: String?
    private var diarElapsed: Double = 0                        // seconds since the last diar chunk
    var lastDiarBoundary: Double = 0                   // recording-time where this diar chunk began

    // MARK: Remote stream (dual-stream phase 3)
    //
    // Everything below is inert unless a Remote channel resolved for THIS session
    // (`remoteStreamActive`). The remote side never influences timing: it has no
    // cadence, no VAD, no RMS and no clock of its own — it rides the office chunk
    // boundary and the single `recordingElapsed`, which is the whole reason both
    // inputs must come from one Aggregate Device (see `MicrophoneSettings`).

    /// True when this session resolved a usable Remote channel. Published so the
    /// status chips can show the Remote speaker count for a dual-stream session
    /// ONLY — with `mic.remoteChannel` unset this stays false for the whole app
    /// lifetime and the chip row is exactly what it was before dual-stream.
    /// It survives stop on purpose: the finished session's remote count stays
    /// readable until the next start re-evaluates it.
    @Published private(set) var remoteStreamActive = false
    /// 16 kHz remote samples accumulated since the last chunk boundary.
    var remoteChunkAudio: [Float] = []
    /// Transcribed remote chunks, merged into `displayRows` by start time.
    var remoteSegments: [RemoteSegment] = []
    /// Remote file-transcribe requests currently in flight.
    var remotePendingChunks = 0
    /// Last remote failure, shown on the stop step. Never fatal — a remote
    /// problem must not cost the user their office transcript.
    var remoteChunkError: String?
    /// Stop gate, mirroring `lastChunkDone`. Starts true so a single-stream
    /// session's gate is complete before it is ever consulted.
    var remoteLastChunkDone = true
    var remoteStopWatchdog: Task<Void, Never>?

    // MARK: Remote diarization (dual-stream phase 4)
    //
    // A SECOND identity space over the SAME sidecar process: the jobs below carry
    // `stream: .remote`, which selects the sidecar's remote ProfileStore. Nothing
    // here may touch `liveTurns`, `sessionSpeakerIDs`, `speakerCount`,
    // `overlapRegions`, `repairWindows` or the ATND/position path — those are all
    // Office-only, and remote ids (>= remoteIDBase) reaching any of them is the
    // corruption this split exists to prevent.

    /// The Remote WAV for this session, diarized as a whole at stop when
    /// `diarization.continueOnStop` is OFF (tail mode needs no file). Held
    /// separately from `lastRecordingURL`, which every Office-only consumer
    /// (final pass, overlap repair, DiCoW) reads and must keep reading.
    var remoteRecordingURL: URL?
    /// Remote-space turns collected so far — the remote twin of `liveTurns`.
    /// Ids are already offset by `remoteIDBase` (the sidecar applies it).
    var remoteLiveTurns: [DiarizationService.Turn] = []
    /// 16 kHz remote samples pending live diarization. Separate from
    /// `remoteChunkAudio` because the diarization cadence is its own setting.
    var remoteDiarAudio: [Float] = []
    /// Temp chunk WAVs awaiting a remote result. Keyed the same way as
    /// `chunkFileByWindow` but kept apart, since both streams use the SAME
    /// window starts and would otherwise delete each other's files.
    var remoteChunkFileByWindow: [Double: URL] = [:]
    var remoteSessionSpeakerIDs = Set<Int>()
    /// Remote-space speaker count. `speakerCount` stays Office-only.
    @Published var remoteSpeakerCount: Int?
    /// Stop gate for the remote final pass. Starts true so a single-stream
    /// session's gate is already complete before it is ever consulted.
    var remoteFinalDiarDone = true
    var remoteFinalDiarWatchdog: Task<Void, Never>?
    /// Window start of the remote TAIL chunk the stop gate is waiting on, or nil
    /// when the remote stop pass is not a tail (full pass / no pass). The remote
    /// twin of `awaitingTailWindowStart`; kept apart because both streams use the
    /// SAME window starts and one would otherwise settle the other's gate.
    var awaitingRemoteTailWindowStart: Double? = nil

    /// Publishes per-model progress for the loading overlay.
    let modelLoader = ModelLoader()

    /// Future hook: realtime ASR (Nemotron) consumes mono buffers of the selected channel.
    var onBuffer: ((AVAudioPCMBuffer) -> Void)?

    private var engine: AVAudioEngine?
    private var file: AVAudioFile?
    /// Second mono WAV for the Remote (conferencing) stream, or nil for today's
    /// single-stream behaviour. Written from the SAME tap callback as `file`, so
    /// the two files are sample-aligned by construction.
    ///
    /// Two mono files rather than one stereo file: `lastRecordingURL` feeds the
    /// final diarization pass, overlap repair (`maybeStartOverlapRepair`,
    /// `OverlapRepairService.separate`) and DiCoW — all Office-only consumers that
    /// must keep reading an unchanged mono office file. A stereo file would force
    /// every one of them to learn channel extraction for no gain.
    private var remoteFile: AVAudioFile?
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
        // Hard refusal BEFORE any model loads: a Remote channel with Voxtral
        // selected cannot work (see `dualStreamRefusalMessage`). Checked here
        // rather than in `beginCapture` so the user is told before waiting out a
        // 4B model load, and it lands in the same overlay every startup failure
        // uses. Refusal, never a silent model swap — the owner picks chunked
        // models deliberately, on measured WER.
        let mic = MicrophoneSettings.current()
        let chunkedID = UserDefaults.standard.string(forKey: "chunked.model") ?? "qwen3"
        if let refusal = Self.dualStreamRefusalMessage(remoteChannel: mic.remoteChannel,
                                                       chunkedModelID: chunkedID) {
            dualStreamLog("REFUSED start — \(refusal)")
            modelLoader.failStartup(step: "Remote stream + chunked model", message: refusal)
            errorMessage = refusal
            state = .idle
            return
        }
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
        // Remote (conferencing) channel, re-validated against the format the engine
        // ACTUALLY delivers. `MicrophoneSettings.resolve` already checked it against
        // the device's advertised channel count, but with an Aggregate Device the
        // live format is the authority — sub-devices can present fewer channels than
        // advertised. If it does not survive, degrade to single-stream and log it:
        // a recording that captures the room beats no recording at all.
        var remoteChannel = Self.resolveRemoteChannel(mic.remoteChannel,
                                                      officeChannel: channel,
                                                      liveChannelCount: Int(format.channelCount))
        if let wanted = mic.remoteChannel, remoteChannel == nil {
            dualStreamLog("Remote channel \(wanted) unusable against the live format "
                          + "(\(format.channelCount) ch, office \(channel)) — recording Office only.")
        }

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
        // 3b. Second mono file for Remote, same format and directory as Office (see
        // `remoteFile`'s doc comment for why two files and not one stereo file).
        // Nothing consumes it yet — phase 2 is capture only. Failing to create it
        // must not kill a working Office recording, so it degrades to single-stream.
        remoteRecordingURL = nil
        if remoteChannel != nil {
            let remoteURL = Self.remoteURL(forOffice: url)
            do {
                remoteFile = try AVAudioFile(forWriting: remoteURL, settings: monoFormat.settings)
                remoteRecordingURL = remoteURL   // the stop-time remote final pass reads this
            } catch {
                remoteFile = nil
                remoteChannel = nil   // keep the tap on the inert single-stream path
                dualStreamLog("Could not create the Remote recording file "
                              + "(\(error.localizedDescription)) — recording Office only.")
            }
        }

        // 4. VAD — fresh instance per session; uses Silero sidecar if it loaded
        let vadOn = UserDefaults.standard.object(forKey: "vad.enabled") as? Bool ?? true
        vadEnabled = vadOn
        let vad = vadOn ? VoiceActivityDetector(silero: modelLoader.sileroVAD) : nil
        self.vad = vad
        let sampleRate = format.sampleRate
        let resampler = AudioResampler(inputFormat: monoFormat) // → 16 kHz for Silero/ASR
        // Remote gets its OWN resampler: `AudioResampler` keeps converter state
        // between buffers, so feeding two interleaved streams through one instance
        // would corrupt both. Created only when there is both a Remote channel and
        // a chunked sidecar to transcribe it with — nil keeps the tap on the
        // single-stream path, and (just as important) stops the remote 16 kHz
        // buffer from growing all meeting for audio nothing would ever consume.
        let remoteWanted = remoteChannel != nil && modelLoader.chunkedASR != nil
        let remoteResampler = remoteWanted ? AudioResampler(inputFormat: monoFormat) : nil

        // 5. Realtime ASR — stream audio in, receive partial/final transcripts
        let realtimeOn = UserDefaults.standard.object(forKey: "realtime.enabled") as? Bool ?? true
        // `asr` is the OFFICE LANE of the one realtime sidecar (the remote lane
        // of the same process is picked up in 5b). Same `feed`/`flush`/
        // `onTranscript` calls as when this was a whole service.
        let asr = realtimeOn ? modelLoader.nemotronASR?.office : nil
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
                    // In `pyannote` source mode the position layer is not displayed
                    // anywhere, so the live partial must not carry its label either.
                    self.partialSpeakerName = self.positionSource.usesPosition
                        ? self.positionDiarizer?.label(
                            for: max(0, self.recordingElapsed - 1.0)...self.recordingElapsed,
                            minSamples: 3)?.name
                        : nil
                }
            }
        }

        // 5b. Realtime ASR for the Remote stream — the SECOND LANE of the same
        // sidecar, captioning only. It gets audio and flushes; it never touches
        // `recordingElapsed`, the VAD, the RMS meter, either cadence or the ATND
        // position path. Nil unless this session has remote 16 kHz audio to give
        // it (the same `remoteResampler` condition the rest of the remote side
        // hangs off) — so a single-stream session never installs this callback at
        // all, and (see the `else` below) never leaves a stale one installed.
        let remoteASR = (realtimeOn && remoteResampler != nil)
            ? modelLoader.nemotronASR?.remote : nil
        if remoteASR == nil {
            // Sharing one process means the remote lane outlives the session that
            // wanted it. Detach so a previous meeting's closure can never fire.
            modelLoader.nemotronASR?.detachRemoteLane()
        }
        remoteASR?.onTranscript = { [weak self] text, _ in
            Task { @MainActor in
                guard let self else { return }
                // Partial and final are handled identically — see `RemoteCaption`
                // for why a final is kept rather than cleared. After Stop the
                // remote chunk pass owns the remaining audio, exactly as the
                // office branch above reasons about its own trailing final.
                guard !self.stopped else { self.remoteCaption.commit(); return }
                self.remoteCaption.update(to: text)
            }
        }

        // 5c. Chunked ASR — rolling accurate pass every N seconds.
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
        // Fresh remote state. `remoteStreamActive` also requires the resampler:
        // without it there is no 16 kHz audio to transcribe, so the remote side
        // stays off (the WAV is still written — capture is phase 2's job).
        remoteStreamActive = remoteResampler != nil
        if remoteWanted && remoteResampler == nil {
            dualStreamLog("Could not create the Remote resampler — the Remote WAV is still "
                          + "written, but remote audio will not be transcribed this session.")
        }
        remoteChunkAudio = []
        remoteCaption.commit()
        remoteSegments = []
        remotePendingChunks = 0
        remoteChunkError = nil
        remoteLastChunkDone = !remoteStreamActive
        remoteStopWatchdog?.cancel()
        remoteStopWatchdog = nil
        // Fresh remote diarization state. Gated on `remoteStreamActive` for the
        // same reason the transcription side is: without remote 16 kHz audio
        // there are no remote rows for the labels to land on.
        remoteLiveTurns = []
        remoteDiarAudio = []
        remoteChunkFileByWindow = [:]
        remoteSessionSpeakerIDs = []
        remoteSpeakerCount = nil
        remoteFinalDiarDone = true   // flipped to false in stop() iff a pass is dispatched
        remoteFinalDiarWatchdog?.cancel()
        remoteFinalDiarWatchdog = nil
        awaitingRemoteTailWindowStart = nil
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
        // Office-only on purpose, and now spelled out by the `.office` lane: the
        // beam describes the ROOM, so a cluster change says nothing about the
        // conferencing stream. Flushing the remote lane here would cut its caption
        // on an event from the other stream — and it is the first step towards
        // remote audio reaching the position path, which it must never do.
        positionDiarizer?.onClusterChange = { [weak self] in
            guard let self, !self.stopped else { return }
            self.modelLoader.nemotronASR?.office.flush()  // end the old speaker's realtime segment now
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
        chunked?.onChunkTranscript = { [weak self] text, words, chunkDuration in
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
                    // Words are carried through untouched (chunk-relative) for
                    // word-exact attribution; nothing consumes them yet, so row
                    // building below is unchanged.
                    self.segments.append(TranscriptSegment(text: text, confirmed: true, window: window,
                                                           words: words,
                                                           alignedChunkDuration: chunkDuration))
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
        // Immutable copy for the escaping tap closure; nil = single-stream, in which
        // case the Remote block below is skipped entirely and the callback does
        // exactly the work it did before dual-stream existed.
        let remoteTapChannel = remoteChannel
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self,
                  let mono = AudioBufferProcessor.extractChannel(buffer, channel: channel)
            else { return }

            try? self.file?.write(from: mono)
            // Remote is written from THIS callback so the two files stay
            // sample-aligned, and resampled here so the 16 kHz stream is derived
            // from exactly the same buffer as the office one. It feeds NOTHING
            // that keeps time — not recordingElapsed, not the VAD, not the RMS
            // meter, not the chunk/diar cadence, not ATND. A failed extraction
            // just skips this buffer rather than aborting the office write above.
            var remoteSamples16k: [Float] = []
            if let remoteTapChannel,
               let remoteMono = AudioBufferProcessor.extractChannel(buffer, channel: remoteTapChannel) {
                try? self.remoteFile?.write(from: remoteMono)
                remoteSamples16k = remoteResampler?.resample(remoteMono) ?? []
                // Live captions for the conferencing audio. Fed from here (not
                // from the main-actor hop below) for the same reason the office
                // engine is: `feed` hands the samples to its own write queue, and
                // the caption should not wait on the main actor to be scheduled.
                remoteASR?.feed(remoteSamples16k)
            }
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
                // Remote accumulates in parallel. Empty for a single-stream
                // session, so this is a no-op there.
                if !remoteSamples16k.isEmpty {
                    self.remoteChunkAudio.append(contentsOf: remoteSamples16k)
                    // Second, independent buffer for remote diarization: the two
                    // cadences are separate settings, so one buffer cleared on the
                    // ASR boundary could not also feed the diarization boundary.
                    self.remoteDiarAudio.append(contentsOf: remoteSamples16k)
                }

                if boundary, chunked != nil {
                    self.chunkElapsed = 0
                    // Flush Nemotron too — keeps both services aligned at the
                    // same audio position so replacement is exact.
                    asr?.flush()
                    let windowStart = self.lastChunkBoundary
                    self.pendingChunkWindows.append(windowStart...self.recordingElapsed)
                    self.lastChunkBoundary = self.recordingElapsed
                    self.startChunkFlush(chunked)
                    // Remote rides the SAME boundary — one cadence, so the two
                    // streams' windows stay aligned and comparable. The office
                    // FLUSH (n=0) is queued first and the remote `-2` frame
                    // second; the sidecar is single-threaded and processes its
                    // stdin strictly in order, so they run sequentially with no
                    // new concurrency of our own.
                    //
                    // The remote realtime engine is flushed on the SAME boundary
                    // as the office one, for the same reason: its caption covers
                    // exactly the audio the chunk below is about to confirm.
                    remoteASR?.flush()
                    self.flushRemoteChunk(window: windowStart...self.recordingElapsed,
                                          chunked: chunked)
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
                    // Remote rides the SAME diarization cadence, dispatched as a
                    // second job on the same stdin. One process, two stores: the
                    // sidecar is single-threaded and drains stdin in order, so the
                    // office job above always runs first and the interleaving is
                    // deterministic. No-op for a single-stream session.
                    self.diarizeRemoteLiveChunk(windowStart: diarWindowStart)
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
            remoteFile = nil   // never leave the Remote handle open on a failed start
            remoteRecordingURL = nil
            remoteStreamActive = false
            remoteLastChunkDone = true
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
        modelLoader.nemotronASR?.office.flush() // finalize any trailing speech
        // Remote's tail window is the office tail window — read BEFORE the office
        // branch below advances `lastChunkBoundary`, so both streams' last windows
        // still line up exactly as they did at every live boundary.
        let tailStart = lastChunkBoundary
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
        remoteFile = nil   // both files close here; releasing the AVAudioFile flushes it
        vad = nil
        rms = 0
        isSpeaking = false

        // Remote tail, on the office tail's window. Inert for a single-stream
        // session (`remoteLastChunkDone` is already true and stays true).
        if remoteStreamActive {
            // Flush parity with the office lane above: finalize the remote
            // caption's trailing speech before its tail chunk is queued. Its own
            // opcode, so this resets only the remote buffer in the sidecar.
            modelLoader.nemotronASR?.remote.flush()
            flushRemoteChunk(window: tailStart...max(recordingElapsed, tailStart + 0.01),
                             chunked: modelLoader.chunkedASR)
            startRemoteStopWatchdog()
            // Nothing in flight (idle channel, everything gated as silent) →
            // complete the gate now instead of waiting for a callback that
            // will never come.
            checkRemoteChunksDone()
        }

        // Who spoke when — either append a tail (continue from live labels) or
        // re-diarize the whole recording (best global clustering).
        let finalOn = UserDefaults.standard.object(forKey: "diarization.finalPass") as? Bool ?? true
        let continueOnStop = UserDefaults.standard.object(forKey: "diarization.continueOnStop") as? Bool ?? true
        let willRunStopPass = finalOn && modelLoader.diarization != nil
        // The remote pass is dispatched HERE, before the overlay is built, so the
        // step list knows whether to show a remote-diarization row. Queued ahead
        // of the office stop pass on the same stdin; the sidecar drains it in
        // order, so both run to completion regardless of who is first.
        let willRunRemoteDiar = startRemoteDiarization()

        // Everything below lands asynchronously; block the controls until it does.
        buildStopSteps(willRunStopPass: willRunStopPass,
                       willRunRemoteDiar: willRunRemoteDiar)
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
}
