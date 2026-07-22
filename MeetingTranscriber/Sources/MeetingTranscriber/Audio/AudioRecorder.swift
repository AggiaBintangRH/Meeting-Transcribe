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
        // Word timestamps from the optional forced aligner (nil when alignment
        // is off or failed). Times are CHUNK-RELATIVE seconds exactly as the
        // sidecar sent them; `alignedChunkDuration` is that chunk buffer's
        // length, kept so the conversion to recording time can be sanity-checked
        // against `window`. Consumed by `WordAttribution` in `derivedRows`.
        var words: [ChunkedASRService.AlignedWord]? = nil
        var alignedChunkDuration: Double? = nil
    }

    /// One transcribed Remote (conferencing) chunk. Deliberately NOT a
    /// `TranscriptSegment`: remote text never goes through the office display
    /// pipeline (`derivedRows` → speaker ranges → position gap-fill → word
    /// attribution). It IS speaker-split (phase 4), but against `remoteLiveTurns`
    /// in the remote identity space, and it must never reach the ATND/position
    /// path — so it is a separate collection that only meets office rows at the
    /// final sort in `rebuildDisplayRows`.
    ///
    /// Text only, by design: the transcript comes from the sidecar's `-2`
    /// file-transcribe frame, which calls `transcribe_path()` directly and never
    /// runs the forced aligner — so there are no `words`/`dur` and remote
    /// attribution stays sentence-level. Acceptable: word-exact timing matters
    /// where ATND and pyannote boundaries compete, which is Office-only.
    struct RemoteSegment: Identifiable, Equatable {
        let id = UUID()
        var text: String
        var window: ClosedRange<Double>   // recording-time span (shared clock)
    }

    /// The live Remote caption — what the Remote realtime engine has produced
    /// since the last remote chunk boundary, shown as its own provisional card
    /// exactly like the office `partialTranscript`.
    ///
    /// A tiny type rather than a bare String because the *rule* is the whole
    /// point and it differs from the office one: a Remote FINAL is KEPT on
    /// screen, not cleared. Office can clear on a final because it immediately
    /// turns that final into an unconfirmed segment; Remote has no unconfirmed
    /// segment (see `RemoteSegment` — remote text never enters the office
    /// pipeline), so clearing on the final would blank the caption for the
    /// several seconds the confirmed remote chunk takes to come back. The
    /// caption instead survives until `commit()`, which every terminal outcome
    /// of that chunk calls — transcribed, empty, skipped as silence, or failed.
    struct RemoteCaption: Equatable {
        /// What the view draws; empty means no caption card at all.
        private(set) var text = ""

        /// A realtime result arrived (partial or final — both are just "the best
        /// text so far" for audio no confirmed row covers yet).
        mutating func update(to incoming: String) {
            text = incoming.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        /// This caption's audio is now represented (or deliberately not) by a
        /// confirmed remote row — drop it so nothing stale lingers underneath.
        mutating func commit() {
            text = ""
        }
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
        /// Row came from the Remote stream, not the room. A display flag: it keeps
        /// the two label spaces visually distinct (amber vs teal) and tells the
        /// rename dialog to edit the remote PROFILE name rather than the composed
        /// "Remote Speaker - …" label. The id range, not this flag, is what routes
        /// a rename to the right store.
        var isRemote: Bool = false
    }

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
    /// Which layer the DISPLAY draws labels from — read once per session in
    /// `configurePositionDiarization()`, like every other setting, so it can't
    /// flip mid-recording and leave half the transcript on each policy.
    private var positionSource: PositionSource = .both
    /// Last SEPARATION line written to the position log, so the per-room tau
    /// calibration diagnostic is emitted on change instead of on every rebuild.
    private var lastLoggedSeparation: String?
    private var diarElapsed: Double = 0                        // seconds since the last diar chunk
    private var lastDiarBoundary: Double = 0                   // recording-time where this diar chunk began

    // MARK: Remote stream (dual-stream phase 3)
    //
    // Everything below is inert unless a Remote channel resolved for THIS session
    // (`remoteStreamActive`). The remote side never influences timing: it has no
    // cadence, no VAD, no RMS and no clock of its own — it rides the office chunk
    // boundary and the single `recordingElapsed`, which is the whole reason both
    // inputs must come from one Aggregate Device (see `MicrophoneSettings`).

    /// True when this session resolved a usable Remote channel.
    private var remoteStreamActive = false
    /// 16 kHz remote samples accumulated since the last chunk boundary.
    private var remoteChunkAudio: [Float] = []
    /// Transcribed remote chunks, merged into `displayRows` by start time.
    private var remoteSegments: [RemoteSegment] = []
    /// Remote file-transcribe requests currently in flight.
    private var remotePendingChunks = 0
    /// Last remote failure, shown on the stop step. Never fatal — a remote
    /// problem must not cost the user their office transcript.
    private var remoteChunkError: String?
    /// Stop gate, mirroring `lastChunkDone`. Starts true so a single-stream
    /// session's gate is complete before it is ever consulted.
    private var remoteLastChunkDone = true
    private var remoteStopWatchdog: Task<Void, Never>?

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
    private var remoteRecordingURL: URL?
    /// Remote-space turns collected so far — the remote twin of `liveTurns`.
    /// Ids are already offset by `remoteIDBase` (the sidecar applies it).
    private var remoteLiveTurns: [DiarizationService.Turn] = []
    /// 16 kHz remote samples pending live diarization. Separate from
    /// `remoteChunkAudio` because the diarization cadence is its own setting.
    private var remoteDiarAudio: [Float] = []
    /// Temp chunk WAVs awaiting a remote result. Keyed the same way as
    /// `chunkFileByWindow` but kept apart, since both streams use the SAME
    /// window starts and would otherwise delete each other's files.
    private var remoteChunkFileByWindow: [Double: URL] = [:]
    private var remoteSessionSpeakerIDs = Set<Int>()
    /// Remote-space speaker count. `speakerCount` stays Office-only.
    @Published var remoteSpeakerCount: Int?
    /// Stop gate for the remote final pass. Starts true so a single-stream
    /// session's gate is already complete before it is ever consulted.
    private var remoteFinalDiarDone = true
    private var remoteFinalDiarWatchdog: Task<Void, Never>?
    /// Window start of the remote TAIL chunk the stop gate is waiting on, or nil
    /// when the remote stop pass is not a tail (full pass / no pass). The remote
    /// twin of `awaitingTailWindowStart`; kept apart because both streams use the
    /// SAME window starts and one would otherwise settle the other's gate.
    private var awaitingRemoteTailWindowStart: Double? = nil

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

    // MARK: - Stop processing gate (the blocking overlay)

    /// The legs the overlay lists for this stop, in the order they finish.
    /// `repair` only appears when the feature is on AND its engine loaded —
    /// otherwise there is nothing to wait for.
    private func buildStopSteps(willRunStopPass: Bool, willRunRemoteDiar: Bool = false) {
        var steps = [StopStep(id: "chunk", name: "Transcribing final audio",
                              state: lastChunkDone ? .done : .loading)]
        // Remote only appears for a dual-stream session. Its state is read the
        // same way the chunk step's is, because the remote tail may already have
        // completed (or been gated as silent) before this list is built.
        if remoteStreamActive {
            steps.append(StopStep(id: "remote", name: "Transcribing remote audio",
                                  state: remoteLastChunkDone
                                      ? (remoteChunkError.map { .failed($0) } ?? .done)
                                      : .loading))
        }
        if willRunStopPass {
            steps.append(StopStep(id: "diarize", name: "Identifying speakers", state: .loading))
        }
        // Its own row: Office and Remote are separate identity spaces, so their
        // progress is separate too — and a failed remote pass must read as a
        // remote failure, not as "speakers could not be identified".
        if willRunRemoteDiar {
            steps.append(StopStep(id: "remote-diarize",
                                  name: "Identifying remote speakers", state: .loading))
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
        guard state == .processing, lastChunkDone, remoteLastChunkDone,
              finalDiarDone, remoteFinalDiarDone,
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

        service.onChunkResult = { [weak self] windowStart, turns, stream in
            Task { @MainActor in
                guard let self else { return }
                // Turn times are chunk-local — offset to absolute recording time.
                let absolute = turns.map {
                    DiarizationService.Turn(start: $0.start + windowStart,
                                            end: $0.end + windowStart,
                                            id: $0.id, name: $0.name)
                }
                // The Remote branch returns early ON PURPOSE: it must not touch
                // liveTurns, sessionSpeakerIDs, speakerCount or the office tail
                // gate. Its ids are >= remoteIDBase and belong to the other space.
                guard stream == .office else {
                    if let file = self.remoteChunkFileByWindow.removeValue(forKey: windowStart) {
                        try? FileManager.default.removeItem(at: file)
                    }
                    self.remoteLiveTurns.append(contentsOf: absolute)
                    for turn in absolute { self.remoteSessionSpeakerIDs.insert(turn.id) }
                    self.remoteSpeakerCount = self.remoteSessionSpeakerIDs.count
                    self.rebuildDisplayRows()
                    // Remote tail-mode stop: THIS chunk result is the remote tail —
                    // settle the remote leg of the gate. Matched on the remote
                    // window start only, so an office chunk sharing the same window
                    // can never settle it (and vice versa).
                    if let expected = self.awaitingRemoteTailWindowStart,
                       abs(windowStart - expected) < 0.001 {
                        self.completeRemoteDiarization()
                    }
                    return
                }
                // Clean up the temp chunk file
                if let file = self.chunkFileByWindow.removeValue(forKey: windowStart) {
                    try? FileManager.default.removeItem(at: file)
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

        service.onFinalResult = { [weak self] turns, stream in
            Task { @MainActor in
                guard let self else { return }
                guard stream == .office else {
                    self.applyRemoteFinalSpeakers(turns)
                    self.completeRemoteDiarization()
                    return
                }
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

        service.onError = { [weak self] message, stream in
            Task { @MainActor in
                guard let self else { return }
                // A remote job's failure settles only the remote leg. Remote
                // trouble must never cost the user their office transcript —
                // the same rule the remote chunk path already follows.
                guard stream == .office else {
                    self.dualStreamLog("remote diarization failed: \(message)")
                    self.completeRemoteDiarization(error: message)
                    return
                }
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

    // MARK: - Remote diarization (the second identity space)

    /// Live: hand the remote audio accumulated since the last diarization
    /// boundary to the sidecar as a `remote` job. Mirrors `diarizeLiveChunk`,
    /// with no position/ATND involvement of any kind.
    private func diarizeRemoteLiveChunk(windowStart: Double) {
        guard remoteStreamActive, let service = modelLoader.diarization else { return }
        let liveOn = UserDefaults.standard.object(forKey: "diarization.live") as? Bool ?? true
        guard liveOn else {
            // Exactly `diarizeLiveChunk`'s rule, and for the same reason: with live
            // labels off but "continue from live labels (tail only)" on, the whole
            // recording becomes the single tail diarized at stop, so the accumulated
            // audio must be KEPT. (Before the 2026-07-22 tail change this always
            // cleared, because the remote stop pass was always a full re-diarization
            // of the remote WAV — see `startRemoteDiarization`.)
            let continueOnStop = UserDefaults.standard.object(forKey: "diarization.continueOnStop") as? Bool ?? true
            if !(continueOnStop && !liveOn) { remoteDiarAudio = [] }
            return
        }
        let samples = remoteDiarAudio
        remoteDiarAudio = []
        guard samples.count > 16_000 else { return }   // skip chunks under 1s
        dispatchRemoteDiarChunk(samples: samples, windowStart: windowStart, service: service)
    }

    /// Shared: write a chunk of 16 kHz REMOTE samples to a temp WAV off-thread and
    /// hand it to the sidecar as a `remote` job. The remote twin of
    /// `dispatchDiarChunk` — live calls pass no failure handler (silent, as
    /// before); the stop-time tail passes one so the remote gate still settles.
    private func dispatchRemoteDiarChunk(samples: [Float], windowStart: Double,
                                         service: DiarizationService,
                                         onDispatchFailure: (() -> Void)? = nil) {
        let detectOverlap = UserDefaults.standard.object(forKey: "diarization.detectOverlap") as? Bool ?? true
        Task.detached(priority: .utility) { [weak self] in
            guard let url = Self.writeTempWAV(samples: samples, prefix: "remote-diar") else {
                await MainActor.run { onDispatchFailure?() }
                return
            }
            await MainActor.run { [weak self] in
                self?.remoteChunkFileByWindow[windowStart] = url
            }
            service.diarizeChunk(audio: url, windowStart: windowStart,
                                 exclusive: !detectOverlap, stream: .remote)
        }
    }

    /// What the remote stop pass is for this session. Pure, so the branch itself
    /// is unit-testable without an engine, a sidecar or an Aggregate Device.
    enum RemoteStopMode: Hashable {
        case none   // no remote pass at all — the gate is never taken
        case tail   // diarize only the audio since the last remote live boundary
        case full   // re-diarize the whole Remote WAV
    }

    /// The stop-pass decision. Remote now honours `diarization.continueOnStop`
    /// exactly as Office does — see `startRemoteDiarization` for WHY this reverses
    /// the phase-4 rule.
    nonisolated static func remoteStopMode(finalPass: Bool,
                                           continueOnStop: Bool,
                                           remoteStreamActive: Bool,
                                           hasDiarizationService: Bool,
                                           hasRemoteRecording: Bool,
                                           tailSamples: Int) -> RemoteStopMode {
        guard finalPass, remoteStreamActive, hasDiarizationService else { return .none }
        if continueOnStop {
            // < 1 s of tail is not worth a job — the same early-out (and the same
            // 16 000-sample threshold) as `diarizeTailChunk`. Returning `.none`
            // rather than dispatching keeps the stop gate untaken, so there is
            // nothing left to settle.
            return tailSamples > 16_000 ? .tail : .none
        }
        return hasRemoteRecording ? .full : .none
    }

    /// At stop: the remote twin of the office stop branch, and it now honours the
    /// SAME `diarization.continueOnStop` setting.
    ///
    /// Phase 4 made this pass ALWAYS a full re-diarization of the Remote WAV, on
    /// the reasoning that a clean separate waveform clusters better globally. The
    /// owner overruled that on 2026-07-22 for a stronger reason: LABEL STABILITY.
    /// A full pass re-embeds voices the live passes already enrolled, and one real
    /// session produced a second profile (R2) for a voice that had matched profile
    /// 1 at sim=0.89 four seconds earlier — one person shown as two speakers, with
    /// the transcript rendering the final pass's mapping. Why that embedding
    /// collapsed below SIM_THRESHOLD=0.5 against a centroid it had just matched at
    /// 0.89 is UNEXPLAINED — nobody has accounted for it, so do not assume it was
    /// understood. Tail mode sidesteps it structurally: the tail never re-embeds
    /// already-enrolled voices, so it cannot mint a duplicate profile for them.
    ///
    /// Returns whether a pass was actually dispatched, so `stop()` can decide
    /// whether the overlay gets a remote-diarization row. This MUST stay decidable
    /// synchronously — `stop()` calls it before `buildStopSteps`.
    @discardableResult
    private func startRemoteDiarization() -> Bool {
        let d = UserDefaults.standard
        let finalOn = d.object(forKey: "diarization.finalPass") as? Bool ?? true
        let continueOnStop = d.object(forKey: "diarization.continueOnStop") as? Bool ?? true
        let mode = Self.remoteStopMode(finalPass: finalOn,
                                       continueOnStop: continueOnStop,
                                       remoteStreamActive: remoteStreamActive,
                                       hasDiarizationService: modelLoader.diarization != nil,
                                       hasRemoteRecording: remoteRecordingURL != nil,
                                       tailSamples: remoteDiarAudio.count)
        guard let service = modelLoader.diarization, mode != .none else {
            // Nothing dispatched → the gate was never taken (`remoteFinalDiarDone`
            // is still true) and no overlay row is added. Drop any pending tail
            // audio so it cannot outlive the session.
            remoteDiarAudio = []
            return false
        }
        remoteFinalDiarDone = false
        switch mode {
        case .tail:  startRemoteTailDiarization(service: service)
        case .full:  startRemoteFullDiarization(service: service)
        case .none:  break   // unreachable, guarded above
        }
        return true
    }

    /// `continueOnStop == true`: diarize only the remote audio accumulated since
    /// the last remote live boundary, as a `chunk` job on the remote stream, so
    /// every label the live passes assigned survives Stop untouched. Mirrors
    /// `diarizeTailChunk`, including its window-start reasoning.
    private func startRemoteTailDiarization(service: DiarizationService) {
        let samples = remoteDiarAudio
        remoteDiarAudio = []
        // Same rule as the office tail: with live labels on, the pending audio
        // began at the last live diarization boundary — which is the SAME
        // `lastDiarBoundary`, because remote rides the office diarization cadence
        // (see `diarizeRemoteLiveChunk`'s call site). With live off (+
        // continueOnStop) nothing was ever cleared, so this is the whole recording
        // and begins at 0.
        let liveOn = UserDefaults.standard.object(forKey: "diarization.live") as? Bool ?? true
        let windowStart = liveOn ? lastDiarBoundary : 0
        awaitingRemoteTailWindowStart = windowStart
        // The tail is a chunk job, not a full pass, so it is bounded by the office
        // tail's limit rather than the recording length — but it still queues
        // BEHIND the office stop job on one stdin, hence the doubling.
        startRemoteDiarWatchdog(seconds: 240, message: "Remote tail diarization timed out")
        dispatchRemoteDiarChunk(samples: samples, windowStart: windowStart, service: service,
                                onDispatchFailure: { [weak self] in
                                    self?.dualStreamLog("could not write remote tail audio for diarization")
                                    self?.completeRemoteDiarization(
                                        error: "Could not write remote tail audio for diarization")
                                })
    }

    /// `continueOnStop == false`: one batch pass over the whole Remote WAV — the
    /// remote twin of `startDiarization`, and phase 4's original behaviour.
    private func startRemoteFullDiarization(service: DiarizationService) {
        guard let recording = remoteRecordingURL else {
            // `remoteStopMode` already proved this non-nil; belt-and-braces so the
            // gate can never be left open by a future edit.
            completeRemoteDiarization(error: "Remote recording is missing")
            return
        }
        let numSpeakers = UserDefaults.standard.integer(forKey: "diarization.numSpeakers")
        let detectOverlap = UserDefaults.standard.object(forKey: "diarization.detectOverlap") as? Bool ?? true
        service.diarizeFinal(audio: recording, numSpeakers: numSpeakers,
                             exclusive: !detectOverlap, stream: .remote)
        // Same scale rule as the office final pass: the remote job is queued BEHIND
        // the office one on a single stdin, so it can legitimately wait out the
        // office pass before it even starts.
        startRemoteDiarWatchdog(seconds: max(180, recordingElapsed) * 2,
                                message: "Remote diarization timed out")
    }

    /// One watchdog for both remote stop modes — whichever path stalls, the gate
    /// still settles and the overlay still drops.
    private func startRemoteDiarWatchdog(seconds: Double, message: String) {
        remoteFinalDiarWatchdog?.cancel()
        remoteFinalDiarWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.remoteFinalDiarDone else { return }
                self.completeRemoteDiarization(error: message)
            }
        }
    }

    /// The remote final result replaces the running remote set, exactly as
    /// `applyFinalSpeakers` does for Office — but into `remoteLiveTurns`, so the
    /// two spaces never share a collection.
    private func applyRemoteFinalSpeakers(_ turns: [DiarizationService.Turn]) {
        remoteSpeakerCount = Set(turns.map(\.id)).count
        remoteLiveTurns = turns
        rebuildDisplayRows()
    }

    /// Settle the remote leg of the stop gate exactly once. Idempotent; every
    /// remote diarization exit path (result, error, timeout) calls it.
    private func completeRemoteDiarization(error: String? = nil) {
        remoteFinalDiarWatchdog?.cancel()
        remoteFinalDiarWatchdog = nil
        awaitingRemoteTailWindowStart = nil
        guard !remoteFinalDiarDone else { return }
        remoteFinalDiarDone = true
        setStopStep("remote-diarize", error.map { .failed($0) } ?? .done)
        checkStopProcessingDone()
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
        lastLoggedSeparation = nil
        let d = UserDefaults.standard
        // Read once, here — the display policy is fixed for the whole session.
        // Reset to `both` first so a session that bails out below (feature off /
        // ATND down) is on the default policy, matching its nil diarizer.
        positionSource = .both
        guard d.bool(forKey: "atnd.position.enabled"),
              ATNDBeamService.shared.state == .listening else { return }

        let tauDeg = d.object(forKey: "atnd.position.tauDeg") as? Double ?? 15
        let smoothingMs = d.object(forKey: "atnd.position.smoothingMs") as? Double ?? 400
        let mode: PositionDiarizer.Mode =
            (d.string(forKey: "atnd.position.mode") == "enrollment") ? .enrollment : .firstCome
        positionSource = PositionSource.current(d)

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
            // One fill PER SPAN — a beam change mid-gap splits into multiple rows.
            // The boundary timeline TILES the gap from its first boundary onward:
            // every stretch of elapsed time belongs to whoever the beam had last
            // settled on, so a fill can no longer leave a hole in the middle of a
            // gap. That is the whole point of the switch away from reconstructed
            // turns — the old density gate (minDurationSec/minSamples) DISCARDED a
            // short run, and discarding a run discarded a stretch of TIME, whose
            // words then had no range to land in and rendered as SPEAKER UNKNOWN.
            // Debounce still decides WHETHER a boundary exists, upstream in
            // `ClusterChangeDetector`; it can no longer delete elapsed time.
            //
            // `labeledSpans` already clips to the queried range, so no snapping,
            // hole-closing or dominant-label fallback is needed here any more.
            let spans = pos.labeledSpans(in: a...b)
            if spans.isEmpty {
                // The only way a gap yields nothing now: it lies entirely BEFORE
                // the first beam boundary of the session (pre-speech silence, or
                // ATND not streaming yet). Leaving it UNKNOWN is correct — there is
                // no talker to attribute it to.
                positionLog("SKIP gap=[\(fmt3(a))..\(fmt3(b))] samples=\(pos.sampleCount(in: a...b)) no-spans")
                continue
            }
            for s in spans {
                positionLog("FILL gap=[\(fmt3(a))..\(fmt3(b))] -> \(s.id):\(s.name) [\(fmt3(s.start))..\(fmt3(s.end))]")
                fills.append((s.start, s.end, s.id, s.name))
            }
        }
        // Calibration instrument, not noise: the owner tunes `atnd.position.tauDeg`
        // per room ("the tables differ in every room"), and this is the only line
        // that shows how far apart the seats actually landed — e.g. a phantom
        // cluster 16.2° from Speaker 1 against a 15° threshold while the real seats
        // sit 33–41° apart. Logged only when it CHANGES: `positionGapFill` runs once
        // per segment per rebuild, so logging it unconditionally repeated the same
        // line after every FILL.
        if !fills.isEmpty {
            let tau = UserDefaults.standard.object(forKey: "atnd.position.tauDeg") as? Double ?? 15
            let line = "SEPARATION \(pos.separationDescription()) (threshold \(Int(tau))°)"
            if line != lastLoggedSeparation {
                lastLoggedSeparation = line
                positionLog(line)
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
        // Office and Remote are two independent label spaces sharing one clock;
        // they only ever meet here, at the final sort. With no remote segments
        // the merge returns `rows` untouched — the single-stream path is exactly
        // what it was.
        displayRows = Self.mergeRowsByStartTime(
            office: rows,
            remote: Self.remoteRows(remoteSegments, turns: remoteLiveTurns))
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
        // The one switch on the display source (see `PositionSource.plan`):
        //   both       → pyannote shows, ATND fills pyannote's own complement
        //   atnd       → nothing pyannote shows, empty coverage ⇒ ATND tiles it all
        //   pyannote   → pyannote shows, no gap-fill runs at all
        //   atndTiming → ATND tiles it all, then each span takes the identity of
        //                the pyannote turn it overlaps most (`relabelFromPyannote`)
        // `ranges` itself is untouched in every mode, and the position ids the
        // fills carry never leave `filled` — liveTurns/overlapRegions/
        // speakerCount/SpeakerProfileStore stay pure pyannote throughout.
        let plan = positionSource.plan(pyannoteRanges: ranges)
        // Off/silent ATND → fills is [] regardless of the source.
        let fills = plan.gapFillCoverage.map { positionGapFill(window: window, covered: $0) } ?? []
        var filled = (plan.displayRanges + fills).sorted { $0.start < $1.start }
        // Timing from ATND, identity from pyannote: rename in place, boundaries
        // untouched. Only ids move here, and only pyannote → display, never back.
        if plan.relabelFromPyannote {
            filled = PositionRelabel.fromPyannote(filled, pyannote: ranges)
        }

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
                return Self.assignSentences(seg.text, window: window, ranges: filled,
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
        // Word-exact path: when the aligner ran, each word goes to the turn that
        // covers it in time instead of to the turn its character offset guesses.
        // Any failed sanity gate returns nil → the estimate below, unchanged.
        if let words = seg.words,
           let pieces = WordAttribution.attribute(text: seg.text, words: words,
                                                  chunkDuration: seg.alignedChunkDuration,
                                                  window: window, ranges: filled,
                                                  log: { self.positionLog($0) }) {
            return pieces.enumerated().map { i, p in
                let overlapped = regions.contains { max($0.start, p.start) < min($0.end, p.end) }
                return SpeakerUtterance(id: "\(seg.id.uuidString)-\(i)", speaker: p.name,
                                        speakerID: p.id, start: p.start, end: p.end,
                                        text: p.text, confirmed: true, overlapped: overlapped)
            }
        }
        return Self.assignSentences(seg.text, window: window, ranges: filled,
                               segID: seg.id.uuidString, regions: regions)
    }

    /// Assign whole sentences to speakers. Each sentence is placed in time by its
    /// character position in the chunk, then handed to the speaker turn it most
    /// overlaps — so text is never cut mid-sentence onto the wrong speaker.
    /// Consecutive sentences by the same speaker merge into one row.
    ///
    /// Pure and `static` so the Remote stream can reuse it verbatim with its own
    /// `ranges` (built from `remoteLiveTurns`) — the split logic is identical,
    /// only the identity space differs.
    nonisolated static func assignSentences(_ text: String,
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
    nonisolated static func splitSentences(_ text: String) -> [String] {
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
    /// The office view: always `liveTurns`, never the remote ones.
    private func speakerRanges(in window: ClosedRange<Double>)
        -> [(start: Double, end: Double, id: Int, name: String)] {
        Self.speakerRanges(in: window, turns: liveTurns)
    }

    /// The same clipping/merging, parameterised by the turn set — so the Remote
    /// stream can reuse it against `remoteLiveTurns` without any chance of the
    /// two identity spaces meeting (each call sees exactly one of them).
    nonisolated static func speakerRanges(in window: ClosedRange<Double>,
                                          turns: [DiarizationService.Turn])
        -> [(start: Double, end: Double, id: Int, name: String)] {
        let clipped = turns.compactMap { t -> (start: Double, end: Double, id: Int, name: String)? in
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
    ///
    /// THREE-WAY on the id range alone — the whole point of the disjoint bases
    /// (pyannote < 10_000 ≤ remote < 100_000 ≤ position). A position id must
    /// never reach the Python-owned profile stores at all; a remote id must
    /// never reach profiles.json, and an office id never profiles-remote.json.
    func renameSpeaker(id: Int, to name: String) {
        if id >= PositionDiarizer.positionIDBase {
            positionDiarizer?.rename(clusterID: id - PositionDiarizer.positionIDBase, to: name)
        } else {
            // `SpeakerProfileStore` picks the file and strips `remoteIDBase`
            // itself, so the file choice cannot drift from the id range.
            SpeakerProfileStore.rename(id: id, to: name)
        }
        // Only ONE of the two turn collections can contain this id (their ranges
        // are disjoint), so mapping both is safe and keeps the branch out.
        liveTurns = liveTurns.map {
            $0.id == id
                ? DiarizationService.Turn(start: $0.start, end: $0.end, id: $0.id, name: name)
                : $0
        }
        remoteLiveTurns = remoteLiveTurns.map {
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

    // MARK: - Dual-stream (Office + Remote) helpers

    /// Final validation of the Remote channel against the format the engine is
    /// actually delivering, run after `MicrophoneSettings.resolve` has already
    /// checked it against the device's *advertised* channel count.
    ///
    /// The re-check exists because of Aggregate Devices: the owner's Office array
    /// and the loopback input are combined in Audio MIDI Setup, and the aggregate
    /// can present a different channel layout than the saved selection assumed
    /// (a sub-device unplugged, reordered, or not yet running). Returning nil
    /// degrades to single-stream — the same reject-don't-relocate rule as
    /// `MicrophoneSettings.resolve`: silently moving Remote to another channel
    /// would record whatever happens to be there without telling anyone.
    ///
    /// - Parameters:
    ///   - wanted: the resolved `MicrophoneSettings.remoteChannel`, or nil.
    ///   - officeChannel: the office channel ALREADY clamped to the live format.
    ///   - liveChannelCount: `input.outputFormat(forBus: 0).channelCount`.
    nonisolated static func resolveRemoteChannel(_ wanted: Int?,
                                                 officeChannel: Int,
                                                 liveChannelCount: Int) -> Int? {
        guard let wanted, wanted >= 0 else { return nil }
        guard wanted < liveChannelCount else { return nil }   // aggregate presents fewer channels
        guard wanted != officeChannel else { return nil }     // one channel cannot be both roles
        return wanted
    }

    /// Remote file URL derived from the Office one: `meeting-<stamp>.wav` →
    /// `meeting-<stamp>-remote.wav`, same directory, same stamp. Derived rather
    /// than re-formatted from `Date()` so the pair can never carry two stamps.
    nonisolated static func remoteURL(forOffice office: URL) -> URL {
        let ext = office.pathExtension
        let base = office.deletingPathExtension().lastPathComponent
        let name = ext.isEmpty ? "\(base)-remote" : "\(base)-remote.\(ext)"
        return office.deletingLastPathComponent().appendingPathComponent(name)
    }

    // MARK: Remote stream — transcription, gating and rows

    /// Prefix every Remote row's speaker label carries. Office and Remote are
    /// two SEPARATE identity spaces on purpose (owner requirement): office
    /// "Speaker 1" and remote "R1" are different people, and the label must make
    /// that impossible to misread.
    nonisolated static let remoteNamePrefix = "Remote Speaker - "

    /// Label a Remote row shows before (or without) remote diarization: the
    /// stream is known to be "not the room" and nothing more.
    nonisolated static let remoteSpeakerLabel = remoteNamePrefix + "Speaker Unknown"

    /// Compose the displayed Remote label from a remote profile name ("R1" →
    /// "Remote Speaker - R1").
    nonisolated static func remoteDisplayName(_ name: String) -> String {
        remoteNamePrefix + name
    }

    /// Inverse of `remoteDisplayName`, for the rename dialog: the user edits the
    /// PROFILE name ("R1"), not the composed row label, so the prefix is not
    /// saved into profiles-remote.json and then re-prefixed on the next rebuild.
    nonisolated static func remoteBaseName(_ display: String) -> String {
        display.hasPrefix(remoteNamePrefix)
            ? String(display.dropFirst(remoteNamePrefix.count))
            : display
    }

    /// Offset the sidecar adds to remote-local speaker ids, mirroring the proven
    /// `PositionDiarizer.positionIDBase = 100_000` split: a downstream consumer
    /// tells the three label spaces apart with one integer comparison
    /// (pyannote < 10_000 ≤ remote < 100_000 ≤ position). It is what keeps a
    /// remote id out of profiles.json and an office id out of profiles-remote.json
    /// — see `renameSpeaker` and `SpeakerProfileStore.Space`.
    nonisolated static let remoteIDBase = 10_000

    /// RMS below which a remote chunk is treated as silence and never sent to the
    /// sidecar. An idle conferencing channel is otherwise a standing ~14 % GPU
    /// cost every chunk interval for a guaranteed-empty transcript (measured duty
    /// per 30 s chunk on this M4: Qwen3 4.3 s, Whisper 4.2 s, Granite 5.6 s).
    ///
    /// 0.004 ≈ −48 dBFS: a digital loopback carrying nothing sits at or very near
    /// 0.0, and even room-noise-through-a-codec stays far below this, while
    /// ordinary speech is an order of magnitude above it. Erring low is the safe
    /// direction — a false "not silent" costs one wasted transcription, a false
    /// "silent" would drop real speech.
    nonisolated static let remoteSilenceRMS: Float = 0.004

    /// Why this remote chunk is not worth transcribing, or nil to go ahead.
    /// Pure, so the gate can be tested without audio hardware or a sidecar.
    nonisolated static func remoteChunkSkipReason(_ samples: [Float],
                                                  threshold: Float = remoteSilenceRMS) -> String? {
        // Under half a second there is nothing an ASR model can usefully say, and
        // a stray fragment would still cost a full round trip.
        guard samples.count >= 8_000 else {
            return "under 0.5s (\(samples.count) samples)"
        }
        let level = AudioBufferProcessor.rms(samples)
        guard level >= threshold else {
            return String(format: "near-silent (rms %.5f < %.5f)", level, threshold)
        }
        return nil
    }

    /// The display rows the remote segments render as, split by the REMOTE
    /// diarization turns (`remoteLiveTurns`) — never by `liveTurns`.
    ///
    /// The splitting machinery is the office one (`speakerRanges` →
    /// `assignSentences`), parameterised by the remote turns; what is
    /// deliberately absent is everything position-shaped: no `positionGapFill`,
    /// no `PositionSource`, no `PositionDiarizer`, no ATND. The beam describes
    /// the ROOM, so it says nothing whatsoever about the conferencing stream.
    /// `regions: []` for the same reason overlap repair is Office-only — remote
    /// rows are never tagged from office overlap windows.
    ///
    /// With no remote turns (feature idle, or the remote pass has not landed
    /// yet) each segment stays ONE row labelled `remoteSpeakerLabel` with no
    /// speaker id — exactly the phase-3 behaviour.
    ///
    /// Sorted by start time here rather than relying on append order: segments
    /// are appended when their transcription lands, which is chronological today
    /// (one sidecar, one queue) but is not something the merge should depend on.
    nonisolated static func remoteRows(_ segments: [RemoteSegment],
                                       turns: [DiarizationService.Turn] = [])
        -> [SpeakerUtterance] {
        segments.sorted { $0.window.lowerBound < $1.window.lowerBound }
            .flatMap { seg -> [SpeakerUtterance] in
                let text = seg.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { return [] }
                let ranges = speakerRanges(in: seg.window, turns: turns)
                guard !ranges.isEmpty else {
                    return [SpeakerUtterance(id: seg.id.uuidString,
                                             speaker: remoteSpeakerLabel, speakerID: nil,
                                             start: seg.window.lowerBound,
                                             end: seg.window.upperBound,
                                             text: text, confirmed: true, overlapped: false,
                                             isRemote: true)]
                }
                return assignSentences(text, window: seg.window, ranges: ranges,
                                       segID: seg.id.uuidString, regions: [])
                    .map { row in
                        var row = row
                        row.speaker = row.speaker.map(remoteDisplayName)
                        row.isRemote = true
                        return row
                    }
            }
    }

    /// Interleave office and remote rows by start time into one transcript.
    ///
    /// A merge walk, NOT a sort of the combined list: office rows keep their own
    /// order exactly as `rebuildDisplayRows` produced it, and each remote row is
    /// placed before the first office row that starts later. Re-sorting the whole
    /// list would silently reorder office rows the office pipeline placed
    /// deliberately — separation-debug segments, for instance, carry real windows
    /// but are pinned to the end on purpose.
    ///
    /// A tie puts Office first (the room is the primary record). A row with no
    /// start time — an unconfirmed realtime segment with no window yet — sorts as
    /// +∞, the same convention `insertPinnedSorted` uses, so remote rows go ahead
    /// of it; it still keeps its place among the office rows.
    ///
    /// With no remote rows this returns `office` untouched: the single-stream
    /// transcript is not re-sorted, re-ordered or otherwise disturbed by this
    /// feature existing. `remote` is expected sorted (see `remoteRows`).
    nonisolated static func mergeRowsByStartTime(office: [SpeakerUtterance],
                                                 remote: [SpeakerUtterance]) -> [SpeakerUtterance] {
        guard !remote.isEmpty else { return office }
        func key(_ r: SpeakerUtterance) -> Double { r.start ?? .greatestFiniteMagnitude }
        var out: [SpeakerUtterance] = []
        out.reserveCapacity(office.count + remote.count)
        var i = 0, j = 0
        while i < office.count, j < remote.count {
            if key(remote[j]) < key(office[i]) {
                out.append(remote[j]); j += 1
            } else {
                out.append(office[i]); i += 1
            }
        }
        out.append(contentsOf: office[i...])
        out.append(contentsOf: remote[j...])
        return out
    }

    /// Startup refusal for the one dual-stream configuration that cannot work:
    /// Voxtral needs ~27 s to transcribe a 30 s chunk (Qwen3 4.3 s, Whisper 4.2 s,
    /// Granite 5.6 s, all measured on this M4), i.e. ~90 % duty for a SINGLE
    /// stream. A second stream pushes it past 100 %: chunk N+1 arrives before N
    /// has finished and the sidecar's queue grows without bound for the rest of
    /// the meeting. Nil = start normally.
    ///
    /// Refusal rather than a silent fallback to another model: the owner selects
    /// chunked models deliberately, on measured WER, and quietly substituting one
    /// would make the transcript's provenance a lie.
    nonisolated static func dualStreamRefusalMessage(remoteChannel: Int?,
                                                     chunkedModelID: String) -> String? {
        guard remoteChannel != nil, chunkedModelID == "voxtral" else { return nil }
        return "Voxtral cannot transcribe two streams. It needs about 27 s per 30 s chunk "
             + "(Qwen3 4.3 s, Whisper 4.2 s, Granite 5.6 s), so the Remote stream would fall "
             + "permanently behind. Pick another chunked model in Settings → Models → Chunked, "
             + "or turn the Remote channel off in Settings → Microphone."
    }

    /// Hand the remote audio accumulated since the last boundary to the chunked
    /// sidecar's file-transcribe path, unless the silence gate rejects it.
    /// Always clears `remoteChunkAudio` — a skipped chunk's audio is dropped, not
    /// carried into the next window, so remote windows keep matching office ones.
    private func flushRemoteChunk(window: ClosedRange<Double>, chunked: ChunkedASRService?) {
        guard remoteStreamActive else { return }
        let samples = remoteChunkAudio
        remoteChunkAudio = []
        guard let chunked else { return }
        if let reason = Self.remoteChunkSkipReason(samples) {
            dualStreamLog("SKIP remote [\(fmt(window.lowerBound))-\(fmt(window.upperBound))] \(reason)")
            // No confirmed row is coming for this window, so nothing would ever
            // replace the caption — drop it rather than leave it hanging under
            // the transcript for the whole of the next chunk interval.
            remoteCaption.commit()
            return
        }
        remotePendingChunks += 1
        Task { [weak self] in
            guard let self else { return }
            let url = await Task.detached(priority: .utility) {
                Self.writeTempWAV(samples: samples, prefix: "remote-chunk")
            }.value
            guard let url else {
                self.remoteChunkError = "Could not write remote audio for transcription"
                self.dualStreamLog("FAIL remote [\(self.fmt(window.lowerBound))-"
                                   + "\(self.fmt(window.upperBound))] could not write a temp WAV")
                self.remoteCaption.commit()   // nothing will confirm this window
                self.finishRemoteChunk()
                return
            }
            do {
                let text = try await chunked.transcribeFile(path: url.path)
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    self.dualStreamLog("remote [\(self.fmt(window.lowerBound))-"
                                       + "\(self.fmt(window.upperBound))] empty transcript")
                } else {
                    self.remoteSegments.append(RemoteSegment(text: trimmed, window: window))
                    self.rebuildDisplayRows()
                    self.dualStreamLog("remote [\(self.fmt(window.lowerBound))-"
                                       + "\(self.fmt(window.upperBound))] \(trimmed)")
                }
            } catch {
                // Same philosophy as `onChunkError`: log and carry on. Remote
                // trouble must never cost the user their office transcript.
                self.remoteChunkError = "Some remote audio could not be transcribed — "
                                      + "see logs/dual-stream.log"
                self.dualStreamLog("FAIL remote [\(self.fmt(window.lowerBound))-"
                                   + "\(self.fmt(window.upperBound))] \(error.localizedDescription)")
            }
            try? FileManager.default.removeItem(at: url)
            // This window is settled either way (transcribed, empty or failed):
            // the caption has served its purpose and must not outlive the row —
            // or the absence of one — that answers for the same audio.
            self.remoteCaption.commit()
            self.finishRemoteChunk()
        }
    }

    /// One remote request settled (either way) — maybe complete the stop gate.
    private func finishRemoteChunk() {
        remotePendingChunks = max(0, remotePendingChunks - 1)
        checkRemoteChunksDone()
    }

    /// The remote leg of the stop gate, mirroring `checkLastChunkDone`.
    /// Idempotent; every remote exit path calls it.
    private func checkRemoteChunksDone() {
        guard stopped, remotePendingChunks == 0, !remoteLastChunkDone else { return }
        remoteLastChunkDone = true
        remoteStopWatchdog?.cancel()
        remoteStopWatchdog = nil
        setStopStep("remote", remoteChunkError.map { .failed($0) } ?? .done)
        checkStopProcessingDone()
    }

    /// Remote can never hold the blocking stop overlay hostage. Each request
    /// already has the sidecar client's own 120 s timeout; this is the backstop
    /// for anything that never resolves at all, on the same pattern as
    /// `chunkWatchdog` / `finalDiarWatchdog`.
    private func startRemoteStopWatchdog() {
        remoteStopWatchdog?.cancel()
        remoteStopWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(180))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.remoteLastChunkDone else { return }
                let message = "Remote transcription timed out — see logs/dual-stream.log"
                self.remoteChunkError = message
                self.remotePendingChunks = 0
                self.remoteLastChunkDone = true
                self.remoteStopWatchdog = nil
                self.dualStreamLog(message)
                self.setStopStep("remote", .failed(message))
                self.checkStopProcessingDone()
            }
        }
    }

    /// Append a line to logs/dual-stream.log. Mirrors `overlapLog`: one line per
    /// decision, so a Remote stream that silently degraded to single-stream on the
    /// owner's machine is diagnosable after the fact.
    private func dualStreamLog(_ message: String) {
        let dir = PythonRuntime.dataDir.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("dual-stream.log")
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

    /// Write 16 kHz mono float samples to a temp WAV (chunk diarization, and the
    /// remote chunk handed to the sidecar's file-transcribe frame — `prefix` only
    /// names the file, so the two are told apart in the temp dir).
    /// Pure/self-contained, so it runs off the main actor from the detached task.
    private nonisolated static func writeTempWAV(samples: [Float],
                                                 prefix: String = "diar-chunk") -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString).wav")
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
