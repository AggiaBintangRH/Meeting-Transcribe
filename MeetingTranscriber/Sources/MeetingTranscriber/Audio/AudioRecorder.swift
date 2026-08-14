import AVFoundation
import Foundation

/// Records from the microphone + channel chosen in Settings → Microphone.
/// Publishes live RMS of the selected channel and saves a mono WAV.
/// Transcription models will hook into `onBuffer` later.
@MainActor
final class AudioRecorder: ObservableObject {

    enum State { case idle, preparing, recording, processing }

    @Published var state: State = .idle

    /// A meeting has been recorded and every stop pass has finished (owner,
    /// 2026-08-10). While this is true the mic is LOCKED and the two meeting
    /// actions are offered instead: `Start Over` clears the panel and unlocks the
    /// mic; `Export to PDF` writes the transcript out.
    ///
    /// A FLAG, NOT `state == .idle && !segments.isEmpty` — which is what this
    /// started as, and it was wrong in the one case that matters most. A meeting
    /// that transcribed NOTHING is still a finished meeting, and it is exactly
    /// then that the user needs Start Over; deriving from `segments` would have
    /// left them with a dead panel and a mic that pretends the session never
    /// happened. Export handles the empty case on its own (`canExport`).
    ///
    /// Raised by ALL THREE `.processing` → `.idle` transitions, which is more than
    /// it looks and was nearly missed:
    ///   1. `checkStopProcessingDone` — every leg landed. The happy path.
    ///   2. `startStopWatchdog` — the backstop fired and marked legs "timed out".
    ///   3. `continueInBackground` — the user chose to stop waiting; work carries on.
    /// Two and three still leave a RECORDED MEETING on screen and an unblocked
    /// app, so skipping them would leave the mic live over a transcript the user
    /// never exported — and a new recording would collide with passes still in
    /// flight in case 3.
    ///
    /// The other five `state = .idle` assignments are START failures (permission
    /// denied, refused configuration, `beginCapture` threw). Those never recorded
    /// anything and must NOT raise this.
    @Published private(set) var meetingFinished = false

    /// Is there anything to export? Export is offered on a finished meeting even
    /// when it is empty, but pressing it then would write a blank document, so the
    /// button is disabled and the user is told why (owner, 2026-08-10).
    var canExport: Bool { !displayRows.isEmpty }

    /// Raise `meetingFinished` from the stop-gate extension.
    ///
    /// A METHOD rather than a looser access level: `private(set)` keeps the setter
    /// in this file, and widening it to internal would let any of the twenty-odd
    /// files in `Audio/` set the flag from anywhere. The three legitimate callers
    /// all live in `AudioRecorder+StopGate.swift` and all mean the same thing, so
    /// they get one named door instead.
    ///
    /// Idempotent by construction — every caller is already on a path that runs
    /// once per stop, and setting `true` twice is harmless anyway.
    func markMeetingFinished() { meetingFinished = true }

    @Published var rms: Float = 0.0
    @Published var isSpeaking = false
    @Published var vadEnabled = true
    @Published var activeDeviceName: String?
    @Published var lastRecordingURL: URL?
    @Published var errorMessage: String?

    /// The `errorMessage` the user has already read and closed.
    ///
    /// The popup is driven by comparing the CURRENT message against this one
    /// rather than by a `hasBeenDismissed` Bool, and that is the whole point: a
    /// Bool would have to be reset at every one of the nine sites that assign
    /// `errorMessage`, and a site added later would set an error the popup then
    /// refused to show — a failure with no error and no trace, which is the
    /// shape this codebase treats as worst. Comparing the text needs nothing at
    /// the assignment sites: a DIFFERENT error re-opens the popup on its own.
    ///
    /// The two places that clear `errorMessage` clear this with it, so pressing
    /// Start again and hitting the SAME wall (a denied microphone, say) shows
    /// the popup again instead of failing silently.
    @Published var dismissedErrorMessage: String?

    /// Whether the error popup is on screen. Never true while `.preparing`:
    /// `LoadingOverlayView` is already up then and owns startup refusals through
    /// `ModelLoader.failureMessage`, so two overlays would stack on one failure.
    var showsErrorPopup: Bool {
        guard state != .preparing, let message = errorMessage else { return false }
        return message != dismissedErrorMessage
    }

    func dismissError() { dismissedErrorMessage = errorMessage }

    /// Set when the user closes a stop panel that ended in red. Reset only by
    /// `clearVisibleMeetingState`, i.e. by the next meeting or Start Over —
    /// which is also the one place `stopSteps` is emptied, so the flag and the
    /// rows it describes can never disagree about which meeting they belong to.
    @Published var stopFailureAcknowledged = false

    /// Did any leg of this stop end in red?
    var stopFailed: Bool {
        stopSteps.contains { if case .failed = $0.state { return true } else { return false } }
    }

    /// Whether the stop panel is on screen.
    ///
    /// Before 2026-08-12 this was simply `state == .processing`, so the panel
    /// vanished the instant the gate settled — including when a leg had just
    /// failed. The owner reported exactly that: a red row appeared and the panel
    /// closed before it could be read, leaving the failure recorded only in a
    /// log file inside `~/Library/Application Support`.
    ///
    /// A failed stop now holds the panel until the user closes it. The mic is
    /// already unlocked underneath (`leaveProcessing` ran), so this blocks
    /// nothing that matters — it only refuses to throw the evidence away.
    var showsStopOverlay: Bool {
        state == .processing || (stopFailed && !stopFailureAcknowledged)
    }

    func dismissStopFailure() { stopFailureAcknowledged = true }

    @Published var segments: [TranscriptSegment] = []
    @Published var displayRows: [SpeakerUtterance] = []
    @Published var partialTranscript = ""

    /// The office twin of `RemoteCaption.startedAt` — when the current office
    /// provisional utterance began. Written ONLY by `setPartialTranscript`, so the
    /// text and its start time cannot disagree about which utterance they describe.
    @Published private(set) var partialStartedAt: Double?
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
    /// Set the office provisional caption and, with it, when this utterance began.
    /// EVERY assignment goes through here — a bare `partialTranscript = …` would
    /// leave a start time describing a different utterance, and the card would sort
    /// against the far end using a stale number.
    func setPartialTranscript(_ text: String) {
        if text.isEmpty { partialStartedAt = nil }
        else if partialTranscript.isEmpty { partialStartedAt = recordingElapsed }
        partialTranscript = text
    }

    /// Which provisional card is drawn FIRST: whoever started speaking first, so
    /// the two cards read down the page in the same time order the rows above them
    /// do. Pure, so the rule is testable without a view or a session.
    ///
    /// It used to be "office always, remote below", justified as *the room is the
    /// primary record*. That is a fine rule for a TIE and a wrong one for
    /// everything else — the owner watched the far end speak into a card pinned
    /// under a room caption the room had not produced.
    ///
    /// Absent start times sort LAST, which is the safe direction: a card with no
    /// timing claim never displaces one that has one.
    nonisolated static func remoteCaptionComesFirst(officeStartedAt: Double?,
                                                    remoteStartedAt: Double?) -> Bool {
        guard let remote = remoteStartedAt else { return false }
        guard let office = officeStartedAt else { return true }
        // Strictly earlier: an exact tie keeps Office first, which is the one case
        // the original rule got right.
        return remote < office
    }

    /// Signature of the last row order written to `logs/row-order.log`, so the log
    /// records CHANGES rather than state. `rebuildDisplayRows` runs on every
    /// realtime partial — logging each one would bury the moment a row moved,
    /// which is the only thing that file exists to show.
    var lastLoggedRowOrder: String?

    @Published var diarizationError: String?

    /// A speaker count that looks like clustering fragmentation — see
    /// `implausibleSpeakerCount`. Deliberately SEPARATE from `diarizationError`:
    /// nothing failed, the engine answered, and the answer is merely suspect.
    /// Rendering it as an error would report a working app as broken, which is
    /// the mistake the 2026-08-12 no-speech work already had to correct once.
    /// Cleared with the rest of the visible meeting state.
    @Published var diarizationCaution: String?

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

    // MARK: The stop-time chunked pass (`chunked.finalPass`; always tail-only
    // since 2026-08-06 — see `chunkedTailOnly` in `stop()`)
    //
    // See AudioRecorder+ChunkedStop.swift. All three default to today's
    // behaviour, and `beginCapture` resets them per session.

    /// Whether `checkLastChunkDone` may delete the leftover unconfirmed
    /// (realtime) segments. Written once per session from
    /// `chunkedStopPlan(_:).sweepsUnconfirmedTail`. TRUE — the sweep as it has
    /// always been — unless `chunked.finalPass` is off, in which case the
    /// realtime tail is the only text that audio will ever have.
    var chunkedSweepsUnconfirmed = true
    /// The full re-transcription driver, or nil. Cancelled by its watchdog.
    var chunkedFullPassTask: Task<Void, Never>?
    var fullPassWatchdog: Task<Void, Never>?
    /// The full pass has taken the remote leg of the stop gate and still owes it
    /// back. Taken in `stop()` rather than in the driver, because the remote
    /// block's `checkRemoteChunksDone()` runs in between and would otherwise
    /// settle the leg before the pass had claimed it.
    var fullPassHoldsRemoteLeg = false

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

    /// The Remote twin of `lastRealtimeFinalElapsed` — its own marker, never the
    /// office one. The two lanes flush independently, so sharing a marker would
    /// give one stream's utterance the other stream's start time.
    var lastRemoteRealtimeFinalElapsed: Double = 0

    /// When the far end has finished an utterance — see `RemoteUtteranceGate`.
    var remoteUtteranceGate = RemoteUtteranceGate()

    /// Temp-WAV path → the recording time that window began, for the four
    /// whole-file engines running live. Keyed by PATH because that is what their
    /// replies echo, so two windows can be in flight and each still finds its own
    /// start. See `AudioRecorder+BatchLiveDiarization`.
    var liveDiarWindowByPath: [String: Double] = [:]

    /// Hands out the arrival order of LIVE rows, across both streams.
    ///
    /// **A row's place is decided when it first appears, and then it stays there**
    /// (owner, 2026-08-13: *"remote speaker ini udah row nya, itu gak pindah
    /// pindah kemana mana pas realtime … office itu row ke dua, gak bisa jadi row
    /// pertama"*).
    ///
    /// Everything before this ordered live rows by TIME, and the times during
    /// recording are estimates that keep being corrected — a whole-chunk span, then
    /// a character-position guess, then late word times. Every correction re-sorted
    /// the list under the reader. Three separate fixes made those numbers better
    /// and none of them made the list stop moving, because the list should not have
    /// depended on the numbers at all.
    ///
    /// Arrival order is right HERE and only here: a realtime final fires when that
    /// stream's speaker stops, so for live rows arrival IS speaking order. It is
    /// NOT true of the confirmed chunk replies — both streams go through one queue
    /// with office always enqueued first — which is why confirmed rows keep their
    /// time ordering, and why the two are ordered separately.
    var nextLiveSeq = 0

    // Live chunked diarization — runs on its OWN interval, independent of ASR
    var chunkAudio: [Float] = []                       // 16k samples pending diarization
    /// Recording time at which the FIRST sample now in `chunkAudio` was captured.
    ///
    /// Exists because the tail pass used to DERIVE this from a setting instead of
    /// recording what happened: `diarizeTailChunk` computed
    /// `windowStart = liveOn ? lastDiarBoundary : 0`, reading `diarization.live`
    /// at STOP time. Settings are reachable while recording (the gear button is
    /// not disabled), and `diarization.live` is re-read per chunk, so turning
    /// live labels OFF mid-meeting with continue-on-stop ON makes the buffer
    /// start accumulating at that MOMENT — while the stop-time read then says
    /// "live is off, so this buffer starts at 0". Every turn in the tail would be
    /// reported shifted earlier by however long the meeting had already run, with
    /// nothing anywhere saying so.
    ///
    /// Updated wherever `chunkAudio` is emptied, and deliberately NOT where it is
    /// retained. A fact about the buffer cannot disagree with the buffer; a
    /// setting read minutes later can.
    var chunkAudioStart: Double = 0

    /// Overlap regions found by the standalone detector at Stop, in recording
    /// seconds. Empty unless `overlap.detect.enabled` is on and a detection ran.
    ///
    /// Kept SEPARATE from `overlapRegions()` rather than merged into it: that
    /// function derives regions from intersecting pyannote turns, and under
    /// pyannote both sources exist and would double-count. The display path adds
    /// the two together; overlap REPAIR takes whichever source its engine has —
    /// turns under pyannote, these under MOSS and spectral, which assign one
    /// speaker per instant and can never produce an intersecting turn of their own.
    var detectedOverlapRegions: [(start: Double, end: Double)] = []

    /// The same, for the REMOTE stream (owner, 2026-08-13).
    ///
    /// Its own collection rather than a shared one, for the reason every other
    /// remote/office pair here is split: the two streams share one clock, so a
    /// region from the far end would land on office rows at the same timestamp
    /// and mark a room conversation as overlapping when nobody in the room spoke
    /// twice. Kept apart, that is not a rule anyone has to remember.
    ///
    /// DISPLAY ONLY. Overlap repair never reads this — see `remoteRows`.
    var remoteDetectedOverlapRegions: [(start: Double, end: Double)] = []

    /// Detection jobs still in flight, office and remote counted together.
    ///
    /// ⚠ **A COUNTER, not two flags.** `overlapDetectDone` releases overlap
    /// repair, and repair holds the blocking stop overlay; releasing it while the
    /// second stream is still running would let repair read half the regions, and
    /// NEVER releasing it hangs the overlay until the 600 s watchdog. Every exit
    /// — result AND error, on either stream — decrements exactly once, and the
    /// flag is raised only at zero.
    var overlapDetectPending = 0

    /// Whether the detector has finished (or will never run) this session.
    ///
    /// Overlap REPAIR reads `detectedOverlapRegions` under MOSS and spectral, so
    /// under those engines it must not start until this is true — otherwise it
    /// reads an empty list, logs "no windows", and the regions land a second later
    /// with nothing left to use them. Under pyannote repair does not read them at
    /// all and does not wait.
    ///
    /// Set true on the detector's result AND on its error: a failed detection must
    /// release repair rather than hang the leg. Starts true, so a session with the
    /// detector switched off never waits for something that will not happen.
    var overlapDetectDone = true

    // MARK: Diarization settings, LOCKED for the session
    //
    // These three used to be read from UserDefaults BOTH per chunk and again at
    // Stop, and Settings is reachable while recording (the gear button carries no
    // `.disabled`). Changing one mid-meeting therefore did not just alter later
    // behaviour — it made the two reads DISAGREE about the same recording:
    //
    //   * `live` + `continueOnStop` decide per chunk whether `chunkAudio` is kept
    //     for a tail pass (`willBeConsumed`), and decide again at Stop which pass
    //     runs. Toggle either mid-meeting and audio is dropped that the tail then
    //     needs, or kept for a full pass that ignores it.
    //   * `detectOverlap` becomes `exclusive` on the wire, so half a meeting could
    //     be diarized with overlap detection and half without — one transcript,
    //     two rules, nothing saying where the seam is.
    //
    // Locked at `beginCapture` for the same reason `configureSpectral` locks the
    // engine: half a transcript under each rule is not a state any display path
    // can render honestly. The user's change still applies — to the NEXT session,
    // which is when a setting can be honoured coherently.
    //
    // `finalPass` and `numSpeakers` are deliberately NOT here. They are read only
    // at Stop, there is no second read to disagree with, and `finalPass` is
    // documented in `runsBatchOfficePass` as intentionally late — a rule that
    // trusted a start-of-session value would dispatch a pass the user had since
    // switched off.
    var diarLiveEnabled = true
    var diarContinueOnStop = false
    var diarDetectOverlap = true

    /// Speakers to ask each diarizer for. **0 = auto**, and the default.
    ///
    /// This was a pinned constant from 2026-08-06 (the picker had been removed)
    /// until the owner asked for the control back on 2026-08-10, on evidence: auto
    /// vs pinned, measured on the same files, showed **spectral counting 13
    /// speakers on a 3-person clip and 20 on a 67-minute meeting**, both fixed by
    /// pinning. The old constant's own comment left this door open for exactly
    /// that — "pinning a count later is one edit" — and this is that edit.
    ///
    /// Read LIVE rather than locked into the session, and that is deliberate: it
    /// reaches the sidecars only in the stop passes, so there is no second reader
    /// during recording for it to disagree with. It is the same reasoning
    /// `runsSpectralOfficePass` gives for reading `finalPass` late, and the reason
    /// `lockDiarizationSettings` does NOT own this key.
    ///
    /// Still ONE place rather than eight literals, so the dispatch sites cannot
    /// drift. Which engines actually act on it is
    /// `ModelLoader.honoursSpeakerCount(diarEngine:)` — DiariZen's sidecar never
    /// reads the field, so the number travels the whole path and dies where the
    /// truth is (the language-picker rule, 2026-07-31).
    static var diarNumSpeakers: Int {
        max(0, UserDefaults.standard.integer(forKey: "diarization.numSpeakers"))
    }

    /// What the REMOTE stop pass asks for: **always auto, on every engine.**
    ///
    /// `diarNumSpeakers` counts the ROOM. The chip says so in its own doc — *"you
    /// know before you press record how many people are in the room"* — it sits
    /// beside the room's RMS meter, and nothing anywhere asks how many people are
    /// on the far end of the call. So the number is simply not a fact about the
    /// Remote WAV, and the two streams are separate identity spaces precisely
    /// because they hold DIFFERENT people.
    ///
    /// Sending it anyway was not inert. A pinned count is an EXACT constraint on
    /// three of the four pipeline engines — pyannote and spectral pass it as
    /// `num_speakers=`, NeMo turns it into `oracle_num_speakers` — so a 5-person
    /// room with one caller on the line had that caller split into five "Remote
    /// Speaker" profiles, and a 2-person room with six callers had the six merged
    /// into two. Fabricating a conversation nobody had is the direction this
    /// project treats as the worse one everywhere else (the ASR hallucination
    /// gates, DiariZen's `min_cluster_size`).
    ///
    /// **This is the same rule the pyannote CHUNK jobs already follow** and for the
    /// same reason (2026-08-10 connection audit): a window that need not contain
    /// everyone must not be told the meeting's headcount. Auto is the safe
    /// direction — the engines that count well count well here too, and spectral,
    /// the one that counts badly, is no worse off than it was before the control
    /// existed.
    ///
    /// **A separate remote count IS now offered (owner, 2026-08-13)**, and this
    /// reads it. It replaces a pinned `0`, whose doc argued the opposite — kept
    /// here because the argument is still the risk, not because it was overruled
    /// in error: *the room is visible from the chair, the far end is not*, so this
    /// picker can collect a number the user is guessing at and hand it to an
    /// engine as a certainty. Defaulting to Auto and saying so in the control is
    /// what keeps that a choice rather than a trap.
    ///
    /// ⚠ **THE INVARIANT THAT MATTERS IS UNCHANGED, and it is not "remote is
    /// always automatic".** It is that a remote pass must never be handed the
    /// OFFICE count — two streams, two identity spaces, different people. That is
    /// what `layout/remote-passes-never-send-the-room-count` pins, and it still
    /// pins it: the passes reference THIS property, never `diarNumSpeakers`.
    /// A remote number the user typed for the remote stream is a different thing
    /// entirely from the room's headcount leaking across.
    static var remoteNumSpeakers: Int {
        UserDefaults.standard.integer(forKey: "diarization.remoteNumSpeakers")
    }

    /// Read the three session-scoped diarization settings once. Called from
    /// `beginCapture` before anything can consume them, and unconditionally —
    /// unlike `configureDiarization`, which returns early when there is no
    /// pyannote service and so cannot own settings the spectral path also reads.
    func lockDiarizationSettings() {
        let d = UserDefaults.standard
        // `diarization.live` and `.detectOverlap` had toggles until 2026-08-06,
        // when the owner removed both. FIXED here rather than left reading their
        // keys: a stored value must not outlive the control that set it, or
        // someone who once switched live labels off would keep getting none with
        // nothing in the UI able to change it back. Both constants are the
        // defaults these keys already had, so this is today's behaviour made
        // fixed. They stay in this function because it is the one place that
        // answers "what are this session's diarization settings".
        diarLiveEnabled = true
        diarContinueOnStop = d.object(forKey: "diarization.continueOnStop") as? Bool ?? false
        diarDetectOverlap = true
        // DIARIZEN'S OWN OVERLAP MARKING, and it belongs here for the same reason
        // the three above do: it is read by the DISPLAY path (`overlapRegions()`,
        // which rebuilds continuously while recording) and again at Stop by
        // `repairWindows`. Settings is reachable mid-meeting, so a live read would
        // let those two disagree about one meeting — half a transcript tagged and
        // half not, with nothing marking the seam. That is the 2026-08-05 finding,
        // and this key has exactly its shape.
        //
        // Under every other engine this is inert: `overlapRegions()` consults it
        // only when `diarizenDiarizationActive`, so pyannote keeps tagging overlap
        // whatever the Detect overlap switch says, byte-for-byte as before.
        diarizenOverlapMarking = d.object(forKey: "overlap.detect.enabled") as? Bool ?? false
    }
    var chunkFileByWindow: [Double: URL] = [:]

    // MARK: Forced alignment (its own sidecar since 2026-07-29)
    //
    // Two cadences, two buffers — the same rule `remoteChunkAudio` vs
    // `remoteDiarAudio` already follows. `chunkAudio` above LOOKS reusable and is
    // not: it is cleared on the DIARIZATION cadence (`diarization.intervalSec`,
    // in `diarizeLiveChunk`/`diarizeTailChunk`), and with live diarization off
    // plus continue-on-stop it is never cleared at all. The aligner needs exactly
    // the audio of ONE ASR chunk, so it gets a buffer cleared at the ASR chunk
    // boundary and nowhere else.

    /// 16 kHz office samples accumulated since the last ASR chunk boundary.
    /// Only filled when this session has an aligner — otherwise it stays empty
    /// and every path below is inert. Capped at `alignMaxBufferSamples`.
    var alignChunkAudio: [Float] = []
    /// Chunk audio parked under its window start, waiting for that chunk's text
    /// to come back so the two can be sent to the aligner together. Keyed like
    /// `chunkFileByWindow`; drained on EVERY path that pops
    /// `pendingChunkWindows`, or a failed chunk would leak a 30 s buffer.
    var alignAudioByWindow: [Double: [Float]] = [:]
    var sessionSpeakerIDs = Set<Int>()
    var liveTurns: [SpeakerTurn] = []      // absolute-time turns collected so far
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
    var remoteLiveTurns: [SpeakerTurn] = []
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

    // MARK: MOSS speaker-attributed ASR (diarization.engine == "moss")
    //
    // Storage only — every rule about these lives in `AudioRecorder+Moss`, which
    // is where the whole engine is implemented. They are declared HERE and not
    // there for the plain Swift reason the remote state above is: an extension
    // cannot hold stored properties.
    //
    // Inert for a pyannote session: `mossDiarizationActive` stays false, so the
    // callbacks return immediately, `mossTurns` stays empty and the display path
    // reads `liveTurns` exactly as it always has.

    /// True when THIS session is taking its speaker labels from MOSS. Read once
    /// in `beginCapture` like every other setting, so the engine cannot change
    /// mid-recording and leave half the transcript labelled by each.
    var mossDiarizationActive = false

    /// True when THIS session takes its speaker labels from the SPECTRAL engine.
    /// Read once in `beginCapture` (via `configureSpectral`) like every other
    /// setting, so the engine cannot change mid-recording and leave half the
    /// transcript labelled by each.
    ///
    /// Declared here rather than in `AudioRecorder+Spectral` for the plain Swift
    /// reason the MOSS and remote state above is: an extension cannot hold stored
    /// properties. Inert for every other session — it stays false, so every guard
    /// that reads it is a no-op.
    var spectralDiarizationActive = false

    /// True when THIS session takes its speaker labels from the NEMO engine.
    /// Read once in `beginCapture` (via `configureNemo`) like every other setting,
    /// so the engine cannot change mid-recording and leave half the transcript
    /// labelled by each.
    ///
    /// Declared here rather than in `AudioRecorder+Nemo` for the plain Swift reason
    /// the MOSS, spectral and remote state above is: an extension cannot hold
    /// stored properties. Inert for every other session — it stays false, so every
    /// guard that reads it is a no-op.
    var nemoDiarizationActive = false
    /// This session runs the DiariZen engine. Locked at `beginCapture` like the
    /// other engine flags — half a transcript labelled by each engine is not a
    /// state any display path can render honestly.
    var diarizenDiarizationActive = false
    /// Whether DiariZen's own overlap marking is switched on for this session
    /// (Settings → Models → Detect overlap, where DiariZen is the detector).
    /// Read once in `lockDiarizationSettings`; see there for why. Inert unless
    /// `diarizenDiarizationActive`.
    var diarizenOverlapMarking = false
    /// This session runs the CAM++ engine. Locked at `beginCapture` like the
    /// other engine flags — half a transcript labelled by each engine is not a
    /// state any display path can render honestly. Declared here rather than in
    /// `AudioRecorder+CamPlus` for the plain Swift reason the flags above are:
    /// an extension cannot hold stored properties. Inert for every other
    /// session — it stays false, so every guard that reads it is a no-op.
    var camPlusDiarizationActive = false
    /// True when the chunked ASR model IS MOSS, so one process fills both roles
    /// and the segments arriving on `onChunkSegments` describe the very text
    /// `onChunkTranscript` is about to deliver.
    var mossIsChunkedModel = false

    /// Guards `startMossIdentifyForOwnASR` against running twice — `checkLastChunkDone`
    /// is re-entered by its own completion handler. Reset per session in `configureMoss`.
    var mossIdentifyStarted = false
    /// MOSS turns — the engine's own per-chunk speaker spans, in recording time.
    /// A SEPARATE collection from `liveTurns` on purpose: office-only state must
    /// stay pure pyannote, so every `officeTurnsOnly` assert, `overlapRegions`,
    /// `applyFinalSpeakers` and `speakerCount` keeps holding without being
    /// relaxed for this engine.
    var mossTurns: [SpeakerTurn] = []

    /// The stop-time MOSS re-diarization, when `moss.continueOnStop` is off.
    /// Held so it can be cancelled; see AudioRecorder+MossStop.swift.
    var mossFullPassTask: Task<Void, Never>?
    /// How many MOSS chunks this session has labelled. Part of every wire id, so
    /// chunk N's speakers can never be confused with chunk N+1's.
    var mossChunkIndex = 0
    /// Segments from a `final` that has arrived, waiting for the transcript
    /// callback of that SAME `final` to consume them.
    ///
    /// One variable serves both modes because at most ONE MOSS process ever
    /// exists (`ModelLoader.needsSecondMossProcess`): either the chunked sidecar
    /// is MOSS, or a second one is, never both. The split exists so that exactly
    /// one callback owns the window FIFO — the segments callback only records,
    /// the transcript callback pops and applies, and the sidecar guarantees that
    /// order for a given `final`.
    var mossIncomingSegments: [ChunkedASRService.MossSegment]?
    /// Window FIFO for the SECOND MOSS process (other ASR + MOSS diarization).
    /// Its own queue, never `pendingChunkWindows`: two sidecars flush on the same
    /// boundary and each has to pop the window it was actually given.
    var mossPendingWindows: [ClosedRange<Double>] = []

    /// How many replies still in flight belong to the LIVE pass and must be
    /// thrown away when they land, because the stop-time full pass has already
    /// replaced the labels they would join.
    ///
    /// THE BUG (found 2026-08-05 in a real session's log). `startMossFullPass`
    /// clears `mossTurns` but not `mossPendingWindows` — that FIFO is only
    /// emptied in `configureMoss`, at session start. So a live chunk dispatched
    /// seconds before Stop was still in the sidecar when the full pass began; its
    /// reply popped the FIFO, was applied to the freshly-cleared set, and the full
    /// pass then re-labelled the SAME audio. Measured, from the log:
    ///
    ///     FULL PASS start — 2 window(s) of 120s over 149.8s
    ///     chunk #0 [0.0-120.1] 36 turns     <- the live chunk, upper bound 120.1
    ///     chunk #1 [0.0-120.0] 36 turns     <- full-pass window 1
    ///     chunk #2 [120.0-149.8] 0 turns
    ///     FULL PASS done — 72 turns         <- 36 counted twice
    ///
    /// The sidecar confirms the duplication: 1886 and 1887 characters, 36
    /// segments, `S01..S05` — the same audio transcribed twice. And because a
    /// MOSS id embeds its chunk index ON PURPOSE, the two sets carry different
    /// ids, so `coalesceAdjacentSameSpeaker` cannot merge them: the first two
    /// minutes end up with two overlapping sets of speaker spans.
    ///
    /// A COUNT rather than a flag, and it is counted BEFORE the pass queues
    /// anything: it drops exactly the replies that predate the pass and keeps the
    /// FIFO aligned, where clearing the queue outright would let a late live
    /// reply pop a FULL-PASS window and be applied under that window's times —
    /// the same bug wearing better clothes.
    var mossStaleReplies = 0
    /// Stop gate leg for that second process, mirroring `lastChunkDone`. Starts
    /// true so a session without one is already complete before it is consulted.
    var mossLastChunkDone = true
    var mossChunkWatchdog: Task<Void, Never>?

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

    /// The recorder that currently holds OPEN recording files, or nil when none
    /// does. Exists solely so `AppDelegate` can reach it while the app is being
    /// told to quit — `ContentView` owns the instance as a `@StateObject` and the
    /// delegate has no path to it otherwise.
    ///
    /// Weak, so this can never keep a finished recorder alive, and cleared in
    /// `stop()` as well, so "non-nil" really means "there are files to close".
    @MainActor private(set) static weak var active: AudioRecorder?

    /// Close the recording files WITHOUT running the stop pipeline.
    ///
    /// **THE BUG THIS EXISTS FOR (found 2026-08-05).** `AVAudioFile` writes the
    /// WAV `data` chunk size when it is released — that is why `stop()` says
    /// "releasing the AVAudioFile flushes it". If the app goes away while
    /// recording, that release never happens, the size field stays **0**, and
    /// every stage in this app then reads the file as having ZERO frames:
    /// `sf.info` says 0.0 s, the diarizers return no speakers, and the audio
    /// looks gone. It is not gone — the samples are all there behind a header
    /// that never learned how many there were.
    ///
    /// Measured on the owner's machine when this was found: **13 recordings,
    /// ~1.7 GB**, one of them 34.1 minutes of real speech (RMS 0.0063, peaks
    /// ±0.5), every one of them unreadable. Recoverable with
    /// `scripts/tools/repair-wav-header.py`.
    ///
    /// Deliberately NOT `stop()`. Stop starts full passes, watchdogs and overlay
    /// steps; the user asked to QUIT. The one thing that cannot be undone later
    /// is the header, so that is the only thing done here.
    ///
    /// The 2026-07-31 audit concluded no terminate hook was needed. That was
    /// right about what it examined — sidecars die on stdin EOF — but it only
    /// ever asked about SIDECARS, and the recording file was never in scope.
    @MainActor func finalizeRecordingFiles() {
        guard file != nil || remoteFile != nil else { return }
        file = nil
        remoteFile = nil
    }

    /// Recordings are stored under the data dir (project folder in dev,
    /// Application Support when bundled).
    private var recordingsDir: URL {
        let dir = PythonRuntime.dataDir.appendingPathComponent("recordings")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func toggle() {
        // A finished meeting LOCKS the mic until `startOver()` clears it (owner,
        // 2026-08-10). The guard is here as well as on the button because a view
        // is not a rule: `.disabled` stops a click, and this stops everything
        // else — a keyboard path, a future menu item, a second window.
        guard !meetingFinished else { return }
        switch state {
        case .idle:       start()
        case .recording:  stop()
        case .preparing:  break // ignore taps while loading
        case .processing: break // ignore taps while the stop work finishes
        }
    }

    /// Everything the transcript panel SHOWS, cleared in one place.
    ///
    /// THE ONE LIST. `startOver()` and `beginCapture` both need a blank panel, and
    /// until 2026-08-10 they each carried their own version — already disagreeing:
    /// `beginCapture` cleared `speakerCount` and `stopSteps` while `startOver` also
    /// cleared `partialTranscript`, `errorMessage` and `chunkedError`, which
    /// `beginCapture` did not. A new `@Published` that a view reads would have been
    /// added to whichever list its author happened to be looking at, and shown
    /// stale on the other path.
    ///
    /// Its membership is decided by ONE question — does a view read it? — over
    /// `TranscriptView`, `StatusChipsView`, `ProcessingOverlayView` and
    /// `RecordCardView`. Internal session state (buffers, watchdogs, boundaries,
    /// turn collections) is NOT here: `beginCapture` owns that, nothing reads it
    /// while idle, and the next recording cannot start without going through it.
    private func clearVisibleMeetingState() {
        // The transcript itself.
        segments = []
        displayRows = []
        setPartialTranscript("")
        partialSpeakerName = nil
        remoteCaption = RemoteCaption()

        // Everything the header chips, the overlay and the record card would
        // otherwise keep reporting about a meeting that is no longer on screen.
        speakerCount = nil
        remoteSpeakerCount = nil
        errorMessage = nil
        // Both "already read this" markers travel with the state they describe.
        // Left behind, they would suppress the popup for an identical error in
        // the NEXT meeting and hide a stop panel whose red rows had just been
        // rebuilt — in both cases a failure the user is never shown.
        dismissedErrorMessage = nil
        stopFailureAcknowledged = false
        // …and the row-order signature, or the first order of the NEXT meeting
        // matches the last order of this one and is never written down.
        lastLoggedRowOrder = nil
        chunkedError = nil
        diarizationError = nil
        // Travels with the transcript it describes: a caution about the LAST
        // meeting's speaker count, left standing over a fresh one, accuses a
        // result nobody has produced yet.
        diarizationCaution = nil
        overlapRepairError = nil
        overlapRepairProgress = nil
        stopSteps = []
    }

    /// When this meeting happened — the date the exported PDF is stamped with.
    ///
    /// A MODEL fact, so it lives on the model. It was derived inside the export
    /// button's action until 2026-08-10: a filesystem stat plus a fallback policy,
    /// in a SwiftUI closure, where no test could reach it — while `startOver()`
    /// nils `lastRecordingURL` specifically for that code's benefit. One fact,
    /// three files.
    ///
    /// Falls back to now when there is no recording file, which is the honest
    /// answer for a meeting whose audio was never written.
    var meetingRecordedAt: Date {
        lastRecordingURL
            .flatMap { try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate }
            ?? Date()
    }

    /// Clear the finished meeting so a new one can start.
    func startOver() {
        guard meetingFinished else { return }
        clearVisibleMeetingState()
        // The recording this panel was showing. Cleared HERE and not in the shared
        // helper: `beginCapture` sets this to the new session's file moments later,
        // so clearing it there would be undone in the same breath. Only Start Over
        // means "there is no recording", which is what stops Export acting on a
        // file the user has just dismissed.
        lastRecordingURL = nil
        meetingFinished = false
    }

    // MARK: - Start

    private func start() {
        errorMessage = nil
        // Paired with the line above — see `dismissedErrorMessage`. Pressing
        // Start again after closing the popup must be able to show the same
        // wall again, because hitting it twice is the commonest case (a denied
        // microphone is still denied the second time).
        dismissedErrorMessage = nil
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
        // Same rule, same place: a Remote channel while the accurate transcript
        // pass is off. The remote stream has no other source of text, and the
        // existing `remoteWanted` guard would drop it SILENTLY.
        let chunkedOn = UserDefaults.standard.object(forKey: "chunked.enabled") as? Bool ?? true
        if let refusal = Self.chunkedOffRefusalMessage(remoteChannel: mic.remoteChannel,
                                                       chunkedEnabled: chunkedOn) {
            dualStreamLog("REFUSED start — \(refusal)")
            modelLoader.failStartup(step: "Remote stream + chunked pass off", message: refusal)
            errorMessage = refusal
            state = .idle
            return
        }
        // Same rule, same place, for the other combination that cannot keep up:
        // Voxtral as the chunked model while MOSS is the diarization engine.
        // Checked before `loadAll` for the same reason — the user should not
        // wait out a 4B load plus a 3.6 GB one to be told it will not work.
        let diarEngine = UserDefaults.standard.string(forKey: "diarization.engine")
            ?? ModelLoader.pyannoteEngineID
        if let refusal = Self.mossRefusalMessage(chunkedModelID: chunkedID,
                                                 diarizationEngine: diarEngine,
                                                 remoteChannel: mic.remoteChannel) {
            mossLog("REFUSED start — \(refusal)")
            modelLoader.failStartup(step: "MOSS diarization + chunked model", message: refusal)
            errorMessage = refusal
            state = .idle
            return
        }
        // Fourth refusal, same rule and same place: "Run a transcription pass at stop" with
        // "Continue from live text (tail only)" OFF asks for a full re-transcription
        // of the whole recording, which MOSS cannot do at all and Voxtral cannot
        // do in reasonable time. Told before the meeting, not after it — the cost
        // of finding out at Stop is an hour of processing or a truncated
        // transcript. Refusal, never a silent downgrade to the tail: the user
        // asked for the whole recording and would have no way to know they did
        // not get it.
        // NO full-pass refusal here any more. `chunkedFullPassRefusalMessage`
        // guards a mode the tail-only pin (see `stop()`) makes unreachable, so the
        // `if` that called it — and the two locals it read — were provably dead,
        // and a CLEAN build said so ("will never be executed"). The pure function
        // and its tests are KEPT: they are what restoring the toggle would need.
        // A branch that can never run is not a safety net, it is a warning on
        // every build.
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
        let asr = realtimeOn ? modelLoader.realtimeASR?.office : nil
        asr?.onTranscript = { [weak self] text, isFinal in
            Task { @MainActor in
                guard let self else { return }
                if isFinal {
                    // Once stopped, the chunked pass is authoritative for all
                    // remaining audio. Skip realtime finals so a trailing one
                    // (fired by the flush in stop()) can't land after the last
                    // chunk's cleanup and survive as an orphan "SPEAKER UNKNOWN"
                    // row next to the confirmed, speaker-split transcript.
                    guard !self.stopped else { self.setPartialTranscript(""); self.partialSpeakerName = nil; return }
                    let end = self.recordingElapsed
                    let start = min(self.lastRealtimeFinalElapsed, end)
                    // Advance the marker on every final (incl. empty ones), since
                    // the sidecar's buffer resets on each flush.
                    self.lastRealtimeFinalElapsed = end
                    if !text.isEmpty {
                        self.nextLiveSeq += 1
                        self.segments.append(TranscriptSegment(text: text, confirmed: false,
                                                               window: start...end,
                                                               seq: self.nextLiveSeq))
                        self.rebuildDisplayRows()
                    }
                    self.setPartialTranscript("")
                    self.partialSpeakerName = nil
                } else {
                    self.setPartialTranscript(text)
                    // Trailing 1s window ≈ the beam active right now (after smoothing).
                    // Short so the live-partial label flips quickly on a talker switch.
                    // In `pyannote` source mode the position layer is not displayed
                    // anywhere, so the live partial must not carry its label either.
                    // The MOSS engine is the same case for the same reason
                    // (`derivedRows` forces the pyannote-pure plan under it): without
                    // this the caption would show an ATND position name while every
                    // confirmed row below it showed a MOSS label — two naming systems
                    // on screen at once, the top one vanishing as it commits.
                    self.partialSpeakerName = !self.mossDiarizationActive
                        && self.positionSource.usesPosition
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
            ? modelLoader.realtimeASR?.remote : nil
        if remoteASR == nil {
            // Sharing one process means the remote lane outlives the session that
            // wanted it. Detach so a previous meeting's closure can never fire.
            modelLoader.realtimeASR?.detachRemoteLane()
        }
        remoteASR?.onTranscript = { [weak self] text, isFinal in
            Task { @MainActor in
                guard let self else { return }
                // After Stop the remote chunk pass owns the remaining audio,
                // exactly as the office branch above reasons about its own
                // trailing final.
                guard !self.stopped else { self.remoteCaption.commit(); return }
                guard isFinal else {
                    self.remoteCaption.update(to: text, at: self.recordingElapsed)
                    return
                }
                // A FINAL becomes its own unconfirmed ROW, the office branch's
                // shape. Until 2026-08-13 finals were treated exactly like
                // partials and only ever grew the caption, so the far end's whole
                // meeting accumulated into one card that could not interleave with
                // the room — and at a 120 s chunk interval that card lived for two
                // minutes at a time.
                let end = self.recordingElapsed
                let start = min(self.lastRemoteRealtimeFinalElapsed, end)
                // Advanced on EVERY final, empty ones included: the sidecar's
                // buffer resets on each flush, so the next utterance starts here
                // whether or not this one had words.
                self.lastRemoteRealtimeFinalElapsed = end
                if !text.isEmpty {
                    self.nextLiveSeq += 1
                    self.remoteSegments.append(RemoteSegment(text: text,
                                                             window: start...max(end, start),
                                                             seq: self.nextLiveSeq,
                                                             confirmed: false))
                    self.rebuildDisplayRows()
                }
                self.remoteCaption.commit()
            }
        }

        // 5c. Chunked ASR — rolling accurate pass every N seconds.
        // Its result REPLACES the unconfirmed Nemotron segments in place.
        let chunked = modelLoader.chunkedASR
        chunkedModelName = chunked?.config.modelName ?? ""
        // Everything a view shows, from the ONE list that Start Over also uses —
        // segments, rows, partials, the chips, the errors, the stop steps. Keeping
        // a second copy of that set here is what let the two paths disagree.
        clearVisibleMeetingState()
        chunkedBusy = false
        chunkElapsed = 0
        recordingElapsed = 0
        lastChunkBoundary = 0
        pendingChunkWindows = []
        lastRealtimeFinalElapsed = 0
        lastRemoteRealtimeFinalElapsed = 0
        remoteUtteranceGate.reset()
        nextLiveSeq = 0
        diarizing = false
        // Fresh overlap-repair state for this session. `overlapRepairProgress` and
        // `overlapRepairError` are display fields and went to the shared helper;
        // the task and the in-flight flag are not, and stay here.
        repairTask?.cancel()
        repairTask = nil
        overlapRepairing = false
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
        // Stop-time chunked pass: back to today's behaviour for the new session.
        // `stop()` writes the real values from this session's settings.
        chunkedSweepsUnconfirmed = true
        chunkedFullPassTask?.cancel()
        chunkedFullPassTask = nil
        fullPassWatchdog?.cancel()
        fullPassWatchdog = nil
        fullPassHoldsRemoteLeg = false
        // Fresh per-session speaker state, then wire the live/final callbacks.
        chunkFileByWindow = [:]
        sessionSpeakerIDs = []
        liveTurns = []
        chunkAudio = []
        chunkAudioStart = 0
        alignChunkAudio = []
        alignAudioByWindow = [:]
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
        lockDiarizationSettings()   // before anything can consume them
        configureDiarization()
        configurePositionDiarization()
        // AFTER the position config: `configureMoss` reads `positionSource` to
        // log, once, that the position layer contributes nothing under this
        // engine. It also resets every MOSS collection for the session and wires
        // the second process's callbacks when there is one.
        configureMoss()
        // The third engine, read in exactly the same place and the same way, so
        // all three are fixed for the whole recording. It is the LAST of the three
        // deliberately: `configureDiarization` returns immediately when
        // `modelLoader.pyannote` is nil (which it always is under this engine) and
        // `configureMoss` sets `mossDiarizationActive = false` for the same reason,
        // so the three calls in this order leave exactly one of the flags true —
        // and a spectral session has already had the other two no-op cleanly
        // rather than half-wiring a service that is not loaded.
        configureSpectral()
        // The FOURTH engine, in the same place and the same way, and for the same
        // reason it is last: each of the four `configure*` calls sets its own flag
        // false unless its engine is the selected one, so this order leaves exactly
        // one of the four true and the other three have already no-op'd cleanly.
        configureNemo()
        configureDiarizen()
        configureCamPlus()
        configureOverlapDetect()
        // The second MOSS process for this session, or nil — captured once, like
        // `chunked`, so the escaping tap closure never touches the recorder to
        // find it.
        let mossDiar = mossDiarService
        // Real-time speaker split: when the beam settles on a different talker,
        // end the old speaker's realtime segment now and relabel existing rows.
        // Detection lags the real switch by ~0.7s (0.4s smoother warm-up + 3
        // confirmation samples at 10 Hz), so the new speaker's first fraction of a
        // second lands in the old segment — same class of approximation as the
        // existing time→char sentence split. No-op when the feature is off
        // (positionDiarizer == nil → this optional-chain never installs anything).
        // The realtime engine is modelLoader.realtimeASR (there is no `self.asr`
        // property — `asr` is a local in beginCapture); flush() is a safe no-op
        // when idle, and empty-text finals are already dropped in onTranscript.
        // Office-only on purpose, and now spelled out by the `.office` lane: the
        // beam describes the ROOM, so a cluster change says nothing about the
        // conferencing stream. Flushing the remote lane here would cut its caption
        // on an event from the other stream — and it is the first step towards
        // remote audio reaching the position path, which it must never do.
        positionDiarizer?.onClusterChange = { [weak self] in
            guard let self, !self.stopped else { return }
            self.modelLoader.realtimeASR?.office.flush()  // end the old speaker's realtime segment now
            self.rebuildDisplayRows()              // relabel existing rows' fills instantly
        }
        // Optionally start each recording with a clean speaker store. Addressed to
        // the STORE'S OWNER since the 2026-07-30 split — the pyannote sidecar has
        // no profiles to reset any more. Semantics are unchanged: the reset rides
        // the same FIFO stdin as the identify jobs, so it is necessarily handled
        // before the first job of this session, exactly as before.
        // ALWAYS fresh (owner, 2026-08-06). `diarization.resetOnStart` had a
        // toggle and is no longer read — same rule as the two above.
        modelLoader.embedding?.resetProfiles()
        let chunkInterval = Double(
            UserDefaults.standard.object(forKey: "chunked.intervalSec") as? Int ?? 30
        )
        // Diarization runs on its own cadence, separate from chunked ASR.
        let diarInterval = Double(
            UserDefaults.standard.object(forKey: "diarization.intervalSec") as? Int ?? 30
        )
        chunkedError = nil
        // MOSS as the chunked model AND the diarizer — the one-process mode. The
        // segments arrive just before the transcript of the same `final`; record
        // them here and let that transcript callback apply them, so exactly one
        // callback owns the window FIFO. Explicitly cleared otherwise: the service
        // outlives the session, so a stale closure from a previous meeting must
        // not survive a switch back to pyannote (and MOSS-as-ASR under pyannote
        // must ignore the segments entirely).
        if mossDiarizationActive, mossIsChunkedModel {
            chunked?.onChunkSegments = { [weak self] segments in
                Task { @MainActor in self?.mossIncomingSegments = segments }
            }
        } else {
            chunked?.onChunkSegments = nil
        }
        // The aligner for THIS session, captured once like `chunked` — nil means
        // no alignment happens at all: no buffer is accumulated in the tap, no
        // window is parked, and no request is ever sent.
        let aligner = modelLoader.aligner
        chunked?.onChunkTranscript = { [weak self] text, conf in
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
                // Popped here, and used below only if this chunk is actually
                // alignable. Removing it unconditionally is the point: every
                // path that pops a window must also drop its audio.
                let alignSamples = window.flatMap {
                    self.alignAudioByWindow.removeValue(forKey: $0.lowerBound)
                }
                self.segments.removeAll { !$0.confirmed }
                // MOSS filling BOTH roles: the `final` that carried this text also
                // carried the model's own per-speaker segmentation of it, which
                // arrived on `onChunkSegments` a moment ago. Append those as pinned
                // per-speaker rows instead of this one joined, unattributed row —
                // appending both would duplicate every word in the chunk. Returns
                // false in every other configuration, leaving the line below to run
                // exactly as it always has.
                let handledByMoss = self.applyMossChunk(window: window)
                if !text.isEmpty, !handledByMoss {
                    // No `words` yet, ON PURPOSE: the row is shown NOW with the
                    // estimated character-proportional split, and the aligner is
                    // asked separately below. `applyAlignedWords` fills them in
                    // when (if) the reply lands.
                    let segment = TranscriptSegment(text: text, confirmed: true,
                                                    window: window, asrConf: conf)
                    self.segments.append(segment)
                    if let aligner, let samples = alignSamples, !samples.isEmpty {
                        self.requestAlignment(aligner: aligner, samples: samples,
                                              segmentID: segment.id, text: text)
                    }
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
                    let window = self.pendingChunkWindows.removeFirst()
                    // …and its parked alignment audio with it: there will be no
                    // text for this window, so nothing would ever collect it.
                    self.alignAudioByWindow.removeValue(forKey: window.lowerBound)
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
            // The SECOND MOSS process gets the identical office samples — nil in
            // every other configuration, including MOSS+MOSS where `chunked` above
            // already IS the MOSS process. Fed from the captured local, off the
            // main actor, for the same reason `chunked` is: `feed` hands the
            // samples straight to the sidecar's own write queue.
            mossDiar?.feed(samples16k)

            Task { @MainActor in
                self.rms = level
                self.recordingElapsed += duration
                // VAD speech → silence edge: utterance ended, finalize it
                if self.isSpeaking && !speaking {
                    asr?.flush()
                }
                self.isSpeaking = speaking
                // Same verdict, same clock: the ATND beam follows any sound, so
                // position only collects direction while this says speech.
                self.positionDiarizer?.noteSpeech(speaking, at: self.recordingElapsed)

                // Chunk boundary: interval elapsed AND we're in silence
                // (avoids cutting mid-word). Hard cap at 1.5x interval if
                // someone talks non-stop.
                self.chunkElapsed += duration
                let boundary = (self.chunkElapsed >= chunkInterval && !speaking)
                            || self.chunkElapsed >= chunkInterval * 1.5
                // Accumulate audio for live diarization (its own cadence below)
                self.chunkAudio.append(contentsOf: samples16k)
                // …and, separately, for the aligner, which needs exactly ONE ASR
                // chunk. Only when this session has an aligner: otherwise not a
                // sample is copied and the feature costs nothing.
                if aligner != nil {
                    self.alignChunkAudio.append(contentsOf: samples16k)
                    if self.alignChunkAudio.count > Self.alignMaxBufferSamples {
                        self.alignChunkAudio.removeFirst(
                            self.alignChunkAudio.count - Self.alignMaxBufferSamples)
                    }
                }
                // Remote accumulates in parallel. Empty for a single-stream
                // session, so this is a no-op there.
                if !remoteSamples16k.isEmpty {
                    self.remoteChunkAudio.append(contentsOf: remoteSamples16k)
                    // Second, independent buffer for remote diarization: the two
                    // cadences are separate settings, so one buffer cleared on the
                    // ASR boundary could not also feed the diarization boundary.
                    self.remoteDiarAudio.append(contentsOf: remoteSamples16k)

                    // END OF A REMOTE UTTERANCE — flush the far end's realtime lane
                    // so its text becomes a ROW placed in time, instead of piling
                    // into one caption that grows for a whole chunk interval. See
                    // `remoteSilenceElapsed` for why the office trigger cannot be
                    // reused here.
                    //
                    // The threshold is `utterancePauseSec`, the SAME 1.0 s that
                    // decides where one utterance ends when a chunk is split.
                    // "When has someone finished speaking" is one question and must
                    // not get two answers. The level gate is `remoteSilenceRMS`, the
                    // same one that already decides a remote chunk is silent.
                    if self.remoteUtteranceGate.note(
                        level: AudioBufferProcessor.rms(remoteSamples16k),
                        duration: duration) {
                        remoteASR?.flush()
                    }
                }

                // The ASR boundary is also what feeds the SECOND MOSS process, so
                // the condition is "does anything ride this cadence?", not "is
                // there a chunked model?". With the chunked pass switched off and
                // the MOSS engine selected, `needsSecondMossProcess` loads that
                // process precisely because there is no chunked one to borrow
                // segments from — and the old `chunked != nil` test would then
                // have starved it, producing a meeting with no speakers at all
                // and nothing in any log to say why.
                if boundary, chunked != nil || self.mossDiarService != nil {
                    self.chunkElapsed = 0
                    // Flush Nemotron too — keeps both services aligned at the
                    // same audio position so replacement is exact. Worth doing
                    // even with no chunked model: its text is then the transcript.
                    asr?.flush()
                    let windowStart = self.lastChunkBoundary
                    self.lastChunkBoundary = self.recordingElapsed
                    // Queued ONLY when a chunk is really dispatched. `pendingChunkWindows`
                    // is drained by chunked replies, so an entry with no request
                    // behind it is never removed and every later reply is paired
                    // with the wrong window.
                    if chunked != nil {
                        self.pendingChunkWindows.append(windowStart...self.recordingElapsed)
                        // Park this chunk's audio under its window BEFORE the flush,
                        // so it is already there when the transcript comes back.
                        self.stashAlignAudio(windowStart: windowStart)
                    }
                    self.startChunkFlush(chunked)
                    // The second MOSS process rides the SAME boundary and gets the
                    // SAME window — deliberately, not `diarization.intervalSec`:
                    // identical windows are what let its turns split the ASR
                    // model's text exactly. No-op when there is no second process.
                    self.flushMossDiarChunk(window: windowStart...self.recordingElapsed)
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
            // Registered only once the engine really started, so `active` means
            // "files are open and unflushed" and never merely "a recorder exists".
            Self.active = self
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
    /// clear the spinner and surface a timeout (details in that sidecar's log —
    /// each ASR service owns its own file, so the name comes from the service
    /// rather than being hard-coded to the old shared one).
    private func startChunkFlush(_ service: ChunkedASRService?) {
        guard let service else { return }
        chunkedBusy = true
        service.flush()
        chunkWatchdog?.cancel()
        let logName = service.config.logName
        chunkWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(180))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.chunkedBusy else { return }
                self.chunkedBusy = false
                self.chunkedError = "Chunk transcription timed out — see logs/\(logName).log"
                // Drain the queued window on timeout too — same reasoning as onChunkError.
                if !self.pendingChunkWindows.isEmpty {
                    let window = self.pendingChunkWindows.removeFirst()
                    self.alignAudioByWindow.removeValue(forKey: window.lowerBound)
                }
                self.checkLastChunkDone()
            }
        }
    }

    private func stop() {
        stopped = true
        // DETACH THE TAP BEFORE ANY FLUSH. The tap body has no `stopped` guard —
        // it feeds `chunked`, `mossDiar` and the realtime lanes straight from the
        // audio thread — so while it was still installed, a callback firing
        // between a FLUSH frame and the teardown further down handed the sidecar
        // samples it would never be asked to transcribe. Bounded (one 4096-frame
        // buffer ≈ 85 ms at 48 kHz) and always at the very end of the recording,
        // but silently dropped: the sidecar transcribes on FLUSH, and no second
        // FLUSH ever came. Removing the tap here makes every sidecar's buffer
        // final before the first flush is sent, so what is captured is what is
        // transcribed. The engine itself is torn down further below, where the
        // files are closed — only the tap moves.
        engine?.inputNode.removeTap(onBus: 0)
        // With the tap gone, the WAV is final — and it, not `recordingElapsed`,
        // is the authority on how long the recording is.
        //
        // `recordingElapsed` is advanced inside `Task { @MainActor }` from the tap
        // callback, while the SAMPLES are written and fed synchronously on the
        // audio thread. So a callback whose task had not run yet left the counter
        // behind the audio by up to one tap buffer (~85 ms at 48 kHz). Every stop
        // window is built from that counter — the tail window, and every window of
        // a full pass — so the last ~85 ms sat outside them all: present in the
        // recording, absent from the transcript. Reading the file closes the gap
        // at its only authoritative source.
        if let file, file.fileFormat.sampleRate > 0 {
            let written = Double(file.length) / file.fileFormat.sampleRate
            if written > recordingElapsed { recordingElapsed = written }
        }
        // Stop ingesting beam notices, but KEEP the collected data — display-time
        // gap-fill (positionGapFill → label(for:)) still queries it afterward.
        positionDiarizer?.stop()
        modelLoader.realtimeASR?.office.flush() // finalize any trailing speech
        // Remote's tail window is the office tail window — read BEFORE the office
        // branch below advances `lastChunkBoundary`, so both streams' last windows
        // still line up exactly as they did at every live boundary.
        let tailStart = lastChunkBoundary
        // The second MOSS process's tail is the office tail — same window, read
        // before `lastChunkBoundary` moves, exactly as the remote tail above is.
        // There is no stop-time MOSS pass of any kind beyond this: the last
        // chunk's result IS the tail, because the engine labels as it transcribes.
        // What the MOSS DIARIZATION engine does at stop. Both keys absent →
        // `.tail`, the branch this line has always taken. Only ever `.full` when
        // a SECOND MOSS process exists (another model does the ASR) — in
        // MOSS+MOSS the tail is the chunked tail and `chunked.*` governs it.
        // See AudioRecorder+MossStop.swift.
        let mossPlan = Self.mossStopPlan(Self.mossStopMode(
            finalPass: UserDefaults.standard.object(forKey: "moss.finalPass") as? Bool ?? true,
            continueOnStop: UserDefaults.standard.object(forKey: "moss.continueOnStop") as? Bool ?? true,
            hasDiarService: mossDiarService != nil,
            hasRecording: lastRecordingURL != nil))
        if mossPlan.flushesTail {
            flushMossDiarChunk(window: tailStart...max(recordingElapsed, tailStart + 0.01))
        }
        if mossPlan.settlesImmediately {
            mossLastChunkDone = true
        }
        if mossPlan.runsFullPass, let recording = lastRecordingURL {
            startMossFullPass(recording: recording)
        }
        // What the chunked pass does at stop. Both keys absent → `.tail`, which
        // is the branch this block has always taken; the "no chunked model"
        // early-out is now `.none`, decided in the same place rather than by an
        // `if` here. See AudioRecorder+ChunkedStop.swift.
        let chunkedFinalPass = UserDefaults.standard.object(forKey: "chunked.finalPass") as? Bool ?? true
        // ALWAYS tail-only. The "Continue from live text (tail only)" toggle was
        // removed on 2026-08-06 (owner), and its stored key is deliberately NOT
        // read any more: leaving the read in place would let a value set before
        // the control disappeared keep choosing the full pass, with nothing in
        // the UI able to change it back. `true` is that toggle's own default and
        // what the app has always done, so this is today's behaviour made fixed
        // rather than a new one. `.full` and its machinery are kept — they are
        // still covered by tests and reachable if the toggle ever returns.
        let chunkedTailOnly = true
        let chunkedMode = Self.chunkedStopMode(
            finalPass: chunkedFinalPass,
            continueOnStop: chunkedTailOnly,
            hasChunkedModel: modelLoader.chunkedASR != nil,
            hasRecording: lastRecordingURL != nil,
            chunkedModelID: UserDefaults.standard.string(forKey: "chunked.model") ?? "qwen3")
        let chunkedPlan = Self.chunkedStopPlan(chunkedMode)
        chunkedSweepsUnconfirmed = chunkedPlan.sweepsUnconfirmedTail
        if chunkedPlan.queuesTailWindow {
            pendingChunkWindows.append(lastChunkBoundary...max(recordingElapsed, lastChunkBoundary + 0.01))
            // Same stash as every live boundary — the tail chunk is aligned like
            // any other. One helper for both sites so they cannot drift apart.
            stashAlignAudio(windowStart: lastChunkBoundary)
            lastChunkBoundary = recordingElapsed
        }
        if chunkedPlan.flushesSidecar {
            startChunkFlush(modelLoader.chunkedASR) // transcribe the last partial chunk
        }
        if chunkedPlan.settlesImmediately {
            // Nothing will ever drain a queued window (no chunked model, or the
            // pass is switched off), so don't queue one — the gate has to
            // complete here instead.
            lastChunkDone = true
        }
        // Taken HERE, before the remote block below calls `checkRemoteChunksDone()`
        // with nothing in flight — that call would otherwise settle the remote leg
        // a moment before the full pass claimed it.
        if chunkedPlan.runsFullPass, remoteStreamActive, remoteRecordingURL != nil {
            remotePendingChunks += 1
            fullPassHoldsRemoteLeg = true
        }
        // Watchdog budgets. A full pass legitimately takes MINUTES (Qwen3 ≈ 9 for
        // a 60-minute meeting), so the fixed 180 s remote and 600 s stop
        // watchdogs would fire in the middle of healthy work and mark it timed
        // out. Both are therefore scaled by the number of windows; when no full
        // pass runs, `fullPassBudget` is 0 and both keep their original values
        // exactly.
        let fullPassWindowCount = chunkedPlan.runsFullPass
            ? Self.fullPassWindows(
                recordingLength: recordingElapsed,
                intervalSec: Double(UserDefaults.standard.object(forKey: "chunked.intervalSec") as? Int ?? 30)
              ).count * (fullPassHoldsRemoteLeg ? 2 : 1)
            : 0
        // The MOSS re-diarization pass is a SECOND stop-time pass and was missing
        // from this budget entirely — found in the 2026-07-31 re-audit, in code
        // written the same day. With only MOSS re-labelling (chunked on tail), the
        // budget was 0 and the stop watchdog stayed at its 600 s floor, while a
        // 60-minute meeting is ~30 windows of 120 s at ~26 s each ≈ 13 minutes.
        // The watchdog would have fired mid-pass and marked every leg timed out
        // while the sidecar was working normally — the same failure the tail
        // diarization watchdog had.
        //
        // SUMMED, not maxed: both passes can run at once and they contend for the
        // same GPU, so their durations add rather than overlap.
        let mossFullPassWindowCount = mossPlan.runsFullPass
            ? Self.fullPassWindows(recordingLength: recordingElapsed,
                                   intervalSec: Self.mossFullPassWindowSec).count
            : 0
        let mossFullPassBudget = mossFullPassWindowCount > 0
            ? Self.fullPassWatchdogSeconds(windowCount: mossFullPassWindowCount) : 0
        let fullPassBudget = Self.fullPassWatchdogSeconds(
            windowCount: fullPassWindowCount) * (chunkedPlan.runsFullPass ? 1 : 0)
            + mossFullPassBudget
        let remoteStopWatchdogSeconds = max(180.0, fullPassBudget + 120)
        // The tap is already gone — detached at the top of `stop()` so that no
        // sample could arrive after a FLUSH. Removing it twice is harmless but
        // would misstate where the boundary is, so it is not repeated here.
        engine?.stop()
        engine = nil
        file = nil
        remoteFile = nil   // both files close here; releasing the AVAudioFile flushes it
        // The headers are written now, so there is nothing left for the quit hook
        // to rescue. Cleared here rather than in `deinit` so "active != nil"
        // stays exactly "there are unflushed files".
        if Self.active === self { Self.active = nil }
        vad = nil
        rms = 0
        isSpeaking = false

        // Remote tail, on the office tail's window. Inert for a single-stream
        // session (`remoteLastChunkDone` is already true and stays true).
        if remoteStreamActive {
            // Flush parity with the office lane above: finalize the remote
            // caption's trailing speech before its tail chunk is queued. Its own
            // opcode, so this resets only the remote buffer in the sidecar.
            modelLoader.realtimeASR?.remote.flush()
            // Remote follows the SAME two toggles the office lane does, because
            // they are pipeline-level settings and the alternative is a silent
            // asymmetry: "re-transcribe the recording" that leaves half the
            // transcript on the live text, or "no pass at stop" that still runs
            // one for Remote. With the full pass, Remote's windows are re-run
            // inside `startChunkedFullPass` instead of here.
            if chunkedPlan.queuesTailWindow {
                flushRemoteChunk(window: tailStart...max(recordingElapsed, tailStart + 0.01),
                                 chunked: modelLoader.chunkedASR)
            } else if !chunkedPlan.runsFullPass {
                // No remote row is coming for this window — drop the caption
                // rather than leave it hanging, exactly as a skipped chunk does.
                remoteCaption.commit()
            }
            startRemoteStopWatchdog(seconds: remoteStopWatchdogSeconds)
            // Nothing in flight (idle channel, everything gated as silent) →
            // complete the gate now instead of waiting for a callback that
            // will never come.
            checkRemoteChunksDone()
        }

        // Who spoke when — either append a tail (continue from live labels) or
        // re-diarize the whole recording (best global clustering).
        let finalOn = UserDefaults.standard.object(forKey: "diarization.finalPass") as? Bool ?? true
        let continueOnStop = diarContinueOnStop
        // The two WHOLE-FILE BATCH engines' office passes, each decided by the SAME
        // pure rule called with its own flag. All three paths are mutually
        // exclusive by construction — at most one of `pyannote`, `spectral` and
        // `nemo` is non-nil in a session — but each is asked explicitly so the
        // overlay's "diarize" row exists for exactly one of them and can never be
        // listed twice or not at all.
        let runsSpectralPass = Self.runsBatchOfficePass(
            batchActive: spectralDiarizationActive,
            hasService: modelLoader.spectral != nil,
            hasRecording: lastRecordingURL != nil,
            finalPass: finalOn)
        let runsNemoPass = Self.runsBatchOfficePass(
            batchActive: nemoDiarizationActive,
            hasService: modelLoader.nemo != nil,
            hasRecording: lastRecordingURL != nil,
            finalPass: finalOn)
        // Its OWN rule, not folded into NeMo's: `runsBatchOfficePass` takes the
        // engine's flag AND its service, and sharing one call would have asked
        // whether NeMo's sidecar was up for a DiariZen session.
        let runsDiarizenPass = Self.runsBatchOfficePass(
            batchActive: diarizenDiarizationActive,
            hasService: modelLoader.diarizen != nil,
            hasRecording: lastRecordingURL != nil,
            finalPass: finalOn)
        // Its OWN rule again, for the reason stated above DiariZen's: sharing one
        // call would ask whether ANOTHER engine's sidecar was up for this session.
        let runsCamPlusPass = Self.runsBatchOfficePass(
            batchActive: camPlusDiarizationActive,
            hasService: modelLoader.camPlus != nil,
            hasRecording: lastRecordingURL != nil,
            finalPass: finalOn)
        let willRunStopPass = runsSpectralPass || runsNemoPass || runsDiarizenPass
            || runsCamPlusPass
            || (finalOn && modelLoader.pyannote != nil)
        // The remote pass is dispatched HERE, before the overlay is built, so the
        // step list knows whether to show a remote-diarization row. Queued ahead
        // of the office stop pass on the same stdin; the sidecar drains it in
        // order, so both run to completion regardless of who is first.
        //
        // Routed by the engine, not merged into one function: the three dispatchers
        // talk to DIFFERENT sidecars and take different modes (both batch engines
        // pass `supportsTail: false`). Each returns whether it took the remote gate,
        // so the overlay contract is identical whichever answered.
        let willRunRemoteDiar: Bool
        if spectralDiarizationActive {
            willRunRemoteDiar = startRemoteSpectralDiarization()
        } else if nemoDiarizationActive {
            willRunRemoteDiar = startRemoteNemoDiarization()
        } else if diarizenDiarizationActive {
            willRunRemoteDiar = startRemoteDiarizenDiarization()
        } else if camPlusDiarizationActive {
            willRunRemoteDiar = startRemoteCamPlusDiarization()
        } else {
            willRunRemoteDiar = startRemoteDiarization()
        }

        // Everything below lands asynchronously; block the controls until it does.
        buildStopSteps(willRunStopPass: willRunStopPass,
                       willRunRemoteDiar: willRunRemoteDiar,
                       willRunMossDiar: mossDiarService != nil)
        state = .processing
        // One place decides how long the overlay may wait — see
        // `stopWatchdogSeconds`, which exists because two watchdogs in a row
        // failed to scale with the work they were guarding.
        // The batch legs' own watchdogs are what the overlay must outlast — the
        // remote one is deliberately DOUBLE the office one (both passes queue on a
        // single stdin, so the remote job can wait out the office job before it
        // even starts), so the budget takes the larger of the two whenever a remote
        // pass was actually dispatched. Zero for every other engine, which is what
        // keeps all three pre-existing budgets identical.
        // Either leg alone is enough to need the budget: a dual-stream session can
        // legitimately dispatch the remote pass while the office one is skipped.
        // ONE budget for both batch engines rather than two, because a session runs
        // at most one of them — see `stopWatchdogSeconds(batchPassSeconds:)`.
        let batchEngineActive = spectralDiarizationActive || nemoDiarizationActive
            || diarizenDiarizationActive || camPlusDiarizationActive
        let batchBudget = batchEngineActive
            && (runsSpectralPass || runsNemoPass || runsDiarizenPass
                || runsCamPlusPass || willRunRemoteDiar)
            ? Self.batchPassWatchdogSeconds(recordingLength: recordingElapsed)
                * (willRunRemoteDiar ? 2 : 1)
            : 0
        startStopWatchdog(seconds: Self.stopWatchdogSeconds(
            chunkedFullPassWindows: chunkedPlan.runsFullPass ? fullPassWindowCount : 0,
            mossFullPassWindows: mossFullPassWindowCount,
            batchPassSeconds: batchBudget))

        // Started AFTER the step list exists, because the pass reports its
        // progress into the "chunk" (and, dual-stream, "remote") rows.
        if chunkedPlan.runsFullPass {
            if let recording = lastRecordingURL, let service = modelLoader.chunkedASR {
                startChunkedFullPass(recording: recording, service: service)
            } else {
                // Unreachable by construction (`chunkedStopMode` returns `.full`
                // only when both exist), but the chunk leg must never be left
                // un-settleable — that hangs the blocking overlay for 10 minutes.
                chunkedStopLog("FULL PASS could not start — no recording or no chunked model")
                chunkedError = "The recording could not be re-transcribed"
                releaseFullPassRemoteLeg()
                checkLastChunkDone()
            }
        }

        // Independent of every stop-gate leg: it marks rows that already exist,
        // so it must not be able to hold the overlay.
        if let recording = lastRecordingURL { startOverlapDetection(recording) }

        if willRunStopPass {
            // The two BATCH branches are FIRST and neither consults
            // `continueOnStop`: there is no tail to continue from, so the
            // whole-file pass is the only mode. `runsSpectralPass` / `runsNemoPass`
            // already proved the service and the recording exist, which is why
            // these legs have no fallback arm — the `else if` chain below keeps
            // pyannote's untouched.
            if runsSpectralPass, let recording = lastRecordingURL {
                startSpectralDiarization(recording)
            } else if runsNemoPass, let recording = lastRecordingURL {
                startNemoDiarization(recording)
            } else if runsDiarizenPass, let recording = lastRecordingURL {
                startDiarizenDiarization(recording)
            } else if runsCamPlusPass, let recording = lastRecordingURL {
                startCamPlusDiarization(recording)
            } else if continueOnStop {
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
