import Foundation

/// Loads (prepares) all models needed for a recording session, step by step,
/// publishing per-model progress for the loading overlay.
///
/// Currently each step verifies the model is downloaded and reserves the slot
/// where the real MLX / Silero initialization will run when the ASR pipeline
/// is wired in — the UI and call sites won't change.
@MainActor
final class ModelLoader: ObservableObject {

    enum ItemState: Equatable {
        case pending, loading, done
        case failed(String)
        /// The step ran and correctly had nothing to do.
        ///
        /// A THIRD outcome, because the other two both lie about this case. The
        /// owner recorded two silent seconds on 2026-08-12 to watch the progress
        /// panel, and NeMo's VAD said — correctly — that there was no speech to
        /// identify speakers in. That is not a failure, and painting it red
        /// reported a broken app to someone whose app was working. A green tick
        /// would be the opposite lie: nothing was identified.
        ///
        /// `AudioRecorder.stopFailed` counts `.failed` ONLY, so a skipped step
        /// does not hold the panel open — a correct outcome must not demand an
        /// acknowledgement.
        case skipped(String)

        /// Red, and therefore something the user must be able to get out of.
        ///
        /// `.skipped` deliberately does NOT count: it is a correct outcome, and a
        /// correct outcome must not demand an acknowledgement — the same rule
        /// `AudioRecorder.stopFailed` follows for the stop panel.
        var isFailed: Bool {
            if case .failed = self { return true }
            return false
        }
    }

    struct Item: Identifiable {
        let id: String
        let name: String
        var state: ItemState = .pending
    }

    @Published private(set) var items: [Item] = []
    @Published private(set) var isLoading = false

    /// Set when a step fails; keeps the overlay on screen so the user can
    /// read what went wrong. Cleared via `dismissFailure()`.
    @Published private(set) var failureMessage: String?

    /// Close the startup overlay.
    ///
    /// ⚠ IT CLEARS THE ROWS TOO, and that is not tidiness — without it this
    /// button does nothing. `hasFailure` now also reads the rows (see below), so
    /// clearing only the message would leave the red row asserting a failure the
    /// user had just dismissed, the overlay would stay, and Close would be a dead
    /// control: the exact symptom this whole change exists to remove, rebuilt one
    /// commit later. Caught by `testDismissingReallyCloses`, which was written
    /// asserting the bug before it was noticed.
    ///
    /// Safe because `items` belongs to the overlay alone and `loadAll` rebuilds
    /// it from scratch on the next attempt.
    func dismissFailure() {
        failureMessage = nil
        items = []
    }

    /// Is there anything the user must be able to dismiss?
    ///
    /// ⚠ ANY RED ROW COUNTS, not only `failureMessage`. A client Mac showed the
    /// startup overlay with CAM++ marked failed, the models below it never
    /// attempted, the header still reading "Loading models" and NO Close button —
    /// the app unusable and Settings unreachable (2026-08-18).
    ///
    /// The two are set one line apart in `loadAll`, so how they came to disagree
    /// there is NOT established, and that is the reason this reads the rows rather
    /// than being "fixed" by trusting the variable harder: the way out must depend
    /// on what the user can SEE. A red row with no way out is a trap whatever set
    /// it, and a second variable that has to agree with the rows is one more thing
    /// that can fail to.
    ///
    /// Lives here, not in the view, so it is testable and so any other surface
    /// asking the same question gets the same answer.
    var hasFailure: Bool {
        failureMessage != nil || items.contains { $0.state.isFailed }
    }

    /// Rows for models this session TORE DOWN, shown above the load rows.
    ///
    /// Switching a model used to be invisible: the old sidecar was terminated
    /// silently and the overlay only ever listed what was being loaded, so a
    /// user changing engine had no way to see that the previous one had gone —
    /// the same complaint the owner raised on 2026-08-12.
    ///
    /// They are recorded as `.done` rather than animated, and that is honest
    /// rather than lazy: `terminate()` closes stdin and signals the process, it
    /// does not wait, so there is no interval to animate. The value is that the
    /// rows STAY on screen beside the load rows for the seconds a model load
    /// takes — "Unloaded Nemotron ASR" sitting above "Parakeet TDT 0.6b v3" is
    /// the whole answer to what just happened.
    private var unloadRows: [Item] = []

    /// Record one teardown. Call ONLY when something was really loaded — a row
    /// for a service that was already nil would claim work that never happened.
    private func noteUnload(_ name: String) {
        unloadRows.append(Item(id: "unload-\(unloadRows.count)-\(name)",
                               name: "Unloaded \(name)", state: .done))
    }

    /// Rows for the models a model SWITCH is about to replace.
    ///
    /// The teardown block in `loadAll` never sees these. Switching realtime from
    /// Nemotron to Parakeet leaves realtime ENABLED, so no `if !wants…` branch
    /// fires; the old sidecar is replaced inside its own load step instead. That
    /// is the commonest way a user changes a model, and without this it was the
    /// one case that showed no unload at all.
    ///
    /// Detected HERE, before the overlay list is built, so the unload gets its
    /// own row ABOVE the load rows rather than being folded into the loading
    /// model's line.
    ///
    /// Safe to compare against `fromSettings()` twice — here and again in the
    /// load step — because it is a pure read of settings that are already locked
    /// for this session. Both calls see the same values by construction.
    ///
    /// Runs AFTER the teardown block, which is what stops a double row: a
    /// service being switched OFF has already been terminated and nil'd there,
    /// so `if let` finds nothing here.
    private func noteReplacements() {
        if let old = realtimeASR, old.config != RealtimeASRService.Config.fromSettings() {
            noteUnload(ModelCatalog.realtimeModel(id: old.config.modelID).name)
        }
        if let old = chunkedASR, old.config != ChunkedASRService.Config.fromSettings() {
            noteUnload(old.config.modelName)
        }
        if let old = mossDiarization,
           old.config != ChunkedASRService.Config.mossDiarization() {
            noteUnload(ModelCatalog.mossDiarization.name)
        }
    }

    /// Refuse to start the session BEFORE any model loads, showing the reason in
    /// the same overlay a load failure uses (one failed row + `failureMessage`,
    /// which is what keeps the overlay on screen). The only caller today is the
    /// dual-stream + Voxtral refusal in `AudioRecorder.prepareAndCapture`: the
    /// configuration is unworkable, so loading a 4B model first would waste ~30 s
    /// before saying so.
    func failStartup(step: String, message: String) {
        items = [Item(id: "startup", name: step, state: .failed(message))]
        failureMessage = message
    }

    /// Silero sidecar, started during loading; nil → heuristic VAD fallback.
    /// Kept alive across sessions so the model only loads once.
    private(set) var sileroVAD: SileroVADService?

    /// Realtime ASR sidecar (Nemotron or Parakeet, per `realtime.model`);
    /// recreated only when its settings change — INCLUDING the engine, which is
    /// part of `Config` precisely so that switching it replaces the process
    /// rather than leaving the previous model answering.
    /// ONE process serving both streams: its `office` and `remote` lanes keep
    /// separate buffers inside the sidecar, so a Remote channel costs no second
    /// model load and no second process — see `RealtimeASRService`.
    private(set) var realtimeASR: RealtimeASRService?

    /// Chunked ASR sidecar (Qwen3/Whisper/Voxtral per settings); persistent.
    private(set) var chunkedASR: ChunkedASRService?

    /// Forced-aligner sidecar (Qwen3-ForcedAligner); persistent, kept alive
    /// across sessions like `diarization` so its 1.2 GB loads once. nil whenever
    /// this session does not use word alignment — which is what makes the whole
    /// alignment path a natural no-op rather than a set of scattered guards.
    private(set) var aligner: AlignerService?

    /// Diarization sidecar (pyannote community-1 on MPS); persistent.
    /// nil whenever `diarization.engine` is MOSS — there is no pyannote process
    /// at all then, which is what makes every pyannote-dependent path
    /// (remote diarization, overlap repair, profiles) a natural no-op.
    ///
    /// Turns only, no identity: since 2026-07-30 this sidecar answers who-spoke-when
    /// with run-local labels and `embedding` says who those people are.
    private(set) var pyannote: PyannoteService?

    /// SPECTRAL diarization sidecar (vendored `diarize`, CPU); persistent.
    /// nil unless `diarization.engine` is `spectral` — which is what makes every
    /// pyannote-dependent path (live chunks, tail passes, overlap repair) a
    /// natural no-op under this engine, exactly as it is under MOSS.
    ///
    /// It answers a WHOLE-FILE pass and nothing else, so it is asked exactly once
    /// per stream per session, at Stop. `embedding` is loaded alongside it — see
    /// `DiarizationStack.usesSpeakerIdentity`.
    private(set) var spectral: SpectralService?

    /// NEMO diarization sidecar (NVIDIA NeMo `ClusteringDiarizer`, MPS);
    /// persistent. nil unless `diarization.engine` is `nemo` — which is what makes
    /// every pyannote-dependent path (live chunks, tail passes, overlap repair from
    /// intersecting turns) a natural no-op under this engine, exactly as it is
    /// under spectral and MOSS.
    ///
    /// Like spectral it answers a WHOLE-FILE pass and nothing else, so it is asked
    /// exactly once per stream per session, at Stop. `embedding` is loaded
    /// alongside it — see `DiarizationStack.usesSpeakerIdentity`. Runs in its OWN
    /// interpreter `.venv-nemo`; a missing one is a loud setup error, never a
    /// silent fall-back to the main `.venv`.
    private(set) var nemo: NemoService?

    /// The DiariZen engine's sidecar, in `.venv-diarizen`. Persistent; nil unless
    /// `diarization.engine` is `diarizen`.
    private(set) var diarizen: DiarizenService?
    /// The CAM++ engine's sidecar, in the MAIN `.venv` — unlike NeMo, DiCoW and
    /// DiariZen it needs no interpreter of its own. Persistent; nil unless
    /// `diarization.engine` is `campplus`.
    private(set) var camPlus: CamPlusService?

    /// The VibeVoice engine's sidecar. The ONLY diarization engine with its own
    /// interpreter (`.venv-vibevoice`): transformers <5.0 against the MLX
    /// stack's 5.x. Persistent; nil unless `diarization.engine` is `vibevoice`.
    private(set) var vibeVoiceDiar: VibeVoiceDiarizationService?

    /// Speaker-identity sidecar (WeSpeaker + the two profile stores); persistent.
    /// The other half of the former `diarize-service.py`.
    ///
    /// Loaded and torn down in LOCKSTEP with `pyannote`, and that is not a
    /// convenience: a pyannote turn has no name until this service supplies one,
    /// and `SpeakerTurn` has no representation for an unidentified turn. So both
    /// exist or neither does — see `buildSteps` (two steps, embedder first) and
    /// the MOSS-engine teardown below (both terminated).
    private(set) var embedding: WeSpeakerService?

    /// A SECOND MOSS process, used only for its speaker segmentation while a
    /// different model does the ASR. nil in the primary MOSS+MOSS mode, where
    /// `chunkedASR` IS the MOSS process and one 3.6 GB load serves both roles —
    /// see `needsSecondMossProcess`.
    private(set) var mossDiarization: ChunkedASRService?

    /// Overlap-repair sidecar (MossFormer2 librimix-2spk); persistent.
    /// Only started when overlap repair is enabled in Settings.
    private(set) var overlapRepair: OverlapRepairService?

    /// Overlap-repair sidecar (DiCoW v3.3 target-speaker ASR); persistent.
    /// The alternative to `overlapRepair` — only one engine loads per session,
    /// whichever is picked in Settings → Models → Overlap.
    private(set) var dicowRepair: DicowService?

    /// Overlap DETECTION sidecar (pyannote segmentation, 32 MB, CPU); persistent.
    /// Marks where two people spoke at once; never recovers words. Loaded only
    /// when the feature is on AND there are rows to mark — see `wantsOverlapDetect`.
    private(set) var overlapDetect: OverlapDetectService?

    /// Whether this session should drive the realtime sidecar's REMOTE lane.
    /// Both conditions are necessary: no Remote channel means there is no second
    /// stream at all, and with realtime captions off there is nothing for either
    /// lane to draw. Pure, so the "inert without a Remote channel" guarantee is
    /// testable without audio hardware.
    nonisolated static func wantsRemoteRealtime(remoteChannel: Int?,
                                                realtimeEnabled: Bool) -> Bool {
        remoteChannel != nil && realtimeEnabled
    }

    /// Whether this session keeps the REALTIME sidecar alive.
    ///
    /// One line, and it earns its own function for the same reason `wantsAligner`
    /// and `wantedOverlapEngine` do: it is read by BOTH `buildSteps` (does the
    /// overlay show a row, and does the process start?) and the teardown in
    /// `loadAll` (is a leftover process stopped?). Those two computing the rule
    /// separately is exactly how the bug below happened, twice now in this
    /// codebase.
    ///
    /// THE BUG (found 2026-08-05, fixed here). `buildSteps` has always loaded
    /// Nemotron only when `realtime.enabled` is on, but `loadAll` had NO
    /// `terminate()` for it anywhere — the only one in the file sat INSIDE the
    /// load step, to replace the process when its settings changed. Services are
    /// kept alive across sessions, so switching realtime captions off left the
    /// sidecar resident until the app quit, with nothing able to ask it anything:
    /// `AudioRecorder` reads `modelLoader.realtimeASR?.office` only when the same
    /// flag is on, so the behaviour was already correct and ONLY the memory was
    /// wrong — which is why it never surfaced. Measured on the owner's Mac:
    /// **1.70 GB**, the second-largest process in a default session.
    ///
    /// This was the LAST persistent service without a want/teardown pair. The VAD
    /// got one on 2026-07-31, the two overlap engines the same day, and both
    /// diarization stacks with `wantedDiarizationStack`.
    ///
    /// `detachRemoteLane` is NOT folded in here. It is the narrower rule — keep
    /// the process, drop one lane's callback — and it applies only when the
    /// process survives; see the call site, where terminating makes detaching
    /// moot.
    nonisolated static func wantsRealtime(realtimeEnabled: Bool) -> Bool {
        realtimeEnabled
    }

    /// The MOSS diarization engine id (`diarization.engine`).
    nonisolated static let mossEngineID = "moss"
    /// The pyannote diarization engine id — the default.
    nonisolated static let pyannoteEngineID = "pyannote"
    /// The SPECTRAL diarization engine id — the third engine (2026-08-04).
    /// Same string as its catalog id, unlike MOSS; see
    /// `ModelCatalog.diarizationEngineValue` for why that difference is not an
    /// inconsistency.
    nonisolated static let spectralEngineID = "spectral"
    /// The NEMO diarization engine id — the fourth engine (2026-08-07).
    /// Same string as its catalog id, like spectral and pyannote and unlike MOSS;
    /// see `ModelCatalog.diarizationEngineValue` for why that difference is not an
    /// inconsistency.
    nonisolated static let nemoEngineID = "nemo"
    nonisolated static let diarizenEngineID = "diarizen"
    /// The CAM++ diarization engine id — the sixth engine (2026-08-11). Same
    /// string as its catalog id, like every engine except MOSS.
    nonisolated static let camPlusEngineID = "campplus"
 nonisolated static let vibeVoiceEngineID = "vibevoice"

    /// Whether this session loads the forced aligner.
    ///
    /// ONE rule, read by both `buildSteps` (does the overlay show a row?) and
    /// `loadAll` (is a leftover sidecar torn down?). They must agree: a step
    /// without a teardown leaves a 1.2 GB process alive for a session that will
    /// never ask it anything, and a teardown without a step would kill the
    /// service the session is about to use. Pure, like `wantsRemoteRealtime`.
    ///
    /// MOSS is excluded: it runs in its own sidecar and already returns its own
    /// per-segment times. The split makes aligning MOSS text POSSIBLE for the
    /// first time (the aligner no longer lives inside the MLX ASR process), but
    /// the owner deferred cross-chunk MOSS work, so it stays excluded and the
    /// Aligner tab says so.
    /// Also false with the chunked pass switched off: the aligner splits CHUNKED
    /// segments into words, and there are no chunked segments then. Loading 1.2 GB
    /// to align text that will never arrive is the same waste every other rule
    /// here exists to prevent.
    nonisolated static func wantsAligner(alignEnabled: Bool,
                                         chunkedID: String,
                                         chunkedEnabled: Bool) -> Bool {
        alignEnabled && chunkedEnabled && chunkedID != "moss"
    }

    /// The aligner does not load AND its job is being done anyway.
    ///
    /// `wantsAligner` returning false has TWO causes and the rail must not treat
    /// them alike (owner, 2026-08-12 — *"ini masih gak di atas aligner tabnya"*):
    ///
    ///   * **the chunked pass is off** — no text, no segments, no attribution.
    ///     Nothing happens, so the tab is genuinely NOT USED;
    ///   * **the model is MOSS** — the work HAPPENS, MOSS just does it itself, by
    ///     emitting a separate timed segment per speaker instead of handing the
    ///     app words to sort by time. Filing that under "NOT USED BY YOUR MODELS"
    ///     tells the user the job is not being done, when it is.
    ///
    /// This is the Detect-overlap lesson applied a second time: DiariZen's own
    /// detection was filed under NOT USED on 2026-08-10 and reverted the same
    /// day, because a heading that says "nothing here applies" is wrong about a
    /// feature that is working. The rail shows `built in` instead, exactly as it
    /// does for diarization under MOSS (`included`) and for pyannote's own
    /// overlap marking.
    ///
    /// ONE function, read by the rail's grouping, the rail's status and the tab —
    /// three readers, so this is precisely where two half-rules living apart
    /// would come to disagree.
    nonisolated static func alignmentIsBuiltIn(chunkedID: String,
                                               chunkedEnabled: Bool) -> Bool {
        chunkedEnabled && chunkedID == "moss"
    }

    /// Whether this session loads the CHUNKED ASR sidecar at all.
    ///
    /// THE MASTER SWITCH, owner-requested 2026-08-06, and it is not the same thing
    /// as `chunked.finalPass` — which was already in the Chunked tab and reads
    /// like an on/off but is not one. That key governs only the extra pass AFTER
    /// Stop; with it off, chunked ASR still runs all meeting and still produces
    /// the transcript. This switch is the one that stops it.
    ///
    /// WHAT TURNING IT OFF COSTS, stated here because the cost is the feature:
    /// the meeting has no accurate transcript at all. The realtime engine's text
    /// becomes the only text that audio will ever have, and the existing
    /// `.none` stop plan already refuses to sweep it away for exactly that reason.
    /// The Chunked tab says so in as many words before the toggle.
    ///
    /// Pure and static, read by BOTH `buildSteps` and the teardown in `loadAll`,
    /// like every other want-rule here since the three want/teardown leaks.
    nonisolated static func wantsChunked(chunkedEnabled: Bool) -> Bool {
        chunkedEnabled
    }

    /// Which diarization engines the Settings UI offers for a chunked model.
    ///
    /// Owner, 2026-08-06, and it is a BICONDITIONAL — MOSS chunked ⟺ MOSS
    /// diarization — written as one function precisely because two half-rules
    /// living apart is how they come to disagree:
    ///
    ///   * with MOSS as the ASR, ONLY MOSS is offered;
    ///   * with any other ASR, MOSS is NOT offered.
    ///
    /// The second half stops a second 3.6 GB process and ~6.4 s more per chunk
    /// being spent to re-label text another model already produced. The first
    /// half is the owner's decision to keep MOSS as the single all-in-one pairing
    /// its authors built.
    ///
    /// THE COST OF THE FIRST HALF, stated because it removes something that
    /// worked: "MOSS as ASR + pyannote as diarizer" was a real mode, and the
    /// better one on paper — it keeps saved profiles, renaming, `spk` confidence,
    /// overlap tags and overlap repair, all of which MOSS-as-diarizer gives up.
    /// It is no longer reachable from Settings.
    ///
    /// UI AVAILABILITY ONLY, deliberately NOT a loader rule.
    /// `needsSecondMossProcess` and `wantedDiarizationStack` still support every
    /// combination and their tests still pin them, so relaxing this one function
    /// restores both modes intact. What must never happen is an engine staying
    /// STORED while no card on screen selects it — a setting in force with
    /// nothing showing it — which is why `selectChunkedModel` and
    /// `DiarizationTab.onAppear` both correct it, and say so.
    nonisolated static func diarizationEngineIsSelectable(_ engineValue: String,
                                                          chunkedID: String) -> Bool {
        chunkedID == "moss" ? engineValue == mossEngineID : engineValue != mossEngineID
    }

    /// The engine to fall back to when the stored one is no longer offered.
    nonisolated static func fallbackDiarizationEngine(chunkedID: String) -> String {
        chunkedID == "moss" ? mossEngineID : pyannoteEngineID
    }

    /// Whether this session needs a SECOND MOSS process for diarization.
    ///
    /// False when the chunked model is already MOSS: that one sidecar's `final`
    /// carries both the text and the segments, so a second process would be a
    /// second 3.6 GB load and a second ~6.4 s per chunk for output we already
    /// have. It is also why the overlay must not show two rows for one process.
    ///
    /// TRUE AGAIN WHEN THE CHUNKED PASS IS SWITCHED OFF, even with the chunked
    /// MODEL still set to MOSS — and this is the coupling that makes the master
    /// switch safe. In MOSS+MOSS mode the diarizer borrows its segments from the
    /// chunked process; with no chunked process there is nothing to borrow, so
    /// the diarizer needs its own. Without this the MOSS engine would load
    /// nothing and produce no speakers at all, silently.
    ///
    /// Pure, so the "one process in the primary mode" guarantee is testable
    /// without loading anything (the `wantsRemoteRealtime` precedent).
    /// Which diarization stack this session should keep alive, or nil for none.
    ///
    /// Pure and static, and fed to BOTH the load step and the teardown — the two
    /// computing this separately is EXACTLY how the bug below happened.
    ///
    /// THE BUG (found 2026-07-31, fixed here). The load step asked
    /// `diarization.enabled`; the teardown only asked whether the engine was
    /// MOSS. So turning the master switch OFF stopped pyannote being LOADED but
    /// never STOPPED one already running, and services are kept alive across
    /// sessions. The next recording then found `modelLoader.pyannote != nil`,
    /// which is the only thing `diarizeLiveChunk` guards on besides
    /// `diarization.live` — so **diarization kept running after the user
    /// switched it off**, and 1.17 GB stayed resident with it. It only bit when
    /// the app was not restarted in between, which is why it survived so long.
    ///
    /// The MOSS branch had the memory half of the same bug: `mossDiarization`
    /// was kept whenever `needsSecondMossProcess` said so, and that function
    /// never knew about `diarization.enabled` either. Its behaviour half was
    /// already covered, because `configureMoss` reads the switch itself.
    nonisolated static func wantedDiarizationStack(diarizationEnabled: Bool,
                                                   engine: String,
                                                   chunkedID: String,
                                                   chunkedEnabled: Bool) -> DiarizationStack? {
        guard diarizationEnabled else { return nil }
        switch engine {
        case mossEngineID:
            // NEVER nil any more. MOSS+MOSS needs no second process — one sidecar
            // already returns text and segments — but it does need the embedder,
            // so that mode has its own case rather than falling through to
            // "nothing", which used to deny it identity by accident.
            return needsSecondMossProcess(chunkedID: chunkedID, engine: engine,
                                          chunkedEnabled: chunkedEnabled)
                ? .mossSecondProcess : .mossOwnASR
        case spectralEngineID:
            return .spectral
        case nemoEngineID:
            return .nemo
        case diarizenEngineID:
            return .diarizen
        case camPlusEngineID:
            return .camPlus
        case vibeVoiceEngineID:
            return .vibeVoiceDiar
        default:
            // An UNKNOWN stored value still resolves to pyannote, which is what
            // `diarizationEngine(forEngine:)` shows and what the recorder's
            // `?? pyannoteEngineID` defaults produce. Keeping the fall-through
            // here (rather than an `== pyannoteEngineID` test) is what stops a
            // stale/garbage setting from silently loading nothing at all.
            return .pyannote
        }
    }

    /// What a session's diarization is served by. `nil` — the fourth case — means
    /// "nothing", which is both "switched off" and "MOSS doing both jobs in the
    /// chunked process".
    enum DiarizationStack: Equatable, Hashable {
        case pyannote           // pyannote-service + wespeaker-service
        case spectral           // spectral-service + wespeaker-service
        case nemo               // nemo-service (.venv-nemo) + wespeaker-service
        case diarizen           // diarizen-service (.venv-diarizen) + wespeaker-service
        case camPlus            // campplus-service (main .venv) + wespeaker-service
        case vibeVoiceDiar      // vibevoice-diar-service (.venv-vibevoice) + wespeaker
        case mossSecondProcess  // a second MOSS process, ASR done by another model
        case mossOwnASR         // MOSS is ALSO the chunked model — no second
                                // process to load, but identity is still wanted

        /// Whether this stack needs the WeSpeaker identity sidecar.
        ///
        /// ALL FIVE do, since 2026-08-05. Spectral emits run-local labels exactly
        /// as pyannote does, so profiles, renaming and `spk` worked under it from
        /// the day of the pyannote/wespeaker split with no new identity code —
        /// and MOSS has now joined them for the same reason, one call later:
        /// its per-call `S01` labels are ALSO run-local, so feeding a whole
        /// meeting's MOSS turns through `identify` is what makes `Speaker 3` mean
        /// one person across every window instead of one window.
        ///
        /// Before that, MOSS "named its own speakers and never reached the
        /// stores", which is why the stop step could only honestly be called
        /// RE-LABELLING: it re-numbered, it did not recognise. That one gap was
        /// also why a MOSS session had no saved profiles, no renaming and no `spk`
        /// — four complaints, one cause.
        ///
        /// Read by the teardown so the embedder is dropped only when diarization
        /// is off entirely.
        var usesSpeakerIdentity: Bool { true }
    }

    /// The overlap-repair engine id this session should keep alive, or nil for
    /// "neither" — the single source of truth for BOTH the load step and the
    /// teardown above, so the two can never disagree.
    ///
    /// Pure and static for the same reason `wantsAligner` and
    /// `needsSecondMossProcess` are: the rule is what matters, and it should be
    /// testable without a loader, a sidecar process or UserDefaults.
    ///
    /// MOSS and SPECTRAL used to return nil unconditionally, and the reason was
    /// never "not pyannote" — it was that both repair engines need OVERLAP
    /// REGIONS, and the only source of them was `AudioRecorder.overlapRegions()`,
    /// which finds turns whose spans intersect. That needs a diarizer able to put
    /// two speakers on one instant. MOSS's segments tile exactly and spectral's
    /// Viterbi smoothing assigns one label per frame, so under either engine
    /// `repairWindows` always came back empty and loading a 6 GB DiCoW to discover
    /// that once per meeting was pure waste.
    ///
    /// THE OVERLAP DETECTOR CHANGES THAT PREMISE, which is why `detectEnabled` is
    /// a parameter now. It reads the audio directly with pyannote's segmentation
    /// network and needs no turns at all, so under those two engines regions exist
    /// exactly when it is switched on. The rule therefore says what it always
    /// meant: repair runs when there is somewhere to run it.
    ///
    /// pyannote is untouched — it marks overlap itself, so its regions never
    /// depended on the detector and `detectEnabled` cannot take repair away from
    /// it. Written as two `!=` tests rather than `== pyannoteEngineID` so an
    /// unknown stored engine value behaves the same here as it does in
    /// `wantedDiarizationStack` (which resolves it to pyannote): a session that
    /// really is running pyannote must not silently lose repair.
    ///
    /// `detectEnabled` deliberately has NO default value. A default would let a
    /// call site forget it and silently fall back to "no repair under MOSS" —
    /// exactly the invisible failure direction this project keeps paying for.
    nonisolated static func wantedOverlapEngine(repairEnabled: Bool,
                                                engineID: String,
                                                diarEngine: String,
                                                detectEnabled: Bool) -> String? {
        guard repairEnabled else { return nil }
        // NeMo joins MOSS and spectral for the SAME structural reason, not by
        // analogy: NME-SC assigns exactly one label per instant, so its turns can
        // never intersect and `overlapRegions()` is empty by construction under it.
        //
        // DIARIZEN IS DELIBERATELY ABSENT FROM THIS LIST. It was added here when it
        // landed, on the assumption that a whole-file batch engine must resemble the
        // other whole-file batch engines. It does not, on this question: its powerset
        // head predicts two-speaker frames DIRECTLY (11 classes = 1 + 4 + 6 pairs),
        // and its turns were measured intersecting — 11 pairs over 13.30 s on
        // `recordings/Overlap123.wav`. Requiring the detector under it meant a 32 MB
        // second opinion had to be switched on before repair would load at all,
        // while the regions the engine had already produced were thrown away. See
        // `AudioRecorder.usesDetectedRegionsForRepair` for the full measurement.
        // CAM++ joins this set, not DiariZen's: its clustering assigns exactly
        // one label per window, so its turns never intersect and it can no more
        // mark overlap than spectral or NeMo can. Stated PER ENGINE rather than
        // derived, per the rule the DiariZen work established — "does this
        // engine mark its own overlap" has the same answer for pyannote and
        // DiariZen but different verdicts, so deriving one from the other would
        // silently move pyannote the next time someone tidies this.
        // VibeVoice joins this set too, and again STRUCTURALLY rather than by
        // resemblance to the other whole-file engines: its spans are cut on the
        // chunk grid and then merged by label, so two speakers can never hold one
        // instant and `overlapRegions()` is empty under it whatever the audio
        // does. It is speaker-attributed ASR like MOSS and lands with MOSS.
        if diarEngine == mossEngineID || diarEngine == spectralEngineID
            || diarEngine == nemoEngineID || diarEngine == camPlusEngineID
            || diarEngine == vibeVoiceEngineID {
            guard detectEnabled else { return nil }
        }
        return engineID
    }

    /// Whether this session keeps the overlap DETECTOR alive.
    ///
    /// Pure and static, and read by BOTH the load step and the teardown — the
    /// pattern every other service here follows since the three want/teardown
    /// bugs of 2026-07-31 and 2026-08-05.
    ///
    /// Needs diarization ON, because the detector marks ROWS and a session with no
    /// speakers has none to mark.
    ///
    /// **THIS ANSWERS ONE NARROW QUESTION: does the pyannote segmentation SIDECAR
    /// start?** Not "is overlap detected" — under DiariZen it very much is.
    ///
    /// False for DiariZen (owner, 2026-08-10) because that engine IS the detector:
    /// its powerset head predicts two-speaker frames directly (11 classes =
    /// 1 + 4 + 6 pairs), so the regions fall out of the diarization pass that
    /// already ran. A second 32 MB network over the same audio would answer a
    /// question already answered. The Detect overlap TAB stays fully available —
    /// it shows DiariZen as the detection model and its switch turns the marking
    /// on and off (`AudioRecorder.diarizenOverlapMarking`). Do not read this
    /// function as "the tab does not apply"; that was tried and reverted.
    ///
    /// **pyannote is deliberately NOT refused, and the asymmetry is the owner's
    /// call, not an oversight.** It also reports overlap itself, so the detector is
    /// redundant there too — but that engine has offered it since 2026-08-06 with
    /// the reason written down (a user switching it on has asked for a second
    /// opinion, and refusing silently is the substitution this project forbids),
    /// and taking it away would change long-standing behaviour to buy consistency
    /// nobody asked for. The rule is therefore stated per engine, not derived from
    /// "does this engine mark its own overlap" — those two questions have the same
    /// answer for pyannote and DiariZen and different verdicts, so deriving one
    /// from the other would silently move pyannote the next time this is touched.
    ///
    /// `diarEngine` has NO default value on purpose, the `detectEnabled` rule: a
    /// default lets a call site forget it and quietly load a detector for a session
    /// that must not have one.
    nonisolated static func wantsOverlapDetect(detectEnabled: Bool,
                                               diarizationEnabled: Bool,
                                               diarEngine: String) -> Bool {
        guard diarEngine != diarizenEngineID else { return false }
        return detectEnabled && diarizationEnabled
    }

    /// Whether this engine ACTS on a pinned speaker count.
    ///
    /// Measured 2026-08-10, auto vs pinned on the same two files, rather than read
    /// off the wire protocol — three sidecars accept `num_speakers` and only some
    /// of them are changed by it:
    ///
    /// | engine | Meeting5People (5) | Overlap123 (3) |
    /// |---|---|---|
    /// | pyannote | 5 → 5 | 3 → 3 |
    /// | NeMo | 5 → 5 | 3 → 3 |
    /// | **spectral** | 5 → 5 | **13 → 3** |
    ///
    /// So the count is worth setting for SPECTRAL above all: its GMM-BIC counting
    /// is the weak stage (13 speakers on a 3-person clip; 20 on a 67-minute
    /// meeting), while its clustering is fine. pyannote and NeMo count correctly
    /// on their own, but both honour the number, so both stay true here — an
    /// engine that would ACT on a count belongs in this set whether or not our two
    /// short fixtures happen to need it.
    ///
    /// **DiariZen is false, and that is a property of the sidecar, not a policy:**
    /// `diarizen-service.py` does not read `num_speakers` at all. Its count comes
    /// from the checkpoint's own bounds and `min_cluster_size`, and pinning through
    /// those was measured ONE-DIRECTIONAL — it obeys a smaller count and ignores a
    /// larger one, so the only thing a number could do there is merge people who
    /// should be separate (owner: the count must stay automatic).
    ///
    /// MOSS is false because it names its own speakers and takes no count at all.
    ///
    /// **CAM++ is true**, and like pyannote and NeMo it is true on the sidecar's
    /// behaviour rather than on need: it counts correctly unaided on both
    /// known-answer fixtures (5 and 3), and it reads `num_speakers` and clusters
    /// to exactly that many when one is sent — measured, pinning `Overlap123` to
    /// 2 returns 2. An engine that would ACT on a count belongs in this set
    /// whether or not our two short fixtures happen to need it.
    /// Does this engine report overlapping speech in its OWN turns?
    ///
    /// Two do. **pyannote** has always done it, and **DiariZen**'s powerset
    /// Conformer head predicts two-speaker frames directly (11 classes = 1
    /// silence + 4 single-speaker + 6 PAIRS) — measured, 11 intersecting pairs
    /// over 13.30 s on `recordings/Overlap123.wav`. The other four assign
    /// exactly one speaker per instant, so their turns can never intersect and
    /// `overlapRegions()` is empty under them however the audio sounds.
    ///
    /// THE COMPLEMENT OF `AudioRecorder.usesDetectedRegionsForRepair`, and
    /// `testTheOverlapRuleAgreesWithTheRecorder` sweeps every engine to keep the
    /// two from drifting. It exists as a pure function because the recorder's
    /// version is keyed on per-session `*Active` flags, which a Settings tab has
    /// no access to — and the tab had been telling users only TWO engines were
    /// affected since NeMo landed, then CAM++ made it two of four.
    nonisolated static func marksItsOwnOverlap(diarEngine: String) -> Bool {
        diarEngine == pyannoteEngineID || diarEngine == diarizenEngineID
    }

    /// Does this engine's SIDECAR read `num_speakers` at all?
    ///
    /// FIVE OF SIX, since DiariZen joined on 2026-08-14 (owner: the picker must
    /// work on every engine). Its sidecar had ignored the field by an earlier
    /// decision of the owner's own — *"saya ingin speakernya itu auto"* — and now
    /// applies it as instance bounds; see `diarizen-service.py`.
    ///
    /// ⚠ **MOSS IS THE ONE THAT CANNOT, and it is not an omission to be closed
    /// later.** It is speaker-attributed ASR: the labels come out of a 0.9B
    /// language model as `[Sxx]` tags in generated text. There is no clustering
    /// stage to bound and no count parameter anywhere in the checkpoint — the only
    /// lever is the prompt, and steering it was MEASURED inert on 2026-07-31
    /// (default vs "output in Indonesian" vs "output in Chinese" all returned
    /// identical English output). Wiring a control to it would be the Granite /
    /// Voxtral language-picker defect rebuilt knowingly.
    ///
    /// Stated as a LIST rather than `!= mossEngineID` on purpose: a seventh engine
    /// must be examined and added, not admitted by default. That is the
    /// `default:`-fall-through trap this project has already paid for twice
    /// (`ChunkedASRModel.scriptName`, and the rail printing `pyannote` for a
    /// DiariZen session).
    nonisolated static func honoursSpeakerCount(diarEngine: String) -> Bool {
        diarEngine == pyannoteEngineID || diarEngine == spectralEngineID
            || diarEngine == nemoEngineID || diarEngine == camPlusEngineID
            || diarEngine == diarizenEngineID
    }

    /// Whether a pinned speaker count REACHES the engine in THIS session.
    ///
    /// `honoursSpeakerCount` answers a narrower question — does the sidecar read
    /// the field at all — and it is a property of the engine. This one adds the
    /// only case where the answer depends on how the session is configured, and it
    /// exists because the audit of 2026-08-10 found the SPK chip lit for a session
    /// that silently discards the number.
    ///
    /// **pyannote sends `num_speakers` only on a FULL stop pass.** Its live windows
    /// and its tail are `chunk` jobs, and `diarizeChunk` carries no count —
    /// deliberately, and correctly: a 30 s window or a few seconds of tail need not
    /// contain every speaker in the meeting, so forcing the meeting's count onto
    /// one would make it invent people.
    ///
    /// | `finalPass` | pass at stop | count sent |
    /// |---|---|---|
    /// | false | none, or a tail (`chunk`) | **no** |
    /// | true | full (`final`) | yes |
    ///
    /// ⚠ **`continueOnStop` USED TO BE A THIRD ROW HERE and is no longer a
    /// parameter** (owner, 2026-08-14: *"pas On mah diulang diarize dari awal
    /// sampai akhir"*). A stop pass is now unconditionally a full pass, so
    /// `true`/`true` — which sent no count while the toggle that set it was
    /// invisible — cannot be expressed any more. Removing the parameter rather
    /// than passing a constant is the `spectralRefusalMessage` precedent: a
    /// configuration that can no longer occur should not still be testable, or the
    /// dead leg outlives the reason anyone remembers for it.
    ///
    /// The BATCH engines never read `finalPass` for this (their stop pass IS the
    /// labels), so they pass straight through on `honoursSpeakerCount`.
    nonisolated static func speakerCountReachesEngine(diarEngine: String,
                                                      diarizationEnabled: Bool,
                                                      finalPass: Bool) -> Bool {
        guard diarizationEnabled, honoursSpeakerCount(diarEngine: diarEngine) else {
            return false
        }
        guard diarEngine == pyannoteEngineID else { return true }
        return finalPass
    }

    nonisolated static func needsSecondMossProcess(chunkedID: String,
                                                   engine: String,
                                                   chunkedEnabled: Bool) -> Bool {
        guard engine == mossEngineID else { return false }
        return !chunkedEnabled || chunkedID != "moss"
    }

    /// Stop every sidecar and drop every reference — the app holds no model
    /// process afterwards (owner, 2026-09-02: "pas stop selesai semua di unload
    /// model modelnya jadi gak akan ada proses apapun").
    ///
    /// ⚠ WHERE THIS MAY BE CALLED FROM, and it is not `leaveProcessing()`.
    /// Three paths leave `.processing`, and TWO OF THEM DELIBERATELY LEAVE WORK
    /// RUNNING: the 600 s watchdog ("the meeting still HAPPENED and the app is
    /// unblocked") and `continueInBackground` ("the user stopped WAITING; the
    /// passes did not stop RUNNING"). Terminating a sidecar mid-pass on either
    /// would delete text that was still going to land — the over-deletion
    /// direction this project ranks worst, and invisible because the transcript
    /// simply ends early.
    ///
    /// The one safe caller is `checkStopProcessingDone()`, whose own guard is the
    /// proof: `lastChunkDone`, `remoteLastChunkDone`, `finalDiarDone`,
    /// `remoteFinalDiarDone`, `mossLastChunkDone`, `!overlapRepairing` and
    /// `repairTask == nil` all hold, so nothing is in flight to kill.
    ///
    /// ⚠ THE COST, stated rather than discovered later: the next recording pays
    /// the full load again — measured ~22 s for a default six-sidecar session on
    /// the 64 GB Mac, and MOSS alone is 9.2 s. What it buys is the whole working
    /// set back between meetings (~10 GB measured), which is why it is worth it
    /// on the client's 16 GB machine and why the owner asked for it.
    ///
    /// Nothing after Stop needs a live process, checked rather than assumed:
    /// PDF export is pure Swift, `SpeakerProfileStore.rename` reads and writes
    /// the JSON itself, and `PositionDiarizer.rename` is an in-memory object.
    func unloadAll() {
        // ⚠ NO `noteUnload` CALLS HERE, and their absence is the point. `items`
        // is built from `unloadRows` at exactly one place — inside `loadAll`,
        // AFTER that same function has cleared `unloadRows` on its first line. So
        // a row appended from outside `loadAll` is wiped before anything can
        // render it. The first version of this function called `noteUnload`
        // fourteen times and every one was provably dead: appended at Stop,
        // discarded at the next Start, never on screen.
        //
        // Left out rather than given a surface, because there is no moment to
        // show them: the stop overlay is already coming down when this runs. If
        // one is ever wanted ("unloaded after the last meeting" above the load
        // rows, explaining the ~22 s wait), `items` is the thing to build, not
        // `unloadRows` to fill.
        //
        // Named individually rather than looped because they are distinct types;
        // `layout/every-service-is-unloaded-at-stop` DERIVES the population from
        // the `?.terminate()` calls in this file and fails if one is missing, so
        // the list cannot silently fall behind the way a hand-written list does.
        realtimeASR?.terminate();      realtimeASR = nil
        chunkedASR?.terminate();       chunkedASR = nil
        aligner?.terminate();          aligner = nil
        pyannote?.terminate();         pyannote = nil
        spectral?.terminate();         spectral = nil
        nemo?.terminate();             nemo = nil
        diarizen?.terminate();         diarizen = nil
        camPlus?.terminate();          camPlus = nil
        vibeVoiceDiar?.terminate();    vibeVoiceDiar = nil
        embedding?.terminate();        embedding = nil
        mossDiarization?.terminate();  mossDiarization = nil
        overlapRepair?.terminate();    overlapRepair = nil
        dicowRepair?.terminate();      dicowRepair = nil
        overlapDetect?.terminate();    overlapDetect = nil
        // No public terminate: it kills its process in `deinit`, so dropping the
        // last reference IS the teardown. The session's `VoiceActivityDetector`
        // is gone by the time this runs, so this is that last reference.
        sileroVAD = nil
    }

    /// Load everything the session needs. Returns true if all succeeded.
    func loadAll() async -> Bool {
        isLoading = true
        failureMessage = nil
        // Cleared per session, or the second start would still be showing what
        // the first one unloaded.
        unloadRows = []
        defer { isLoading = false }

        let d = UserDefaults.standard
        let realtimeOn = d.object(forKey: "realtime.enabled") as? Bool ?? true
        let wantsRemote = Self.wantsRemoteRealtime(
            remoteChannel: MicrophoneSettings.current().remoteChannel,
            realtimeEnabled: realtimeOn)
        // Realtime off → stop the sidecar, don't merely stop loading it. See
        // `wantsRealtime` for the bug this fixes and why the rule is a function.
        //
        // `else if` rather than two independent tests: `detachRemoteLane` keeps
        // the process and drops one lane's callback, which is meaningless once
        // the process is gone. A session that does not want remote captions but
        // DOES want office ones still takes the second branch — there is no
        // second process to orphan any more, but the lane's callback would hold
        // the finished recorder, so detach it. Nothing feeds the lane afterwards,
        // so it stays silent.
        if !Self.wantsRealtime(realtimeEnabled: realtimeOn) {
            if let loaded = realtimeASR {
                noteUnload(ModelCatalog.realtimeModel(id: loaded.config.modelID).name)
            }
            realtimeASR?.terminate()
            realtimeASR = nil
        } else if !wantsRemote {
            realtimeASR?.detachRemoteLane()
        }

        // Diarization engines are exclusive, and both services are kept alive
        // across sessions — so the one this session is NOT using has to be torn
        // down here, not merely left unloaded. Every pyannote-dependent path
        // (live/tail/final passes, remote diarization, overlap repair, the
        // profile store) guards on `diarization != nil`, which is exactly what
        // makes them natural no-ops under the MOSS engine; a leftover service
        // from a previous session would quietly re-enable all of them.
        let diarEngine = d.string(forKey: "diarization.engine") ?? ShippedDefaults.diarizationEngine
        let chunkedID = d.string(forKey: "chunked.model") ?? ShippedDefaults.chunkedModel
        let chunkedOn = d.object(forKey: "chunked.enabled") as? Bool ?? true
        // BOTH halves of the split go together: leaving the embedder alive under
        // the MOSS engine would keep a process resident that nothing can ask
        // anything (MOSS turns never reach the profile stores), and leaving it
        // alive while pyannote was torn down is the state no code path expects.
        // The VAD was the last persistent service with no teardown at all —
        // every other one has had a rule since 2026-07-31. Turning it off is
        // already honoured at record time (`AudioRecorder` builds no
        // `VoiceActivityDetector`), so this is memory only (~0.20 GB), but a
        // service nothing can ask anything is exactly what the other rules exist
        // to stop.
        if !(d.object(forKey: "vad.enabled") as? Bool ?? true) {
            // Released rather than `terminate()`d: this service has no public
            // terminate — it kills its process in `deinit` — so dropping the last
            // reference IS the teardown. `AudioRecorder` only holds one for the
            // duration of a session (inside `VoiceActivityDetector`), and this
            // runs before capture starts, so the reference here is the last one.
            sileroVAD = nil
        }
        let wantedDiar = Self.wantedDiarizationStack(
            diarizationEnabled: d.object(forKey: "diarization.enabled") as? Bool ?? true,
            engine: diarEngine, chunkedID: chunkedID, chunkedEnabled: chunkedOn)
        // BOTH halves of the pyannote split go together: leaving the embedder
        // alive without the pipeline is a state no code path expects, and
        // leaving either alive when diarization is off is what let a switched-off
        // feature keep running. See `wantedDiarizationStack`.
        if wantedDiar != .pyannote, pyannote != nil {
            noteUnload(ModelCatalog.diarizationEngine(forEngine: Self.pyannoteEngineID).name)
            pyannote?.terminate()
            pyannote = nil
        }
        if wantedDiar != .spectral, spectral != nil {
            noteUnload(ModelCatalog.diarizationEngine(forEngine: Self.spectralEngineID).name)
            spectral?.terminate()
            spectral = nil
        }
        // The same want/teardown PAIR, from the same function, as every other
        // service here. A load rule and a teardown rule computed apart is what left
        // pyannote, both overlap engines and Nemotron resident and unreachable — and
        // this is the largest offender yet if it happened again: NeMo's peak RSS
        // scales with the audio it was last given (measured 1.15 GB for 98 s,
        // 7.02 GB for 48 min, 13.33 GB for 67 min).
        if wantedDiar != .nemo, nemo != nil {
            noteUnload(ModelCatalog.diarizationEngine(forEngine: Self.nemoEngineID).name)
            nemo?.terminate()
            nemo = nil
        }
        // The same want/teardown PAIR, from the same function. DiariZen holds a
        // WavLM encoder plus the pyannote 3.1 stack, measured at 1.7-3.7 GB — the
        // third-largest idle process this app can hold, after MOSS and DiCoW.
        if wantedDiar != .diarizen, diarizen != nil {
            noteUnload(ModelCatalog.diarizationEngine(forEngine: Self.diarizenEngineID).name)
            diarizen?.terminate()
            diarizen = nil
        }
        // The same want/teardown PAIR, from the same function — the rule every
        // persistent service here now has, after the realtime sidecar was found
        // being loaded by a switch and stopped by nothing (1.70 GB resident until
        // quit). CAM++ is the smallest of them (66 MB of weights plus torch), but
        // "small" is not a reason to be the one service without a teardown.
        if wantedDiar != .vibeVoiceDiar, vibeVoiceDiar != nil {
            noteUnload(ModelCatalog.diarizationEngine(forEngine: Self.vibeVoiceEngineID).name)
            vibeVoiceDiar?.terminate()
            vibeVoiceDiar = nil
        }
        if wantedDiar != .camPlus, camPlus != nil {
            noteUnload(ModelCatalog.diarizationEngine(forEngine: Self.camPlusEngineID).name)
            camPlus?.terminate()
            camPlus = nil
        }
        // The embedder serves BOTH pipeline engines, so it is dropped only when
        // NEITHER is selected. Written as one question — "does this session's
        // stack use identity?" — rather than as a second `!=` beside each
        // pipeline's own, because two independent tests would each have dropped
        // it for the other engine's session.
        if wantedDiar?.usesSpeakerIdentity != true, embedding != nil {
            noteUnload(ModelCatalog.speakerEmbedding.name)
            embedding?.terminate()
            embedding = nil
        }
        if wantedDiar != .mossSecondProcess, mossDiarization != nil {
            noteUnload(ModelCatalog.mossDiarization.name)
            mossDiarization?.terminate()
            mossDiarization = nil
        }
        // The chunked sidecar had NO teardown at all before the master switch
        // existed — the only `terminate()` for it sits inside its own load step,
        // to replace the process when its settings change. That is the same shape
        // as the Nemotron leak of 2026-08-05 and the two overlap engines before
        // it: harmless only while the service was unconditional. The moment it
        // became optional it needed this, or switching the pass off would leave
        // the largest process in the app (Qwen3 4.29 GB, MOSS 5.65 GB) resident
        // and unreachable until quit.
        if !Self.wantsChunked(chunkedEnabled: chunkedOn) {
            // `modelName` off the live config, not a catalog lookup by id: it
            // names the model this process ACTUALLY loaded, which is the honest
            // answer even if the stored setting has since changed.
            if let loaded = chunkedASR { noteUnload(loaded.config.modelName) }
            chunkedASR?.terminate()
            chunkedASR = nil
        }
        // Same reasoning for the aligner, which is now a process of its own: a
        // session that does not align must not keep a previous session's sidecar
        // alive, because `AudioRecorder` decides whether to align at all by
        // asking whether this service exists.
        if !Self.wantsAligner(alignEnabled: d.object(forKey: "align.enabled") as? Bool ?? true,
                              chunkedID: chunkedID, chunkedEnabled: chunkedOn) {
            if aligner != nil { noteUnload(ModelCatalog.wordAligner.name) }
            aligner?.terminate()
            aligner = nil
        }
        // The two OVERLAP engines are mutually exclusive in exactly the way the
        // two diarization engines above are, and are equally kept alive across
        // sessions — so the same rule has to apply, and until 2026-07-31 it did
        // not. `overlapRepair` and `dicowRepair` had no `terminate()` call
        // anywhere in the app, so:
        //   * turning overlap repair OFF left its sidecar resident until quit;
        //   * switching MossFormer2 ↔ DiCoW left BOTH resident at once;
        //   * selecting the MOSS diarization engine (which cannot run repair at
        //     all) left whichever engine was loaded running with nothing able to
        //     ask it anything.
        // DiCoW is a ~6 GB model in its own interpreter, so this was the largest
        // idle process the app could hold. The condition mirrors the load step's
        // exactly — same four inputs, one function — because a
        // teardown rule that disagrees with the load rule would either drop a
        // service this session needs or keep one it cannot use.
        let wantedRepair = Self.wantedOverlapEngine(
            repairEnabled: d.object(forKey: "overlap.repair.enabled") as? Bool ?? ShippedDefaults.overlapRepair,
            engineID: d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id,
            diarEngine: diarEngine,
            detectEnabled: d.object(forKey: "overlap.detect.enabled") as? Bool ?? ShippedDefaults.overlapDetect)
        if wantedRepair != ModelCatalog.overlapSeparation.id {
            if overlapRepair != nil { noteUnload(ModelCatalog.overlapSeparation.name) }
            overlapRepair?.terminate()
            overlapRepair = nil
        }
        if wantedRepair != ModelCatalog.overlapDicow.id {
            if dicowRepair != nil { noteUnload(ModelCatalog.overlapDicow.name) }
            dicowRepair?.terminate()
            dicowRepair = nil
        }
        // Same want/teardown pair as everything else — a detector left running for
        // a session that switched it off is the exact bug this file has now had
        // three times.
        if !Self.wantsOverlapDetect(
            detectEnabled: d.object(forKey: "overlap.detect.enabled") as? Bool ?? ShippedDefaults.overlapDetect,
            diarizationEnabled: d.object(forKey: "diarization.enabled") as? Bool ?? true,
            diarEngine: d.string(forKey: "diarization.engine") ?? ShippedDefaults.diarizationEngine) {
            overlapDetect?.terminate()
            overlapDetect = nil
        }

        // Built once and reused for both the overlay list and the run below, so
        // the indices into `items` can never disagree with the steps executed.
        // The remote lane adds NO step: it rides the one realtime sidecar.
        // Must run AFTER every teardown above and BEFORE the overlay is built —
        // see `noteReplacements()` for both halves of that ordering.
        noteReplacements()

        let steps = buildSteps()
        // The unload rows come FIRST, so the overlay reads in the order things
        // actually happened: what went, then what is arriving.
        //
        // ⚠ `loadOffset` is why this stayed correct. The loop below indexes
        // `items` by step position, and that alignment is the invariant the
        // comment above `buildSteps` exists to protect — prepending rows without
        // it would silently mark the WRONG row as loading/failed, which looks
        // like a load failure in a model that was never touched.
        let loadOffset = unloadRows.count
        items = unloadRows + steps.map { Item(id: $0.model.id, name: $0.model.name) }
        var allOK = true

        for (index, step) in steps.enumerated() {
            items[loadOffset + index].state = .loading
            do {
                try await load(step)
                items[loadOffset + index].state = .done
            } catch {
                items[loadOffset + index].state = .failed(error.localizedDescription)
                failureMessage = error.localizedDescription
                allOK = false
                break
            }
        }
        return allOK
    }

    // MARK: - Steps

    private struct Step {
        let model: ModelInfo
        let checkInstalled: Bool
    }

    /// Which models this session needs, based on current settings.
    private func buildSteps() -> [Step] {
        let d = UserDefaults.standard
        var steps: [Step] = []
        let chunkedID = d.string(forKey: "chunked.model") ?? ShippedDefaults.chunkedModel
        let chunkedOn = d.object(forKey: "chunked.enabled") as? Bool ?? true
        let diarEngine = d.string(forKey: "diarization.engine") ?? ShippedDefaults.diarizationEngine

        if d.object(forKey: "vad.enabled") as? Bool ?? true {
            steps.append(Step(model: ModelCatalog.vad, checkInstalled: false)) // energy VAD needs no files yet
        }
        // Same rule as the teardown in `loadAll`, from the SAME function.
        if Self.wantsRealtime(
            realtimeEnabled: d.object(forKey: "realtime.enabled") as? Bool ?? true) {
            // One step whether or not there is a Remote stream — the sidecar's
            // second lane needs no extra weights and no extra process.
            steps.append(Step(model: ModelCatalog.realtimeModel(
                id: d.string(forKey: "realtime.model") ?? RealtimeASRService.defaultModelID),
                checkInstalled: true))
        }
        // Word aligner — its OWN sidecar since 2026-07-29, so this step both
        // verifies the weights and starts the process. Kept BEFORE the chunked
        // step so "aligner not downloaded" is reported before a multi-GB ASR
        // load, not after it. Skipped for MOSS — see `wantsAligner`.
        if Self.wantsAligner(alignEnabled: d.object(forKey: "align.enabled") as? Bool ?? true,
                             chunkedID: chunkedID, chunkedEnabled: chunkedOn) {
            steps.append(Step(model: ModelCatalog.wordAligner, checkInstalled: true))
        }
        // Chunked model (rolling accurate pass) — per settings selection, and only
        // when the master switch is on. Off, the session genuinely has no accurate
        // transcript: `chunkedStopMode(hasChunkedModel:)` already resolves that to
        // the `.none` plan, which settles the stop gate here and deliberately does
        // NOT sweep the realtime tail. That state predates this switch — it was
        // the "no chunked model" early-out — which is why turning it off does not
        // need a single new branch downstream.
        if Self.wantsChunked(chunkedEnabled: chunkedOn) {
            steps.append(Step(model: ModelCatalog.chunkedModel(id: chunkedID), checkInstalled: true))
        }
        // Diarization (runs after recording ends, but loads up front).
        // Engine-aware: pyannote gets a step only when it is the selected engine,
        // and MOSS gets one only when it needs a SECOND process — with MOSS in
        // both roles the chunked step above already IS that load, and two rows
        // for one process would misrepresent what is happening.
        // Same rule as the teardown in `prepare`, from the SAME function.
        switch Self.wantedDiarizationStack(
            diarizationEnabled: d.object(forKey: "diarization.enabled") as? Bool ?? true,
            engine: diarEngine, chunkedID: chunkedID, chunkedEnabled: chunkedOn) {
        case .none:
            break
        case .mossSecondProcess:
            // Embedder FIRST, the same order and the same reason as the other
            // stacks: it is the cheap check, so a broken one is reported in
            // seconds rather than after a 3.6 GB load that was going to fail.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
            steps.append(Step(model: ModelCatalog.mossDiarization, checkInstalled: true))
        case .mossOwnASR:
            // The chunked step above IS the MOSS load; only identity is added.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
        case .spectral:
            // Two steps for the same reason the pyannote engine has two: this
            // engine is the pipeline half only, and identity is a separate
            // process. Embedder FIRST, again because it is the cheap check —
            // and here the two steps genuinely check the SAME weights
            // (`spectralDiarization.hfRepo` IS the WeSpeaker checkpoint), so a
            // missing download fails at the first row rather than the second.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
            steps.append(Step(model: ModelCatalog.spectralDiarization, checkInstalled: true))
        case .nemo:
            // Two steps for the same reason spectral and pyannote have two: this
            // engine is the pipeline half only, and identity is a separate process.
            // Embedder FIRST again, and here it matters most — NeMo's cold import
            // alone is ~51 s, so a broken embedder is reported in seconds rather
            // than after the slowest load in the app.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
            steps.append(Step(model: ModelCatalog.nemoDiarization, checkInstalled: true))
        case .diarizen:
            // Embedder first, as every pipeline engine does: identity is a separate
            // process, and a broken embedder is reported in seconds rather than
            // after DiariZen's WavLM load.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
            steps.append(Step(model: ModelCatalog.diarizenDiarization, checkInstalled: true))
        case .camPlus:
            // Embedder first, as every pipeline engine does. Both are quick here
            // (26 MB and 66 MB), so the ordering buys less than it does for NeMo
            // or DiariZen — it is kept identical anyway, because an engine that
            // orders its steps differently is a difference someone has to explain.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
            steps.append(Step(model: ModelCatalog.camPlusDiarization, checkInstalled: true))
        case .vibeVoiceDiar:
            // Embedder first, as every pipeline engine does — and here the
            // ordering genuinely buys something: 26 MB against 5.6 GB, so a
            // missing embedder is reported in seconds rather than after the
            // slowest model load in the app.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
            steps.append(Step(model: ModelCatalog.vibeVoiceDiarization, checkInstalled: true))
        case .pyannote:
            // TWO steps for the pyannote engine since the 2026-07-30 split: the
            // embedder and the pipeline are separate processes now, and the
            // overlay says so rather than hiding one inside the other. Embedder
            // FIRST on purpose — it is ~26 MB against the pipeline's minute, so a
            // missing/broken embedder is reported in seconds instead of after a
            // long load that was going to fail anyway.
            steps.append(Step(model: ModelCatalog.speakerEmbedding, checkInstalled: true))
            steps.append(Step(model: ModelCatalog.diarization, checkInstalled: true))
        }
        // Overlap repair (runs at stop) — off by default. Loads whichever engine
        // is picked in Settings → Overlap (MossFormer2 or DiCoW). Both engines
        // locate their windows from pyannote turns, so neither can run under the
        // MOSS diarization engine — don't load a 6 GB model to do nothing.
        // Same rule as the teardown in `prepare`, from the SAME function — a load
        // step and a teardown that computed this separately is exactly how the
        // engines came to be started and never stopped.
        if Self.wantsOverlapDetect(
            detectEnabled: d.object(forKey: "overlap.detect.enabled") as? Bool ?? ShippedDefaults.overlapDetect,
            diarizationEnabled: d.object(forKey: "diarization.enabled") as? Bool ?? true,
            diarEngine: d.string(forKey: "diarization.engine") ?? ShippedDefaults.diarizationEngine) {
            // Resolved from the STORED choice, not hard-coded to the one entry.
            // `overlap.detect.model` had a picker and nothing read it — the
            // Granite/Voxtral language-picker trap, harmless only while the list
            // has a single entry, and a silent defect the moment it has two.
            steps.append(Step(model: ModelCatalog.overlapDetector(
                id: d.string(forKey: "overlap.detect.model") ?? ModelCatalog.overlapDetectPyannote.id),
                              checkInstalled: true))
        }
        if let engineID = Self.wantedOverlapEngine(
            repairEnabled: d.object(forKey: "overlap.repair.enabled") as? Bool ?? ShippedDefaults.overlapRepair,
            engineID: d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id,
            diarEngine: diarEngine,
            detectEnabled: d.object(forKey: "overlap.detect.enabled") as? Bool ?? ShippedDefaults.overlapDetect) {
            steps.append(Step(model: ModelCatalog.overlapEngine(id: engineID), checkInstalled: true))
        }
        return steps
    }

    private enum LoadError: LocalizedError {
        case notDownloaded(String)
        case sidecarFailed(String)
        var errorDescription: String? {
            switch self {
            case .notDownloaded(let name):
                return "\(name) is not downloaded. Run download-best-models.sh first."
            case .sidecarFailed(let name):
                return "\(name) failed to start — mlx-audio is probably missing. "
                     + "Run download-best-models.sh again (it now installs mlx-audio), then retry."
            }
        }
    }

    private func load(_ step: Step) async throws {
        if step.checkInstalled, !ModelCatalog.isInstalled(step.model) {
            throw LoadError.notDownloaded(step.model.name)
        }

        // Word aligner: start its persistent sidecar. It used to load inside the
        // chunked ASR process, so this step was a check and nothing else; the
        // aligner is now a process of its own and this is where it comes up.
        // Failure IS fatal for the session, exactly as it was before: the load
        // used to happen inside the ASR sidecar's startup, so a broken aligner
        // already stopped the meeting from starting rather than surfacing
        // mid-recording. Keeping that means the user is told at the overlay.
        if step.model.id == ModelCatalog.wordAligner.id {
            let config = AlignerService.Config.fromSettings()
            if let existing = aligner, existing.config == config {
                return
            }
            aligner?.terminate()
            aligner = nil
            aligner = try await Task.detached(priority: .userInitiated) {
                try AlignerService(config: config)
            }.value
            return
        }

        // Silero VAD: start the Python sidecar (real model load).
        // Failure is non-fatal — the heuristic VAD takes over.
        if step.model.id == ModelCatalog.vad.id {
            if sileroVAD == nil {
                sileroVAD = await Task.detached(priority: .userInitiated) {
                    SileroVADService()
                }.value
            }
            return
        }

        // Realtime ASR: start the MLX sidecar for the SELECTED engine.
        // Reused across sessions; recreated only if its settings changed — and
        // the engine id is one of those settings, so this is also the
        // engine-switch mechanism. Matched against the LIST, never against one
        // id: `== ModelCatalog.realtime.id` was true for exactly one engine and
        // would have left a second one loading nothing at all.
        if ModelCatalog.realtimeModels.contains(where: { $0.id == step.model.id }) {
            let config = RealtimeASRService.Config.fromSettings()
            if let existing = realtimeASR, existing.config == config {
                return
            }
            // The unload already has its OWN row above, added by
            // `noteReplacements()` before the overlay was built.
            realtimeASR?.terminate()
            realtimeASR = nil
            // Throws with the sidecar's exact error message (shown in overlay)
            realtimeASR = try await Task.detached(priority: .userInitiated) {
                try RealtimeASRService(config: config)
            }.value
            return
        }

        // Chunked ASR: start the persistent sidecar for the selected model.
        if ModelCatalog.chunked.contains(where: { $0.id == step.model.id }) {
            let config = ChunkedASRService.Config.fromSettings()
            if let existing = chunkedASR, existing.config == config {
                return
            }
            // Its unload row was added by `noteReplacements()` above.
            chunkedASR?.terminate()
            chunkedASR = nil
            chunkedASR = try await Task.detached(priority: .userInitiated) {
                try ChunkedASRService(config: config)
            }.value
            return
        }

        // MOSS as the DIARIZATION engine while another model does the ASR: a
        // second MOSS process, driven by the same client and fed the same audio.
        // This step exists only when `needsSecondMossProcess` said so.
        if step.model.id == ModelCatalog.mossDiarization.id {
            let config = ChunkedASRService.Config.mossDiarization()
            if let existing = mossDiarization, existing.config == config {
                return
            }
            mossDiarization?.terminate()
            mossDiarization = nil
            mossDiarization = try await Task.detached(priority: .userInitiated) {
                try ChunkedASRService(config: config)
            }.value
            return
        }

        // Speaker identity: start the persistent WeSpeaker sidecar (loads on MPS).
        // Failure IS fatal for the session, deliberately, and this is the one
        // place to say why: `SpeakerTurn` has no representation for a turn whose
        // speaker is unknown, so there is nothing for a failed identify to
        // degrade TO. The house rule is refuse loudly, never substitute — the
        // same rule Voxtral + dual-stream follows.
        if step.model.id == ModelCatalog.speakerEmbedding.id {
            let config = WeSpeakerService.Config.fromSettings()
            if let existing = embedding, existing.config == config {
                return
            }
            embedding?.terminate()
            embedding = nil
            embedding = try await Task.detached(priority: .userInitiated) {
                try WeSpeakerService(config: config)
            }.value
            return
        }

        // Diarization: start the persistent pyannote sidecar (loads on MPS).
        if step.model.id == ModelCatalog.diarization.id {
            if pyannote == nil {
                pyannote = try await Task.detached(priority: .userInitiated) {
                    try PyannoteService()
                }.value
            }
            return
        }

        // Diarization: start the persistent SPECTRAL sidecar (Silero + WeSpeaker,
        // both CPU). Fatal on failure, like every other diarization stack member:
        // this engine has no live path, so a broken sidecar would otherwise
        // surface as a meeting that produces no speakers at all, at Stop, with
        // nothing left to re-run.
        if step.model.id == ModelCatalog.spectralDiarization.id {
            let config = SpectralService.Config.fromSettings()
            if let existing = spectral, existing.config == config {
                return
            }
            spectral?.terminate()
            spectral = nil
            spectral = try await Task.detached(priority: .userInitiated) {
                try SpectralService(config: config)
            }.value
            return
        }

        // Diarization: start the persistent DIARIZEN sidecar, in `.venv-diarizen`.
        // Fatal on failure for the same reason as every other diarization stack
        // member: no live path, so a broken sidecar surfaces as a meeting with no
        // speakers at all, at Stop, with nothing left to re-run.
        if step.model.id == ModelCatalog.diarizenDiarization.id {
            let config = DiarizenService.Config()
            if let existing = diarizen, existing.config == config {
                return
            }
            diarizen?.terminate()
            diarizen = nil
            diarizen = try await Task.detached(priority: .userInitiated) {
                try DiarizenService(config: config)
            }.value
            return
        }

        // Diarization: start the persistent CAM++ sidecar, in the MAIN `.venv`.
        // Fatal on failure for the same reason as every other diarization stack
        // member: no live path, so a broken sidecar surfaces as a meeting with no
        // speakers at all, at Stop, with nothing left to re-run.
        // Diarization: start the persistent VIBEVOICE sidecar, in its OWN
        // `.venv-vibevoice`. Fatal on failure like every other stack member, and
        // here the failure is more likely than most: 5.6 GB onto MPS, ~5 s, in an
        // interpreter that exists only for this engine.
        if step.model.id == ModelCatalog.vibeVoiceDiarization.id {
            let config = VibeVoiceDiarizationService.Config.fromSettings()
            if let existing = vibeVoiceDiar, existing.config == config {
                return
            }
            vibeVoiceDiar?.terminate()
            vibeVoiceDiar = nil
            vibeVoiceDiar = try await Task.detached(priority: .userInitiated) {
                try VibeVoiceDiarizationService(config: config)
            }.value
            return
        }

        if step.model.id == ModelCatalog.camPlusDiarization.id {
            let config = CamPlusService.Config.fromSettings()
            if let existing = camPlus, existing.config == config {
                return
            }
            camPlus?.terminate()
            camPlus = nil
            camPlus = try await Task.detached(priority: .userInitiated) {
                try CamPlusService(config: config)
            }.value
            return
        }

        // Diarization: start the persistent NEMO sidecar, in `.venv-nemo`. Fatal on
        // failure, like every other diarization stack member: this engine has no
        // live path, so a broken sidecar would otherwise surface as a meeting that
        // produces no speakers at all, at Stop, with nothing left to re-run. A
        // missing `.venv-nemo` throws the setup error that names it.
        if step.model.id == ModelCatalog.nemoDiarization.id {
            let config = NemoService.Config()
            if let existing = nemo, existing.config == config {
                return
            }
            nemo?.terminate()
            nemo = nil
            nemo = try await Task.detached(priority: .userInitiated) {
                try NemoService(config: config)
            }.value
            return
        }

        // Overlap detection: start the persistent segmentation sidecar. Failure is
        // NOT fatal — a missing mark costs the user a hint, not their transcript,
        // so this is the one diarization-adjacent service that degrades instead of
        // refusing the session.
        if ModelCatalog.overlapDetectors.contains(where: { $0.id == step.model.id }) {
            if overlapDetect == nil {
                overlapDetect = try? await Task.detached(priority: .userInitiated) {
                    try OverlapDetectService()
                }.value
            }
            return
        }

        // Overlap repair: start the persistent DiCoW sidecar (loads once).
        // Runs in .venv-dicow, so it throws a setup error if that venv is absent.
        if step.model.id == ModelCatalog.overlapDicow.id {
            if dicowRepair == nil {
                dicowRepair = try await Task.detached(priority: .userInitiated) {
                    try DicowService()
                }.value
            }
            return
        }

        // Overlap repair: start the persistent MossFormer2 sidecar (loads once).
        if step.model.id == ModelCatalog.overlapSeparation.id {
            if overlapRepair == nil {
                overlapRepair = try await Task.detached(priority: .userInitiated) {
                    try OverlapRepairService()
                }.value
            }
            return
        }

    }
}
