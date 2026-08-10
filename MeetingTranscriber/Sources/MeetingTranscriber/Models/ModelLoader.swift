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

    func dismissFailure() {
        failureMessage = nil
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

    /// Nemotron realtime ASR sidecar; recreated only when its settings change.
    /// ONE process serving both streams: its `office` and `remote` lanes keep
    /// separate buffers inside the sidecar, so a Remote channel costs no second
    /// model load and no second process — see `NemotronASRService`.
    private(set) var nemotronASR: NemotronASRService?

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
    /// `AudioRecorder` reads `modelLoader.nemotronASR?.office` only when the same
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

    /// Whether this session loads the CHUNKED ASR sidecar at all.
    ///
    /// THE MASTER SWITCH, owner-requested 2026-08-06, and it is not the same thing
    /// as `chunked.finalPass` — which was already in the Chunked tab and reads
    /// like an on/off but is not one. That key governs only the extra pass AFTER
    /// Stop; with it off, chunked ASR still runs all meeting and still produces
    /// the transcript. This switch is the one that stops it.
    ///
    /// WHAT TURNING IT OFF COSTS, stated here because the cost is the feature:
    /// the meeting has no accurate transcript at all. The realtime Nemotron text
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
        if diarEngine == mossEngineID || diarEngine == spectralEngineID
            || diarEngine == nemoEngineID {
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
    /// speakers has none to mark. It is deliberately NOT restricted to the MOSS
    /// and spectral engines: under pyannote it is redundant rather than wrong
    /// (that engine reports overlap itself), and the Diarization tab says so — but
    /// a user who switches it on there has asked for a second opinion, and
    /// refusing it silently would be the substitution this project forbids.
    nonisolated static func wantsOverlapDetect(detectEnabled: Bool,
                                               diarizationEnabled: Bool) -> Bool {
        detectEnabled && diarizationEnabled
    }

    nonisolated static func needsSecondMossProcess(chunkedID: String,
                                                   engine: String,
                                                   chunkedEnabled: Bool) -> Bool {
        guard engine == mossEngineID else { return false }
        return !chunkedEnabled || chunkedID != "moss"
    }

    /// Load everything the session needs. Returns true if all succeeded.
    func loadAll() async -> Bool {
        isLoading = true
        failureMessage = nil
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
            nemotronASR?.terminate()
            nemotronASR = nil
        } else if !wantsRemote {
            nemotronASR?.detachRemoteLane()
        }

        // Diarization engines are exclusive, and both services are kept alive
        // across sessions — so the one this session is NOT using has to be torn
        // down here, not merely left unloaded. Every pyannote-dependent path
        // (live/tail/final passes, remote diarization, overlap repair, the
        // profile store) guards on `diarization != nil`, which is exactly what
        // makes them natural no-ops under the MOSS engine; a leftover service
        // from a previous session would quietly re-enable all of them.
        let diarEngine = d.string(forKey: "diarization.engine") ?? Self.pyannoteEngineID
        let chunkedID = d.string(forKey: "chunked.model") ?? "qwen3"
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
        if wantedDiar != .pyannote {
            pyannote?.terminate()
            pyannote = nil
        }
        if wantedDiar != .spectral {
            spectral?.terminate()
            spectral = nil
        }
        // The same want/teardown PAIR, from the same function, as every other
        // service here. A load rule and a teardown rule computed apart is what left
        // pyannote, both overlap engines and Nemotron resident and unreachable — and
        // this is the largest offender yet if it happened again: NeMo's peak RSS
        // scales with the audio it was last given (measured 1.15 GB for 98 s,
        // 7.02 GB for 48 min, 13.33 GB for 67 min).
        if wantedDiar != .nemo {
            nemo?.terminate()
            nemo = nil
        }
        // The embedder serves BOTH pipeline engines, so it is dropped only when
        // NEITHER is selected. Written as one question — "does this session's
        // stack use identity?" — rather than as a second `!=` beside each
        // pipeline's own, because two independent tests would each have dropped
        // it for the other engine's session.
        if wantedDiar?.usesSpeakerIdentity != true {
            embedding?.terminate()
            embedding = nil
        }
        if wantedDiar != .mossSecondProcess {
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
            chunkedASR?.terminate()
            chunkedASR = nil
        }
        // Same reasoning for the aligner, which is now a process of its own: a
        // session that does not align must not keep a previous session's sidecar
        // alive, because `AudioRecorder` decides whether to align at all by
        // asking whether this service exists.
        if !Self.wantsAligner(alignEnabled: d.object(forKey: "align.enabled") as? Bool ?? false,
                              chunkedID: chunkedID, chunkedEnabled: chunkedOn) {
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
            repairEnabled: d.object(forKey: "overlap.repair.enabled") as? Bool ?? false,
            engineID: d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id,
            diarEngine: diarEngine,
            detectEnabled: d.object(forKey: "overlap.detect.enabled") as? Bool ?? false)
        if wantedRepair != ModelCatalog.overlapSeparation.id {
            overlapRepair?.terminate()
            overlapRepair = nil
        }
        if wantedRepair != ModelCatalog.overlapDicow.id {
            dicowRepair?.terminate()
            dicowRepair = nil
        }
        // Same want/teardown pair as everything else — a detector left running for
        // a session that switched it off is the exact bug this file has now had
        // three times.
        if !Self.wantsOverlapDetect(
            detectEnabled: d.object(forKey: "overlap.detect.enabled") as? Bool ?? false,
            diarizationEnabled: d.object(forKey: "diarization.enabled") as? Bool ?? true) {
            overlapDetect?.terminate()
            overlapDetect = nil
        }

        // Built once and reused for both the overlay list and the run below, so
        // the indices into `items` can never disagree with the steps executed.
        // The remote lane adds NO step: it rides the one realtime sidecar.
        let steps = buildSteps()
        items = steps.map { Item(id: $0.model.id, name: $0.model.name) }
        var allOK = true

        for (index, step) in steps.enumerated() {
            items[index].state = .loading
            do {
                try await load(step)
                items[index].state = .done
            } catch {
                items[index].state = .failed(error.localizedDescription)
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
        let chunkedID = d.string(forKey: "chunked.model") ?? "qwen3"
        let chunkedOn = d.object(forKey: "chunked.enabled") as? Bool ?? true
        let diarEngine = d.string(forKey: "diarization.engine") ?? Self.pyannoteEngineID

        if d.object(forKey: "vad.enabled") as? Bool ?? true {
            steps.append(Step(model: ModelCatalog.vad, checkInstalled: false)) // energy VAD needs no files yet
        }
        // Same rule as the teardown in `loadAll`, from the SAME function.
        if Self.wantsRealtime(
            realtimeEnabled: d.object(forKey: "realtime.enabled") as? Bool ?? true) {
            // One step whether or not there is a Remote stream — the sidecar's
            // second lane needs no extra weights and no extra process.
            steps.append(Step(model: ModelCatalog.realtime, checkInstalled: true))
        }
        // Word aligner — its OWN sidecar since 2026-07-29, so this step both
        // verifies the weights and starts the process. Kept BEFORE the chunked
        // step so "aligner not downloaded" is reported before a multi-GB ASR
        // load, not after it. Skipped for MOSS — see `wantsAligner`.
        if Self.wantsAligner(alignEnabled: d.object(forKey: "align.enabled") as? Bool ?? false,
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
            detectEnabled: d.object(forKey: "overlap.detect.enabled") as? Bool ?? false,
            diarizationEnabled: d.object(forKey: "diarization.enabled") as? Bool ?? true) {
            // Resolved from the STORED choice, not hard-coded to the one entry.
            // `overlap.detect.model` had a picker and nothing read it — the
            // Granite/Voxtral language-picker trap, harmless only while the list
            // has a single entry, and a silent defect the moment it has two.
            steps.append(Step(model: ModelCatalog.overlapDetector(
                id: d.string(forKey: "overlap.detect.model") ?? ModelCatalog.overlapDetectPyannote.id),
                              checkInstalled: true))
        }
        if let engineID = Self.wantedOverlapEngine(
            repairEnabled: d.object(forKey: "overlap.repair.enabled") as? Bool ?? false,
            engineID: d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id,
            diarEngine: diarEngine,
            detectEnabled: d.object(forKey: "overlap.detect.enabled") as? Bool ?? false) {
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

        // Nemotron realtime ASR: start the MLX sidecar.
        // Reused across sessions; recreated only if its settings changed.
        if step.model.id == ModelCatalog.realtime.id {
            let config = NemotronASRService.Config.fromSettings()
            if let existing = nemotronASR, existing.config == config {
                return
            }
            nemotronASR?.terminate()
            nemotronASR = nil
            // Throws with the sidecar's exact error message (shown in overlay)
            nemotronASR = try await Task.detached(priority: .userInitiated) {
                try NemotronASRService(config: config)
            }.value
            return
        }

        // Chunked ASR: start the persistent sidecar for the selected model.
        if ModelCatalog.chunked.contains(where: { $0.id == step.model.id }) {
            let config = ChunkedASRService.Config.fromSettings()
            if let existing = chunkedASR, existing.config == config {
                return
            }
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
