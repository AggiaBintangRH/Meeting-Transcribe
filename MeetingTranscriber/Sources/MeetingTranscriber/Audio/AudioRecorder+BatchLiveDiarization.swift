import Foundation

// LIVE, per-interval diarization for the four WHOLE-FILE engines — spectral,
// NeMo, DiariZen and CAM++ (owner, 2026-08-13).
//
// WHAT CHANGED, AND WHAT DID NOT
// ------------------------------
// Until now these four had no live path at all: record with no speaker labels,
// then one pass per stream at Stop. The owner asked for the pyannote shape
// everywhere — "Run a diarization pass at stop" for every engine, and when it is
// OFF the engine keeps running per interval instead.
//
// The objection those engines' own comments raise is real and is NOT waved away:
// each counts and clusters GLOBALLY, so a 30 s window's `SPEAKER_00` has nothing
// to do with the next window's. What makes this work anyway is that the labels
// were never what carried continuity — `WeSpeakerService.identify` is, and it
// already runs for all four at Stop. This path sends each window through the
// SAME identify → `composeTurns` → `applyChunkTurns` chain pyannote's live
// chunks use, so a voice keeps its profile across windows for the same reason
// pyannote's does. No new id space, no new identity code.
//
// NO SIDECAR WAS TOUCHED, deliberately. A window is just a short recording, so
// it is sent as the ordinary `cmd: "final"` on a temp WAV. `spectral/…` and
// `campplus/no-live-chunk-branch` pin that those sidecars have no chunk branch,
// and they still do not — the check stays true rather than being relaxed.
//
// ⚠ THE MEASURED RISK, and the tab states it rather than hiding it. On 25–30 s
// clips of two people, measured 2026-08-13, the AUTO speaker count of three of
// the four is wrong: spectral 12 and 14, CAM++ 9 and 15, NeMo 12 and 2, against
// DiariZen's 2 and 2. A live window IS a 25–30 s clip, so auto-counting repeats
// that failure every window. Pinned to the real count, every engine answered
// correctly on both files — so the SPK picker beside the record button is what
// makes this mode trustworthy, and `liveCountWarning` says so.
extension AudioRecorder {

    /// Does this session run the four whole-file engines per interval?
    ///
    /// PURE and static so the rule can be tested without a session, and stated in
    /// ONE place so the dispatch, the buffer's retain/clear decision and the tab
    /// cannot disagree about it — the shape of the 2026-08-05 bug where two reads
    /// of one setting described the same meeting differently.
    ///
    /// Tied to the STOP PASS being off, which is what the owner asked for: with
    /// the pass on, these engines behave exactly as they always have, so nothing
    /// about the shipped default changes.
    nonisolated static func runsBatchLiveDiarization(isBatchEngine: Bool,
                                                     finalPass: Bool) -> Bool {
        isBatchEngine && !finalPass
    }

    /// Does a TAIL-ONLY pass run at stop?
    ///
    /// ⚠ THIS IS THE OTHER HALF OF `runsBatchLiveDiarization`'s condition, and the
    /// two now agree with the TAB (owner, 2026-08-14): *"Tail hanya muncul dan
    /// dipakai pas Run At Stop = OFF … pas On mah diulang diarize dari awal sampai
    /// akhir."*
    ///
    /// Until then `continueOnStop` was read only inside `if willRunStopPass`,
    /// i.e. only when the stop pass was ON — while its toggle was shown only when
    /// the stop pass was OFF. Visible where inert, active where invisible. The
    /// owner paid that cost: a stale `true` turned the office pass into a tail, a
    /// tail is a `chunk` job carrying no `num_speakers`, and the SPK picker went
    /// dim with nothing in the UI able to explain or undo it.
    ///
    /// So the stop pass is now unconditionally a FULL pass, and the tail is what
    /// finishes a live-labelled meeting: the audio after the last live window,
    /// which otherwise reaches Stop with no labels at all.
    ///
    /// `hasLivePath` is what makes the mode coherent — a tail continues from live
    /// labels, so with nothing live there is nothing to continue from.
    nonisolated static func runsTailPassAtStop(finalPass: Bool,
                                               continueOnStop: Bool,
                                               hasLivePath: Bool) -> Bool {
        !finalPass && continueOnStop && hasLivePath
    }

    /// True for this session, from the flags the engines' `configure*` calls set.
    var batchLiveDiarizationActive: Bool {
        Self.runsBatchLiveDiarization(
            isBatchEngine: spectralDiarizationActive || nemoDiarizationActive
                        || diarizenDiarizationActive || camPlusDiarizationActive,
            finalPass: UserDefaults.standard.object(forKey: "diarization.finalPass")
                        as? Bool ?? true)
    }

    /// Send one window's audio to whichever whole-file engine this session runs.
    ///
    /// The window is remembered against the temp file's PATH, because that is what
    /// the reply echoes — the same trick `chunkFileByWindow` uses for pyannote,
    /// and the reason no FIFO is needed: two windows can be in flight and each
    /// reply still finds its own start time.
    func dispatchBatchLiveWindow(windowStart: Double, samples: [Float]) {
        // THE SAME 1 s FLOOR `diarizeTailChunk` APPLIES, and it was missing here
        // until the 2026-08-14 audit. A sub-second window is the worst possible
        // input for these engines: it is too short to cluster, it is what makes
        // the auto count fabricate, and NeMo's VAD raises outright on one. The
        // guard was `!samples.isEmpty`, which admitted all of that.
        guard samples.count > 16_000 else { return }
        Task { [weak self] in
            guard let self else { return }
            let url = await Task.detached(priority: .utility) {
                Self.writeTempWAV(samples: samples, prefix: "batch-live")
            }.value
            guard let url else {
                self.batchLiveLog("could not write a temp WAV for "
                                  + "[\(self.fmt(windowStart))] — window dropped")
                return
            }
            self.liveDiarWindowByPath[url.path] = windowStart
            // THE COUNT IS SENT, unlike pyannote's live chunks. That looks like a
            // contradiction of the 2026-08-11 rule ("a window that need not contain
            // everyone must not be told the meeting's headcount") and is its
            // opposite case: pyannote's per-window clustering is measured GOOD on
            // its own, while these engines' auto count on a 25–30 s clip is
            // measured BAD — 9 to 15 speakers for two people. Auto here is the
            // failure mode, not the safe default, so the user's own number is the
            // better answer when they have given one. Zero means auto, unchanged.
            self.dispatchBatchWindow(audio: url, numSpeakers: Self.diarNumSpeakers)
        }
    }

    /// Hand one window to whichever whole-file engine this session runs.
    ///
    /// The four are mutually exclusive by construction (`configure*` sets exactly
    /// one flag), and each is asked for its OWN service rather than through a
    /// shared protocol: they are four separate processes with four separate
    /// lifetimes, and a protocol would let a nil one look like a working one.
    func dispatchBatchWindow(audio: URL, numSpeakers: Int) {
        if spectralDiarizationActive, let s = modelLoader.spectral {
            s.diarizeFinal(audio: audio, numSpeakers: numSpeakers)
        } else if nemoDiarizationActive, let s = modelLoader.nemo {
            s.diarizeFinal(audio: audio, numSpeakers: numSpeakers)
        } else if diarizenDiarizationActive, let s = modelLoader.diarizen {
            s.diarizeFinal(audio: audio, numSpeakers: numSpeakers)
        } else if camPlusDiarizationActive, let s = modelLoader.camPlus {
            s.diarizeFinal(audio: audio, numSpeakers: numSpeakers)
        } else {
            // The engine's sidecar died mid-session. Drop the window rather than
            // leaving its temp file and its map entry behind.
            liveDiarWindowByPath.removeValue(forKey: audio.path)
            try? FileManager.default.removeItem(at: audio)
            batchLiveLog("no engine to take the window — sidecar gone")
        }
    }

    /// Route a whole-file reply: a LIVE window if this path is one we sent, and
    /// the stop pass otherwise. Returns whether it was handled here.
    ///
    /// One router for all four engines rather than a branch inside each of their
    /// handlers: the four are identical on this question, and four copies is how
    /// one of them comes to be fixed and the others missed — the failure the
    /// `diarizeLiveChunk` guard's own comment already warns about.
    func handleBatchLiveResult(audio: String, localTurns: [PyannoteService.LocalTurn],
                               stream: PyannoteService.Stream) -> Bool {
        guard let windowStart = liveDiarWindowByPath.removeValue(forKey: audio) else {
            return false                      // the stop pass — the caller's own path
        }
        Task { [weak self] in
            guard let self else { return }
            defer { try? FileManager.default.removeItem(atPath: audio) }
            guard let embedding = modelLoader.embedding else {
                self.batchLiveLog("no identity sidecar — window "
                                  + "[\(self.fmt(windowStart))] keeps no labels")
                return
            }
            do {
                let identity = try await embedding.identify(audio: audio, turns: localTurns,
                                                            stream: stream)
                let turns = Self.composeTurns(localTurns, identity: identity)
                self.batchLiveLog("window [\(self.fmt(windowStart))] — "
                                  + "\(localTurns.count) local turn(s) → "
                                  + "\(turns.count) identified, "
                                  + "\(Set(turns.map(\.id)).count) speaker(s)")
                self.applyChunkTurns(turns, windowStart: windowStart, stream: stream)
                // THE CAUTION THE STOP PASSES ALREADY RAISE, now raised here too
                // (2026-08-14 audit). It was reachable only from `applyFinalTurns`
                // and the remote pass — i.e. only in the mode this one replaces —
                // so the ONE configuration where the auto count is measured to fail
                // was the one configuration with nothing watching.
                //
                // Measured on a 30 s slice of `Meeting5People.wav` with the shipped
                // default (count = Auto): CAM++ returned **11 speakers in 14 turns**
                // and spectral **12**, against 5 real people. Pinned to 5, spectral
                // returned 5. A live window IS a 30 s clip, so that is not an edge
                // case here — it is every window.
                //
                // It CAUTIONS and never corrects, exactly as it does at Stop: the
                // count came from the engine, and overriding a model's answer is
                // the substitution this project refuses everywhere else.
                //
                // ⚠ LOG-ONLY since 2026-08-24 (owner request). It fired on every
                // live window under the shipped Auto count, so the amber row was
                // permanently on screen; the verdict now lands in
                // logs/position-diarization.log instead. The rule and its
                // measurements are untouched.
                self.noteImplausibleSpeakerCount(turns, stream: stream)
                // A window landed, so whatever went wrong before is over. Reset
                // here rather than at dispatch: dispatching proves nothing about
                // the engine, and a counter cleared by the act of asking could
                // never reach its limit against a sidecar that fails every time.
                self.batchLiveFailureRun = 0
            } catch {
                // NEVER FATAL. A failed window costs labels for that stretch and
                // nothing else; the transcript is untouched and the next window is
                // unaffected. Same rule the aligner and the overlap detector follow.
                self.batchLiveLog("window [\(self.fmt(windowStart))] identify failed: "
                                  + error.localizedDescription)
            }
        }
        return true
    }

    /// A sidecar error that belongs to a LIVE WINDOW, not to a stop pass.
    ///
    /// 🔴 THE DEFECT THIS FIXES (2026-08-14 audit). Every engine's `onError` and
    /// `onNoSpeech` went straight to `handleDiarizationFailure`, which is the
    /// STOP-GATE handler: it sets `diarizationError` — a red banner across the
    /// running transcript — plus `finalDiarDone = true`, `awaitingTailWindowStart
    /// = nil`, both watchdogs cancelled, and `setStopStep("diarize", .failed)`.
    /// All of that fired DURING recording, for one window.
    ///
    /// And it is not a rare path. Measured: NeMo returns
    /// `{"type":"error","kind":"no_speech",…}` for 30 s of silence, which in live
    /// mode is an ordinary quiet stretch of a meeting:
    ///
    ///     echo '{"cmd":"final","audio":"silent30.wav","num_speakers":0}' | nemo-service.py
    ///     -> {"type": "error", "kind": "no_speech", "text": "No speech found …"}
    ///
    /// ⚠ WHY `liveMode` IS A SUFFICIENT TEST, rather than a guess at which job
    /// failed (every call site passes `batchLiveDiarizationActive`): in this mode
    /// NEITHER stop pass runs for these engines. `runsBatchOfficePass` requires `finalPass`, and the remote side
    /// returns `.none` because all four pass `supportsTail: false` into
    /// `remoteStopMode`'s `!finalPass` branch. So there is no other job an error
    /// could belong to, and `handleDiarizationFailure` can never be the right
    /// destination while this is true.
    ///
    /// RELEASES EVERY OUTSTANDING WINDOW, because the error reply names `stream`
    /// and not `audio`, so it cannot say which one failed. That is safe rather
    /// than lossy: measured per-window cost is 0.52 s (CAM++), 1.04–1.40 s
    /// (spectral) and 1.65–3.24 s (DiariZen) against a minimum interval of 15 s,
    /// so a second window is essentially never in flight — and losing one
    /// window's labels is the documented non-fatal outcome anyway, while leaking
    /// its temp WAV and its map entry is the ~230 MB/h shape that has bitten
    /// twice. Returns whether it handled the error.
    /// ⚠ `liveMode` IS A PARAMETER WITH NO DEFAULT, and both halves are deliberate.
    ///
    /// No default, because a defaulted one lets a new call site forget it and fall
    /// back to "not live" — which restores the original defect silently, with the
    /// compiler saying nothing. That is the `detectEnabled` precedent (2026-08-06),
    /// where the same reasoning caught all five call sites plus nine test sites.
    ///
    /// A parameter at all, because the only other way to reach this rule is
    /// `batchLiveDiarizationActive`, which reads `diarization.finalPass` from
    /// `UserDefaults.standard` — and this project forbids a test writing there,
    /// since the suite runs in the owner's REAL preference domain. Reading the flag
    /// inside made the whole rule untestable: the first version of these tests
    /// bailed out with "live mode is off in this environment" on the owner's own
    /// machine, which is a test that passes by not running.
    func handleBatchLiveFailure(_ message: String, stream: PyannoteService.Stream,
                                noSpeech: Bool, liveMode: Bool) -> Bool {
        guard liveMode else { return false }
        let outstanding = liveDiarWindowByPath.count
        releaseBatchLiveWindows()
        batchLiveLog((noSpeech ? "no speech in a live window" : "live window failed")
                     + " (\(stream == .office ? "office" : "remote")): \(message)"
                     + " — \(outstanding) window(s) released, transcript untouched")

        // NO SPEECH IS NOT A FAILURE. It is the correct answer for a quiet
        // stretch, it happens routinely, and counting it would make the run below
        // fire on an ordinary silent meeting — the `.skipped` vs `.failed`
        // distinction the stop panel already draws, applied to the live path.
        guard !noSpeech else { batchLiveFailureRun = 0; return true }

        // 🟡 …AND THE OVER-CORRECTION THIS GUARDS, found by the second audit pass
        // the same day. Swallowing the error fixed the loud-but-wrong failure
        // above and created a QUIET one: `onError` also carries "The … sidecar is
        // not running — restart the recording session"
        // (`CamPlusService.swift:170`, `NemoService.swift:190`). A dead sidecar
        // fails EVERY window, so the user recorded an hour, got no speaker labels
        // at all, and saw nothing. This project ranks a silent wrong result above
        // a loud one, so trading the first defect for that was the wrong direction.
        //
        // A RUN, not a single error. One window failing is genuinely non-fatal —
        // that is the whole reason this handler exists — while three in a row is
        // no longer "a window", it is the engine. The counter resets on any
        // success, so a meeting that recovers never accuses anyone.
        //
        // `diarizationCaution`, NOT `diarizationError`: the caution channel is a
        // banner and nothing more. `diarizationError` would settle the stop gate
        // and mark the stop panel failed, which is exactly the mid-recording state
        // mutation this whole handler exists to prevent.
        batchLiveFailureRun += 1
        guard batchLiveFailureRun >= Self.batchLiveFailureRunLimit else { return true }
        diarizationCaution =
            "Speaker labelling has failed \(batchLiveFailureRun) times in a row, so "
            + "this recording may end up with no speaker names. The words are "
            + "unaffected. See logs/batch-live-diarization.log — last reason: \(message)"
        return true
    }

    /// How many CONSECUTIVE live-window failures mean the engine is gone rather
    /// than one window being awkward.
    ///
    /// Three, because a single failure is the ordinary non-fatal case this handler
    /// is built around, and a dead sidecar fails every window — so the run reaches
    /// any threshold almost immediately while a one-off never does. Deliberately
    /// not 1: at the shortest interval that would put a banner up for one bad
    /// window, and a banner that fires on the ordinary case is a banner nobody
    /// reads (the 2026-08-06 rule, from the Detect overlap tab).
    static let batchLiveFailureRunLimit = 3

    /// Drop every outstanding window: its temp WAV and its map entry together.
    ///
    /// Called on a live-window failure and at session start. The map had NO
    /// lifecycle clear at all before the 2026-08-14 audit — not at
    /// `configure*`, not at Stop, not in `startOver()` — so a failed window's
    /// entry and its 1.9 MB WAV survived until the app quit.
    func releaseBatchLiveWindows() {
        for path in liveDiarWindowByPath.keys {
            try? FileManager.default.removeItem(atPath: path)
        }
        liveDiarWindowByPath = [:]
    }

    // `liveCountWarning` LIVED HERE and was removed on 2026-08-13, the same day it
    // was added: the owner saw the tab print it and asked for it gone. Removed
    // rather than left unread, so it cannot become a rule nothing keeps in step.
    // The measurement it carried is not lost — it is in this file's header, which
    // is where a future edit to `dispatchBatchLiveWindow`'s count argument will be
    // read anyway.

    /// One line per decision, its own file, one writer — the house rule.
    func batchLiveLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "batch-live-diarization", message: message)
    }
}
