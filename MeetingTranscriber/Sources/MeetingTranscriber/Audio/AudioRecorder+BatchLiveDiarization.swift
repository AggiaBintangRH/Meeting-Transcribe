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
        guard !samples.isEmpty else { return }
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
                                  + "\(turns.count) identified")
                self.applyChunkTurns(turns, windowStart: windowStart, stream: stream)
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

    /// The warning the tab shows when this mode is on with the count on Auto.
    /// Nil when there is nothing to warn about, so the caller renders nothing.
    nonisolated static func liveCountWarning(isBatchEngine: Bool, finalPass: Bool,
                                             numSpeakers: Int) -> String? {
        guard runsBatchLiveDiarization(isBatchEngine: isBatchEngine,
                                       finalPass: finalPass), numSpeakers == 0 else {
            return nil
        }
        return "Live labelling runs on windows this short, and on 25–30 s clips of "
             + "two people the automatic speaker count of these engines was measured "
             + "wrong — 9 to 15 speakers. Set the speaker count beside the record "
             + "button; pinned to the real number, every engine answered correctly."
    }

    /// One line per decision, its own file, one writer — the house rule.
    func batchLiveLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "batch-live-diarization", message: message)
    }
}
