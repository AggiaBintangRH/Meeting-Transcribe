import AVFoundation
import Foundation

// The stop-time chunked-ASR pass, and the two settings that govern it
// (owner-requested 2026-08-03):
//
//   chunked.finalPass       "Run a transcription pass at stop"
//   chunked.continueOnStop  RETIRED 2026-08-06 (owner). Its toggle was removed
//                           and the key is no longer read: the pass is always
//                           tail-only, which was this key's own default. The
//                           `.full` mode below is KEPT — it is still pinned by
//                           tests and is what the toggle would restore.
//
// They mirror the Diarization tab's existing pair EXACTLY in shape, and
// deliberately NOT in default: `diarization.continueOnStop` defaults to FALSE
// (a stop pass re-diarizes the whole recording), the retired `chunked.continueOnStop`
// defaults to TRUE (the ASR pass covers only the tail). TRUE is what the app has
// always done, and requirement #1 of this change is that both keys ABSENT
// reproduce today's stop path exactly — so the default has to follow behaviour,
// not symmetry. The Settings copy says so, because a mirrored pair with opposite
// defaults is otherwise a trap.
//
//   finalPass ON  + tail ON   → today's behaviour: flush the sidecar's buffer,
//                               one tail window, one `final`.
//   finalPass ON  + tail OFF  → re-transcribe the WHOLE recording, window by
//                               window, over the recorded WAV.
//   finalPass OFF             → no ASR pass at stop at all. The realtime
//                               (Nemotron) text for the tail is KEPT — see
//                               `sweepsUnconfirmedTail` below, which is the one
//                               genuinely dangerous corner of this change.
//
// ── WHY THE FULL PASS IS A WINDOW SEQUENCE, NOT ONE WHOLE-FILE CALL ──────────
//
// `ChunkedASRService.transcribeFile` (the `-2` FILE-TRANSCRIBE frame) returns
// TEXT ONLY — no window, no segmentation. One call over a 60-minute meeting
// would come back as an undifferentiated blob, and `speakerRanges`
// (AudioRecorder+RowDerivation) would then have to split it
// character-proportionally across the entire meeting: far worse attribution than
// the app already has, and no way to notice it went wrong. Re-running the SAME
// window sequence instead keeps windows, per-window speaker attribution,
// per-window `asr` confidence and the aligner path all working exactly as they
// do live, and needs NO new sidecar protocol — the `-2` frame already exists and
// is already driven by the Remote stream and by overlap repair.
extension AudioRecorder {

    // MARK: - The decision (pure, so the branch is testable without audio)

    /// What the chunked pass does at stop for this session. The `RemoteStopMode`
    /// precedent: the branch itself is a pure function, so which legs of the stop
    /// gate are taken can be enumerated in a test rather than reasoned about.
    enum ChunkedStopMode: Hashable {
        case none   // no ASR pass at stop; the realtime tail text is kept
        case tail   // flush the sidecar's live buffer — today's behaviour
        case full   // re-transcribe the whole recording, window by window
    }

    /// The stop-pass decision.
    ///
    /// Note the two fall-backs to `.tail` rather than `.none`: a missing
    /// recording, and a model whose full pass is refused. `.tail` is what the
    /// user had before they touched the setting and it never leaves audio
    /// untranscribed, whereas `.none` would silently drop the tail. Neither
    /// fall-back is reachable through the normal path — startup refuses the
    /// model combinations (`chunkedFullPassRefusalMessage`, checked in
    /// `prepareAndCapture` beside the dual-stream and MOSS refusals) and a
    /// session always has a recording URL — they exist for the one hole those
    /// checks cannot close, which is the setting being flipped mid-recording.
    nonisolated static func chunkedStopMode(finalPass: Bool,
                                            continueOnStop: Bool,
                                            hasChunkedModel: Bool,
                                            hasRecording: Bool,
                                            chunkedModelID: String) -> ChunkedStopMode {
        // No chunked model → nothing would ever drain a queued window. This is
        // the pre-existing branch at `stop()`, moved into the decision so there
        // is exactly one place that decides.
        guard hasChunkedModel else { return .none }
        guard finalPass else { return .none }
        if continueOnStop { return .tail }
        guard chunkedFullPassRefusalMessage(chunkedModelID: chunkedModelID) == nil else { return .tail }
        return hasRecording ? .full : .tail
    }

    /// Exactly what `stop()` does for a given mode. A value rather than a set of
    /// `if`s in `stop()` because the dangerous item is `sweepsUnconfirmedTail`,
    /// and a pure value is the only way to assert it in both directions.
    struct ChunkedStopPlan: Equatable {
        /// Append the tail window to `pendingChunkWindows`.
        var queuesTailWindow = false
        /// Park the tail's audio for the aligner (`stashAlignAudio`).
        var stashesAlignAudio = false
        /// Send the sidecar a FLUSH (n = 0) for its live buffer.
        var flushesSidecar = false
        /// Settle `lastChunkDone` right here in `stop()`, because nothing
        /// asynchronous is going to.
        var settlesImmediately = false
        /// Whether `checkLastChunkDone` may delete the leftover unconfirmed
        /// (realtime) segments. **THE dangerous flag of this change.**
        ///
        /// That sweep exists because "the chunked pass is authoritative for all
        /// audio" — true when a pass runs, false when one does not. With
        /// `finalPass` OFF it would silently delete up to 1.5× the chunk
        /// interval of real speech (45 s at the default, 180 s at a 120 s
        /// interval) leaving no trace anywhere: the text is simply gone from the
        /// transcript. Over-deletion is this project's established dangerous
        /// direction, so the sweep is OFF whenever no pass ran.
        var sweepsUnconfirmedTail = false
        /// Drive `startChunkedFullPass`.
        var runsFullPass = false
    }

    nonisolated static func chunkedStopPlan(_ mode: ChunkedStopMode) -> ChunkedStopPlan {
        switch mode {
        case .tail:
            // Byte-for-byte today's stop path, and the default with both keys absent.
            return ChunkedStopPlan(queuesTailWindow: true, stashesAlignAudio: true,
                                   flushesSidecar: true, settlesImmediately: false,
                                   sweepsUnconfirmedTail: true, runsFullPass: false)
        case .full:
            // The full pass replaces every confirmed segment, so the realtime
            // leftovers are genuinely superseded — the sweep stays on.
            return ChunkedStopPlan(queuesTailWindow: false, stashesAlignAudio: false,
                                   flushesSidecar: false, settlesImmediately: false,
                                   sweepsUnconfirmedTail: true, runsFullPass: true)
        case .none:
            // Nothing is coming, so settle here — the same shape as the
            // pre-existing "no chunked model" branch. And do NOT sweep: with no
            // pass, the realtime tail IS the transcript for that audio.
            return ChunkedStopPlan(queuesTailWindow: false, stashesAlignAudio: false,
                                   flushesSidecar: false, settlesImmediately: true,
                                   sweepsUnconfirmedTail: false, runsFullPass: false)
        }
    }

    // MARK: - The window sequence (pure)

    /// The full pass's windows, in order, over a recording of `recordingLength`
    /// seconds at `intervalSec` per window.
    ///
    /// Uniform windows, deliberately unlike the LIVE boundaries, which are cut at
    /// silence (`chunkElapsed >= interval && !speaking`, hard cap at 1.5×). There
    /// is no record of where those cuts fell — `pendingChunkWindows` is drained
    /// as each chunk lands — and reconstructing them would mean re-running the
    /// VAD over the file. A uniform partition can cut mid-word at a boundary,
    /// which is the one real cost of the full pass and is stated in the UI copy.
    ///
    /// `minTailSec` drops a final sliver: under a quarter second there is nothing
    /// an ASR model can usefully say, and it would cost a whole round trip.
    nonisolated static func fullPassWindows(recordingLength: Double,
                                            intervalSec: Double,
                                            minTailSec: Double = 0.25) -> [ClosedRange<Double>] {
        guard recordingLength > minTailSec, intervalSec > 0 else { return [] }
        var windows: [ClosedRange<Double>] = []
        var start = 0.0
        while start < recordingLength {
            let end = min(start + intervalSec, recordingLength)
            if end - start >= minTailSec { windows.append(start...end) }
            start = end
        }
        return windows
    }

    // MARK: - Cutting at silence, the way the live pass does

    /// One RMS value per `silenceFrameSec` of the recording, streamed so a long
    /// meeting never sits in memory as samples (a 60-minute recording is ~230 MB
    /// of 16 kHz float; its energy profile is under 1 MB).
    ///
    /// Returns nil when the file cannot be read — the caller then falls back to
    /// uniform windows, which is what this whole file did before.
    nonisolated static let silenceFrameSec: Double = 0.02

    nonisolated static func fullPassEnergyProfile(from url: URL) -> [Float]? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let format = file.processingFormat
        let rate = format.sampleRate
        guard rate > 0, file.length > 0 else { return nil }
        let frameLen = max(1, Int(rate * silenceFrameSec))
        var energies: [Float] = []
        energies.reserveCapacity(Int(Double(file.length) / Double(frameLen)) + 8)

        var carry: [Float] = []
        let chunkFrames: AVAudioFrameCount = 32_768
        while file.framePosition < file.length {
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunkFrames)
            else { break }
            do { try file.read(into: buffer, frameCount: chunkFrames) } catch { break }
            let got = Int(buffer.frameLength)
            guard got > 0 else { break }
            // Mono mix — the energy profile only needs loudness, not channels.
            var mono = [Float](repeating: 0, count: got)
            if let channels = buffer.floatChannelData {
                let count = Int(buffer.format.channelCount)
                for c in 0..<count {
                    for i in 0..<got { mono[i] += channels[c][i] }
                }
                if count > 1 { for i in 0..<got { mono[i] /= Float(count) } }
            }
            carry.append(contentsOf: mono)
            while carry.count >= frameLen {
                let frame = carry[0..<frameLen]
                var sum: Float = 0
                for v in frame { sum += v * v }
                energies.append((sum / Float(frameLen)).squareRoot())
                carry.removeFirst(frameLen)
            }
        }
        return energies.isEmpty ? nil : energies
    }

    /// Window boundaries snapped to silence, mirroring the LIVE rule.
    ///
    /// WHY THIS EXISTS. The live path cuts a chunk only when the interval has
    /// elapsed AND the VAD says nobody is speaking, with a hard cap at 1.5x:
    ///
    ///     let boundary = (chunkElapsed >= chunkInterval && !speaking)
    ///                 || chunkElapsed >= chunkInterval * 1.5
    ///
    /// — which is what the Chunked tab means by "cut at silence, never mid-word".
    /// The full pass originally used UNIFORM boundaries, so it could cut straight
    /// through a word, and a word split across two windows is mis-transcribed at
    /// both ends. A re-transcription that damages boundaries the live pass got
    /// right is worse than not re-transcribing at all, so the rule is reproduced
    /// here rather than approximated.
    ///
    /// It is computed from an ENERGY PROFILE rather than by driving the Silero
    /// sidecar: the live VAD is fed ~32 ms at a time as audio arrives, and asking
    /// it for a 60-minute recording would be ~112 000 round trips. The threshold
    /// is the project's existing one — a frame is silence below 15 % of the
    /// recording's speech RMS — the same rule `WordAttribution.snapRanges` was
    /// derived from and the same one used to measure word timings (CLAUDE.md,
    /// "Whisper's native word_timestamps").
    ///
    /// The hard cap is what makes this safe on continuous speech: if no silence
    /// is found in range, the boundary lands at 1.5x the interval exactly as the
    /// live path's second clause does.
    nonisolated static func fullPassWindowsAtSilence(recordingLength: Double,
                                                     intervalSec: Double,
                                                     energies: [Float],
                                                     frameSec: Double = silenceFrameSec,
                                                     minTailSec: Double = 0.25)
        -> [ClosedRange<Double>] {
        guard recordingLength > minTailSec, intervalSec > 0, !energies.isEmpty else { return [] }

        // Speech RMS = the mean of the frames that are actually loud enough to be
        // speech, so a recording with long silences is not judged against its own
        // quiet. Falls back to the overall mean when nothing stands out.
        let sorted = energies.sorted()
        let median = sorted[sorted.count / 2]
        let speech = energies.filter { $0 > median }
        let speechRMS = speech.isEmpty
            ? energies.reduce(0, +) / Float(energies.count)
            : speech.reduce(0, +) / Float(speech.count)
        let quiet = speechRMS * 0.15

        func isQuiet(atSecond t: Double) -> Bool {
            let i = Int(t / frameSec)
            guard i >= 0, i < energies.count else { return false }
            return energies[i] < quiet
        }

        var windows: [ClosedRange<Double>] = []
        var start = 0.0
        while start < recordingLength {
            let target = start + intervalSec
            let cap = min(start + intervalSec * 1.5, recordingLength)
            var end = min(target, recordingLength)
            if target < recordingLength {
                // Walk forward from the target to the cap looking for quiet,
                // exactly as the live path waits for `!speaking` after the
                // interval has elapsed. Forward only: cutting EARLY would shorten
                // a window the model was going to get anyway.
                var t = target
                var found = false
                while t <= cap {
                    if isQuiet(atSecond: t) { end = t; found = true; break }
                    t += frameSec
                }
                if !found { end = cap }   // continuous speech — the 1.5x clause
            }
            if end - start >= minTailSec { windows.append(start...end) }
            if end <= start { break }     // no progress ⇒ stop rather than spin
            start = end
        }
        return windows
    }

    // MARK: - Refusal (refuse loudly, never substitute)

    /// Why the FULL pass cannot run with this chunked model, or nil to allow it.
    /// Mirrors `dualStreamRefusalMessage` / `mossRefusalMessage`: shown in
    /// Settings where the choice is made, AND enforced as a hard startup refusal
    /// so the user is told before a meeting rather than after one.
    ///
    /// **MOSS — impossible, not merely slow.** Measured on this M4 (CLAUDE.md,
    /// "MOSS memory"): a 360 s pass hits the 2048-token cap exactly, i.e. the
    /// transcript is truncated mid-sentence and `parse_transcript` drops the
    /// incomplete trailing segment WITHOUT raising; a 600 s pass never finished
    /// in 20+ minutes while the MPS pool climbed toward the machine's limit. A
    /// full pass is a long sequence of exactly those calls.
    ///
    /// **Voxtral — refused rather than confirmed.** ~27 s per 30 s chunk, so a
    /// 60-minute meeting is ~54 minutes of processing behind the blocking stop
    /// overlay. Refusal was chosen over an "are you sure?" for three reasons:
    /// this app has no confirmation-dialog pattern to reuse, Voxtral is already
    /// refused in two other combinations for the same arithmetic
    /// (`dualStreamRefusalMessage`, `mossRefusalMessage`) so a third rule would
    /// be inconsistent, and a confirmation the user clicks through at Stop is a
    /// choice made at the worst possible moment. The tail pass — the default —
    /// still works with Voxtral.
    /// The words on the toggle, stated ONCE.
    ///
    /// ⚠ The refusal below quotes them, the tab draws them, and a test asserts
    /// the quote really is the label — so this is the thing all three read
    /// rather than three copies that agree today. `testRefusalsSayWhatToDoAboutIt`
    /// FAILED on exactly that coupling when the label was reworded on 2026-09-04
    /// and the message still quoted "Continue from live text (tail only)": a
    /// refusal pointing at a control by a name the control no longer has.
    nonisolated static let fullPassToggleLabel = "the whole recording"

    nonisolated static func chunkedFullPassRefusalMessage(chunkedModelID: String) -> String? {
        switch chunkedModelID {
        // ⚠ MOSS USED TO BE REFUSED HERE, AND THE REASON HAD EXPIRED TWICE OVER
        // (lifted 2026-08-18, owner-requested). The message cited a single 360 s
        // pass hitting a 2048-token cap. Both halves are stale: `MAX_NEW_TOKENS`
        // has been 5120 since 2026-08-05 (the checkpoint's own default — the 2048
        // was a copy of upstream's CLI default, never a model limit), and this
        // pass has never sent 360 s of audio in one call. It cuts at
        // `chunked.intervalSec`, so 120 s is its ceiling — measured at ~1638 of
        // the budget even on the OLD cap.
        //
        // What actually blocked it was never the cap: the FILE-TRANSCRIBE frame
        // discarded MOSS's speaker segments, so a full pass produced a
        // re-transcribed meeting with nobody in it. `emit_file` carries them now,
        // and `replaceOfficeSegments` rebuilds pinned rows from them.
        case "voxtral":
            return "Voxtral cannot re-transcribe a whole recording in reasonable time. It needs "
                 + "about 27 s per 30 s chunk (Qwen3 4.3 s, Whisper 0.3–2.1 s, Granite 5.6 s), "
                 + "so a 60-minute meeting would take about 54 minutes of processing after you "
                 + "press Stop. Switch \"\(fullPassToggleLabel)\" off with Voxtral so only the last "
                 + "seconds are re-transcribed, or pick another chunked model."
        default:
            return nil
        }
    }

    /// Plain-language cost of a full pass for this model, for the Settings copy.
    ///
    /// 🔴 EVERY FIGURE HERE WAS WRONG UNTIL 2026-09-04, and one was wrong by 11x.
    /// The old note took each model's per-30 s INFERENCE time from CLAUDE.md and
    /// multiplied by 120, which is the extrapolation CLAUDE.md itself warns about
    /// ("a per-chunk timing does not predict cost at a different chunk length").
    /// It also ignores everything around the inference — writing the window WAV,
    /// the round trip, the decode — which is most of the gap on the fast models.
    ///
    /// These are now MEASURED end to end: every window of a real recording driven
    /// through the real sidecar over the real `-2` file-transcribe frame, at
    /// 16 kHz mono, which is what `loadWindow` writes.
    ///
    /// | model | old claim | measured | |
    /// |---|---|---|---|
    /// | granite | ~11 min | **1.0 min** | 11x overstated |
    /// | qwen3 | ~9 min | **2.1 min** | 4x overstated |
    /// | whisper | 1–4 min | **4.2 min** | UNDERstated |
    /// | moss | ~13 min | **10.2 min** | |
    /// | voxtral | ~54 min | **42.6 min** | still refused |
    ///
    /// ⚠ THE ORDER WAS WRONG TOO, not merely the magnitudes. The old note implied
    /// Whisper was the quickest (0.3–2.1 s per 30 s against Qwen3's 4.3 s); it is
    /// the SLOWEST of the three fast models here. Anyone re-measuring should drive
    /// the sidecar, not time `transcribe()`.
    ///
    /// ⚠ Measured at the shipped 30 s `chunked.intervalSec` on one recording
    /// (`meeting-2026-08-19T02-30-22Z.wav`, 5 min of it). Longer windows change
    /// the arithmetic for the autoregressive models — the same warning that made
    /// the old figures wrong.
    nonisolated static func fullPassCostNote(chunkedModelID: String) -> String {
        switch chunkedModelID {
        case "whisper":
            return "Measured on this Mac: about 14x realtime — roughly 4 minutes to re-transcribe "
                 + "a 60-minute meeting."
        case "qwen3":
            return "Measured on this Mac: about 28x realtime — roughly 2 minutes to re-transcribe "
                 + "a 60-minute meeting, the quickest of the accurate models."
        case "granite":
            return "Measured on this Mac: about 58x realtime — roughly 1 minute to re-transcribe a "
                 + "60-minute meeting, the fastest here. It writes no capitals and almost no "
                 + "punctuation, which is the reason to weigh against the speed."
        case "voxtral":
            return "Measured on this Mac: about 1.4x realtime — roughly 43 minutes to re-transcribe "
                 + "a 60-minute meeting, which is why the whole-recording pass is refused with it."
        case "moss":
            return "Measured on this Mac: about 6x realtime — roughly 10 minutes to re-transcribe a "
                 + "60-minute meeting. It is the one model where this also REDIARIZES: MOSS writes "
                 + "speaker labels with the text, so the whole meeting is re-labelled from the "
                 + "file. Measured on a 5-person recording, longer windows are what fix its speaker "
                 + "count (30 s gave 6 speakers, 120 s gave the correct 5)."
        default:
            return ""
        }
    }

    // MARK: - Watchdog budgets

    /// Seconds the stop overlay must be willing to wait, given every stop-time
    /// pass this session actually scheduled.
    ///
    /// PURE AND CENTRAL ON PURPOSE. Two bugs on 2026-07-31 were the same shape —
    /// a watchdog that did not scale with the work in front of it:
    ///
    ///   * the TAIL diarization used a flat 120 s, and in the one mode where the
    ///     "tail" is the whole recording a 60-minute meeting was marked
    ///     "timed out" while the sidecar worked normally;
    ///   * the stop budget counted only the CHUNKED full pass, so a session that
    ///     re-labelled with MOSS but transcribed on tail got the 600 s floor for
    ///     ~13 minutes of work.
    ///
    /// The second was found only because the first taught us to look. Three
    /// stop-time passes exist now, and a fourth would repeat it again — so the
    /// arithmetic lives HERE, in one place, with a test that fails if a scheduled
    /// pass contributes nothing.
    ///
    /// Window counts are SUMMED rather than maxed: the passes can run at once and
    /// contend for the same GPU, so their durations add.
    ///
    /// **`batchPassSeconds` is the fourth pass, added 2026-08-04 as
    /// `spectralPassSeconds` and renamed 2026-08-07 when NeMo became the second
    /// engine to use it** — and it is the case the doc comment above predicted. It
    /// is expressed in SECONDS rather than in windows because a whole-file batch
    /// engine has no windows: it is one call over the whole recording, and its cost
    /// is a function of the recording's length, not of a count. Callers pass
    /// exactly the number that engine's own leg watchdog uses
    /// (`batchPassWatchdogSeconds`), so the overlay can never give up before the leg
    /// it is waiting on does. Zero when no batch pass is scheduled, which is what
    /// keeps every pre-existing budget identical.
    ///
    /// ONE parameter for both engines rather than two, because they are mutually
    /// exclusive: a session has at most one diarization engine, so a second field
    /// could only ever be zero while the first was set.
    nonisolated static func stopWatchdogSeconds(chunkedFullPassWindows: Int,
                                                mossFullPassWindows: Int,
                                                batchPassSeconds: Double = 0,
                                                floor: Double = 600,
                                                headroom: Double = 300) -> Double {
        let scheduled = max(0, chunkedFullPassWindows) + max(0, mossFullPassWindows)
        let batch = max(0, batchPassSeconds)
        guard scheduled > 0 || batch > 0 else { return floor }
        let windowed = scheduled > 0 ? fullPassWatchdogSeconds(windowCount: scheduled) : 0
        return max(floor, windowed + batch + headroom)
    }



    /// Seconds to allow a full pass of `windowCount` windows. 120 s each is the
    /// chunked client's OWN per-request timeout (`transcribeFile`), so a window
    /// can never take longer than that without throwing — this is the backstop
    /// for the case where the loop itself wedges, not for a slow model.
    nonisolated static func fullPassWatchdogSeconds(windowCount: Int) -> Double {
        Double(120 * max(1, windowCount) + 60)
    }

    // MARK: - Driving the full pass

    /// Re-transcribe the whole recording, window by window, replacing the
    /// confirmed segments for each window as its result lands.
    ///
    /// No new stop-gate leg, and none is needed: the pass holds `chunkedBusy` for
    /// its whole life, so `checkLastChunkDone`'s existing `!chunkedBusy` guard
    /// already keeps the "chunk" leg open until the last window is in. The FIFO
    /// discipline `pendingChunkWindows` provides for the streaming path is
    /// inherent here — `transcribeFile` is awaited one window at a time and its
    /// replies are id-correlated by the service — so `pendingChunkWindows` stays
    /// empty throughout and no second queue is introduced.
    func startChunkedFullPass(recording: URL, service: ChunkedASRService) {
        let interval = Double(
            UserDefaults.standard.object(forKey: "chunked.intervalSec") as? Int ?? 30
        )
        // Cut at silence when the recording can be profiled, exactly as the live
        // pass does; fall back to uniform windows only if it cannot be read.
        // Uniform boundaries slice through words, and a word split across two
        // windows is mis-transcribed at both ends — see `fullPassWindowsAtSilence`.
        let energies = Self.fullPassEnergyProfile(from: recording)
        let windows: [ClosedRange<Double>]
        if let energies {
            windows = Self.fullPassWindowsAtSilence(recordingLength: recordingElapsed,
                                                    intervalSec: interval,
                                                    energies: energies)
            chunkedStopLog("FULL PASS windows cut at silence "
                           + "(\(energies.count) energy frames profiled)")
        } else {
            windows = Self.fullPassWindows(recordingLength: recordingElapsed,
                                           intervalSec: interval)
            chunkedStopLog("FULL PASS could not profile the recording for silence — "
                           + "falling back to uniform \(Int(interval))s windows, which can "
                           + "cut mid-word")
        }
        let remoteRecording = fullPassHoldsRemoteLeg ? remoteRecordingURL : nil
        guard !windows.isEmpty else {
            chunkedStopLog("FULL PASS skipped — the recording is only "
                           + "\(fmt(recordingElapsed))s, nothing to re-transcribe")
            releaseFullPassRemoteLeg()
            checkLastChunkDone()
            return
        }

        chunkedBusy = true
        chunkedError = nil
        chunkedStopLog("FULL PASS start — \(windows.count) window(s) of \(Int(interval))s over "
                       + "\(fmt(recordingElapsed))s of office audio"
                       + (remoteRecording != nil
                          ? "; the same windows are then re-run over the Remote WAV" : ""))
        // The whole recording is about to be re-transcribed, so the realtime
        // leftovers are superseded now rather than at the end — otherwise a
        // stale unconfirmed row sits under the transcript for the entire pass.
        if segments.contains(where: { !$0.confirmed }) {
            segments.removeAll { !$0.confirmed }
            rebuildDisplayRows()
        }
        // MOSS+MOSS: the labels are about to be replaced wholesale too, because
        // this model writes them WITH the text. Cleared up front for the same
        // reason as the unconfirmed rows — a stale set shown under a transcript
        // being rebuilt is worse than none — and `mossChunkIndex` restarts so the
        // full pass's windows are numbered 0..n rather than continuing the live
        // count. Window numbering is what `identifyMossTurns` groups on, so a
        // continued count would leave it grouping live windows that no longer
        // exist. Inert for every other model: `mossDiarizationActive` is false.
        if mossDiarizationActive, mossIsChunkedModel {
            mossTurns = []
            mossChunkIndex = 0
            mossIdentifyStarted = false
            mossLog("FULL PASS — live labels cleared; the whole recording is "
                    + "re-transcribed and re-labelled from the file")
        }

        let totalWindows = windows.count * (remoteRecording == nil ? 1 : 2)
        startFullPassWatchdog(windowCount: totalWindows)

        chunkedFullPassTask = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.runFullPass(windows: windows, recording: recording,
                                   service: service, isRemote: false)
            if let remoteRecording {
                self.remoteSegments = []
                self.rebuildDisplayRows()
                await self.runFullPass(windows: windows, recording: remoteRecording,
                                       service: service, isRemote: true)
            }
            self.fullPassWatchdog?.cancel()
            self.fullPassWatchdog = nil
            self.chunkedFullPassTask = nil
            self.chunkedBusy = false
            self.chunkedStopLog("FULL PASS done")
            self.releaseFullPassRemoteLeg()
            self.checkLastChunkDone()
        }
    }

    /// One stream's worth of the full pass. Sequential on purpose: the sidecar is
    /// single-threaded and `transcribeFile` is serialized by its own gate, so
    /// issuing windows concurrently would only queue them behind each other while
    /// making the progress count meaningless.
    private func runFullPass(windows: [ClosedRange<Double>], recording: URL,
                             service: ChunkedASRService, isRemote: Bool) async {
        // Office only. Remote rows have never carried word timings (see
        // `RemoteSegment`), and the full pass is not the place to change that.
        let aligner = isRemote ? nil : modelLoader.aligner
        let stepID = isRemote ? "remote" : "chunk"
        let stream = isRemote ? "remote" : "office"
        let label = isRemote ? "Re-transcribing remote audio" : "Re-transcribing the recording"
        let restingName = isRemote ? "Transcribing remote audio" : "Transcribing final audio"

        for (index, window) in windows.enumerated() {
            if Task.isCancelled { break }
            setStopStepName(stepID, "\(label) (\(index + 1)/\(windows.count))")
            let span = "[\(fmt(window.lowerBound))-\(fmt(window.upperBound))] \(stream)"

            let samples = await Task.detached(priority: .utility) {
                Self.loadWindow16k(from: recording,
                                   start: window.lowerBound, end: window.upperBound)
            }.value
            guard let samples, !samples.isEmpty else {
                chunkedStopLog("FULL PASS \(span) could not read this window from the WAV — skipped")
                continue
            }
            // The Remote silence gate applies to the full pass exactly as it does
            // live: an idle conferencing channel is a guaranteed-empty transcript
            // at full model cost, and here there are hundreds of such windows.
            if isRemote, let reason = Self.remoteChunkSkipReason(samples) {
                chunkedStopLog("FULL PASS \(span) \(reason) — skipped")
                continue
            }
            let url = await Task.detached(priority: .utility) {
                Self.writeTempWAV(samples: samples,
                                  prefix: isRemote ? "full-pass-remote" : "full-pass")
            }.value
            guard let url else {
                chunkedStopLog("FULL PASS \(span) could not write a temp WAV — skipped")
                continue
            }
            do {
                let result = try await service.transcribeFile(path: url.path)
                let trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if isRemote {
                    replaceRemoteSegments(window: window, text: trimmed, conf: result.conf)
                } else {
                    replaceOfficeSegments(window: window, text: trimmed, conf: result.conf,
                                          aligner: aligner, samples: samples,
                                          mossSegments: result.segments)
                }
                chunkedStopLog("FULL PASS \(span) "
                               + (trimmed.isEmpty ? "empty transcript" : trimmed))
            } catch {
                // Log and carry on, exactly as `onChunkError` and the remote
                // chunk path do: one failed window must not cost the user the
                // rest of the re-transcription.
                let message = "Some audio could not be re-transcribed — see logs/chunked-stop.log"
                if isRemote { remoteChunkError = message } else { chunkedError = message }
                chunkedStopLog("FULL PASS FAIL \(span) \(error.localizedDescription)")
            }
            // Deleted on BOTH the success and the error path — the placement
            // `flushRemoteChunk` and `requestAlignment` already use. The `continue`
            // paths above never created a file.
            try? FileManager.default.removeItem(at: url)
        }
        setStopStepName(stepID, restingName)
    }

    /// Swap this window's confirmed office text for the re-transcribed version.
    ///
    /// Removal is by time OVERLAP rather than by equality, because the live
    /// windows were cut at silence and the full pass's are uniform — they do not
    /// line up. A live segment straddling this boundary is therefore removed
    /// here and its later half is restored by the NEXT window, which is why the
    /// pass must run in chronological order.
    private func replaceOfficeSegments(window: ClosedRange<Double>, text: String,
                                       conf: Double?, aligner: AlignerService?,
                                       samples: [Float],
                                       mossSegments: [ChunkedASRService.MossSegment]? = nil) {
        segments.removeAll { segment in
            guard segment.confirmed, let existing = segment.window else { return false }
            return existing.overlaps(window)
        }
        // MOSS+MOSS: this model transcribed AND labelled the window in one call,
        // so its reply becomes one PINNED row per speaker plus the turns behind
        // them — the same two products `applyMossChunk` makes from a live chunk,
        // built by the same two pure functions so the id scheme cannot drift.
        //
        // Deliberately BEFORE the plain-text branch and returning early: appending
        // both would put every word in the window on screen twice, which is the
        // exact duplication `applyMossChunk`'s return value exists to prevent on
        // the live path.
        if mossDiarizationActive, mossIsChunkedModel, let mossSegments {
            let index = mossChunkIndex
            mossChunkIndex += 1
            let turns = Self.mossTurns(from: mossSegments, window: window, chunkIndex: index)
            mossTurns.append(contentsOf: turns)
            let pinned = Self.mossPinnedSegments(from: mossSegments,
                                                 window: window, chunkIndex: index)
            for segment in pinned {
                let at = segments.firstIndex {
                    ($0.window?.lowerBound ?? .infinity) > (segment.window?.lowerBound ?? 0)
                } ?? segments.count
                segments.insert(segment, at: at)
            }
            // No aligner request: a pinned row's speaker is already decided, and
            // word times would only re-split a row the model attributed itself.
            mossLog("FULL PASS window #\(index) [\(fmt(window.lowerBound))-"
                    + "\(fmt(window.upperBound))] \(pinned.count) row(s), "
                    + "speakers \(Set(turns.map(\.name)).sorted())")
            rebuildDisplayRows()
            return
        }
        if !text.isEmpty {
            let segment = TranscriptSegment(text: text, confirmed: true,
                                            window: window, asrConf: conf)
            // Insert in time order rather than appending: a window whose text
            // arrives after a later one was already placed (it cannot today, but
            // the invariant should not depend on that) must still read in order.
            let insertAt = segments.firstIndex {
                ($0.window?.lowerBound ?? .infinity) > window.lowerBound
            } ?? segments.count
            segments.insert(segment, at: insertAt)
            // Re-request alignment per replaced window. Any reply still in flight
            // for the segment we just deleted is dropped by
            // `indexForAlignedWords` ("the segment is gone"), and this new request
            // is keyed to the new segment's UUID — so the two cannot cross.
            if let aligner, !samples.isEmpty {
                requestAlignment(aligner: aligner, samples: samples,
                                 segmentID: segment.id, text: text)
            }
        }
        rebuildDisplayRows()
    }

    /// The remote twin of `replaceOfficeSegments`. `remoteSegments` was cleared
    /// before the remote sub-pass began, so this only ever appends — the removal
    /// is kept anyway so the two halves cannot diverge if that ever changes.
    private func replaceRemoteSegments(window: ClosedRange<Double>, text: String, conf: Double?) {
        remoteSegments.removeAll { $0.window.overlaps(window) }
        if !text.isEmpty {
            let insertAt = remoteSegments.firstIndex {
                $0.window.lowerBound > window.lowerBound
            } ?? remoteSegments.count
            remoteSegments.insert(RemoteSegment(text: text, window: window, conf: conf),
                                  at: insertAt)
        }
        rebuildDisplayRows()
    }

    /// Fail loudly and settle the leg rather than hanging the overlay. Scaled by
    /// the number of windows, because the fixed 180 s per-chunk watchdog this
    /// replaces is far too short for a pass that legitimately takes minutes.
    private func startFullPassWatchdog(windowCount: Int) {
        fullPassWatchdog?.cancel()
        let budget = Self.fullPassWatchdogSeconds(windowCount: windowCount)
        fullPassWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(budget))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.chunkedBusy else { return }
                self.chunkedFullPassTask?.cancel()
                self.chunkedFullPassTask = nil
                self.fullPassWatchdog = nil
                self.chunkedBusy = false
                let message = "Re-transcribing the recording timed out after "
                            + "\(Int(budget))s — see logs/chunked-stop.log"
                self.chunkedError = message
                self.chunkedStopLog("FULL PASS TIMEOUT after \(Int(budget))s")
                self.releaseFullPassRemoteLeg()
                self.checkLastChunkDone()
            }
        }
    }

    /// Give back the remote leg of the stop gate, once. The full pass takes it in
    /// `stop()` (before the remote block runs `checkRemoteChunksDone`, or that
    /// call would settle the leg before the pass had claimed it), and both the
    /// completion path and the watchdog can reach here — hence the flag.
    func releaseFullPassRemoteLeg() {
        guard fullPassHoldsRemoteLeg else { return }
        fullPassHoldsRemoteLeg = false
        finishRemoteChunk()
    }

    // MARK: - Reading one window out of the recorded WAV

    /// Load `start..<end` of a recording as 16 kHz mono float samples, or nil.
    ///
    /// Opened per window rather than once for the whole file: the alternative is
    /// holding the entire recording resident (a 60-minute 44.1 kHz mono float
    /// capture is ~630 MB before resampling) across a pass that lasts minutes,
    /// and `AVAudioFile` is not `Sendable` so it could not be carried across the
    /// awaits anyway. The per-window open+seek costs milliseconds against a
    /// transcription that costs seconds — the opposite of the pyannote
    /// crop-vs-full-load measurement, where hundreds of 2 s reads competed with a
    /// single load and lost.
    ///
    /// Pure and static so it runs off the main actor from a detached task.
    nonisolated static func loadWindow16k(from url: URL, start: Double, end: Double) -> [Float]? {
        guard end > start else { return nil }
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let format = file.processingFormat
        let rate = format.sampleRate
        guard rate > 0 else { return nil }

        let startFrame = AVAudioFramePosition(max(0, start) * rate)
        guard startFrame < file.length else { return nil }
        // Clamped to what the file actually holds rather than padded with
        // silence: the last window is short, and padding it would hand the model
        // digital silence — which MOSS answers as a chatbot and the mlx-audio
        // models answer with a canned caption.
        var remaining = Int(min(Double(file.length - startFrame),
                                (end - max(0, start)) * rate))
        guard remaining > 0 else { return nil }
        file.framePosition = startFrame

        // Already the format every ASR sidecar wants → copy the samples straight
        // out rather than building a converter that would only copy them.
        let passthrough = rate == 16_000 && format.channelCount == 1
                       && format.commonFormat == .pcmFormatFloat32
        let resampler = passthrough ? nil : AudioResampler(inputFormat: format)
        guard passthrough || resampler != nil else { return nil }

        // **`AVAudioFile.read` is allowed to return FEWER frames than asked for**
        // — it reads whole packets — so this loops until the window is complete.
        // Measured, not assumed: asking for 32 000 frames of a 16 kHz float WAV
        // comes back with 31 104. A single read would have silently shortened
        // EVERY window of every full pass by ~3 %, losing the last second of
        // audio at each boundary with nothing to show for it but slightly
        // shorter text. One stateful resampler across the reads, deliberately:
        // it is the same continuous signal.
        let chunkFrames = 16_384
        var samples: [Float] = []
        samples.reserveCapacity(Int(Double(remaining) * 16_000 / rate) + 16)
        while remaining > 0 {
            let want = AVAudioFrameCount(min(remaining, chunkFrames))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: want) else { break }
            do { try file.read(into: buffer, frameCount: want) } catch { break }
            let got = Int(buffer.frameLength)
            guard got > 0 else { break }
            remaining -= got
            if let resampler {
                samples.append(contentsOf: resampler.resample(buffer))
            } else if let channel = buffer.floatChannelData {
                samples.append(contentsOf: UnsafeBufferPointer(start: channel[0], count: got))
            }
        }
        return samples.isEmpty ? nil : samples
    }

    // MARK: - Log

    /// Append a line to `logs/chunked-stop.log`. Swift-owned and single-writer,
    /// like `dual-stream.log` and `overlap-repair-decisions.log` — the sidecar
    /// naming rule (`scripts/<name>/<name>-service.py` → `logs/<name>.log`) covers
    /// sidecar logs only, and this file is written by the app.
    ///
    /// Every window's outcome lands here, for the same reason every dropped
    /// segment does in `logs/chunked-asr.log`'s successors: a full pass REPLACES
    /// text, so if it replaces something with less, the log is the only place the
    /// original decision survives.
    func chunkedStopLog(_ message: String) {
        let dir = PythonRuntime.dataDir.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("chunked-stop.log")
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
}
