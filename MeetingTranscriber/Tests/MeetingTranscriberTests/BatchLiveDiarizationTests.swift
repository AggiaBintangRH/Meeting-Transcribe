import XCTest
@testable import MeetingTranscriber

/// The four whole-file engines run per interval when the stop pass is off
/// (owner, 2026-08-13: the pyannote "Run at stop" block, for every engine).
///
/// Until now spectral, NeMo, DiariZen and CAM++ had no live path: record with no
/// labels, one pass per stream at Stop. The toggle was hidden for them precisely
/// because switching it off would have left a meeting with no labels at all.
///
/// What makes the live path possible is not a change to any sidecar — a window is
/// just a short recording, sent as the ordinary whole-file job. It is that the
/// labels never carried continuity in the first place: `WeSpeakerService.identify`
/// does, and all four already went through it at Stop.
@MainActor
final class BatchLiveDiarizationTests: XCTestCase {

    // MARK: - The rule

    /// Live per-interval work happens exactly when the stop pass does not.
    func testLiveRunsPreciselyWhenTheStopPassDoesNot() {
        for finalPass in [true, false] {
            XCTAssertEqual(
                AudioRecorder.runsBatchLiveDiarization(isBatchEngine: true,
                                                       finalPass: finalPass),
                !finalPass)
        }
    }

    /// THE INVARIANT THAT MATTERS MOST: a batch session always has exactly one
    /// source of labels. Never two (duplicated work over the same audio), and
    /// never none — which is the failure that kept this toggle hidden.
    func testABatchSessionAlwaysHasExactlyOneLabelSource() {
        for finalPass in [true, false] {
            let live = AudioRecorder.runsBatchLiveDiarization(isBatchEngine: true,
                                                              finalPass: finalPass)
            let stop = AudioRecorder.runsBatchOfficePass(batchActive: true,
                                                         hasService: true,
                                                         hasRecording: true,
                                                         finalPass: finalPass)
            XCTAssertNotEqual(live, stop,
                              "finalPass=\(finalPass) gave live=\(live) stop=\(stop) — "
                              + "a session must have one label source, not none or both")
        }
    }

    /// pyannote is untouched: it has always had both, and this feature must not
    /// give it a second live path over the same audio.
    func testPyannoteNeverTakesTheBatchLivePath() {
        for finalPass in [true, false] {
            XCTAssertFalse(AudioRecorder.runsBatchLiveDiarization(isBatchEngine: false,
                                                                  finalPass: finalPass))
        }
    }

    // MARK: - The stop pass now honours the toggle

    /// The reversal itself. With the toggle off there is no whole-file pass — that
    /// is what makes the live path the labels rather than extra work beside them.
    func testTheStopPassIsSkippedWhenTheToggleIsOff() {
        XCTAssertFalse(AudioRecorder.runsBatchOfficePass(batchActive: true,
                                                         hasService: true,
                                                         hasRecording: true,
                                                         finalPass: false))
        XCTAssertTrue(AudioRecorder.runsBatchOfficePass(batchActive: true,
                                                        hasService: true,
                                                        hasRecording: true,
                                                        finalPass: true))
    }

    // MARK: - The tail moved to the stop-pass-OFF branch (owner, 2026-08-14)

    /// *"Tail hanya muncul dan dipakai pas Run At Stop = OFF … pas On mah diulang
    /// diarize dari awal sampai akhir."*
    ///
    /// Until then `continueOnStop` was read only inside `if willRunStopPass`,
    /// while its toggle was shown only when the stop pass was OFF — visible where
    /// inert, active where invisible. A `true` set in the branch where it did
    /// nothing then decided behaviour in the branch where nothing could change it,
    /// which cost a real session: the office pass became a tail, a tail is a
    /// `chunk` job carrying no `num_speakers`, and the SPK picker went dim.
    func testTheTailBelongsToTheStopPassOffBranch() {
        for continueOnStop in [true, false] {
            XCTAssertFalse(AudioRecorder.runsTailPassAtStop(finalPass: true,
                                                            continueOnStop: continueOnStop,
                                                            hasLivePath: true),
                           "a stop pass is a FULL pass; continueOnStop=\(continueOnStop) "
                           + "must not turn it into a tail")
        }
        XCTAssertTrue(AudioRecorder.runsTailPassAtStop(finalPass: false,
                                                       continueOnStop: true,
                                                       hasLivePath: true))
        XCTAssertFalse(AudioRecorder.runsTailPassAtStop(finalPass: false,
                                                        continueOnStop: false,
                                                        hasLivePath: true))
    }

    /// A tail CONTINUES FROM live labels, so with no live path there is nothing to
    /// continue from. Without this the mode would dispatch a `chunk` job into an
    /// engine that never produced a live label to attach it to.
    func testNoLivePathMeansNoTail() {
        for finalPass in [true, false] {
            for continueOnStop in [true, false] {
                XCTAssertFalse(AudioRecorder.runsTailPassAtStop(finalPass: finalPass,
                                                                continueOnStop: continueOnStop,
                                                                hasLivePath: false))
            }
        }
    }

    /// THE INVARIANT THE WHOLE CHANGE BUYS: the live path and the tail are the
    /// SAME branch, so a tail can only ever finish work the live passes started.
    /// Asserted as a sweep rather than case by case, because the failure it
    /// guards is the two rules drifting apart again — which is how the setting
    /// came to be read on one side and shown on the other in the first place.
    func testTheTailOnlyEverRunsWhereTheLivePathDoes() {
        for finalPass in [true, false] {
            for continueOnStop in [true, false] {
                let tail = AudioRecorder.runsTailPassAtStop(finalPass: finalPass,
                                                            continueOnStop: continueOnStop,
                                                            hasLivePath: true)
                let live = AudioRecorder.runsBatchLiveDiarization(isBatchEngine: true,
                                                                  finalPass: finalPass)
                if tail {
                    XCTAssertTrue(live,
                                  "a tail ran where the live path did not — it would "
                                  + "have nothing to continue from")
                }
            }
        }
    }

    // MARK: - The 2026-08-14 audit's two 🔴 defects

    /// A LIVE WINDOW'S FAILURE IS NOT THE STOP PASS'S.
    ///
    /// Every engine's `onError`/`onNoSpeech` went straight to
    /// `handleDiarizationFailure` — the stop-gate handler, which sets
    /// `diarizationError` (a red banner over the running transcript) and
    /// `finalDiarDone`. All of it fired DURING recording, for one window, and NeMo
    /// returns an error for 30 s of silence, which is an ordinary quiet stretch.
    ///
    /// The routing test is `batchLiveDiarizationActive`, and this asserts the
    /// PREMISE that makes it sufficient: in that mode no stop pass runs for these
    /// engines, so no error can legitimately need the stop-gate handler.
    func testNoStopPassRunsInLiveModeSoAnErrorCannotBelongToOne() {
        XCTAssertFalse(AudioRecorder.runsBatchOfficePass(batchActive: true,
                                                         hasService: true,
                                                         hasRecording: true,
                                                         finalPass: false),
                       "an office stop pass in live mode would give an engine error "
                       + "two possible owners, and the routing could not tell them apart")
        // The remote half: all four pass `supportsTail: false`, so the
        // `!finalPass` branch returns `.none` rather than a tail.
        XCTAssertEqual(AudioRecorder.remoteStopMode(finalPass: false,
                                                    continueOnStop: true,
                                                    remoteStreamActive: true,
                                                    hasDiarizationService: true,
                                                    hasRemoteRecording: true,
                                                    tailSamples: 480_000,
                                                    supportsTail: false),
                       .none,
                       "a remote pass in live mode would be the second owner")
    }

    /// Outstanding windows are released together with their temp WAVs, and the map
    /// is emptied. Before the audit nothing cleared it — not `configure*`, not
    /// Stop, not `startOver()` — so a failed window's entry and its ~1.9 MB WAV
    /// survived until the app quit.
    @MainActor
    func testReleasingWindowsEmptiesTheMapAndDeletesTheFiles() throws {
        let r = AudioRecorder()
        var paths: [String] = []
        for i in 0..<3 {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("batch-live-test-\(UUID().uuidString)-\(i).wav")
            try Data([0x52, 0x49, 0x46, 0x46]).write(to: url)
            paths.append(url.path)
            r.liveDiarWindowByPath[url.path] = Double(i) * 30
        }
        XCTAssertEqual(r.liveDiarWindowByPath.count, 3)
        r.releaseBatchLiveWindows()
        XCTAssertTrue(r.liveDiarWindowByPath.isEmpty, "the map still holds entries")
        for p in paths {
            XCTAssertFalse(FileManager.default.fileExists(atPath: p),
                           "a temp WAV survived its window: \(p)")
        }
    }

    /// THE CAUTION IS REACHABLE FROM A LIVE WINDOW. Measured on a 30 s slice of
    /// `Meeting5People.wav` with the shipped default (Auto): CAM++ returned 11
    /// speakers in 14 turns, spectral 12 — against 5 real people. This is the
    /// signature that describes: many speakers, most speaking once.
    ///
    /// It was reachable only from the stop passes, i.e. only from the mode the
    /// live path replaces, so the one configuration measured to fail was the one
    /// with nothing watching.
    func testTheMeasuredElevenSpeakerWindowRaisesTheCaution() {
        // 11 speakers over 14 turns: three speak twice, eight speak once.
        var turns: [SpeakerTurn] = []
        var t = 0.0
        for id in 1...11 {
            turns.append(SpeakerTurn(start: t, end: t + 2, id: id, name: "Speaker \(id)"))
            t += 2
        }
        for id in 1...3 {
            turns.append(SpeakerTurn(start: t, end: t + 2, id: id, name: "Speaker \(id)"))
            t += 2
        }
        XCTAssertEqual(turns.count, 14)
        XCTAssertNotNil(AudioRecorder.implausibleSpeakerCount(turns),
                        "the measured 11-speaker/14-turn window must be cautioned")
        // And a plausible window is left alone — without this half the rule could
        // be "always caution", which would train the user to ignore the banner.
        var real: [SpeakerTurn] = []
        for id in 1...3 {
            for k in 0..<5 {
                let s = Double(id * 10 + k)
                real.append(SpeakerTurn(start: s, end: s + 1, id: id,
                                        name: "Speaker \(id)"))
            }
        }
        XCTAssertNil(AudioRecorder.implausibleSpeakerCount(real))
    }

    // MARK: - A dead sidecar must not be silent (the second audit pass)

    /// 🟡 THE OVER-CORRECTION. Routing live-window errors away from the stop gate
    /// fixed a loud-but-wrong failure and created a QUIET one: `onError` also
    /// carries "The … sidecar is not running", which fails EVERY window, so a user
    /// recorded an hour, got no speaker labels and saw nothing at all.
    ///
    /// A RUN is what separates the two. One window failing is the ordinary
    /// non-fatal case the handler exists for; three in a row is the engine.
    @MainActor
    func testARunOfFailuresRaisesACautionButASingleOneDoesNot() {
        let r = AudioRecorder()
        let dead = "The CAM++ diarization sidecar is not running — restart the session."

        XCTAssertTrue(r.handleBatchLiveFailure(dead, stream: .office, noSpeech: false, liveMode: true))
        XCTAssertNil(r.diarizationCaution, "one failed window is not the engine")
        _ = r.handleBatchLiveFailure(dead, stream: .office, noSpeech: false, liveMode: true)
        XCTAssertNil(r.diarizationCaution, "two is still not a run")
        _ = r.handleBatchLiveFailure(dead, stream: .office, noSpeech: false, liveMode: true)
        XCTAssertNotNil(r.diarizationCaution,
                        "three consecutive failures is a dead engine and must be visible")
    }

    /// NO SPEECH IS NOT A FAILURE — it is the correct answer for a quiet stretch,
    /// and NeMo returns it for 30 s of silence. Counting it would put the banner up
    /// on an ordinary quiet meeting, and a banner that fires on the ordinary case
    /// is one nobody reads.
    @MainActor
    func testQuietWindowsNeverAccuseTheEngine() {
        let r = AudioRecorder()
        for _ in 0..<10 {
            _ = r.handleBatchLiveFailure("No speech found in the recording.",
                                         stream: .office, noSpeech: true, liveMode: true)
        }
        XCTAssertNil(r.diarizationCaution)
        XCTAssertEqual(r.batchLiveFailureRun, 0)
    }

    /// A meeting that recovers never accuses anyone: no-speech resets the run, so
    /// two failures either side of a quiet window are not three in a row.
    @MainActor
    func testTheRunResetsSoAnInterruptedSequenceIsNotARun() {
        let r = AudioRecorder()
        _ = r.handleBatchLiveFailure("boom", stream: .office, noSpeech: false, liveMode: true)
        _ = r.handleBatchLiveFailure("boom", stream: .office, noSpeech: false, liveMode: true)
        _ = r.handleBatchLiveFailure("quiet", stream: .office, noSpeech: true, liveMode: true)
        _ = r.handleBatchLiveFailure("boom", stream: .office, noSpeech: false, liveMode: true)
        _ = r.handleBatchLiveFailure("boom", stream: .office, noSpeech: false, liveMode: true)
        XCTAssertNil(r.diarizationCaution,
                     "the run was broken, so this is 2 + 2 and not 4")
    }

    /// It uses the CAUTION channel, never `diarizationError`. The error channel
    /// settles the stop gate and marks the stop panel failed — the mid-recording
    /// state mutation this whole handler exists to prevent.
    @MainActor
    func testTheRunNeverTouchesTheErrorChannelOrTheStopGate() {
        let r = AudioRecorder()
        for _ in 0..<5 {
            _ = r.handleBatchLiveFailure("boom", stream: .office, noSpeech: false, liveMode: true)
        }
        XCTAssertNotNil(r.diarizationCaution)
        XCTAssertNil(r.diarizationError,
                     "a live window may never raise the transcript's error banner")
        XCTAssertFalse(r.finalDiarDone,
                       "a live window may never settle the stop gate")
    }

    /// The pre-existing guards still bite regardless of the toggle: no engine, no
    /// recording, or the engine not selected all mean no pass.
    func testTheOlderGuardsStillHold() {
        for finalPass in [true, false] {
            XCTAssertFalse(AudioRecorder.runsBatchOfficePass(batchActive: false,
                                                             hasService: true,
                                                             hasRecording: true,
                                                             finalPass: finalPass))
            XCTAssertFalse(AudioRecorder.runsBatchOfficePass(batchActive: true,
                                                             hasService: false,
                                                             hasRecording: true,
                                                             finalPass: finalPass))
            XCTAssertFalse(AudioRecorder.runsBatchOfficePass(batchActive: true,
                                                             hasService: true,
                                                             hasRecording: false,
                                                             finalPass: finalPass))
        }
    }
}
