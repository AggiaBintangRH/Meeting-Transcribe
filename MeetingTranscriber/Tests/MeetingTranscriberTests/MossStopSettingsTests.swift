import XCTest
@testable import MeetingTranscriber

/// `moss.finalPass` / `moss.continueOnStop` — what the MOSS DIARIZATION engine
/// does at Stop, mirroring the chunked pair.
///
/// The rule these pin is the same one `ChunkedStopSettingsTests` pins: a new
/// setting must default to EXACTLY today's behaviour, and the dangerous branch
/// must be asserted in both directions rather than assumed.
final class MossStopSettingsTests: XCTestCase {

    private func mode(finalPass: Bool = true, tail: Bool = true,
                      service: Bool = true, recording: Bool = true)
        -> AudioRecorder.MossStopMode {
        AudioRecorder.mossStopMode(finalPass: finalPass, continueOnStop: tail,
                                   hasDiarService: service, hasRecording: recording)
    }

    // MARK: - Defaults

    /// ⚠ RE-AIMED 2026-08-14, and this one is a real BEHAVIOUR change to MOSS's
    /// shipped default, recorded rather than smoothed over. It used to assert that
    /// both keys absent (`?? true` / `?? true`) gave a TAIL — "what `stop()` has
    /// always done", and the whole safety argument for that change.
    ///
    /// The owner's new rule is that `continueOnStop` belongs to the stop-pass-OFF
    /// branch only (*"pas On mah diulang diarize dari awal sampai akhir"*), and
    /// one control governs both pairs — a MOSS exception would make the same
    /// toggle mean two things. So a default MOSS session now ends in a FULL
    /// re-diarization.
    func testAbsentKeysNowGiveTheFullPass() {
        XCTAssertEqual(mode(), .full)
        let plan = AudioRecorder.mossStopPlan(.full)
        XCTAssertTrue(plan.runsFullPass)
        XCTAssertFalse(plan.flushesTail, "the full pass covers the tail itself")
        XCTAssertFalse(plan.settlesImmediately,
                       "the pass settles the leg asynchronously, as the tail did")
    }

    /// The tail is still reachable — it moved, it was not removed. With the stop
    /// pass off it is what finishes a live-labelled MOSS meeting.
    func testTheTailIsNowTheStopPassOffMode() {
        XCTAssertEqual(mode(finalPass: false, tail: true), .tail)
        let plan = AudioRecorder.mossStopPlan(.tail)
        XCTAssertTrue(plan.flushesTail)
        XCTAssertFalse(plan.runsFullPass)
        XCTAssertFalse(plan.settlesImmediately,
                       "the tail flush settles the leg asynchronously, as it always has")
    }

    /// The mirrored pair has the OPPOSITE default to pyannote's on purpose:
    /// pyannote's `continueOnStop` is false (full re-diarization), MOSS's is true
    /// (tail), because true is what MOSS already did.
    func testTailDefaultIsTrueUnlikePyannotes() {
        let d = UserDefaults.standard
        d.removeObject(forKey: "moss.continueOnStop")
        XCTAssertTrue(d.object(forKey: "moss.continueOnStop") as? Bool ?? true)
        XCTAssertFalse(d.object(forKey: "diarization.continueOnStop") as? Bool ?? false,
                       "pyannote's default is the opposite — the UI copy must say so")
    }

    // MARK: - The branches

    func testTailOffAsksForTheFullPass() {
        XCTAssertEqual(mode(tail: false), .full)
        let plan = AudioRecorder.mossStopPlan(.full)
        XCTAssertTrue(plan.runsFullPass)
        XCTAssertFalse(plan.flushesTail, "the full pass covers the tail itself")
        XCTAssertFalse(plan.settlesImmediately, "the pass settles the leg when it finishes")
    }

    /// With the stop pass off AND no tail asked for, nothing runs — the `.none`
    /// leg still exists, it just needs both halves now that the tail lives here.
    func testFinalPassOffRunsNothingAndSettlesHere() {
        XCTAssertEqual(mode(finalPass: false, tail: false), .none)
        let plan = AudioRecorder.mossStopPlan(.none)
        XCTAssertFalse(plan.flushesTail)
        XCTAssertFalse(plan.runsFullPass)
        XCTAssertTrue(plan.settlesImmediately,
                      "nothing asynchronous will settle mossLastChunkDone, so stop() must")
    }

    /// No second MOSS process ⇒ these settings govern nothing. That covers BOTH
    /// the pyannote engine and MOSS+MOSS, where the chunked pair owns the tail.
    func testNoSecondProcessMeansTheseSettingsDoNothing() {
        for finalPass in [true, false] {
            for tail in [true, false] {
                XCTAssertEqual(mode(finalPass: finalPass, tail: tail, service: false), .none,
                               "finalPass=\(finalPass) tail=\(tail)")
            }
        }
    }

    /// A missing recording must cost the tail, not the whole pass.
    func testFullPassWithoutARecordingDegradesToTheTail() {
        XCTAssertEqual(mode(tail: false, recording: false), .tail)
    }

    /// Every branch either settles the leg itself or hands it to something that
    /// will. A branch doing neither would hang the stop overlay forever.
    func testEveryBranchEitherSettlesOrHasAJobToSettleIt() {
        for m in [AudioRecorder.MossStopMode.none, .tail, .full] {
            let plan = AudioRecorder.mossStopPlan(m)
            XCTAssertTrue(plan.settlesImmediately || plan.flushesTail || plan.runsFullPass,
                          "\(m) would leave mossLastChunkDone unsettled")
        }
    }

    // MARK: - The window length IS the feature

    /// MOSS is byte-deterministic, so re-running the same windows would return
    /// the same labels. The full pass is only worth anything because it uses a
    /// LONGER window — if this ever equals the ASR cadence, the feature is dead.
    func testFullPassWindowIsLongerThanTheLiveCadence() {
        XCTAssertEqual(AudioRecorder.mossFullPassWindowSec, 120)
        XCTAssertGreaterThan(AudioRecorder.mossFullPassWindowSec, 30,
                             "a same-length re-run is a control that cannot change its answer")
    }

    /// 120 s is the measured ceiling: ~13.2 context tokens per second of audio
    /// puts it near 1 638 of 2 048, while 360 s was measured to hit the cap and
    /// drop the truncated tail silently.
    func testFullPassWindowStaysUnderTheTokenCap() {
        let tokensPerSecond = 13.2
        let used = AudioRecorder.mossFullPassWindowSec * tokensPerSecond
        XCTAssertLessThan(used, 2048 * 0.9,
                          "a window this long risks silent truncation — see the MOSS token gate")
    }

    /// The window sequence covers the whole recording, in order, with no gaps.
    func testFullPassWindowsCoverTheRecording() {
        let windows = AudioRecorder.fullPassWindows(
            recordingLength: 300, intervalSec: AudioRecorder.mossFullPassWindowSec)
        XCTAssertEqual(windows.count, 3)                 // 120 + 120 + 60
        XCTAssertEqual(windows.first?.lowerBound, 0)
        XCTAssertEqual(windows.last?.upperBound, 300)
        for (a, b) in zip(windows, windows.dropFirst()) {
            XCTAssertEqual(a.upperBound, b.lowerBound, accuracy: 1e-9, "gap between windows")
        }
    }
}
