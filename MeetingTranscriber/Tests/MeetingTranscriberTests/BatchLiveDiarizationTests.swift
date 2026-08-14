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

    // MARK: - The warning

    /// It appears only where it is true: this mode, on Auto.
    func testTheAutoCountWarningAppearsOnlyWhereItApplies() {
        XCTAssertNotNil(AudioRecorder.liveCountWarning(isBatchEngine: true,
                                                       finalPass: false, numSpeakers: 0))
        XCTAssertNil(AudioRecorder.liveCountWarning(isBatchEngine: true,
                                                    finalPass: false, numSpeakers: 2),
                     "a pinned count is what makes the mode trustworthy")
        XCTAssertNil(AudioRecorder.liveCountWarning(isBatchEngine: true,
                                                    finalPass: true, numSpeakers: 0),
                     "the stop pass reads the whole file — short-window counting "
                     + "is not in play")
        XCTAssertNil(AudioRecorder.liveCountWarning(isBatchEngine: false,
                                                    finalPass: false, numSpeakers: 0),
                     "pyannote's per-window clustering is measured good on its own")
    }

    /// It has to be actionable, and it has to name the evidence — the numbers are
    /// what turn it from nagging into a reason.
    func testTheWarningNamesTheEvidenceAndTheFix() {
        let text = AudioRecorder.liveCountWarning(isBatchEngine: true, finalPass: false,
                                                  numSpeakers: 0)
        XCTAssertNotNil(text)
        XCTAssertTrue(text!.contains("9 to 15"), "the measured wrong counts")
        XCTAssertTrue(text!.lowercased().contains("speaker count"),
                      "and where to set the right one")
    }
}
