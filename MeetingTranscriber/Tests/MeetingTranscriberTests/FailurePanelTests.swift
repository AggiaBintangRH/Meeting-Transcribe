import XCTest
@testable import MeetingTranscriber

/// A failure the app has already detected must not leave the screen on its own
/// (owner, 2026-08-12: *"jika ada Error pas start recording sedang recording
/// stop recording tampilkan popupnya dengan tombol close jangan langsung
/// close panelnya"*).
///
/// The report behind it: a stop pass went red and the panel closed in the same
/// breath, so the only surviving copy of the reason was a log file inside
/// `~/Library/Application Support` — the owner could not even screenshot it.
///
/// Both directions are pinned throughout. A panel that never closes is as
/// broken as one that closes itself, and the second failure of a session must
/// still be able to raise the popup after the first was dismissed.
@MainActor
final class FailurePanelTests: XCTestCase {

    private func step(_ id: String, _ state: ModelLoader.ItemState) -> AudioRecorder.StopStep {
        AudioRecorder.StopStep(id: id, name: id, state: state)
    }

    // MARK: - The stop panel

    /// The defect itself: processing is over, a leg is red, and the panel stays.
    func testAFailedStopKeepsThePanelUntilItIsClosed() {
        let r = AudioRecorder()
        r.state = .idle
        r.stopSteps = [step("chunk", .done), step("diarize", .failed("silence"))]

        XCTAssertTrue(r.stopFailed)
        XCTAssertTrue(r.showsStopOverlay,
                      "a stop that ended in red must hold the panel — this is the "
                      + "whole defect: the reason existed for one frame and then "
                      + "only in a log file")

        r.dismissStopFailure()
        XCTAssertFalse(r.showsStopOverlay,
                       "and Close must really close it, or the app is unusable "
                       + "after any failed pass")
    }

    /// The other direction. A clean stop must behave exactly as it did before
    /// this change — the panel is tied to `.processing` and nothing else.
    func testACleanStopStillClosesByItself() {
        let r = AudioRecorder()
        r.stopSteps = [step("chunk", .done), step("diarize", .done)]

        r.state = .processing
        XCTAssertTrue(r.showsStopOverlay)

        r.state = .idle
        XCTAssertFalse(r.stopFailed)
        XCTAssertFalse(r.showsStopOverlay,
                       "nothing failed, so there is nothing to read and nothing to "
                       + "close — holding the panel here would make every normal "
                       + "meeting end with an extra click")
    }

    /// A leg that fails while the others are still running must NOT let the user
    /// close a panel that is still working. `ProcessingOverlayView` reads the
    /// same pair of facts to decide whether to draw the Close button.
    func testAFailureDuringProcessingDoesNotEndTheWait() {
        let r = AudioRecorder()
        r.state = .processing
        r.stopSteps = [step("chunk", .loading), step("diarize", .failed("boom"))]

        XCTAssertTrue(r.stopFailed)
        XCTAssertTrue(r.showsStopOverlay)

        // Dismissing is meaningless here and must not take the panel away from
        // work that is still running.
        r.dismissStopFailure()
        XCTAssertTrue(r.showsStopOverlay,
                      "`.processing` alone holds the panel, so an acknowledgement "
                      + "that arrives early cannot cut the wait short")
    }

    /// The acknowledgement travels with the rows it describes. If it survived
    /// into the next meeting, that meeting's red rows would be built into a
    /// panel that had already been told not to show.
    func testStartOverForgetsThatAFailureWasAcknowledged() {
        let r = AudioRecorder()
        r.state = .idle
        r.stopSteps = [step("diarize", .failed("silence"))]
        r.dismissStopFailure()
        r.markMeetingFinished()

        r.startOver()

        XCTAssertTrue(r.stopSteps.isEmpty)
        XCTAssertFalse(r.stopFailureAcknowledged,
                       "cleared with the rows, never after them — a stale `true` "
                       + "would silently swallow the NEXT meeting's failure")
    }

    /// The escape hatch must really let go. A leg that had already gone red
    /// would otherwise hold the panel open the instant `.processing` ended —
    /// the hatch visibly failing to do the one thing it exists for.
    func testContinueInBackgroundReleasesAPanelThatHadAlreadyGoneRed() {
        let r = AudioRecorder()
        r.state = .processing
        r.stopSteps = [step("chunk", .loading), step("diarize", .failed("boom"))]

        r.continueInBackground()

        XCTAssertEqual(r.state, .idle)
        XCTAssertFalse(r.showsStopOverlay,
                       "pressing it IS reading the panel and choosing to stop "
                       + "looking; leaving the red rows up would make the hatch a "
                       + "button that does nothing")
    }

    /// …and the watchdog is the opposite case. Nobody pressed anything, so its
    /// timed-out rows must stay on screen to be read.
    func testATimedOutStopStillHoldsThePanel() {
        let r = AudioRecorder()
        r.state = .idle          // the watchdog has already left `.processing`
        r.stopSteps = [step("diarize", .failed("timed out"))]

        XCTAssertTrue(r.showsStopOverlay,
                      "a timeout is a result the user has to be told about — the "
                      + "backstop firing silently is indistinguishable from a "
                      + "meeting that simply finished")
    }

    // MARK: - The recording-error popup

    /// Start / during-recording errors reach a popup, and Close leaves the
    /// message behind for the record card to keep showing.
    func testARecordingErrorRaisesThePopupAndCloseLeavesTheTrace() {
        let r = AudioRecorder()
        XCTAssertFalse(r.showsErrorPopup, "a fresh recorder has nothing to report")

        r.errorMessage = "Could not open ATND1061. Try reconnecting it."
        XCTAssertTrue(r.showsErrorPopup)

        r.dismissError()
        XCTAssertFalse(r.showsErrorPopup)
        XCTAssertEqual(r.errorMessage, "Could not open ATND1061. Try reconnecting it.",
                       "Close dismisses the popup, not the fact — the record card "
                       + "keeps the line so the reason is still on screen")
    }

    /// The reason the marker stores the MESSAGE rather than a Bool: a second,
    /// different failure must raise the popup again with no help from the nine
    /// sites that assign `errorMessage`.
    func testADifferentErrorReopensThePopupOnItsOwn() {
        let r = AudioRecorder()
        r.errorMessage = "Microphone access denied."
        r.dismissError()
        XCTAssertFalse(r.showsErrorPopup)

        r.errorMessage = "Audio engine failed: device removed"
        XCTAssertTrue(r.showsErrorPopup,
                      "a new error is new evidence; nothing at the assignment site "
                      + "should have to remember to reset a flag for it to be seen")
    }

    /// Hitting the same wall twice is the commonest case — a denied microphone
    /// is still denied on the second press — so the reset that `start()` does
    /// has to cover the marker as well as the message.
    func testPressingStartAgainCanShowTheSameErrorAgain() {
        let r = AudioRecorder()
        r.errorMessage = "Microphone access denied."
        r.dismissError()

        // What `start()` does on entry, and the only thing it needs to do here.
        r.errorMessage = nil
        r.dismissedErrorMessage = nil
        r.errorMessage = "Microphone access denied."

        XCTAssertTrue(r.showsErrorPopup,
                      "identical text, genuinely a second failure — suppressing it "
                      + "would leave a Start press with no visible outcome at all")
    }

    /// `.preparing` belongs to `LoadingOverlayView`, which already keeps startup
    /// refusals on screen through `ModelLoader.failureMessage`. Two cards over
    /// one failure would stack, and the top one would hide the step list that
    /// says which model refused.
    func testThePopupStandsAsideWhileTheLoadingOverlayOwnsTheScreen() {
        let r = AudioRecorder()
        r.errorMessage = "Voxtral cannot run with a Remote channel."

        r.state = .preparing
        XCTAssertFalse(r.showsErrorPopup)

        r.state = .idle
        XCTAssertTrue(r.showsErrorPopup,
                      "and it must appear once the loading overlay is gone, or a "
                      + "refusal raised during preparation is never shown at all")
    }
}
