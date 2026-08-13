import XCTest
@testable import MeetingTranscriber

/// `logs/row-order.log` records row MOVEMENT, not row state (owner, 2026-08-13:
/// *"speaker row office terus pindah pindah ke atas kenapa … kamu bikin lognya,
/// saya test"*).
///
/// Nothing in this app recorded the order the rows ended up in, so a row that
/// jumped left no trace whatsoever — there was nothing to read after a session.
///
/// The change-detection is the load-bearing part and the reason for this file.
/// `rebuildDisplayRows` runs on every realtime partial, several times a second.
/// A log with one entry per rebuild would be megabytes of identical lines with
/// the one interesting moment buried inside it — technically complete and
/// practically useless, which is the failure mode a diagnostic log has.
@MainActor
final class RowOrderLogTests: XCTestCase {

    private func office(_ start: Double, _ text: String) -> AudioRecorder.TranscriptSegment {
        AudioRecorder.TranscriptSegment(text: text, confirmed: true,
                                        window: start...(start + 5))
    }

    /// Rebuilding with the SAME rows must not re-write the signature — that is
    /// what keeps the file readable.
    func testAnUnchangedOrderIsNotLoggedTwice() {
        let r = AudioRecorder()
        r.segments = [office(0, "hello"), office(10, "world")]

        r.rebuildDisplayRows()
        let first = r.lastLoggedRowOrder
        XCTAssertNotNil(first, "the first order is always worth recording")

        r.rebuildDisplayRows()
        XCTAssertEqual(r.lastLoggedRowOrder, first, "nothing moved — nothing to say")
    }

    /// A row arriving between two others IS a change, and must be recorded.
    func testANewRowChangesTheSignature() {
        let r = AudioRecorder()
        r.segments = [office(0, "hello"), office(10, "world")]
        r.rebuildDisplayRows()
        let before = r.lastLoggedRowOrder

        r.segments.insert(office(5, "in between"), at: 1)
        r.rebuildDisplayRows()
        XCTAssertNotEqual(r.lastLoggedRowOrder, before)
    }

    /// THE CASE THIS EXISTS FOR: the same rows in a different ORDER. A signature
    /// built from a Set, or one that ignored position, would call this unchanged
    /// and the log would stay silent about exactly the event being investigated.
    func testTheSameRowsInADifferentOrderIsAChange() {
        let r = AudioRecorder()
        r.remoteSegments = [AudioRecorder.RemoteSegment(text: "from the call",
                                                        window: 2.0...4.0)]
        r.segments = [office(10, "from the room")]
        r.rebuildDisplayRows()
        let remoteFirst = r.lastLoggedRowOrder

        // Same two rows, now with the room speaking first.
        r.segments = [office(0, "from the room")]
        r.rebuildDisplayRows()
        XCTAssertNotEqual(r.lastLoggedRowOrder, remoteFirst,
                          "the rows swapped places — that is the whole subject")
    }

    /// An emptied transcript leaves no stale signature behind, so the first order
    /// of the next meeting is recorded rather than mistaken for "unchanged".
    ///
    /// The session-level reset (`clearVisibleMeetingState`, reached from Start Over
    /// and from `beginCapture`) is not exercised here: both it and `meetingFinished`
    /// are private, and contorting the app's access control to reach a one-line
    /// reset beside a dozen sibling resets would be the test shaping the code.
    func testAnEmptiedTranscriptLeavesNoStaleSignature() {
        let r = AudioRecorder()
        r.segments = [office(0, "hello")]
        r.rebuildDisplayRows()
        XCTAssertEqual(r.lastLoggedRowOrder, "O-@0.0")

        r.segments = []
        r.rebuildDisplayRows()
        XCTAssertEqual(r.lastLoggedRowOrder, "",
                       "no rows means no order — and the next one must not be "
                       + "compared against the old meeting's")
    }
}
