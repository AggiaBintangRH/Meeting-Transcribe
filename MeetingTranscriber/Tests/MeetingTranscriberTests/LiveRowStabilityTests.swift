import XCTest
@testable import MeetingTranscriber

/// A live row's place is decided when it first appears, and then it stays there
/// (owner, 2026-08-13).
///
/// Their words, and they are a specification: *"ketika berbeda speaker maka itu
/// sudah beda row … remote speaker ini udah row nya, itu gak pindah pindah kemana
/// mana pas realtime … terus datang office maka office itu row ke dua, gak bisa
/// jadi row pertama … terus datang remote lagi maka ini row ke 3"*. Only the TEXT
/// may change later, when the chunked model re-transcribes.
///
/// **Everything before this ordered live rows by TIME**, and the times during
/// recording are estimates that keep being corrected — a whole-chunk span, then a
/// character-position guess, then late word times. Every correction re-sorted the
/// list under the reader. Three separate fixes made those numbers better and none
/// made the list stop moving, because the order should never have depended on the
/// numbers.
///
/// Arrival order is right for LIVE rows specifically: a realtime final fires when
/// that stream's speaker stops, so arrival is speaking order. It is NOT true of
/// the confirmed chunk replies — both streams share one queue with office always
/// enqueued first — which is why confirmed rows keep their time ordering.
@MainActor
final class LiveRowStabilityTests: XCTestCase {

    private func officeLive(_ seq: Int, _ text: String,
                            _ w: ClosedRange<Double>) -> AudioRecorder.TranscriptSegment {
        AudioRecorder.TranscriptSegment(text: text, confirmed: false, window: w, seq: seq)
    }
    private func remoteLive(_ seq: Int, _ text: String,
                            _ w: ClosedRange<Double>) -> AudioRecorder.RemoteSegment {
        AudioRecorder.RemoteSegment(text: text, window: w, seq: seq, confirmed: false)
    }

    /// THE SPECIFICATION, verbatim: remote, then office, then remote.
    func testRemoteThenOfficeThenRemoteKeepsThatOrder() {
        let r = AudioRecorder()
        r.remoteSegments = [remoteLive(1, "the call first.", 0.0...5.0)]
        r.rebuildDisplayRows()
        XCTAssertEqual(r.displayRows.map(\.isRemote), [true])

        r.segments = [officeLive(2, "then the room.", 6.0...9.0)]
        r.rebuildDisplayRows()
        XCTAssertEqual(r.displayRows.map(\.isRemote), [true, false],
                       "the room arrived second and is row two")

        r.remoteSegments.append(remoteLive(3, "the call again.", 12.0...15.0))
        r.rebuildDisplayRows()
        XCTAssertEqual(r.displayRows.map(\.isRemote), [true, false, true])
    }

    /// THE ONE THAT WOULD HAVE CAUGHT EVERY EARLIER ATTEMPT. A row's index must
    /// not change when a LATER row arrives — whatever times the new row carries.
    ///
    /// The times here are deliberately hostile: the office row that arrived second
    /// claims an EARLIER span than the remote row that arrived first. Under time
    /// ordering it would jump to the top, which is exactly what the owner kept
    /// seeing. Under arrival ordering it cannot.
    func testAnEarlierTimestampCannotPromoteALaterRow() {
        let r = AudioRecorder()
        r.remoteSegments = [remoteLive(1, "the call first.", 30.0...35.0)]
        r.rebuildDisplayRows()
        let firstRowText = r.displayRows[0].text

        // Arrived second, but claims to have happened first.
        r.segments = [officeLive(2, "the room, timestamped earlier.", 0.0...5.0)]
        r.rebuildDisplayRows()

        XCTAssertEqual(r.displayRows.count, 2)
        XCTAssertEqual(r.displayRows[0].text, firstRowText,
                       "row one was already decided — a later arrival cannot take it")
        XCTAssertFalse(r.displayRows[1].isRemote)
    }

    /// Text may change under a row without moving it — the owner allows exactly
    /// this: *"paling cuma kata katanya saja … karena di transcribe ulang"*.
    func testTextMayChangeWithoutTheRowMoving() {
        let r = AudioRecorder()
        r.remoteSegments = [remoteLive(1, "rough text.", 0.0...5.0)]
        r.segments = [officeLive(2, "the room.", 6.0...9.0)]
        r.rebuildDisplayRows()
        XCTAssertEqual(r.displayRows.map(\.isRemote), [true, false])

        r.remoteSegments[0].text = "the corrected text."
        r.rebuildDisplayRows()

        XCTAssertEqual(r.displayRows.map(\.isRemote), [true, false], "same places")
        XCTAssertEqual(r.displayRows[0].text, "the corrected text.")
    }

    /// Rebuilding repeatedly — which happens several times a second on realtime
    /// partials — must not disturb anything.
    func testRepeatedRebuildsAreStable() {
        let r = AudioRecorder()
        r.remoteSegments = [remoteLive(1, "a.", 0.0...5.0), remoteLive(3, "c.", 20.0...25.0)]
        r.segments = [officeLive(2, "b.", 10.0...15.0)]
        r.rebuildDisplayRows()
        let order = r.displayRows.map(\.text)

        for _ in 0..<5 {
            r.rebuildDisplayRows()
            XCTAssertEqual(r.displayRows.map(\.text), order)
        }
    }

    /// CONFIRMED rows keep TIME ordering, and live rows follow them. Both halves
    /// are asserted together because the danger is applying one rule to both: the
    /// finished transcript must read in time order, and it arrives as one chunk
    /// whose rows have no meaningful arrival order between them.
    func testConfirmedRowsStayInTimeOrderAndLiveRowsFollow() {
        let r = AudioRecorder()
        r.segments = [AudioRecorder.TranscriptSegment(text: "the room, confirmed.",
                                                      confirmed: true, window: 10.0...15.0),
                      officeLive(9, "the room, still live.", 40.0...45.0)]
        r.remoteSegments = [AudioRecorder.RemoteSegment(text: "the call, confirmed.",
                                                        window: 0.0...5.0)]
        r.rebuildDisplayRows()

        XCTAssertEqual(r.displayRows.map(\.confirmed), [true, true, false],
                       "live rows go last — the chunked pass replaces them anyway")
        XCTAssertTrue(r.displayRows[0].isRemote,
                      "confirmed rows are ordered by time, and the call was first")
    }
}
