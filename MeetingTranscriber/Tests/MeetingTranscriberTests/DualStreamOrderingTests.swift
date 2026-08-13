import XCTest
@testable import MeetingTranscriber

/// Office and Remote rows must INTERLEAVE by when people actually spoke
/// (owner, 2026-08-13).
///
/// The report: *"tadi saya Office pertama terus remote terus office terus remote…
/// ini malah jadi row 1 2 office row 3 4 remote"*. Four turns alternating between
/// the room and the far end came out as both office rows, then both remote rows.
///
/// The turns were RIGHT — measured on the owner's own recording
/// (`meeting-2026-08-13T07-52-51Z`, CAM++ pinned to 2), and they alternate
/// exactly as described:
///
/// ```
/// OFFICE   2.88 – 17.31
/// REMOTE  17.83 – 27.45
/// OFFICE  27.71 – 42.59
/// REMOTE  43.07 – 63.10
/// ```
///
/// What was wrong is the TIME each row carried. `assignSentences` placed every
/// sentence by its CHARACTER POSITION across the whole chunk window and kept that
/// as the row's span — so a 63 s window with one office segment and one remote
/// segment spread BOTH streams' text evenly over the same 0–63 s, and
/// `mergeRowsByStartTime` then ordered them by numbers that describe how much
/// text each speaker produced rather than when they said it.
///
/// Single-stream never noticed: within one stream the estimate is monotonic, so
/// rows stay in text order regardless. It takes two streams sharing one clock for
/// the fake times to decide anything.
@MainActor
final class DualStreamOrderingTests: XCTestCase {

    /// The owner's real turns, verbatim from the measurement above.
    private var officeTurns: [SpeakerTurn] {
        [SpeakerTurn(start: 2.88, end: 17.31, id: 1, name: "Speaker 1"),
         SpeakerTurn(start: 27.71, end: 42.59, id: 2, name: "Speaker 2")]
    }
    private var remoteTurns: [SpeakerTurn] {
        [SpeakerTurn(start: 17.83, end: 27.45,
                     id: AudioRecorder.remoteIDBase + 1, name: "R1"),
         SpeakerTurn(start: 43.07, end: 63.10,
                     id: AudioRecorder.remoteIDBase + 2, name: "R2")]
    }

    private let window = 0.0...63.1

    /// THE REPORTED BUG. Rows must come out office, remote, office, remote.
    func testTheTwoStreamsInterleaveInTheOrderPeopleSpoke() {
        let office = AudioRecorder.assignSentences(
            "The room speaks first here. And the room speaks again later on.",
            window: window, ranges: AudioRecorder.speakerRanges(in: window, turns: officeTurns),
            segID: "office", regions: [])
        let remote = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "The far end answers. And the far end closes.",
                                         window: window)],
            turns: remoteTurns)

        let merged = AudioRecorder.mergeRowsByStartTime(office: office, remote: remote)

        XCTAssertEqual(merged.count, 4, "two speakers on each side, four rows")
        XCTAssertEqual(merged.map(\.isRemote), [false, true, false, true],
                       "office, remote, office, remote — the order they were spoken in")
    }

    /// Why the rows land where they do: each row's span must sit inside the TURN
    /// it was attributed to. That is the property the ordering rests on, so it is
    /// asserted directly rather than only through the merge.
    func testEachRowSitsInsideTheTurnItWasAttributedTo() {
        let rows = AudioRecorder.assignSentences(
            "The room speaks first here. And the room speaks again later on.",
            window: window, ranges: AudioRecorder.speakerRanges(in: window, turns: officeTurns),
            segID: "office", regions: [])

        for row in rows {
            guard let id = row.speakerID, let s = row.start, let e = row.end,
                  let turn = officeTurns.first(where: { $0.id == id }) else {
                return XCTFail("row without a speaker or a span")
            }
            XCTAssertGreaterThanOrEqual(s, turn.start - 0.001,
                                        "row starts before its speaker did")
            XCTAssertLessThanOrEqual(e, turn.end + 0.001,
                                     "row ends after its speaker stopped")
        }
    }

    /// A single-stream transcript keeps its text order. This is the regression
    /// guard for the fix itself: clamping row times to turns must not reorder or
    /// drop anything when there is only one stream.
    func testSingleStreamRowsStayInTextOrder() {
        let rows = AudioRecorder.assignSentences(
            "First sentence here. Second sentence here. Third sentence here.",
            window: window, ranges: AudioRecorder.speakerRanges(in: window, turns: officeTurns),
            segID: "office", regions: [])

        XCTAssertFalse(rows.isEmpty)
        let starts = rows.compactMap(\.start)
        XCTAssertEqual(starts, starts.sorted(), "rows must not go backwards in time")
        let text = rows.map(\.text).joined(separator: " ")
        for word in ["First", "Second", "Third"] {
            XCTAssertTrue(text.contains(word), "\(word) sentence was lost")
        }
    }

    /// No turns at all → the estimate is all there is, and the row keeps the whole
    /// window. Clamping must not turn "no information" into a fabricated span.
    func testWithNoTurnsTheRowStillSpansTheWindow() {
        let rows = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "nobody was diarized here.", window: window)],
            turns: [])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].start, window.lowerBound)
        XCTAssertEqual(rows[0].end, window.upperBound)
    }
}
