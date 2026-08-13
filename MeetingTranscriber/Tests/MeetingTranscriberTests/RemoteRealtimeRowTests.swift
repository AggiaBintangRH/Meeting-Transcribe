import XCTest
@testable import MeetingTranscriber

/// A remote realtime utterance becomes its own ROW, like an office one
/// (owner, 2026-08-13).
///
/// The report: *"Row dari Office ketika dibawah Remote itu malah pindah keatas …
/// jadinya ketika ada remote yang ngomong lagi jadinya row remote panjang banget
/// ketika realtime"*.
///
/// **One asymmetry produced both symptoms.** An office realtime FINAL was appended
/// as an unconfirmed segment — a row, placed by time. A remote realtime final was
/// treated exactly like a partial and only ever grew the caption card. So:
///
/// * every remote utterance piled into ONE card that kept growing (at the owner's
///   120 s chunk interval, for two minutes at a time) — *"panjang banget"*;
/// * the caption cards render BELOW the row list, so each new office row appeared
///   above the far end's text — *"office pindah keatas"*.
///
/// Neither was an ordering bug, which is why three ordering fixes did not touch
/// it: the remote text was never in the list being ordered.
@MainActor
final class RemoteRealtimeRowTests: XCTestCase {

    /// A provisional remote row is one row, unsplit, carrying its own span — which
    /// is what lets it be placed among the office rows at all.
    func testAnUnconfirmedRemoteSegmentIsOneProvisionalRow() {
        let rows = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "hello from the call. and again.",
                                         window: 10.0...14.0, confirmed: false)],
            turns: [SpeakerTurn(start: 10, end: 12,
                                id: AudioRecorder.remoteIDBase + 1, name: "R1"),
                    SpeakerTurn(start: 12, end: 14,
                                id: AudioRecorder.remoteIDBase + 2, name: "R2")])

        XCTAssertEqual(rows.count, 1, "provisional text is not split by speaker")
        XCTAssertFalse(rows[0].confirmed)
        XCTAssertTrue(rows[0].isRemote)
        XCTAssertEqual(rows[0].start, 10.0)
        XCTAssertEqual(rows[0].end, 14.0)
    }

    /// THE REPORTED CASE: office, remote, office live rows interleave by time
    /// instead of the remote text sitting under all of them.
    func testLiveRowsFromBothStreamsInterleaveByTime() {
        let r = AudioRecorder()
        r.segments = [
            AudioRecorder.TranscriptSegment(text: "the room first.", confirmed: false,
                                            window: 0.0...5.0),
            AudioRecorder.TranscriptSegment(text: "the room again.", confirmed: false,
                                            window: 20.0...25.0)]
        r.remoteSegments = [AudioRecorder.RemoteSegment(text: "the call in between.",
                                                        window: 10.0...15.0,
                                                        confirmed: false)]
        r.rebuildDisplayRows()

        XCTAssertEqual(r.displayRows.map(\.isRemote), [false, true, false])
        XCTAssertFalse(r.displayRows.contains { $0.confirmed })
    }

    /// Two separate remote utterances are two rows — not one card that grew. This
    /// is the *"panjang banget"* half, and the office row between them is what
    /// proves they are genuinely separate.
    func testTwoRemoteUtterancesAreTwoRowsNotOneLongOne() {
        let r = AudioRecorder()
        r.remoteSegments = [
            AudioRecorder.RemoteSegment(text: "first thing.", window: 2.0...6.0,
                                        confirmed: false),
            AudioRecorder.RemoteSegment(text: "second thing.", window: 30.0...34.0,
                                        confirmed: false)]
        r.segments = [AudioRecorder.TranscriptSegment(text: "the room.", confirmed: false,
                                                      window: 15.0...18.0)]
        r.rebuildDisplayRows()

        XCTAssertEqual(r.displayRows.map(\.isRemote), [true, false, true])
        XCTAssertEqual(r.displayRows[0].text, "first thing.")
        XCTAssertEqual(r.displayRows[2].text, "second thing.")
    }

    /// The confirmed chunk REPLACES the provisional rows it covers. Without this
    /// every remote utterance would appear twice — once realtime, once accurate.
    func testConfirmedRemoteTextReplacesTheProvisionalRows() {
        let r = AudioRecorder()
        r.remoteSegments = [
            AudioRecorder.RemoteSegment(text: "rough live text.", window: 2.0...6.0,
                                        confirmed: false),
            AudioRecorder.RemoteSegment(text: "more rough text.", window: 8.0...12.0,
                                        confirmed: false)]

        // What `flushRemoteChunk` does when the accurate text lands.
        r.remoteSegments.removeAll { !$0.confirmed }
        r.remoteSegments.append(AudioRecorder.RemoteSegment(text: "the accurate version.",
                                                            window: 0.0...30.0))
        r.rebuildDisplayRows()

        XCTAssertEqual(r.displayRows.count, 1)
        XCTAssertEqual(r.displayRows[0].text, "the accurate version.")
        XCTAssertTrue(r.displayRows[0].confirmed)
    }

    /// A confirmed remote segment is unaffected — it still splits by turns. The
    /// unconfirmed branch must not swallow the path everything else uses.
    func testConfirmedRemoteRowsStillSplitByTurns() {
        let rows = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "first part. second part.",
                                         window: 10.0...14.0)],
            turns: [SpeakerTurn(start: 10, end: 12,
                                id: AudioRecorder.remoteIDBase + 1, name: "R1"),
                    SpeakerTurn(start: 12, end: 14,
                                id: AudioRecorder.remoteIDBase + 2, name: "R2")])
        XCTAssertEqual(rows.count, 2)
        XCTAssertTrue(rows.allSatisfy(\.confirmed))
    }
}
