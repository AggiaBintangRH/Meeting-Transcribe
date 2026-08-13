import XCTest
@testable import MeetingTranscriber

/// Remote-vs-remote overlap is MARKED (owner, 2026-08-13: *"saya ingin agar ini
/// juga mendeteksi overlap untuk remote, jadi remote-remote"*).
///
/// Until this change `remoteRows` hardcoded `overlapped: false`, so two remote
/// participants talking over each other produced text with no caution on it,
/// while the same situation in the room was marked. One transcript, two standards.
///
/// ⚠ **Marking only — repair is deliberately NOT extended.** Measured the same
/// day on 16 real overlap windows: separation rescued **0**, lifting one voice and
/// leaving artefacts. A remote repair path would be a pipeline built to reject.
@MainActor
final class RemoteOverlapTests: XCTestCase {

    private func turn(_ id: Int, _ s: Double, _ e: Double) -> SpeakerTurn {
        SpeakerTurn(start: s, end: e, id: id, name: "R\(id - AudioRecorder.remoteIDBase)")
    }

    // MARK: - The regions themselves

    /// Two remote speakers overlapping past the 0.4 s bar is a region.
    func testTwoRemoteSpeakersOverlappingIsARegion() {
        let r = AudioRecorder()
        r.remoteLiveTurns = [turn(AudioRecorder.remoteIDBase + 1, 0, 5),
                             turn(AudioRecorder.remoteIDBase + 2, 4, 9)]

        let regions = r.remoteOverlapRegions()
        XCTAssertEqual(regions.count, 1)
        XCTAssertEqual(regions[0].start, 4, accuracy: 0.001)
        XCTAssertEqual(regions[0].end, 5, accuracy: 0.001)
    }

    /// Under the bar is not overlap — the same 0.4 s the office side uses, and
    /// now from the same function, so the two cannot drift apart.
    func testABriefBrushIsNotOverlap() {
        let r = AudioRecorder()
        r.remoteLiveTurns = [turn(AudioRecorder.remoteIDBase + 1, 0, 5),
                             turn(AudioRecorder.remoteIDBase + 2, 4.8, 9)]
        XCTAssertTrue(r.remoteOverlapRegions().isEmpty)
    }

    /// THE ONE THAT MATTERS MOST: the office stream must be unaffected.
    ///
    /// Both streams share one clock, so a remote region at t=4 would land on an
    /// office row at t=4 and mark a room conversation as overlapping when nobody
    /// in the room spoke twice. The two collections are separate precisely so
    /// that is not a rule anyone has to remember.
    func testRemoteOverlapDoesNotLeakIntoTheOfficeRegions() {
        let r = AudioRecorder()
        r.remoteLiveTurns = [turn(AudioRecorder.remoteIDBase + 1, 0, 5),
                             turn(AudioRecorder.remoteIDBase + 2, 4, 9)]
        r.liveTurns = [SpeakerTurn(start: 0, end: 5, id: 1, name: "Speaker 1")]

        XCTAssertEqual(r.remoteOverlapRegions().count, 1)
        XCTAssertTrue(r.overlapRegions().isEmpty,
                      "the room had one speaker — a remote overlap must not mark it")
    }

    /// …and the reverse, which is the half a one-directional fix would miss.
    func testOfficeOverlapDoesNotLeakIntoTheRemoteRegions() {
        let r = AudioRecorder()
        r.liveTurns = [SpeakerTurn(start: 0, end: 5, id: 1, name: "Speaker 1"),
                       SpeakerTurn(start: 4, end: 9, id: 2, name: "Speaker 2")]
        r.remoteLiveTurns = [turn(AudioRecorder.remoteIDBase + 1, 0, 5)]

        XCTAssertEqual(r.overlapRegions().count, 1)
        XCTAssertTrue(r.remoteOverlapRegions().isEmpty)
    }

    // MARK: - The mark reaching the row

    /// A remote row inside a region is flagged; one outside is not.
    func testRemoteRowsCarryTheMark() {
        let seg = AudioRecorder.RemoteSegment(text: "both at once.", window: 4.0...5.0)
        let clean = AudioRecorder.RemoteSegment(text: "just me.", window: 20.0...21.0)

        let marked = AudioRecorder.remoteRows([seg], turns: [],
                                              regions: [(start: 4.0, end: 5.0)])
        let plain = AudioRecorder.remoteRows([clean], turns: [],
                                             regions: [(start: 4.0, end: 5.0)])

        XCTAssertEqual(marked.count, 1)
        XCTAssertTrue(marked[0].overlapped,
                      "this is the whole feature — it was hardcoded false before")
        XCTAssertTrue(marked[0].isRemote)
        XCTAssertEqual(plain.count, 1)
        XCTAssertFalse(plain[0].overlapped)
    }

    /// With NO regions the rows are exactly what they were, so a single-stream
    /// session and a remote stream with no overlap are both untouched.
    func testNoRegionsMeansNoMarks() {
        let seg = AudioRecorder.RemoteSegment(text: "hello.", window: 4.0...5.0)
        let rows = AudioRecorder.remoteRows([seg], turns: [], regions: [])
        XCTAssertEqual(rows.count, 1)
        XCTAssertFalse(rows[0].overlapped)
    }

    // MARK: - The gate

    /// The detector now runs TWICE in a dual-stream session, and the gate it
    /// releases holds the blocking stop overlay through overlap repair.
    ///
    /// Released early → repair reads half the regions. Never released → the
    /// overlay hangs until the 600 s watchdog. So the counter is the load-bearing
    /// part, and both of its failure directions are asserted here.
    func testTheGateWaitsForBothStreamsThenReleasesExactlyOnce() {
        let r = AudioRecorder()
        r.overlapDetectDone = false
        r.overlapDetectPending = 2

        r.finishOverlapDetectJob()
        XCTAssertFalse(r.overlapDetectDone,
                       "one stream in, one still running — releasing here lets "
                       + "repair read half the regions")

        r.finishOverlapDetectJob()
        XCTAssertTrue(r.overlapDetectDone, "both in — release")
    }

    /// A single-stream session takes the one-job path and is unchanged.
    func testASingleStreamSessionReleasesOnItsOnlyJob() {
        let r = AudioRecorder()
        r.overlapDetectDone = false
        r.overlapDetectPending = 1
        r.finishOverlapDetectJob()
        XCTAssertTrue(r.overlapDetectDone)
    }

    /// An extra call cannot drive the counter negative and strand the gate.
    /// The error path carries no stream, so it is deliberately possible for both
    /// jobs to fail and call this twice with nothing left to count.
    func testAnExtraCompletionIsHarmless() {
        let r = AudioRecorder()
        r.overlapDetectDone = false
        r.overlapDetectPending = 1
        r.finishOverlapDetectJob()
        r.finishOverlapDetectJob()
        XCTAssertEqual(r.overlapDetectPending, 0)
        XCTAssertTrue(r.overlapDetectDone)
    }
}
