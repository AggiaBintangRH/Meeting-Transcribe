import XCTest
@testable import MeetingTranscriber

/// Overlap REPAIR reaches the Remote stream (owner, 2026-08-13).
///
/// It did not, and the reason was a measurement: separation rescued **0 of 16**
/// real office overlap windows, so a repair path was called "a pipeline built to
/// reject". That number stands — and it describes ACOUSTIC mixing in a
/// reverberant room. Remote overlap is two independent streams mixed DIGITALLY:
/// no shared room, no reverb, no crosstalk, which is the condition separation
/// models are actually good at. Re-measured on the owner's own remote recordings,
/// **3 of 3** windows separated cleanly — each track matching one real voice at
/// 0.55–0.74 and the other at ≈0, against a 0.5 same-speaker bar.
///
/// What these tests guard is not the separator (that is measured, not asserted)
/// but the SPLICE: repaired text must land under the right speaker, in the right
/// stream, exactly once.
@MainActor
final class RemoteRepairTests: XCTestCase {

    private func rid(_ n: Int) -> Int { AudioRecorder.remoteIDBase + n }

    private func turn(_ n: Int, _ s: Double, _ e: Double) -> SpeakerTurn {
        SpeakerTurn(start: s, end: e, id: rid(n), name: "R\(n)")
    }

    // MARK: - A pinned segment is finished business

    /// THE ONE THAT MATTERS MOST. A repaired segment's text came from a SEPARATED
    /// track; the turns describe the MIXED audio and no longer describe it. Left
    /// unpinned, `speakerRanges` would split one speaker's recovered words across
    /// both speakers — the exact failure repair exists to fix, reintroduced by the
    /// display path.
    func testAPinnedRemoteSegmentIsNeverResplitByTurns() {
        // TWO sentences, deliberately: the estimate splits on sentence boundaries,
        // so this text WOULD come back as two rows under two speakers if the pin
        // were ignored. A single sentence would pass either way and prove nothing —
        // it did exactly that when this test was first written.
        let text = "first half is mine. second half is also mine."
        let seg = AudioRecorder.RemoteSegment(text: text, window: 0.0...10.0,
                                              pinnedSpeakerID: rid(1),
                                              pinnedSpeakerName: "R1")
        let turns = [turn(1, 0, 5), turn(2, 5, 10)]

        // The control: unpinned, the same text and turns really do split in two.
        let unpinned = AudioRecorder.RemoteSegment(text: text, window: 0.0...10.0)
        let split = AudioRecorder.remoteRows([unpinned], turns: turns)
        XCTAssertEqual(split.count, 2, "precondition: this text is splittable")
        XCTAssertNotEqual(split[0].speakerID, split[1].speakerID)

        let rows = AudioRecorder.remoteRows([seg], turns: turns)
        XCTAssertEqual(rows.count, 1, "a pinned segment is one row, whatever the turns say")
        XCTAssertEqual(rows[0].text, text)
        XCTAssertEqual(rows[0].speakerID, rid(1))
    }

    /// The stored name is the PROFILE name ("R1"); the prefix is added at render
    /// time. Storing the composed label instead would re-prefix it on every
    /// rebuild — "Remote Speaker - Remote Speaker - R1" — and each rebuild would
    /// add another.
    func testAPinnedNameIsPrefixedExactlyOnce() {
        let seg = AudioRecorder.RemoteSegment(text: "hello", window: 0.0...5.0,
                                              pinnedSpeakerID: rid(1),
                                              pinnedSpeakerName: "R1")
        let rows = AudioRecorder.remoteRows([seg])
        XCTAssertEqual(rows[0].speaker, "Remote Speaker - R1")
    }

    /// Debug rows carry a literal label, sort last, and are never marked as
    /// overlapped — they are raw separator output, not a speaker's words.
    func testDebugRowsSortLastAndAreNotSpeakers() {
        let real = AudioRecorder.RemoteSegment(text: "real speech", window: 8.0...9.0)
        let debug = AudioRecorder.RemoteSegment(text: "track one", window: 0.0...5.0,
                                                pinnedSpeakerName: "MossFormer2 Remote Index1",
                                                isSeparationDebug: true)
        let rows = AudioRecorder.remoteRows([debug, real],
                                            regions: [(start: 0.0, end: 5.0)])

        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows[0].text, "real speech", "the debug row sorts last despite "
                       + "starting earlier")
        XCTAssertEqual(rows[1].speaker, "MossFormer2 Remote Index1")
        XCTAssertFalse(rows[1].overlapped,
                       "a raw separated track is not two people talking at once")
    }

    // MARK: - The splice

    /// The invariant: each affected row's text survives EXACTLY once — folded into
    /// the repaired speaker's text, or preserved verbatim. Never both, never
    /// neither.
    func testRepairFoldsOneSpeakerAndPreservesTheOtherExactlyOnce() {
        let r = AudioRecorder()
        r.remoteLiveTurns = [turn(1, 0, 5), turn(2, 5, 10)]
        r.remoteSegments = [AudioRecorder.RemoteSegment(text: "aaa. bbb.",
                                                        window: 0.0...10.0)]
        r.rebuildDisplayRows()

        r.applyRemoteRepair(ws: 0, we: 5, decisions: [rid(1): "aaa recovered"])

        let texts = r.remoteSegments.map(\.text)
        XCTAssertTrue(texts.contains("aaa recovered"), "the repaired speaker's text landed")
        XCTAssertFalse(texts.contains { $0.contains("aaa.") },
                       "the consumed row must not ALSO survive verbatim — that is "
                       + "the duplication this invariant forbids")
        XCTAssertTrue(texts.contains { $0.contains("bbb") },
                      "the other speaker was outside the repair and must be untouched")
        XCTAssertEqual(r.remoteSegments.filter { $0.pinnedSpeakerID == rid(1) }.count, 1)
    }

    /// A remote repair must not touch the room's transcript. The two collections
    /// are separate, so this is structural — which is exactly why it is worth
    /// asserting: nothing would fail loudly if a future edit reached across.
    func testARemoteRepairLeavesTheOfficeTranscriptAlone() {
        let r = AudioRecorder()
        r.liveTurns = [SpeakerTurn(start: 0, end: 10, id: 1, name: "Speaker 1")]
        r.segments = [AudioRecorder.TranscriptSegment(text: "the room said this",
                                                      confirmed: true, window: 0.0...10.0)]
        r.remoteLiveTurns = [turn(1, 0, 5), turn(2, 5, 10)]
        r.remoteSegments = [AudioRecorder.RemoteSegment(text: "aaa. bbb.",
                                                        window: 0.0...10.0)]
        r.rebuildDisplayRows()
        let officeBefore = r.segments

        r.applyRemoteRepair(ws: 0, we: 5, decisions: [rid(1): "aaa recovered"])

        XCTAssertEqual(r.segments.map(\.text), officeBefore.map(\.text))
        XCTAssertEqual(r.segments.count, 1)
    }

    /// Repair windows come from the stream's OWN turns. Feeding office turns to
    /// the remote side would splice room words under a remote speaker, so the
    /// id-space assert guards it — and this pins that remote regions are built
    /// from remote turns alone.
    func testRemoteRegionsComeFromRemoteTurnsOnly() {
        let r = AudioRecorder()
        r.liveTurns = [SpeakerTurn(start: 0, end: 5, id: 1, name: "Speaker 1"),
                       SpeakerTurn(start: 4, end: 9, id: 2, name: "Speaker 2")]
        r.remoteLiveTurns = [turn(1, 0, 5)]

        XCTAssertEqual(r.overlapRegions().count, 1, "the room really did overlap")
        XCTAssertTrue(r.remoteOverlapRegions().isEmpty,
                      "one remote turn cannot overlap itself, whatever the room did")
    }
}
