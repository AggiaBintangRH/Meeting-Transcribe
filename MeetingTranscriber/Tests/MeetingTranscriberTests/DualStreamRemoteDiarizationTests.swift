import XCTest
@testable import MeetingTranscriber

/// Unit tests for phase 4 of dual-stream capture — diarizing the Remote stream
/// into its OWN identity space.
///
/// The two things this phase can get catastrophically wrong are both pure logic
/// and both tested here: (a) an id from one space being written into the other
/// space's profile file, which would silently and permanently corrupt speaker
/// identity, and (b) remote rows being split by the wrong turn set. Everything
/// else (the sidecar round trip, the second `final` job at stop, the Aggregate
/// Device) needs hardware and is listed as an on-device check instead.
@MainActor
final class DualStreamRemoteDiarizationTests: XCTestCase {

    // MARK: - Helpers

    private func turn(_ start: Double, _ end: Double, _ id: Int, _ name: String)
        -> DiarizationService.Turn {
        DiarizationService.Turn(start: start, end: end, id: id, name: name)
    }

    private func remoteSegment(_ start: Double, _ end: Double, _ text: String)
        -> AudioRecorder.RemoteSegment {
        AudioRecorder.RemoteSegment(text: text, window: start...end)
    }

    /// A remote profile id as it arrives from the sidecar (already offset).
    private func remoteID(_ local: Int) -> Int { AudioRecorder.remoteIDBase + local }

    // MARK: - 1. The id ranges are disjoint and route by one comparison

    /// The whole safety argument rests on this layout:
    ///   pyannote/office < 10_000 <= remote < 100_000 <= position
    func testIDRangesAreDisjointAndOrdered() {
        XCTAssertEqual(AudioRecorder.remoteIDBase, 10_000)
        XCTAssertEqual(PositionDiarizer.positionIDBase, 100_000)
        XCTAssertLessThan(AudioRecorder.remoteIDBase, PositionDiarizer.positionIDBase)
    }

    /// Office ids route to profiles.json, remote ids to profiles-remote.json —
    /// decided by the id alone, so a caller cannot pick the wrong file.
    func testWireIDRoutesToTheRightProfileSpace() {
        for id in [1, 2, 42, 9_999] {
            XCTAssertEqual(SpeakerProfileStore.Space.forWireID(id), .office,
                           "id \(id) is an office profile")
        }
        for id in [10_000, 10_001, 12_345, 99_999] {
            XCTAssertEqual(SpeakerProfileStore.Space.forWireID(id), .remote,
                           "id \(id) is a remote profile")
        }
    }

    /// The offset comes back off before anything is written, so the sidecar's own
    /// local ids are what lands in profiles-remote.json. Office ids pass through
    /// untouched — an office rename writes exactly what it always did.
    func testRemoteIDsAreOffsetBackToLocalBeforeWriting() {
        XCTAssertEqual(SpeakerProfileStore.Space.remote.localID(remoteID(1)), 1)
        XCTAssertEqual(SpeakerProfileStore.Space.remote.localID(remoteID(7)), 7)
        XCTAssertEqual(SpeakerProfileStore.Space.office.localID(3), 3)
    }

    /// The two spaces address two different files. Same directory, never the
    /// same file — that is what makes cross-space matching structurally
    /// impossible rather than merely filtered.
    func testTheTwoSpacesUseDifferentFiles() {
        XCTAssertEqual(SpeakerProfileStore.Space.office.fileName, "profiles.json")
        XCTAssertEqual(SpeakerProfileStore.Space.remote.fileName, "profiles-remote.json")
        XCTAssertNotEqual(SpeakerProfileStore.fileURL(.office),
                          SpeakerProfileStore.fileURL(.remote))
        XCTAssertEqual(SpeakerProfileStore.fileURL(.office), SpeakerProfileStore.fileURL,
                       "The unqualified fileURL must still mean the office file")
    }

    /// Position ids never reach the profile stores at all — `renameSpeaker`
    /// intercepts them first. This pins the boundary the interception relies on.
    func testPositionIDsAreAboveBothProfileSpaces() {
        XCTAssertGreaterThan(PositionDiarizer.positionIDBase, AudioRecorder.remoteIDBase)
        XCTAssertEqual(SpeakerProfileStore.Space.forWireID(PositionDiarizer.positionIDBase),
                       .remote,
                       "Position ids would land in the remote space if they ever got "
                       + "this far — renameSpeaker must branch on positionIDBase FIRST")
    }

    // MARK: - 2. Remote rows split by the REMOTE turns

    /// The point of the phase: remote voices become R1, R2 … in their own label
    /// space, carrying remote-range ids.
    func testRemoteRowsSplitBySpeakerAndCarryRemoteLabels() {
        let rows = AudioRecorder.remoteRows(
            [remoteSegment(0, 10, "Hello there. Goodbye now.")],
            turns: [turn(0, 5, remoteID(1), "R1"), turn(5, 10, remoteID(2), "R2")])

        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows.map(\.speaker), ["Remote Speaker - R1", "Remote Speaker - R2"])
        XCTAssertEqual(rows.map(\.speakerID), [remoteID(1), remoteID(2)])
        XCTAssertEqual(rows.map(\.text), ["Hello there.", "Goodbye now."])
        XCTAssertTrue(rows.allSatisfy(\.isRemote))
        XCTAssertTrue(rows.allSatisfy(\.confirmed))
        XCTAssertTrue(rows.allSatisfy { !$0.overlapped },
                      "Remote rows are never tagged from office overlap regions")
    }

    /// Every remote row's id is in the remote range — none of them could ever be
    /// mistaken for an office profile by a downstream consumer.
    func testEveryRemoteRowIDIsInTheRemoteRange() {
        let rows = AudioRecorder.remoteRows(
            [remoteSegment(0, 6, "One. Two. Three.")],
            turns: [turn(0, 2, remoteID(1), "R1"),
                    turn(2, 4, remoteID(2), "R2"),
                    turn(4, 6, remoteID(3), "R3")])
        XCTAssertEqual(rows.count, 3)
        for row in rows {
            guard let id = row.speakerID else { return XCTFail("remote row lost its id") }
            XCTAssertEqual(SpeakerProfileStore.Space.forWireID(id), .remote)
            XCTAssertLessThan(id, PositionDiarizer.positionIDBase)
        }
    }

    /// A renamed remote profile shows through the composed label, and the rename
    /// dialog gets the profile name back — not the composed one, which would be
    /// re-prefixed on the next rebuild.
    func testRenamedRemoteProfileShowsThroughTheLabel() {
        let rows = AudioRecorder.remoteRows([remoteSegment(0, 4, "Hi.")],
                                            turns: [turn(0, 4, remoteID(1), "Priya")])
        XCTAssertEqual(rows.first?.speaker, "Remote Speaker - Priya")
        XCTAssertEqual(AudioRecorder.remoteBaseName("Remote Speaker - Priya"), "Priya")
        XCTAssertEqual(AudioRecorder.remoteDisplayName("R1"), "Remote Speaker - R1")
        XCTAssertEqual(AudioRecorder.remoteBaseName("Speaker 1"), "Speaker 1",
                       "A label without the prefix is returned unchanged")
    }

    /// Office and Remote are separate spaces: the same NUMBER in each is a
    /// different person, and the labels must not collide.
    func testOfficeAndRemoteLabelsNeverCollide() {
        let remote = AudioRecorder.remoteRows([remoteSegment(0, 4, "Hi.")],
                                              turns: [turn(0, 4, remoteID(1), "Speaker 1")])
        XCTAssertEqual(remote.first?.speaker, "Remote Speaker - Speaker 1")
        XCTAssertNotEqual(remote.first?.speaker, "Speaker 1")
        XCTAssertNotEqual(remote.first?.speakerID, 1)
    }

    /// Consecutive sentences by the same remote speaker merge into one row,
    /// exactly as the office path does.
    func testConsecutiveSentencesBySameRemoteSpeakerMerge() {
        let rows = AudioRecorder.remoteRows(
            [remoteSegment(0, 10, "First one. Second one.")],
            turns: [turn(0, 10, remoteID(1), "R1")])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows.first?.text, "First one. Second one.")
    }

    /// Remote turns from OTHER windows must not leak into this segment's rows.
    func testRemoteTurnsOutsideTheWindowAreIgnored() {
        let rows = AudioRecorder.remoteRows(
            [remoteSegment(10, 14, "Only this.")],
            turns: [turn(0, 5, remoteID(1), "R1"), turn(10, 14, remoteID(2), "R2")])
        XCTAssertEqual(rows.map(\.speaker), ["Remote Speaker - R2"])
    }

    /// Multiple remote segments stay chronological after splitting.
    func testMultipleRemoteSegmentsStayChronological() {
        let rows = AudioRecorder.remoteRows(
            [remoteSegment(20, 24, "later"), remoteSegment(0, 4, "earlier")],
            turns: [turn(0, 4, remoteID(1), "R1"), turn(20, 24, remoteID(2), "R2")])
        XCTAssertEqual(rows.map(\.text), ["earlier", "later"])
        XCTAssertEqual(rows.map(\.speaker),
                       ["Remote Speaker - R1", "Remote Speaker - R2"])
    }

    // MARK: - 3. No remote turns ⇒ exactly the phase-3 behaviour

    /// Before the remote pass lands (and forever, if remote diarization never
    /// produces a turn) a remote segment is ONE unlabelled row with no id — so
    /// nothing can be renamed into a profile that does not exist.
    func testNoRemoteTurnsKeepsTheUnknownRow() {
        let rows = AudioRecorder.remoteRows([remoteSegment(4, 9, "hello from the call")])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].speaker, AudioRecorder.remoteSpeakerLabel)
        XCTAssertEqual(rows[0].speaker, "Remote Speaker - Speaker Unknown")
        XCTAssertNil(rows[0].speakerID)
        XCTAssertTrue(rows[0].isRemote)
        XCTAssertEqual(rows[0].start, 4)
        XCTAssertEqual(rows[0].end, 9)
    }

    /// The default argument is what phase 3's call site relied on; keeping it
    /// means a session whose remote pass fails degrades to the old rows rather
    /// than losing the remote text.
    func testTurnsArgumentDefaultsToEmpty() {
        XCTAssertEqual(AudioRecorder.remoteRows([remoteSegment(0, 1, "x")]).map(\.speaker),
                       AudioRecorder.remoteRows([remoteSegment(0, 1, "x")], turns: [])
                        .map(\.speaker))
    }

    /// Blank text is still dropped before any splitting is attempted.
    func testBlankRemoteSegmentIsDroppedEvenWithTurns() {
        XCTAssertTrue(AudioRecorder.remoteRows([remoteSegment(0, 4, "  \n ")],
                                               turns: [turn(0, 4, remoteID(1), "R1")]).isEmpty)
    }

    // MARK: - 4. `speakerRanges` is parameterised, not shared state

    /// The same clipping/merging serves both spaces, and each call sees exactly
    /// one turn set — an office turn can never appear in a remote range.
    func testSpeakerRangesSeesOnlyTheTurnsItIsGiven() {
        let officeTurns = [turn(0, 5, 1, "Speaker 1")]
        let remoteTurns = [turn(0, 5, remoteID(1), "R1")]

        let office = AudioRecorder.speakerRanges(in: 0...5, turns: officeTurns)
        XCTAssertEqual(office.map(\.id), [1])

        let remote = AudioRecorder.speakerRanges(in: 0...5, turns: remoteTurns)
        XCTAssertEqual(remote.map(\.id), [remoteID(1)])
        XCTAssertFalse(remote.contains { $0.id < AudioRecorder.remoteIDBase })
    }

    /// Turns are clipped to the window and same-speaker runs bridge short gaps —
    /// the behaviour the office path has always had, now shared.
    func testSpeakerRangesClipsAndMerges() {
        let ranges = AudioRecorder.speakerRanges(
            in: 2...8,
            turns: [turn(0, 4, remoteID(1), "R1"),
                    turn(4.5, 6, remoteID(1), "R1"),   // 0.5s gap → merges
                    turn(7, 12, remoteID(2), "R2")])
        XCTAssertEqual(ranges.count, 2)
        XCTAssertEqual(ranges[0].id, remoteID(1))
        XCTAssertEqual(ranges[0].start, 2, accuracy: 0.001, "clipped to the window")
        XCTAssertEqual(ranges[0].end, 6, accuracy: 0.001, "merged across the 0.5s gap")
        XCTAssertEqual(ranges[1].id, remoteID(2))
        XCTAssertEqual(ranges[1].end, 8, accuracy: 0.001, "clipped to the window")
    }

    func testSpeakerRangesWithNoTurnsIsEmpty() {
        XCTAssertTrue(AudioRecorder.speakerRanges(in: 0...10, turns: []).isEmpty)
    }

    // MARK: - 5. Office rows are untouched by all of this

    /// The regression bar: with no remote segments the merge still returns the
    /// office rows verbatim, in their own order.
    func testOfficeRowsAreUnaffectedByRemoteDiarization() {
        let officeRows = [
            AudioRecorder.SpeakerUtterance(id: "a", speaker: "Speaker 1", speakerID: 1,
                                           start: 3, end: 4, text: "c", confirmed: true),
            AudioRecorder.SpeakerUtterance(id: "b", speaker: "Speaker 2", speakerID: 2,
                                           start: 1, end: 2, text: "a", confirmed: true)
        ]
        XCTAssertEqual(AudioRecorder.mergeRowsByStartTime(office: officeRows, remote: []),
                       officeRows)
    }

    /// An office row and a remote row can carry the same speaker NUMBER without
    /// ever being confusable — different id ranges, different labels, and the
    /// remote one flagged.
    func testOfficeAndRemoteRowsCoexistInOneTranscript() {
        let officeRow = AudioRecorder.SpeakerUtterance(
            id: "o", speaker: "Speaker 1", speakerID: 1,
            start: 0, end: 4, text: "in the room", confirmed: true)
        let remote = AudioRecorder.remoteRows([remoteSegment(5, 9, "on the call")],
                                              turns: [turn(5, 9, remoteID(1), "R1")])
        let merged = AudioRecorder.mergeRowsByStartTime(office: [officeRow], remote: remote)

        XCTAssertEqual(merged.map(\.text), ["in the room", "on the call"])
        XCTAssertEqual(merged.map(\.isRemote), [false, true])
        XCTAssertEqual(merged.map(\.speaker), ["Speaker 1", "Remote Speaker - R1"])
        XCTAssertEqual(merged.compactMap(\.speakerID).map(SpeakerProfileStore.Space.forWireID),
                       [.office, .remote])
    }
}
