import XCTest
@testable import MeetingTranscriber

/// Phase 5 of dual-stream capture — the id-range invariants, made explicit.
///
/// Phases 1–4 kept the three label spaces apart BY CONSTRUCTION (separate
/// collections, separate files). This suite tests the tripwires added on top:
/// the pure predicates that say which space an id belongs to, and the
/// pass-through wrappers that assert on them at the Office-only boundaries.
///
/// What can be tested here is the predicate side — a tripped `assert` traps the
/// process, so the failing path is proven by the predicate returning the ids the
/// assert would have reported, not by catching the trap. The passing path IS
/// exercised for real: every `officeTurnsOnly`/`remoteTurnsOnly` call below runs
/// the assert in this debug build, so a legitimate id silently returning is the
/// test's own success condition.
@MainActor
final class SpeakerIDSpaceGuardTests: XCTestCase {

    private func turn(_ id: Int) -> DiarizationService.Turn {
        DiarizationService.Turn(start: 0, end: 1, id: id, name: "S\(id)")
    }

    private func remoteID(_ local: Int) -> Int { AudioRecorder.remoteIDBase + local }
    private func positionID(_ local: Int) -> Int { PositionDiarizer.positionIDBase + local }

    // MARK: - The predicates the asserts are written against

    /// Office ids are clean; remote and position ids are exactly what an
    /// Office-only consumer must never be handed.
    func testNonOfficeIDsFindsRemoteAndPositionIDs() {
        XCTAssertTrue(AudioRecorder.nonOfficeIDs(in: [turn(0), turn(1), turn(9_999)]).isEmpty,
                      "Legitimate pyannote ids are not flagged")
        XCTAssertTrue(AudioRecorder.nonOfficeIDs(in: []).isEmpty)
        XCTAssertEqual(AudioRecorder.nonOfficeIDs(in: [turn(1), turn(remoteID(2))]),
                       [remoteID(2)])
        XCTAssertEqual(AudioRecorder.nonOfficeIDs(in: [turn(1), turn(positionID(0))]),
                       [positionID(0)])
        XCTAssertEqual(AudioRecorder.nonOfficeIDs(in: [turn(remoteID(1)), turn(positionID(3))]),
                       [remoteID(1), positionID(3)],
                       "Every offending id is reported, not just the first")
    }

    /// The mirror: profiles-remote.json must not see an office id either, and a
    /// position id is in neither space.
    func testNonRemoteIDsFindsOfficeAndPositionIDs() {
        XCTAssertTrue(AudioRecorder.nonRemoteIDs(in: [turn(remoteID(0)), turn(remoteID(41))]).isEmpty)
        XCTAssertTrue(AudioRecorder.nonRemoteIDs(in: []).isEmpty)
        XCTAssertEqual(AudioRecorder.nonRemoteIDs(in: [turn(remoteID(1)), turn(2)]), [2])
        XCTAssertEqual(AudioRecorder.nonRemoteIDs(in: [turn(remoteID(1)), turn(positionID(0))]),
                       [positionID(0)])
    }

    /// The boundary values themselves, since the whole design is one comparison.
    func testRangeBoundariesAreExact() {
        XCTAssertTrue(AudioRecorder.nonOfficeIDs(in: [turn(AudioRecorder.remoteIDBase - 1)]).isEmpty)
        XCTAssertFalse(AudioRecorder.nonOfficeIDs(in: [turn(AudioRecorder.remoteIDBase)]).isEmpty)
        XCTAssertTrue(AudioRecorder.nonRemoteIDs(in: [turn(AudioRecorder.remoteIDBase)]).isEmpty)
        XCTAssertFalse(AudioRecorder.nonRemoteIDs(in: [turn(AudioRecorder.remoteIDBase - 1)]).isEmpty)
        XCTAssertTrue(AudioRecorder.nonRemoteIDs(
            in: [turn(PositionDiarizer.positionIDBase - 1)]).isEmpty)
        XCTAssertFalse(AudioRecorder.nonRemoteIDs(
            in: [turn(PositionDiarizer.positionIDBase)]).isEmpty)
    }

    // MARK: - The guards themselves

    /// Legitimate office turns pass through the guard untouched — same ids, same
    /// order, same count — and, in this debug build, without tripping the assert.
    /// This is the regression bar: the guard must be invisible to correct code.
    func testOfficeGuardPassesLegitimateTurnsThroughUnchanged() {
        let turns = [turn(0), turn(1), turn(2)]
        let out = AudioRecorder.officeTurnsOnly(turns, "test")
        XCTAssertEqual(out.map(\.id), turns.map(\.id))
        XCTAssertEqual(out.count, 3)
    }

    /// The remote guard, same bar — remote-space ids flow through untouched.
    func testRemoteGuardPassesLegitimateTurnsThroughUnchanged() {
        let turns = [turn(remoteID(1)), turn(remoteID(2))]
        let out = AudioRecorder.remoteTurnsOnly(turns, "test")
        XCTAssertEqual(out.map(\.id), turns.map(\.id))
    }

    /// The failure the guards exist for, stated as the predicate they assert on:
    /// hand an Office-only consumer the REMOTE collection and it is flagged.
    /// (In a debug build the wrapper would trap here, which is the intent — a
    /// wrong-collection edit fails loudly in development.)
    func testWrongCollectionIsDetected() {
        let remoteTurns = [turn(remoteID(1)), turn(remoteID(2))]
        XCTAssertEqual(AudioRecorder.nonOfficeIDs(in: remoteTurns), remoteTurns.map(\.id),
                       "Feeding remoteLiveTurns to overlapRegions/repairWindows/"
                       + "applyFinalSpeakers must be caught")
        let officeTurns = [turn(1), turn(2)]
        XCTAssertEqual(AudioRecorder.nonRemoteIDs(in: officeTurns), officeTurns.map(\.id),
                       "Feeding liveTurns to the remote path must be caught too")
    }

    // MARK: - The profile-store boundary

    /// Both profile spaces are addressable ids; positions are not profiles at all.
    func testIsProfileIDSplitsProfilesFromPositions() {
        for id in [0, 1, 9_999, AudioRecorder.remoteIDBase, 99_999] {
            XCTAssertTrue(SpeakerProfileStore.isProfileID(id), "id \(id) is a voice profile")
        }
        for id in [PositionDiarizer.positionIDBase, positionID(1), 1_000_000] {
            XCTAssertFalse(SpeakerProfileStore.isProfileID(id), "id \(id) is an ATND cluster")
        }
    }

    /// Every id that survives `isProfileID` routes to a file, and to the file its
    /// range says — asked in this order because `forWireID` asserts the first.
    func testProfileIDsStillRouteToTheirOwnFile() {
        for id in [0, 1, 9_999] {
            XCTAssertTrue(SpeakerProfileStore.isProfileID(id))
            XCTAssertEqual(SpeakerProfileStore.Space.forWireID(id), .office)
        }
        for id in [remoteID(0), remoteID(1), 99_999] {
            XCTAssertTrue(SpeakerProfileStore.isProfileID(id))
            XCTAssertEqual(SpeakerProfileStore.Space.forWireID(id), .remote)
        }
    }

    // MARK: - Single-stream regression bar

    /// A recorder that never resolved a Remote channel reports no remote stream,
    /// which is the ONE condition `StatusChipsView.remoteChip` renders on — so a
    /// single-stream user's chip row is unchanged by phase 5.
    func testSingleStreamRecorderReportsNoRemoteStream() {
        let recorder = AudioRecorder()
        XCTAssertFalse(recorder.remoteStreamActive)
        XCTAssertNil(recorder.remoteSpeakerCount)
    }
}
