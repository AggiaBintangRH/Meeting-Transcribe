import XCTest
@testable import MeetingTranscriber

/// The join between the two halves of the 2026-07-30 diarization split: the
/// pyannote sidecar's run-local spans and the WeSpeaker sidecar's identities.
///
/// This is the one place the two wires meet, and it is where the split could
/// silently lose something the merged sidecar used to guarantee. Three encodings
/// have to survive it intact, and each one used to be enforced inside the single
/// Python process:
///   * `conf` present   — the cosine that matched this voice;
///   * `conf` absent    — a brand-new profile, `nil`, NEVER 0;
///   * a `null` identity — unidentifiable, and the turn is DROPPED (the merged
///     sidecar's own `labeled_segments` skip).
/// Plus the id-space invariant, which is the failure CLAUDE.md calls silent and
/// permanent: composed remote turns must still satisfy `remoteTurnsOnly`, and
/// composed office turns `officeTurnsOnly`.
final class ComposeTurnsTests: XCTestCase {

    private func local(_ start: Double, _ end: Double, _ label: String)
        -> PyannoteService.LocalTurn {
        // Decoded rather than constructed: LocalTurn's fields come off the wire,
        // and decoding is the path the app actually takes.
        let line = #"{"start":\#(start),"end":\#(end),"label":"\#(label)"}"#
        return try! JSONDecoder().decode(PyannoteService.LocalTurn.self,
                                         from: Data(line.utf8))
    }

    private func identityMap(_ json: String) throws -> WeSpeakerService.IdentityMap {
        struct Reply: Decodable { let speakers: WeSpeakerService.IdentityMap }
        return try JSONDecoder().decode(Reply.self, from: Data(json.utf8)).speakers
    }

    // MARK: - The three encodings

    func testMatchedVoiceCarriesItsConfidenceThrough() throws {
        let map = try identityMap(
            #"{"speakers":{"SPEAKER_00":{"id":1,"name":"Speaker 1","conf":0.874}}}"#)
        let turns = AudioRecorder.composeTurns([local(1.0, 4.0, "SPEAKER_00")],
                                               identity: map)
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].id, 1)
        XCTAssertEqual(turns[0].name, "Speaker 1")
        XCTAssertEqual(turns[0].conf ?? -1, 0.874, accuracy: 1e-9,
                       "the cosine must survive the join verbatim")
    }

    /// The dangerous direction: a brand-new profile was never scored against
    /// anything, so there is no number. A 0.00 in front of a correctly-named
    /// speaker reads as "definitely the wrong person" — the opposite claim.
    func testBrandNewProfileHasNoConfidenceAndNotZero() throws {
        let map = try identityMap(
            #"{"speakers":{"SPEAKER_00":{"id":7,"name":"Speaker 7"}}}"#)
        let turns = AudioRecorder.composeTurns([local(0.0, 3.0, "SPEAKER_00")],
                                               identity: map)
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].id, 7)
        XCTAssertNil(turns[0].conf, "absent must stay absent — never 0")
    }

    /// `null` identity ⇒ the sidecar could not identify that voice at all (under
    /// its minimum speech length, or a degenerate embedding). Its turns are
    /// dropped, which is exactly what the merged sidecar did before emitting.
    func testUnidentifiableLabelsTurnsAreDropped() throws {
        let map = try identityMap("""
            {"speakers":{"SPEAKER_00":{"id":1,"name":"Speaker 1","conf":0.9},
                         "SPEAKER_01":null}}
            """)
        let turns = AudioRecorder.composeTurns([local(0.0, 3.0, "SPEAKER_00"),
                                                local(3.1, 3.4, "SPEAKER_01"),
                                                local(4.0, 6.0, "SPEAKER_00")],
                                               identity: map)
        XCTAssertEqual(turns.map(\.start), [0.0, 4.0],
                       "the null-identity span must not reach the transcript")
        XCTAssertTrue(turns.allSatisfy { $0.id == 1 })
    }

    /// A label the reply does not mention at all is the same outcome as `null`.
    /// The merged sidecar used `mapping.get(label)`, so this case behaved
    /// identically there and must keep doing so.
    func testLabelMissingFromTheReplyIsAlsoDropped() throws {
        let map = try identityMap(
            #"{"speakers":{"SPEAKER_00":{"id":1,"name":"Speaker 1"}}}"#)
        let turns = AudioRecorder.composeTurns([local(0.0, 3.0, "SPEAKER_00"),
                                                local(3.0, 5.0, "SPEAKER_09")],
                                               identity: map)
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].id, 1)
    }

    func testEmptyIdentityMapDropsEverythingRatherThanInventNames() throws {
        let map = try identityMap(#"{"speakers":{}}"#)
        XCTAssertTrue(AudioRecorder.composeTurns([local(0, 3, "SPEAKER_00")],
                                                 identity: map).isEmpty)
    }

    // MARK: - Decoding contract

    /// `Identity` itself, so the "absent key ⇒ nil" rule is pinned at the type
    /// and not only through `composeTurns`.
    func testIdentityDecodesAbsentConfAsNil() throws {
        let with = try JSONDecoder().decode(
            WeSpeakerService.Identity.self,
            from: Data(#"{"id":3,"name":"R3","conf":0.51}"#.utf8))
        let without = try JSONDecoder().decode(
            WeSpeakerService.Identity.self,
            from: Data(#"{"id":3,"name":"R3"}"#.utf8))
        XCTAssertEqual(with.conf ?? -1, 0.51, accuracy: 1e-9)
        XCTAssertNil(without.conf)
    }

    /// An explicit `null` must decode to a PRESENT key with a nil value, not
    /// vanish and not throw — that is how "unidentifiable" reaches `composeTurns`
    /// as something it can act on.
    func testNullIdentityDecodesAsAPresentNilValue() throws {
        let map = try identityMap(#"{"speakers":{"SPEAKER_02":null}}"#)
        XCTAssertTrue(map.keys.contains("SPEAKER_02"))
        XCTAssertNil(map["SPEAKER_02"] ?? nil)
    }

    /// Times are rounded to 3 dp at the join, because that is where the merged
    /// sidecar rounded them — app-visible turn times are unchanged by the split.
    /// The wire itself carries full precision on purpose (the identity stage
    /// slices the waveform with those numbers).
    func testTimesAreRoundedToThreeDecimalsAtTheJoin() throws {
        let map = try identityMap(#"{"speakers":{"S":{"id":1,"name":"Speaker 1"}}}"#)
        let turns = AudioRecorder.composeTurns([local(1.2345678, 4.9876543, "S")],
                                               identity: map)
        XCTAssertEqual(turns[0].start, 1.235, accuracy: 1e-9)
        XCTAssertEqual(turns[0].end, 4.988, accuracy: 1e-9)
    }

    // MARK: - Id spaces (the silent, permanent failure)

    func testComposedOfficeTurnsSatisfyTheOfficeGuard() throws {
        let map = try identityMap("""
            {"speakers":{"SPEAKER_00":{"id":1,"name":"Speaker 1"},
                         "SPEAKER_01":{"id":9999,"name":"Speaker 9999"}}}
            """)
        let turns = AudioRecorder.composeTurns([local(0, 3, "SPEAKER_00"),
                                                local(3, 6, "SPEAKER_01")],
                                               identity: map)
        XCTAssertTrue(AudioRecorder.nonOfficeIDs(in: turns).isEmpty,
                      "office ids must stay below remoteIDBase")
        XCTAssertEqual(AudioRecorder.officeTurnsOnly(turns, "test").count, 2)
    }

    /// The offset is applied inside the identity sidecar, at the process
    /// boundary, so the app never sees a store-local remote id. Composition must
    /// carry that through untouched.
    func testComposedRemoteTurnsSatisfyTheRemoteGuard() throws {
        let map = try identityMap("""
            {"speakers":{"SPEAKER_00":{"id":10001,"name":"R1","conf":0.77},
                         "SPEAKER_01":{"id":10002,"name":"R2"}}}
            """)
        let turns = AudioRecorder.composeTurns([local(0, 3, "SPEAKER_00"),
                                                local(3, 6, "SPEAKER_01")],
                                               identity: map)
        XCTAssertEqual(turns.map(\.id), [10001, 10002])
        XCTAssertTrue(AudioRecorder.nonRemoteIDs(in: turns).isEmpty,
                      "remote ids must sit in [remoteIDBase, positionIDBase)")
        XCTAssertEqual(AudioRecorder.remoteTurnsOnly(turns, "test").count, 2)
    }

    /// The mirror: a composed remote turn must NOT pass the office guard. Without
    /// this the previous test would also pass on a broken guard.
    func testRemoteIDsAreRejectedByTheOfficeGuard() throws {
        let map = try identityMap(#"{"speakers":{"S":{"id":10001,"name":"R1"}}}"#)
        let turns = AudioRecorder.composeTurns([local(0, 3, "S")], identity: map)
        XCTAssertEqual(AudioRecorder.nonOfficeIDs(in: turns), [10001])
    }
}
