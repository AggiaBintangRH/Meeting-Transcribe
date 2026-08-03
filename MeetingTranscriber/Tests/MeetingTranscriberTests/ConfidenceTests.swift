import XCTest
@testable import MeetingTranscriber

/// The two confidence numbers shown in each transcript row: `spk` (how strongly
/// the voice matched its saved profile) and `asr` (how confident the chunked
/// model was in the words).
///
/// Every check here defends the SAME rule from a different side: a number is
/// shown only where one was actually measured, and "not measured" is `nil` —
/// never 0, never carried over from another row, never invented for symmetry. A
/// wrong confidence is worse than no confidence, because it is presented as the
/// thing the reader should use to judge everything else on the row.
final class ConfidenceWireDecodingTests: XCTestCase {

    // MARK: - Diarization turns

    /// Every pre-confidence payload must still decode. This is the whole reason
    /// `conf` is Optional with a default rather than a required field.
    func testTurnWithoutConfDecodesAsNil() throws {
        let line = #"{"start":1.0,"end":4.5,"id":2,"name":"Speaker 2"}"#
        let turn = try JSONDecoder().decode(SpeakerTurn.self,
                                            from: Data(line.utf8))
        XCTAssertEqual(turn.id, 2)
        XCTAssertNil(turn.conf, "an absent key means NOT MEASURED, not zero")
    }

    func testTurnWithConfDecodes() throws {
        let line = #"{"start":1.0,"end":4.5,"id":2,"name":"Speaker 2","conf":0.873}"#
        let turn = try JSONDecoder().decode(SpeakerTurn.self,
                                            from: Data(line.utf8))
        XCTAssertEqual(turn.conf ?? -1, 0.873, accuracy: 1e-9)
    }

    /// A whole `result` payload, the shape the sidecar actually emits: some turns
    /// matched a profile, one is a brand-new voice that matched nothing.
    func testMixedSegmentsDecodeIndependently() throws {
        let line = #"{"type":"result","segments":[{"start":0.0,"end":3.0,"id":1,"name":"Speaker 1","conf":0.91},{"start":3.0,"end":6.0,"id":2,"name":"Speaker 2"}]}"#
        struct Reply: Decodable { let segments: [SpeakerTurn] }
        let reply = try JSONDecoder().decode(Reply.self, from: Data(line.utf8))
        XCTAssertEqual(reply.segments.count, 2)
        XCTAssertEqual(reply.segments[0].conf ?? -1, 0.91, accuracy: 1e-9)
        XCTAssertNil(reply.segments[1].conf)
    }

    // MARK: - Chunked ASR messages

    func testFinalWithoutConfDecodesAsNil() throws {
        let line = #"{"type":"final","text":"hello there"}"#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertNil(m.conf, "Qwen3/Granite/Voxtral send no conf at all")
    }

    func testFinalWithConfDecodes() throws {
        let line = #"{"type":"final","text":"hello there","conf":0.918}"#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.conf ?? -1, 0.918, accuracy: 1e-9)
    }

    func testFileResultCarriesConfAlongsideItsID() throws {
        let line = #"{"type":"file_result","id":7,"text":"remote words","conf":0.84}"#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.id, 7)
        XCTAssertEqual(m.conf ?? -1, 0.84, accuracy: 1e-9)
    }

    /// `conf` survives alongside keys this decoder does NOT read. The aligner
    /// moved to its own sidecar on 2026-07-29, so `words`/`dur` are gone from
    /// this message — but a stale packaged sidecar can still send them, and the
    /// confidence must come through untouched when it does.
    func testConfSurvivesAlongsideAStaleSidecarsAlignerKeys() throws {
        let line = #"{"type":"final","text":"one-minute break","words":[{"text":"oneminute","start":0.2,"end":0.8,"src":0}],"dur":30.0,"conf":0.77}"#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.text, "one-minute break")
        XCTAssertEqual(m.conf ?? -1, 0.77, accuracy: 1e-9)
    }
}

/// `conf` must survive every place a `Turn` is rebuilt field by field. Those
/// rebuilds are where a newly added field gets silently dropped, and the loss
/// would show only as a number quietly missing from the transcript.
final class TurnConfidenceCarryThroughTests: XCTestCase {
    private func turn(_ id: Int, _ s: Double, _ e: Double,
                      conf: Double? = nil) -> SpeakerTurn {
        SpeakerTurn(start: s, end: e, id: id, name: "S\(id)", conf: conf)
    }

    func testOffsetTurnsShiftsTimeAndKeepsConf() {
        let out = AudioRecorder.offsetTurns([turn(1, 2, 5, conf: 0.9),
                                             turn(2, 5, 7)], by: 30)
        XCTAssertEqual(out[0].start, 32)
        XCTAssertEqual(out[0].end, 35)
        XCTAssertEqual(out[0].conf ?? -1, 0.9, accuracy: 1e-9)
        XCTAssertNil(out[1].conf, "an unmeasured turn stays unmeasured")
    }

    /// Mirrors `renameSpeaker`'s map: the LABEL changes, the evidence behind it
    /// does not.
    func testRenameMappingPreservesConf() {
        let turns = [turn(1, 0, 3, conf: 0.82), turn(2, 3, 6, conf: 0.55)]
        let renamed = turns.map {
            $0.id == 1
                ? SpeakerTurn(start: $0.start, end: $0.end, id: $0.id,
                                          name: "Aggia", conf: $0.conf)
                : $0
        }
        XCTAssertEqual(renamed[0].name, "Aggia")
        XCTAssertEqual(renamed[0].conf ?? -1, 0.82, accuracy: 1e-9)
        XCTAssertEqual(renamed[1].conf ?? -1, 0.55, accuracy: 1e-9)
    }
}

/// `annotateSpeakerConfidence` — the last step of the row build.
final class SpeakerConfidenceAnnotationTests: XCTestCase {
    private func turn(_ id: Int, _ s: Double, _ e: Double,
                      conf: Double? = nil) -> SpeakerTurn {
        SpeakerTurn(start: s, end: e, id: id, name: "S\(id)", conf: conf)
    }
    private func row(_ id: Int?, _ s: Double?, _ e: Double?,
                     remote: Bool = false,
                     confirmed: Bool = true) -> AudioRecorder.SpeakerUtterance {
        AudioRecorder.SpeakerUtterance(
            id: "r\(id ?? -1)-\(s ?? -1)", speaker: id.map { "S\($0)" }, speakerID: id,
            start: s, end: e, text: "words", confirmed: confirmed, isRemote: remote)
    }
    private func annotate(_ rows: [AudioRecorder.SpeakerUtterance],
                          office: [SpeakerTurn] = [],
                          remote: [SpeakerTurn] = [])
        -> [AudioRecorder.SpeakerUtterance] {
        AudioRecorder.annotateSpeakerConfidence(rows: rows, officeTurns: office,
                                                remoteTurns: remote)
    }

    /// A coalesced row spanning several turns is worth its WEAKEST evidence.
    func testMinimumOverOverlappingTurnsOfTheSameSpeaker() {
        let out = annotate([row(1, 0, 30)],
                           office: [turn(1, 0, 10, conf: 0.95),
                                    turn(1, 10, 20, conf: 0.61),
                                    turn(1, 20, 30, conf: 0.88)])
        XCTAssertEqual(out[0].speakerConf ?? -1, 0.61, accuracy: 1e-9,
                       "the row claims one identity across all three turns")
    }

    /// A turn with no measurement is SKIPPED, not counted as zero — otherwise
    /// one first-appearance turn would drag every row it touches to 0.00.
    func testTurnWithoutConfIsSkippedNotTreatedAsZero() {
        let out = annotate([row(1, 0, 20)],
                           office: [turn(1, 0, 10),               // no conf
                                    turn(1, 10, 20, conf: 0.74)])
        XCTAssertEqual(out[0].speakerConf ?? -1, 0.74, accuracy: 1e-9)
    }

    func testNoMeasurementAtAllLeavesNil() {
        let out = annotate([row(1, 0, 20)], office: [turn(1, 0, 20)])
        XCTAssertNil(out[0].speakerConf)
    }

    /// Another speaker's turns say nothing about this row.
    func testOtherSpeakersTurnsAreIgnored() {
        let out = annotate([row(1, 0, 10)],
                           office: [turn(2, 0, 10, conf: 0.30),
                                    turn(1, 0, 10, conf: 0.92)])
        XCTAssertEqual(out[0].speakerConf ?? -1, 0.92, accuracy: 1e-9)
    }

    func testNoOverlapInTimeMeansNoNumber() {
        let out = annotate([row(1, 50, 60)], office: [turn(1, 0, 10, conf: 0.99)])
        XCTAssertNil(out[0].speakerConf,
                     "a turn elsewhere in the meeting is not evidence for this row")
    }

    /// ATND position speakers (id ≥ 100 000) never appear in the pyannote turns,
    /// so they simply find nothing — no position code, and no special case.
    func testPositionSpeakerRowsGetNothing() {
        let positionID = PositionDiarizer.positionIDBase + 3
        let out = annotate([row(positionID, 0, 10)],
                           office: [turn(1, 0, 10, conf: 0.93)])
        XCTAssertNil(out[0].speakerConf)
    }

    /// The id spaces are disjoint, so a remote row must read the REMOTE turns.
    /// The office collection is deliberately seeded with the same numeric id to
    /// prove the routing is by `isRemote`, not by luck.
    func testRemoteRowsReadTheRemoteTurns() {
        let remoteID = AudioRecorder.remoteIDBase + 1
        let out = annotate([row(remoteID, 0, 10, remote: true)],
                           office: [turn(remoteID, 0, 10, conf: 0.10)],
                           remote: [turn(remoteID, 0, 10, conf: 0.86)])
        XCTAssertEqual(out[0].speakerConf ?? -1, 0.86, accuracy: 1e-9)
    }

    func testOfficeRowDoesNotReadRemoteTurns() {
        let out = annotate([row(1, 0, 10)],
                           office: [],
                           remote: [turn(1, 0, 10, conf: 0.99)])
        XCTAssertNil(out[0].speakerConf)
    }

    /// Live text is provisional — its speaker label can still change, so it
    /// carries no confidence claim.
    func testUnconfirmedRowsAreNotAnnotated() {
        let out = annotate([row(1, 0, 10, confirmed: false)],
                           office: [turn(1, 0, 10, conf: 0.9)])
        XCTAssertNil(out[0].speakerConf)
    }

    func testUnknownSpeakerRowIsLeftAlone() {
        let out = annotate([row(nil, 0, 10)], office: [turn(1, 0, 10, conf: 0.9)])
        XCTAssertNil(out[0].speakerConf)
    }

    /// A realtime row with no window at all must not crash or acquire a number.
    func testRowWithoutTimesIsLeftAlone() {
        let out = annotate([row(1, nil, nil)], office: [turn(1, 0, 10, conf: 0.9)])
        XCTAssertNil(out[0].speakerConf)
    }
}
