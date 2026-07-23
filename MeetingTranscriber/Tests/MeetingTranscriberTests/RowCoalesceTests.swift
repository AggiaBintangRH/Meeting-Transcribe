import XCTest
@testable import MeetingTranscriber

/// `coalesceAdjacentSameSpeaker` joins chunk-boundary fragments of one speaker
/// while refusing to swallow a genuine speaker change.
final class RowCoalesceTests: XCTestCase {
    private func row(_ speaker: Int?, _ text: String, _ s: Double, _ e: Double,
                     remote: Bool = false, confirmed: Bool = true,
                     overlapped: Bool = false) -> AudioRecorder.SpeakerUtterance {
        AudioRecorder.SpeakerUtterance(
            id: "\(text)", speaker: speaker.map { "S\($0)" }, speakerID: speaker,
            start: s, end: e, text: text, confirmed: confirmed,
            overlapped: overlapped, isRemote: remote)
    }
    private func fn(_ r: [AudioRecorder.SpeakerUtterance]) -> [AudioRecorder.SpeakerUtterance] {
        AudioRecorder.coalesceAdjacentSameSpeaker(r)
    }

    func testChunkFragmentsOfOneSpeakerMerge() {
        let out = fn([row(1, "...just one minute? You", 5, 14),
                      row(1, "know,", 14, 14)])
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].text, "...just one minute? You know,")
        XCTAssertEqual(out[0].end, 14)
    }

    func testDifferentSpeakersNeverMerge() {
        let out = fn([row(1, "hello there", 0, 5), row(2, "and then I said", 5, 9)])
        XCTAssertEqual(out.count, 2)
    }

    func testRemoteRowBetweenBlocksTheMerge() {
        let out = fn([row(1, "scared by", 0, 5),
                      row(10001, "Thank you.", 3, 6, remote: true),
                      row(1, "a jobless future", 6, 9)])
        XCTAssertEqual(out.count, 3, "a remote utterance sat between the two office pieces")
    }

    func testUnknownRowsDoNotMerge() {
        let out = fn([row(nil, "one", 0, 3), row(nil, "two", 3, 6)])
        XCTAssertEqual(out.count, 2, "two UNKNOWN rows are not known to be the same person")
    }

    func testOfficeAndRemoteSameNumberNeverMerge() {
        // office id and a remote id could coincide numerically only if ranges
        // overlapped, which they don't; the isRemote guard is the real barrier.
        let out = fn([row(1, "room", 0, 3), row(1, "call", 3, 6, remote: true)])
        XCTAssertEqual(out.count, 2)
    }

    func testUnconfirmedRowLeftAlone() {
        let out = fn([row(1, "final text", 0, 5), row(1, "live text", 5, 8, confirmed: false)])
        XCTAssertEqual(out.count, 2)
    }

    func testThreeInARowCollapseAndOverlapPropagates() {
        let out = fn([row(1, "a", 0, 3), row(1, "b", 3, 6, overlapped: true), row(1, "c", 6, 9)])
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].text, "a b c")
        XCTAssertTrue(out[0].overlapped)
    }
}

/// `timeDistance` and the nearest-talker tail fallback.
final class TailAttributionTests: XCTestCase {
    private func turn(_ id: Int, _ s: Double, _ e: Double) -> DiarizationService.Turn {
        DiarizationService.Turn(start: s, end: e, id: id, name: "S\(id)")
    }
    func testOverlapIsZeroDistance() {
        XCTAssertEqual(AudioRecorder.timeDistance(from: turn(1, 0, 10), to: 5...8), 0)
    }
    func testDistanceToTurnBefore() {
        XCTAssertEqual(AudioRecorder.timeDistance(from: turn(1, 0, 10), to: 12...15), 2, accuracy: 1e-9)
    }
    func testDistanceToTurnAfter() {
        XCTAssertEqual(AudioRecorder.timeDistance(from: turn(1, 20, 30), to: 12...15), 5, accuracy: 1e-9)
    }
    func testNearestPicksTheCloserTurn() {
        // A tail window 16..17 sits between a turn ending at 15 and one starting at 25.
        let turns = [turn(1, 0, 15), turn(2, 25, 40)]
        let win = 16.0...17.0
        let nearest = turns.min {
            AudioRecorder.timeDistance(from: $0, to: win) < AudioRecorder.timeDistance(from: $1, to: win)
        }
        XCTAssertEqual(nearest?.id, 1, "1 second away beats 8 seconds away")
    }
}
