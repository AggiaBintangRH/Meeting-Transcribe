import XCTest
@testable import MeetingTranscriber

/// One row per UTTERANCE, not one per chunk (owner, 2026-08-13).
///
/// The report, and it is the root the three earlier ordering fixes kept missing:
/// *"si remote jadi disatuin gak bikin row … remote yang baru ini gak bikin row
/// baru tapi ambil row yang ada"*.
///
/// A chunk produces exactly ONE segment per stream, and with a batch diarization
/// engine there are no live turns to split it. At the owner's measured **120 s**
/// chunk interval (`FLUSH received — transcribing 120.1s chunk`) that means two
/// minutes of a stream's speech collapse into a single row however many separate
/// things were said. The far end speaking three times produced one row — so no
/// amount of ordering could interleave it with the room, because there was only
/// ever one row to order.
///
/// The cut is made at real silences in that stream's OWN word times.
@MainActor
final class UtteranceSplitTests: XCTestCase {

    private func word(_ s: Double, _ e: Double, _ src: Int) -> ChunkedASRService.AlignedWord {
        ChunkedASRService.AlignedWord(text: "w", start: s, end: e, src: src)
    }

    // MARK: - The split

    /// THE REPORTED CASE. Two utterances 30 s apart inside one 120 s chunk become
    /// two rows, each carrying its own time.
    func testTwoUtterancesInOneChunkBecomeTwoRows() {
        let rows = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "hello there. goodbye now.",
                                         window: 0.0...120.0,
                                         words: [word(2, 3, 0), word(3, 4, 1),
                                                 word(40, 41, 2), word(41, 42, 3)],
                                         alignedChunkDuration: 120.0)])

        XCTAssertEqual(rows.count, 2, "a 36 s silence is two things said, not one")
        XCTAssertEqual(rows[0].text, "hello there.")
        XCTAssertEqual(rows[1].text, "goodbye now.")
        XCTAssertEqual(rows[0].start ?? -1, 2.0, accuracy: 0.001)
        XCTAssertEqual(rows[1].start ?? -1, 40.0, accuracy: 0.001)
        XCTAssertTrue(rows.allSatisfy(\.isRemote))
    }

    /// …and that is what finally lets the two streams interleave: remote, office,
    /// remote, exactly the pattern the owner described.
    func testTheSplitRowsInterleaveWithTheOtherStream() {
        let remote = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "first from the call. second from the call.",
                                         window: 0.0...120.0,
                                         words: [word(2, 4, 0), word(40, 42, 1)],
                                         alignedChunkDuration: 120.0)])
        let office = [AudioRecorder.SpeakerUtterance(
            id: "o", speaker: nil, speakerID: nil, start: 20.0, end: 25.0,
            text: "the room in between.", confirmed: true)]

        let merged = AudioRecorder.mergeRowsByStartTime(office: office, remote: remote)
        XCTAssertEqual(merged.map(\.isRemote), [true, false, true],
                       "remote at 2 s, room at 20 s, remote again at 40 s")
    }

    /// A natural pause inside one sentence must NOT split it. The threshold is
    /// twice the largest inter-word gap this project has ever measured inside
    /// continuous speech (0.480 s).
    func testABreathPauseDoesNotSplitASentence() {
        let parts = AudioRecorder.splitAtPauses(
            text: "one two three", words: [word(1, 2, 0), word(2.4, 3, 1), word(3.4, 4, 2)],
            chunkDuration: 120.0, window: 0.0...120.0)
        XCTAssertEqual(parts?.count, 1, "0.4 s gaps are ordinary speech")
    }

    /// Exactly at the threshold counts as a split — the boundary is `>=`, and an
    /// off-by-one here either merges real utterances or shreds sentences.
    func testTheThresholdBoundaryIsInclusive() {
        let atThreshold = AudioRecorder.splitAtPauses(
            text: "a b", words: [word(1, 2, 0), word(3.0, 4, 1)],
            chunkDuration: 120.0, window: 0.0...120.0)
        XCTAssertEqual(atThreshold?.count, 2, "a gap of exactly 1.0 s splits")

        let justUnder = AudioRecorder.splitAtPauses(
            text: "a b", words: [word(1, 2, 0), word(2.99, 4, 1)],
            chunkDuration: 120.0, window: 0.0...120.0)
        XCTAssertEqual(justUnder?.count, 1)
    }

    // MARK: - Nothing may be lost

    /// THE INVARIANT: every word of the original text survives, once, in order.
    /// A splitter that drops a token is the over-deletion direction — invisible in
    /// the transcript and only findable in a log.
    func testEveryWordSurvivesExactlyOnceInOrder() {
        let text = "alpha bravo charlie delta echo"
        let parts = AudioRecorder.splitAtPauses(
            text: text,
            words: [word(1, 2, 0), word(2, 3, 1), word(30, 31, 2),
                    word(31, 32, 3), word(60, 61, 4)],
            chunkDuration: 120.0, window: 0.0...120.0)

        XCTAssertEqual(parts?.count, 3)
        XCTAssertEqual(parts?.map(\.text).joined(separator: " "), text)
    }

    /// Punctuation-only tokens carry no word time. They must stay with the group
    /// they follow rather than vanish.
    func testUnalignedTokensStayWithTheirGroup() {
        let text = "hello — goodbye"
        let parts = AudioRecorder.splitAtPauses(
            text: text, words: [word(1, 2, 0), word(30, 31, 2)],
            chunkDuration: 120.0, window: 0.0...120.0)
        XCTAssertEqual(parts?.map(\.text).joined(separator: " "), text)
    }

    // MARK: - When it must refuse

    /// No word times → no split, and the row stays exactly what it was. This is
    /// the pre-aligner behaviour, unchanged.
    func testWithoutWordTimesNothingIsSplit() {
        XCTAssertNil(AudioRecorder.splitAtPauses(text: "hello there.", words: nil,
                                                 chunkDuration: 120.0,
                                                 window: 0.0...120.0))
        let rows = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "hello there. goodbye now.",
                                         window: 0.0...120.0)])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].start, 0.0)
        XCTAssertEqual(rows[0].end, 120.0)
    }

    /// A duration disagreeing with the window means the aligner and the app are
    /// describing different spans — the same gate every other word-time consumer
    /// applies, so a bad payload cannot cut text at invented places.
    func testADurationMismatchRefusesToSplit() {
        XCTAssertNil(AudioRecorder.splitAtPauses(
            text: "a b", words: [word(1, 2, 0), word(30, 31, 1)],
            chunkDuration: 5.0, window: 0.0...120.0))
    }
}
