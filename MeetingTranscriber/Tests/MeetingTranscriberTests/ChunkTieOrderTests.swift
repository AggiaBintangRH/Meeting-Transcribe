import XCTest
@testable import MeetingTranscriber

/// Two streams inside ONE chunk are ordered by when speech happened, not by a
/// tie-break rule (owner, 2026-08-13).
///
/// The report, after two earlier ordering fixes: *"masih Office diatas … ketika
/// remote, terus office itu aman, tapi ketika remote lagi office jadi diatas"* —
/// and their statement of the wanted behaviour: *"selama speaker berbeda maka dia
/// akan dibawahnya"*.
///
/// **The mechanism, and it explains the odd "sometimes fine" shape exactly.** With
/// a batch diarization engine (CAM++, spectral, NeMo, DiariZen) there are NO turns
/// during recording — the pass runs once at Stop. So every row was given its
/// CHUNK's span, and all rows from one 30 s chunk shared a start time.
/// `mergeRowsByStartTime` breaks that tie with `remote < office`, i.e. Office wins
/// every tie. Two utterances in DIFFERENT chunks therefore order correctly, and
/// two in the SAME chunk always come out office-first.
///
/// ⚠ Ordering by ARRIVAL cannot fix it, which is worth pinning as prose because it
/// is the intuitive answer: both streams go through one sidecar on one queue with
/// office always enqueued first, so office replies always land first. Arrival
/// order would encode our dispatch order and pin Office on top permanently.
@MainActor
final class ChunkTieOrderTests: XCTestCase {

    private func word(_ s: Double, _ e: Double, _ src: Int) -> ChunkedASRService.AlignedWord {
        ChunkedASRService.AlignedWord(text: "w", start: s, end: e, src: src)
    }

    /// One 30 s chunk. The far end speaks at 2–8 s, the room at 15–22 s.
    /// No turns at all — the batch-engine live case.
    func testInsideOneChunkTheEarlierSpeakerIsDrawnFirst() {
        let window = 0.0...30.0
        let remote = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "the far end speaks first.", window: window,
                                         words: [word(2, 4, 0), word(6, 8, 1)],
                                         alignedChunkDuration: 30.0)])
        let office = [AudioRecorder.SpeakerUtterance(
            id: "o", speaker: nil, speakerID: nil,
            start: 15.0, end: 22.0, text: "then the room.", confirmed: true)]

        let merged = AudioRecorder.mergeRowsByStartTime(office: office, remote: remote)
        XCTAssertEqual(merged.map(\.isRemote), [true, false],
                       "the far end spoke at 2 s and the room at 15 s — remote goes first")
    }

    /// The control that makes the test above mean something: WITHOUT word times
    /// the remote row still spans the whole chunk, ties with the office row at the
    /// chunk start, and loses the tie. This is the bug, pinned.
    func testWithoutWordTimesTheRowStillSpansTheChunkAndTies() {
        let window = 0.0...30.0
        let remote = AudioRecorder.remoteRows(
            [AudioRecorder.RemoteSegment(text: "the far end speaks first.", window: window)])
        XCTAssertEqual(remote[0].start, 0.0)
        XCTAssertEqual(remote[0].end, 30.0, "no word times, so no better answer exists")
    }

    // MARK: - The span rule itself

    /// The span is first word to last word, in RECORDING time — the window start
    /// plus the chunk-local word time.
    func testTheSpanIsFirstWordToLastWordInRecordingTime() {
        let span = AudioRecorder.spokenSpan(words: [word(2, 4, 0), word(6, 8, 1)],
                                            chunkDuration: 30.0, window: 30.0...60.0)
        XCTAssertEqual(span?.lowerBound ?? 0, 32.0, accuracy: 0.001)
        XCTAssertEqual(span?.upperBound ?? 0, 38.0, accuracy: 0.001)
    }

    /// The same sanity gate `WordAttribution` applies: a duration disagreeing with
    /// the window means the aligner and the app describe different spans, so the
    /// times would land in the wrong place on the recording clock.
    func testADurationMismatchYieldsNoSpan() {
        XCTAssertNil(AudioRecorder.spokenSpan(words: [word(2, 4, 0)],
                                              chunkDuration: 90.0, window: 0.0...30.0))
    }

    /// No words means no answer — NOT a fabricated one. With no turns and no word
    /// times, nothing in the data knows who spoke first.
    func testNoWordsMeansNoSpan() {
        XCTAssertNil(AudioRecorder.spokenSpan(words: nil, chunkDuration: 30.0,
                                              window: 0.0...30.0))
        XCTAssertNil(AudioRecorder.spokenSpan(words: [], chunkDuration: 30.0,
                                              window: 0.0...30.0))
    }

    /// A word time running past the chunk is clamped rather than producing a row
    /// that claims audio outside its own window.
    func testTimesAreClampedToTheWindow() {
        let span = AudioRecorder.spokenSpan(words: [word(1, 999, 0)],
                                            chunkDuration: nil, window: 0.0...30.0)
        XCTAssertEqual(span?.upperBound ?? 0, 30.0, accuracy: 0.001)
    }

    /// A zero-width result is rejected: it would sort unpredictably and describes
    /// nothing.
    func testAZeroWidthSpanIsRejected() {
        XCTAssertNil(AudioRecorder.spokenSpan(words: [word(5, 5, 0)],
                                              chunkDuration: nil, window: 0.0...30.0))
    }
}
