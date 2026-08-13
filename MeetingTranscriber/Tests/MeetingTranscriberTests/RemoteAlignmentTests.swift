import XCTest
@testable import MeetingTranscriber

/// The Remote stream is word-aligned, like Office (owner, 2026-08-13).
///
/// The report that produced this: the SAME audio fed through both channels came
/// back split two different ways — *"saya test nya sama audio sama dll sama tapi
/// kenapa remote hasilnya kurang bagus"*. The audio matched and the model matched;
/// what did not was the display path. Office placed every word by the turn that
/// COVERS IT IN TIME (`WordAttribution`, plus its boundary snapping); Remote
/// placed whole sentences by their CHARACTER POSITION in the chunk, which is a
/// guess and loses at exactly the speaker changes the user is looking at.
///
/// `RemoteSegment` had no `words` field at all, so the gap was structural rather
/// than a wrong branch — which is why nothing here could have failed before.
@MainActor
final class RemoteAlignmentTests: XCTestCase {

    private func remoteTurn(_ n: Int, _ s: Double, _ e: Double) -> SpeakerTurn {
        SpeakerTurn(start: s, end: e, id: AudioRecorder.remoteIDBase + n, name: "R\(n)")
    }

    private func word(_ src: Int, _ text: String, _ s: Double, _ e: Double)
        -> ChunkedASRService.AlignedWord {
        ChunkedASRService.AlignedWord(text: text, start: s, end: e, src: src)
    }

    /// Two speakers, one chunk, one unpunctuated sentence spanning both.
    ///
    /// The estimate cannot split this — there is one sentence, so it goes to one
    /// speaker whole. The word times can, and that difference IS the feature.
    private let turns = [SpeakerTurn(start: 0, end: 5,
                                     id: AudioRecorder.remoteIDBase + 1, name: "R1"),
                         SpeakerTurn(start: 5, end: 10,
                                     id: AudioRecorder.remoteIDBase + 2, name: "R2")]
    private let text = "one two three four"
    private var words: [ChunkedASRService.AlignedWord] {
        [word(0, "one", 0.2, 0.6), word(1, "two", 0.8, 1.2),
         word(2, "three", 6.0, 6.4), word(3, "four", 6.6, 7.0)]
    }

    // MARK: - The split itself

    /// THE ONE THAT MATTERS: with word times, the text lands on two speakers.
    func testRemoteRowsSplitByWordTimeWhenTheAlignerRan() {
        let seg = AudioRecorder.RemoteSegment(text: text, window: 0.0...10.0,
                                              words: words, alignedChunkDuration: 10.0)
        let rows = AudioRecorder.remoteRows([seg], turns: turns)

        XCTAssertEqual(rows.count, 2, "the two aligned halves belong to two speakers")
        XCTAssertEqual(rows[0].text, "one two")
        XCTAssertEqual(rows[1].text, "three four")
        XCTAssertEqual(rows[0].speakerID, AudioRecorder.remoteIDBase + 1)
        XCTAssertEqual(rows[1].speakerID, AudioRecorder.remoteIDBase + 2)
        XCTAssertTrue(rows.allSatisfy(\.isRemote))
        XCTAssertEqual(rows[0].speaker, "Remote Speaker - R1",
                       "the word path must produce the same display name the "
                       + "estimate does — it is a different branch, not a different "
                       + "naming rule")
    }

    /// The SAME segment without word times — i.e. exactly what shipped before —
    /// gives one row. This is the control: without it, the test above could pass
    /// on a recording the estimate would have split anyway.
    func testWithoutWordTimesItIsTheOldSingleRowEstimate() {
        let seg = AudioRecorder.RemoteSegment(text: text, window: 0.0...10.0)
        let rows = AudioRecorder.remoteRows([seg], turns: turns)

        XCTAssertEqual(rows.count, 1, "one sentence, placed whole — the old behaviour")
        XCTAssertEqual(rows[0].text, text)
    }

    /// Word times must never cost text. A payload that fails `WordAttribution`'s
    /// gates falls back to the estimate rather than dropping the chunk — the
    /// over-deletion direction leaves no trace in the transcript, only in the log.
    func testARejectedAlignmentFallsBackAndKeepsEveryWord() {
        let bad = [word(0, "one", 0.2, 0.6), word(9, "four", 6.6, 7.0)]   // src out of range
        let seg = AudioRecorder.RemoteSegment(text: text, window: 0.0...10.0,
                                              words: bad, alignedChunkDuration: 10.0)
        let rows = AudioRecorder.remoteRows([seg], turns: turns)

        XCTAssertEqual(rows.map(\.text).joined(separator: " "), text)
    }

    /// A duration that disagrees with the window means the aligner and the app are
    /// describing different spans, so the times would land in the wrong place on
    /// the recording clock. Same gate the office side relies on.
    func testADurationMismatchFallsBack() {
        let seg = AudioRecorder.RemoteSegment(text: text, window: 0.0...10.0,
                                              words: words, alignedChunkDuration: 30.0)
        let rows = AudioRecorder.remoteRows([seg], turns: turns)
        XCTAssertEqual(rows.count, 1)
    }

    /// The overlap mark survives the new branch. It is computed per PIECE here
    /// rather than per sentence, so a region touching only the second half marks
    /// only the second row — which the estimate could not express.
    func testTheOverlapMarkIsCarriedPerAlignedPiece() {
        let seg = AudioRecorder.RemoteSegment(text: text, window: 0.0...10.0,
                                              words: words, alignedChunkDuration: 10.0)
        let rows = AudioRecorder.remoteRows([seg], turns: turns,
                                            regions: [(start: 6.0, end: 8.0)])

        XCTAssertEqual(rows.count, 2)
        XCTAssertFalse(rows[0].overlapped, "the region does not reach the first half")
        XCTAssertTrue(rows[1].overlapped)
    }

    // MARK: - Which reply may be applied

    /// The stale-index guard, on the Remote collection. `src` indexes the exact
    /// `text.split()` the request was made against, so a rewritten text silently
    /// puts every later word on the wrong part of the sentence — no gate inside
    /// `WordAttribution` can see it, which is why byte-equality is the check.
    func testALateReplyIsDroppedWhenTheTextChanged() {
        let seg = AudioRecorder.RemoteSegment(text: "the original text", window: 0.0...10.0)
        XCTAssertNil(AudioRecorder.indexForAlignedWords(segments: [seg], id: seg.id,
                                                        requestText: "something else"))
        XCTAssertEqual(AudioRecorder.indexForAlignedWords(segments: [seg], id: seg.id,
                                                          requestText: "the original text"), 0)
    }

    /// …and when the segment is gone entirely (a restarted session).
    func testALateReplyIsDroppedWhenTheSegmentVanished() {
        let seg = AudioRecorder.RemoteSegment(text: "hello", window: 0.0...10.0)
        XCTAssertNil(AudioRecorder.indexForAlignedWords(segments: [seg], id: UUID(),
                                                        requestText: "hello"))
    }

    /// Office and Remote answer the same question the same way. They delegate to
    /// one private rule precisely so they cannot drift; this asserts the result,
    /// so a future edit that splits them back into two bodies has to fail here.
    func testTheOfficeAndRemoteRulesAgree() {
        let office = AudioRecorder.TranscriptSegment(text: "same text", confirmed: true,
                                                     window: 0.0...10.0)
        let remote = AudioRecorder.RemoteSegment(text: "same text", window: 0.0...10.0)

        for (label, changed) in [("matching", "same text"), ("changed", "other text")] {
            let o = AudioRecorder.indexForAlignedWords(segments: [office], id: office.id,
                                                       requestText: changed)
            let r = AudioRecorder.indexForAlignedWords(segments: [remote], id: remote.id,
                                                      requestText: changed)
            XCTAssertEqual(o, r, "the two collections disagreed on a \(label) text")
        }
    }
}
