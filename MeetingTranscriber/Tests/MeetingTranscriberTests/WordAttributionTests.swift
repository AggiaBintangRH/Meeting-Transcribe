import XCTest
@testable import MeetingTranscriber

/// Word-exact attribution. The first group is THE invariant — text preservation;
/// the rest cover placement, the `src` trap, and every sanity gate.
final class WordAttributionTests: XCTestCase {

    // MARK: - Helpers

    private func word(_ text: String, _ start: Double, _ end: Double, _ src: Int)
        -> ChunkedASRService.AlignedWord {
        ChunkedASRService.AlignedWord(text: text, start: start, end: end, src: src)
    }

    private func ranges(_ items: [(Double, Double, Int, String)])
        -> [WordAttribution.SpeakerRange] {
        items.map { (start: $0.0, end: $0.1, id: $0.2, name: $0.3) }
    }

    /// The invariant: rows concatenated and re-split == source text split.
    private func assertTextPreserved(_ text: String, _ pieces: [WordAttribution.Piece]?,
                                     file: StaticString = #filePath, line: UInt = #line) {
        guard let pieces else { return XCTFail("expected pieces, got nil", file: file, line: line) }
        let expected = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        let got = pieces.map(\.text).joined(separator: " ")
            .split(whereSeparator: { $0.isWhitespace }).map(String.init)
        XCTAssertEqual(got, expected, file: file, line: line)
    }

    /// Rows must never carry NaN or inverted spans.
    private func assertSaneTimes(_ pieces: [WordAttribution.Piece]?,
                                 file: StaticString = #filePath, line: UInt = #line) {
        guard let pieces else { return XCTFail("expected pieces, got nil", file: file, line: line) }
        for p in pieces {
            XCTAssertFalse(p.start.isNaN || p.end.isNaN, file: file, line: line)
            XCTAssertLessThanOrEqual(p.start, p.end, file: file, line: line)
        }
    }

    private let twoTurns = [(0.0, 5.0, 1, "Speaker 1"), (5.0, 10.0, 2, "Speaker 2")]

    // MARK: - 1. Text preservation (THE invariant)

    func testTextPreservedOnFullyAlignedText() {
        let text = "one two three four"
        let words = [word("one", 0.0, 1.0, 0), word("two", 1.0, 2.0, 1),
                     word("three", 6.0, 7.0, 2), word("four", 7.0, 8.0, 3)]
        let pieces = WordAttribution.attribute(text: text, words: words, chunkDuration: 10,
                                               window: 0...10, ranges: ranges(twoTurns))
        assertTextPreserved(text, pieces)
        assertSaneTimes(pieces)
        XCTAssertEqual(pieces?.map(\.text), ["one two", "three four"])
    }

    func testTextPreservedWithUnalignedLeadingWords() {
        // The first two source words have no aligner entry at all.
        let text = "um well three four"
        let words = [word("three", 6.0, 7.0, 2), word("four", 7.0, 8.0, 3)]
        let pieces = WordAttribution.attribute(text: text, words: words, chunkDuration: 10,
                                               window: 0...10, ranges: ranges(twoTurns))
        assertTextPreserved(text, pieces)
        assertSaneTimes(pieces)
        // Leading unaligned words inherit the first assigned speaker → one row.
        XCTAssertEqual(pieces?.count, 1)
        XCTAssertEqual(pieces?.first?.id, 2)
    }

    func testTextPreservedWithDroppedTail() {
        // Items force-fitted past the audio are dropped by the sidecar, so the
        // trailing source words have no entry — they must still be emitted.
        let text = "one two three four five"
        let words = [word("one", 0.0, 1.0, 0), word("two", 1.0, 2.0, 1)]
        let pieces = WordAttribution.attribute(text: text, words: words, chunkDuration: 10,
                                               window: 0...10, ranges: ranges(twoTurns))
        assertTextPreserved(text, pieces)
        assertSaneTimes(pieces)
        XCTAssertEqual(pieces?.map(\.text), ["one two three four five"])
    }

    func testTextPreservedWithPunctuationOnlySourceWords() {
        // "—" and "..." never get an aligner entry; they inherit the preceding word.
        let text = "one — two ... three"
        let words = [word("one", 0.0, 1.0, 0), word("two", 1.5, 2.0, 2),
                     word("three", 6.0, 7.0, 4)]
        let pieces = WordAttribution.attribute(text: text, words: words, chunkDuration: 10,
                                               window: 0...10, ranges: ranges(twoTurns))
        assertTextPreserved(text, pieces)
        assertSaneTimes(pieces)
        XCTAssertEqual(pieces?.map(\.text), ["one — two ...", "three"])
    }

    func testTextPreservedWithIrregularWhitespace() {
        // Double spaces and newlines: `src` follows Python's `text.split()`.
        let text = "one  two\nthree   four"
        let words = [word("one", 0.0, 1.0, 0), word("two", 1.0, 2.0, 1),
                     word("three", 6.0, 7.0, 2), word("four", 7.0, 8.0, 3)]
        let pieces = WordAttribution.attribute(text: text, words: words, chunkDuration: 10,
                                               window: 0...10, ranges: ranges(twoTurns))
        assertTextPreserved(text, pieces)
        XCTAssertEqual(pieces?.map(\.text), ["one two", "three four"])
    }

    // MARK: - 2. Placement

    func testStraddlingWordGoesToTheTurnItOverlapsMore() {
        // 4.0–5.8: 1.0s in turn 1, 0.8s in turn 2 → turn 1.
        let more = WordAttribution.attribute(text: "straddle", words: [word("straddle", 4.0, 5.8, 0)],
                                             chunkDuration: 10, window: 0...10,
                                             ranges: ranges(twoTurns))
        XCTAssertEqual(more?.first?.id, 1)
        // 4.8–6.0: 0.2s in turn 1, 1.0s in turn 2 → turn 2.
        let less = WordAttribution.attribute(text: "straddle", words: [word("straddle", 4.8, 6.0, 0)],
                                             chunkDuration: 10, window: 0...10,
                                             ranges: ranges(twoTurns))
        XCTAssertEqual(less?.first?.id, 2)
    }

    func testWordOverlappingNoTurnGoesToNearestTurn() {
        // Turns leave 4.0–8.0 uncovered; the word sits at 7.0–7.5, nearer turn 2.
        let gapped = ranges([(0.0, 4.0, 1, "Speaker 1"), (8.0, 10.0, 2, "Speaker 2")])
        let pieces = WordAttribution.attribute(text: "orphan", words: [word("orphan", 7.0, 7.5, 0)],
                                               chunkDuration: 10, window: 0...10, ranges: gapped)
        XCTAssertEqual(pieces?.first?.id, 2)
        // Same turns, word at 1.0–1.5 → nearer turn 1.
        let early = WordAttribution.attribute(text: "orphan", words: [word("orphan", 1.0, 1.5, 0)],
                                              chunkDuration: 10, window: 0...10, ranges: gapped)
        XCTAssertEqual(early?.first?.id, 1)
    }

    func testWindowOffsetIsAppliedToWordTimes() {
        // Chunk times are chunk-relative; turns live on the recording clock.
        let turns = ranges([(100.0, 105.0, 1, "Speaker 1"), (105.0, 110.0, 2, "Speaker 2")])
        let words = [word("early", 1.0, 2.0, 0), word("late", 8.0, 9.0, 1)]
        let pieces = WordAttribution.attribute(text: "early late", words: words,
                                               chunkDuration: 10, window: 100...110, ranges: turns)
        XCTAssertEqual(pieces?.map(\.id), [1, 2])
        XCTAssertEqual(pieces?.first?.start ?? 0, 101.0, accuracy: 1e-9)
        XCTAssertEqual(pieces?.last?.end ?? 0, 109.0, accuracy: 1e-9)
    }

    // MARK: - 3. The `src` trap (words.count != sourceWords.count)

    func testSrcIndexingSurvivesADroppedPunctuationToken() {
        // Real phase-1 shape: the standalone em-dash gets no aligner item, so
        // `src` jumps 2 → 4. Indexing by array position would shift every later
        // word by one; indexing by `src` keeps them correct.
        let text = "The one-minute pitch — it works, right?"
        let items = [word("The", 0.0, 0.3, 0), word("oneminute", 0.3, 1.0, 1),
                     word("pitch", 1.0, 1.5, 2), word("it", 6.0, 6.2, 4),
                     word("works", 6.2, 6.6, 5), word("right", 6.6, 7.0, 6)]
        let sourceWords = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        XCTAssertEqual(sourceWords.count, 7)
        XCTAssertNotEqual(items.count, sourceWords.count)

        let pieces = WordAttribution.attribute(text: text, words: items, chunkDuration: 10,
                                               window: 0...10, ranges: ranges(twoTurns))
        assertTextPreserved(text, pieces)
        // The word after the dash is the ORIGINAL "it", and it opens speaker 2's row.
        XCTAssertEqual(pieces?.map(\.text), ["The one-minute pitch —", "it works, right?"])
        XCTAssertEqual(pieces?.map(\.id), [1, 2])
        // Display text is never the aligner's normalized token.
        XCTAssertTrue(pieces?.first?.text.contains("one-minute") ?? false)
        XCTAssertFalse(pieces?.first?.text.contains("oneminute") ?? true)
    }

    // MARK: - 4. Sanity gates (each returns nil → caller falls back)

    func testGateEmptyWords() {
        XCTAssertNil(WordAttribution.attribute(text: "one two", words: [], chunkDuration: 10,
                                               window: 0...10, ranges: ranges(twoTurns)))
    }

    func testGateEmptyRanges() {
        XCTAssertNil(WordAttribution.attribute(text: "one two", words: [word("one", 0, 1, 0)],
                                               chunkDuration: 10, window: 0...10, ranges: []))
    }

    func testGateSrcOutOfRange() {
        XCTAssertNil(WordAttribution.attribute(text: "one two",
                                               words: [word("one", 0, 1, 0), word("two", 1, 2, 9)],
                                               chunkDuration: 10, window: 0...10,
                                               ranges: ranges(twoTurns)))
    }

    func testGateSrcDecreasing() {
        XCTAssertNil(WordAttribution.attribute(text: "one two three",
                                               words: [word("two", 1, 2, 1), word("one", 2, 3, 0)],
                                               chunkDuration: 10, window: 0...10,
                                               ranges: ranges(twoTurns)))
    }

    func testGateNonMonotonicTimes() {
        XCTAssertNil(WordAttribution.attribute(text: "one two",
                                               words: [word("one", 5, 6, 0), word("two", 1, 2, 1)],
                                               chunkDuration: 10, window: 0...10,
                                               ranges: ranges(twoTurns)))
    }

    func testGateEndBeforeStart() {
        XCTAssertNil(WordAttribution.attribute(text: "one two",
                                               words: [word("one", 5, 4, 0)],
                                               chunkDuration: 10, window: 0...10,
                                               ranges: ranges(twoTurns)))
    }

    func testGateDurMismatch() {
        // The sidecar trimmed its buffer: dur 300s vs a 10s window.
        XCTAssertNil(WordAttribution.attribute(text: "one two", words: [word("one", 0, 1, 0)],
                                               chunkDuration: 300, window: 0...10,
                                               ranges: ranges(twoTurns)))
        // Within tolerance (0.5s) it still runs.
        XCTAssertNotNil(WordAttribution.attribute(text: "one two", words: [word("one", 0, 1, 0)],
                                                  chunkDuration: 10.5, window: 0...10,
                                                  ranges: ranges(twoTurns)))
    }

    func testGateEmptyText() {
        XCTAssertNil(WordAttribution.attribute(text: "   ", words: [word("one", 0, 1, 0)],
                                               chunkDuration: 10, window: 0...10,
                                               ranges: ranges(twoTurns)))
    }

    func testMissingChunkDurationSkipsOnlyTheDurGate() {
        XCTAssertNotNil(WordAttribution.attribute(text: "one two", words: [word("one", 0, 1, 0)],
                                                  chunkDuration: nil, window: 0...10,
                                                  ranges: ranges(twoTurns)))
    }

    // MARK: - 5. The target case: a mid-sentence speaker switch

    /// Speaker 1 speaks slowly up to 5s, speaker 2 fires off many short words
    /// after it. Character-proportional splitting cuts the chunk near its
    /// character midpoint; word times cut it at the actual switch.
    func testSplitLandsAtTheSwitchNotAtTheCharacterMidpoint() {
        let text = "Do you think the proposal is ready yes it is all good now"
        let sourceWords = text.split(separator: " ").map(String.init)
        // Words 0–6 are speaker 1 (0–5s), words 7–12 speaker 2 (5–10s).
        var items: [ChunkedASRService.AlignedWord] = []
        for i in 0..<7 {
            items.append(word(sourceWords[i], Double(i) * 0.7, Double(i) * 0.7 + 0.6, i))
        }
        for i in 7..<sourceWords.count {
            let s = 5.0 + Double(i - 7) * 0.25
            items.append(word(sourceWords[i], s, s + 0.2, i))
        }

        let pieces = WordAttribution.attribute(text: text, words: items, chunkDuration: 10,
                                               window: 0...10, ranges: ranges(twoTurns))
        assertTextPreserved(text, pieces)
        assertSaneTimes(pieces)
        XCTAssertEqual(pieces?.count, 2)
        XCTAssertEqual(pieces?[0].text, "Do you think the proposal is ready")
        XCTAssertEqual(pieces?[1].text, "yes it is all good now")
        XCTAssertEqual(pieces?.map(\.id), [1, 2])

        // The proportional estimate would have cut elsewhere: the switch sits at
        // 34 of 56 characters (61%), well past the 50% time midpoint the
        // character-count model assumes, so it would hand "is ready" to speaker 2.
        let charsBeforeSwitch = sourceWords[0..<7].joined(separator: " ").count
        let proportionalSplitTime = 10.0 * Double(charsBeforeSwitch) / Double(text.count)
        XCTAssertGreaterThan(proportionalSplitTime, 5.5)
        XCTAssertEqual(pieces?[0].end ?? 0, 4.8, accuracy: 1e-9)
    }
}
