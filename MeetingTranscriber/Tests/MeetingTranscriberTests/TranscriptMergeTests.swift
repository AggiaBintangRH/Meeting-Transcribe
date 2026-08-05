import XCTest
@testable import MeetingTranscriber

/// First coverage of `TranscriptMerge` — it shipped with none, which is why the
/// both-sided-gap duplication bug reached a real transcript.
///
/// THE invariant these tests are built around: a COMBINE emits the EXISTING text
/// verbatim, in order, plus track runs inserted only at gaps where existing had
/// nothing. A gap with words on both sides is a SUBSTITUTION of the same audio,
/// never two things that were both said.
final class TranscriptMergeTests: XCTestCase {

    // MARK: - Helpers

    private func tokens(_ s: String) -> [String] {
        s.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }

    /// The existing token sequence must survive, in order, inside the output.
    /// (Insertions are additions between existing tokens, so `existing` is a
    /// subsequence of the merged text — never reordered, never dropped.)
    private func assertExistingSurvivesInOrder(_ existing: String, _ result: TranscriptMerge.Result,
                                               file: StaticString = #filePath, line: UInt = #line) {
        let want = tokens(existing)
        let got = tokens(result.text)
        var i = 0
        for token in got where i < want.count && token == want[i] { i += 1 }
        XCTAssertEqual(i, want.count,
                       "existing text is not an in-order subsequence of \"\(result.text)\"",
                       file: file, line: line)
    }

    // MARK: - 1. Substitutions must never duplicate (the shipped bug)

    /// Verbatim from logs/overlap-repair-decisions.log: the merge produced
    /// "it didn't did not have this and it didn't did not have anything...".
    /// The track re-words the same audio; nothing about it is new speech.
    func testContractionSubstitutionIsANoopNotADuplication() {
        let existing = "it didn't have this and it didn't have anything under security"
        let track    = "it did not have this and it did not have anything under security"
        let r = TranscriptMerge.merge(existing: existing, track: track)

        XCTAssertEqual(r.kind, .noop, "a pure substitution adds no speech")
        XCTAssertEqual(r.text, existing)
        XCTAssertEqual(r.inserted, 0)
        XCTAssertFalse(r.text.contains("didn't did not"))
        XCTAssertFalse(r.text.contains("did not didn't"))
    }

    /// The owner's second logged example, abbreviated: "because cause",
    /// "doesn't does not", "parity parody", "that's that is" all in one row.
    func testParityParodySubstitutionsKeepTheExistingWording() {
        let existing = "i don't like the verbiage of it because it doesn't have parity so that's where i think"
        let track    = "i do not like the verbiage of it cause it does not have parody so that is where i think"
        let r = TranscriptMerge.merge(existing: existing, track: track)

        XCTAssertFalse(r.text.contains("parity parody"))
        XCTAssertFalse(r.text.contains("parody parity"))
        XCTAssertFalse(r.text.contains("because cause"))
        XCTAssertFalse(r.text.contains("that's that is"))
        XCTAssertFalse(r.text.contains("doesn't does not"))
        // The confirmed chunked-ASR wording is the one that survives.
        XCTAssertTrue(r.text.contains("parity"))
        XCTAssertFalse(r.text.contains("parody"))
        assertExistingSurvivesInOrder(existing, r)
    }

    // MARK: - 2. Genuine insertions still land

    func testTrackWordsAreInsertedWhereExistingIsSilent() {
        let existing = "one two three four seven eight nine ten"
        let track    = "one two three four five six seven eight nine ten"
        let r = TranscriptMerge.merge(existing: existing, track: track)

        XCTAssertEqual(r.kind, .combine)
        XCTAssertEqual(r.text, "one two three four five six seven eight nine ten")
        XCTAssertEqual(r.inserted, 2, "\"five six\" — the only words existing did not have")
        assertExistingSurvivesInOrder(existing, r)
    }

    /// One substitution gap and one genuine insertion gap in the same pair: the
    /// substitution keeps existing, the insertion is applied.
    func testMixedSubstitutionAndInsertion() {
        let existing = "alpha bravo charlie delta didn't echo foxtrot golf hotel india juliet kilo"
        let track    = "alpha bravo charlie delta did not echo foxtrot golf hotel extra words india juliet kilo"
        let r = TranscriptMerge.merge(existing: existing, track: track)

        XCTAssertEqual(r.kind, .combine)
        XCTAssertEqual(r.text,
            "alpha bravo charlie delta didn't echo foxtrot golf hotel extra words india juliet kilo")
        XCTAssertEqual(r.inserted, 2)
        XCTAssertFalse(r.text.contains("didn't did not"))
        assertExistingSurvivesInOrder(existing, r)
    }

    /// Matched runs are emitted from the EXISTING originals: the norms are equal,
    /// the casing and punctuation are not, and existing is the trusted wording.
    func testMatchedRunKeepsTheExistingCasingAndPunctuation() {
        let existing = "Parity, yes we agree on that"
        let track    = "parity, yes we agree on that plus more here"
        let r = TranscriptMerge.merge(existing: existing, track: track)

        XCTAssertEqual(r.kind, .combine)
        XCTAssertTrue(r.text.hasPrefix("Parity,"), "got \"\(r.text)\"")
        XCTAssertEqual(r.text, "Parity, yes we agree on that plus more here")
        XCTAssertEqual(r.inserted, 3)
        assertExistingSurvivesInOrder(existing, r)
    }

    // MARK: - 3. `.replace` and `.noop` are unaffected

    func testDisjointTextsReplaceVerbatim() {
        let r = TranscriptMerge.merge(existing: "alpha bravo charlie delta",
                                      track: "zulu yankee xray whiskey")
        XCTAssertEqual(r.kind, .replace)
        XCTAssertEqual(r.text, "zulu yankee xray whiskey")
        XCTAssertEqual(r.inserted, 0)
    }

    func testTrackFullyContainedIsANoop() {
        let existing = "one two three four five six"
        let r = TranscriptMerge.merge(existing: existing, track: "two three four five")
        XCTAssertEqual(r.kind, .noop)
        XCTAssertEqual(r.text, existing)
        XCTAssertEqual(r.inserted, 0)
    }

    /// Punctuation-only tokens have an empty norm and never participate in
    /// matching, so they cannot build an anchoring run of their own.
    ///
    /// CORRECTED 2026-08-05. This case used to assert `.replace` — i.e. that
    /// `"x — — — — y"` may be replaced by `"— — — —"`, DELETING the words `x` and
    /// `y`. That is the punctuation-only data-loss bug, and this test was pinning
    /// it, which is a large part of why it survived: the merge behaved exactly as
    /// the suite said it should.
    ///
    /// The stated intent — punctuation cannot anchor a run — is about
    /// `longestRun`, and that half is unchanged and still the point. Only the
    /// conclusion drawn from it was wrong: no anchor does not mean "replace with
    /// this", it means "this track has nothing to say".
    func testPunctuationCannotAnchorARun() {
        let r = TranscriptMerge.merge(existing: "x — — — — y", track: "— — — —")
        XCTAssertEqual(r.longestRun, 0, "four matching dashes are not four matching words")
        XCTAssertEqual(r.kind, .noop, "a wordless track may never delete real words")
        XCTAssertEqual(r.text, "x — — — — y")
    }

    // MARK: - A wordless track may never delete (2026-08-05)

    /// THE BUG: a punctuation-only track REPLACED a real sentence with a full
    /// stop. Empty norms never match, so no block anchors the two texts,
    /// `significant` was false, and `.replace(text: ".")` came back. Measured
    /// before the fix: existing "we should ship it on friday", track "." →
    /// `kind=replace text="."`.
    ///
    /// Reachable, which is why it is a bug and not a footgun: BOTH call sites in
    /// `AudioRecorder+OverlapRepair` test only `trimmingCharacters(...).isEmpty`,
    /// which "." passes, and a re-ASR of a near-silent separated track emitting
    /// bare punctuation is ordinary behaviour.
    func testAPunctuationOnlyTrackCannotReplaceRealText() {
        let existing = "we should ship it on friday"
        for track in [".", ",", " . ", "…", "-", "?!"] {
            let r = TranscriptMerge.merge(existing: existing, track: track)
            XCTAssertEqual(r.kind, .noop, "track \(track.debugDescription) must not edit anything")
            XCTAssertEqual(r.text, existing,
                           "a wordless track deleted real text (track \(track.debugDescription))")
            XCTAssertEqual(r.inserted, 0)
        }
    }

    /// The empty case, which the callers do gate today — pinned so the type's own
    /// guarantee does not depend on them continuing to.
    func testAnEmptyTrackKeepsTheExistingTextVerbatim() {
        let existing = "we should ship it on friday"
        let r = TranscriptMerge.merge(existing: existing, track: "")
        XCTAssertEqual(r.kind, .noop)
        XCTAssertEqual(r.text, existing)
    }

    /// The guard must not swallow a track that carries ONE real word alongside
    /// punctuation — that is genuine recovery, and refusing it would be the
    /// over-deletion this fix exists to prevent, in the other direction.
    func testASingleRealWordStillCounts() {
        let r = TranscriptMerge.merge(existing: "", track: ". friday .")
        XCTAssertEqual(r.kind, .replace)
        XCTAssertEqual(r.text, ". friday .")
    }
}
