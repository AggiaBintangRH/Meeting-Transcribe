import XCTest
@testable import MeetingTranscriber

/// `coalesceAdjacentSameSpeaker` joins chunk-boundary fragments of one speaker
/// while refusing to swallow a genuine speaker change.
final class RowCoalesceTests: XCTestCase {
    private func row(_ speaker: Int?, _ text: String, _ s: Double, _ e: Double,
                     remote: Bool = false, confirmed: Bool = true,
                     overlapped: Bool = false,
                     asrConf: Double? = nil) -> AudioRecorder.SpeakerUtterance {
        AudioRecorder.SpeakerUtterance(
            id: "\(text)", speaker: speaker.map { "S\($0)" }, speakerID: speaker,
            start: s, end: e, text: text, confirmed: confirmed,
            overlapped: overlapped, isRemote: remote, asrConf: asrConf)
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

    // MARK: - asr confidence across the merge

    /// A merged row covers both chunks, so it is worth no more than the worse of
    /// them. Averaging would let a good chunk vouch for a bad one.
    func testAsrConfTakesTheMinimumOfMergedChunks() {
        let out = fn([row(1, "a", 0, 3, asrConf: 0.94),
                      row(1, "b", 3, 6, asrConf: 0.62),
                      row(1, "c", 6, 9, asrConf: 0.81)])
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].asrConf ?? -1, 0.62, accuracy: 1e-9)
    }

    /// nil means "this model reports nothing", not "zero" — it must neither win
    /// the minimum nor erase the measurement beside it.
    func testNilAsrConfIsIgnoredRatherThanTreatedAsZero() {
        let out = fn([row(1, "a", 0, 3, asrConf: 0.7), row(1, "b", 3, 6)])
        XCTAssertEqual(out[0].asrConf ?? -1, 0.7, accuracy: 1e-9)

        let reversed = fn([row(1, "a", 0, 3), row(1, "b", 3, 6, asrConf: 0.7)])
        XCTAssertEqual(reversed[0].asrConf ?? -1, 0.7, accuracy: 1e-9)
    }

    func testAllNilStaysNil() {
        let out = fn([row(1, "a", 0, 3), row(1, "b", 3, 6)])
        XCTAssertNil(out[0].asrConf, "Qwen3/Granite rows show no asr number at all")
    }

    /// Rows that do NOT merge keep their own numbers — no leakage across a
    /// speaker change.
    func testUnmergedRowsKeepTheirOwnConfidence() {
        let out = fn([row(1, "a", 0, 3, asrConf: 0.9),
                      row(2, "b", 3, 6, asrConf: 0.4)])
        XCTAssertEqual(out.count, 2)
        XCTAssertEqual(out[0].asrConf ?? -1, 0.9, accuracy: 1e-9)
        XCTAssertEqual(out[1].asrConf ?? -1, 0.4, accuracy: 1e-9)
    }
}

/// `timeDistance` and the nearest-talker tail fallback.
final class TailAttributionTests: XCTestCase {
    private func turn(_ id: Int, _ s: Double, _ e: Double) -> SpeakerTurn {
        SpeakerTurn(start: s, end: e, id: id, name: "S\(id)")
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

/// `absorbShortSpeakerIslands` — a sub-second wrong-speaker word between two
/// same-speaker rows is relabelled; genuine turns are left alone.
final class SpeakerIslandTests: XCTestCase {
    private func row(_ speaker: Int?, _ text: String, _ s: Double, _ e: Double,
                     remote: Bool = false, confirmed: Bool = true,
                     speakerConf: Double? = nil) -> AudioRecorder.SpeakerUtterance {
        AudioRecorder.SpeakerUtterance(
            id: "\(text)-\(s)", speaker: speaker.map { "S\($0)" }, speakerID: speaker,
            start: s, end: e, text: text, confirmed: confirmed, isRemote: remote,
            speakerConf: speakerConf)
    }
    private func absorbThenCoalesce(_ r: [AudioRecorder.SpeakerUtterance],
                                    mossSession: Bool = false) -> [AudioRecorder.SpeakerUtterance] {
        AudioRecorder.coalesceAdjacentSameSpeaker(
            AudioRecorder.absorbShortSpeakerIslands(r, mossSession: mossSession))
    }

    func testTheToBugFromTheLog() {
        // Speaker 2 "...speech" / Speaker 1 "to" (0.2s) / Speaker 2 "deliver..."
        let out = absorbThenCoalesce([
            row(2, "If we have a one minute speech", 88, 94.0),
            row(1, "to", 94.0, 94.2),
            row(2, "deliver, the main challenge", 94.2, 100),
        ])
        XCTAssertEqual(out.count, 1, "the flicker word rejoins Speaker 2's sentence")
        XCTAssertEqual(out[0].speakerID, 2)
        XCTAssertEqual(out[0].text, "If we have a one minute speech to deliver, the main challenge")
    }

    /// The measured A-B-A-B sliver pattern, with the real durations from
    /// `recordings/meeting-2026-07-28T03-13-37Z.wav`: pyannote emitted
    /// 1.735..18.863 S1, then an 0.237s S2 and an 0.287s S1, then 19.387.. S2.
    ///
    /// This pins the claim that boundary snapping deliberately does NOT touch
    /// slivers (they are shorter than its 0.5s minimum turn): absorb + coalesce
    /// already resolve them, and the net output is the correct two rows. The
    /// FIRST island fires (both flanks are S1) and the second then has differing
    /// flanks — but it is itself S1, so the coalesce sweeps all three together.
    func testMeasuredSliverPatternCollapsesToTwoRowsWithoutSnapping() {
        let out = absorbThenCoalesce([
            row(1, "one", 1.735, 18.863),
            row(2, "two", 18.863, 19.100),     // 0.237s
            row(1, "three", 19.100, 19.387),   // 0.287s
            row(2, "four", 19.387, 39.923),
        ])
        XCTAssertEqual(out.count, 2, "net A, B — the two slivers are absorbed, not snapped")
        XCTAssertEqual(out[0].speakerID, 1)
        XCTAssertEqual(out[0].text, "one two three")
        XCTAssertEqual(out[1].speakerID, 2)
        XCTAssertEqual(out[1].text, "four")
    }

    /// A MOSS SESSION is exempt from the absorber, and this is a behaviour fix
    /// rather than defensive tidiness — it was caught by an actual failing case
    /// during integration. Absorbing is an ATND beam-FLICKER heuristic: it assumes
    /// a sub-second wrong-speaker row is a timing artefact. MOSS is not a timing
    /// source; it attributes the words itself, so a short MOSS row is a short
    /// utterance, not a glitch. Without the exemption a genuine 0.8s "Hi." was
    /// relabelled onto its neighbours and coalesce then merged all three rows
    /// into one — silently discarding the model's own attribution.
    ///
    /// KEYED ON THE SESSION since 2026-08-05, not on the id range. Once MOSS turns
    /// go through `identify`, their ids are PROFILE ids (< 10 000): the old
    /// `id < mossIDBase` test would have quietly stopped exempting anything and
    /// started absorbing the very rows it protects. The session is the honest
    /// signal — in a MOSS session `liveTurns` is empty and every row comes from
    /// `mossTurns`, so no ATND flicker can be present to fix.
    func testMossIslandsAreNeverAbsorbed() {
        // Post-identify ids, i.e. ordinary profile ids — the case the id-range
        // test could not have covered.
        let identified = absorbThenCoalesce([
            row(3, "Hello there.", 0, 4.0),
            row(7, "Hi.", 4.0, 4.8),
            row(3, "How are you?", 4.8, 9.0),
        ], mossSession: true)
        XCTAssertEqual(identified.count, 3,
                       "a stitched MOSS session must still keep its own short rows")
        XCTAssertEqual(identified.map(\.speakerID), [3, 7, 3])

        let base = AudioRecorder.mossIDBase
        let out = absorbThenCoalesce([
            row(base + 1, "Hello there.", 0, 4.0),
            row(base + 2, "Hi.", 4.0, 4.8),        // 0.8s — well under the absorb limit
            row(base + 1, "How are you?", 4.8, 9.0),
        ], mossSession: true)
        XCTAssertEqual(out.count, 3, "MOSS attributed these itself; nothing may be merged away")
        XCTAssertEqual(out.map(\.speakerID), [base + 1, base + 2, base + 1])
        XCTAssertEqual(out[1].text, "Hi.")

        // The identical shape with pyannote ids still absorbs — the exemption
        // must be scoped to MOSS, not a weakening of the rule itself.
        let pyannote = absorbThenCoalesce([
            row(1, "Hello there.", 0, 4.0),
            row(2, "Hi.", 4.0, 4.8),
            row(1, "How are you?", 4.8, 9.0),
        ])
        XCTAssertEqual(pyannote.count, 1, "pyannote path unchanged: the island is still absorbed")
    }

    func testLongIslandIsNotAbsorbed() {
        // A 2s Speaker-1 turn between Speaker-2 rows is a real interjection.
        let out = AudioRecorder.absorbShortSpeakerIslands([
            row(2, "so anyway", 0, 5),
            row(1, "no I disagree completely", 5, 7),
            row(2, "well okay", 7, 10),
        ])
        XCTAssertEqual(out[1].speakerID, 1, "2s is too long to be a flicker")
    }

    func testDifferentFlankingSpeakersNotAbsorbed() {
        let out = AudioRecorder.absorbShortSpeakerIslands([
            row(1, "hello", 0, 3),
            row(3, "hi", 3, 3.2),
            row(2, "hey", 3.2, 6),
        ])
        XCTAssertEqual(out[1].speakerID, 3, "flanks differ (1 vs 2) — not an island")
    }

    func testRemoteIslandNotMergedIntoOffice() {
        let out = AudioRecorder.absorbShortSpeakerIslands([
            row(2, "room speech", 0, 5),
            row(10001, "to", 5, 5.2, remote: true),
            row(2, "continues", 5.2, 8),
        ])
        XCTAssertEqual(out[1].speakerID, 10001, "a remote word is not an office flicker")
    }

    func testUnknownFlanksDoNotAbsorb() {
        let out = AudioRecorder.absorbShortSpeakerIslands([
            row(nil, "one", 0, 3),
            row(1, "two", 3, 3.2),
            row(nil, "three", 3.2, 6),
        ])
        XCTAssertEqual(out[1].speakerID, 1, "nil-id flanks are not a known same speaker")
    }

    /// An absorbed island's identity comes from a HEURISTIC (flanked on both
    /// sides), not from a voice matched against a profile — so whatever
    /// confidence it carried described the speaker it no longer claims to be,
    /// and must not follow the new label.
    func testAbsorbedIslandLosesItsSpeakerConfidence() {
        let out = AudioRecorder.absorbShortSpeakerIslands([
            row(2, "speech", 88, 94.0, speakerConf: 0.90),
            row(1, "to", 94.0, 94.2, speakerConf: 0.51),
            row(2, "deliver", 94.2, 100, speakerConf: 0.90),
        ])
        XCTAssertEqual(out[1].speakerID, 2)
        XCTAssertNil(out[1].speakerConf, "0.51 was Speaker 1's score, not Speaker 2's")
    }

    /// A row that is NOT absorbed keeps everything it had.
    func testUnabsorbedRowKeepsItsSpeakerConfidence() {
        let out = AudioRecorder.absorbShortSpeakerIslands([
            row(2, "so anyway", 0, 5, speakerConf: 0.9),
            row(1, "no I disagree completely", 5, 7, speakerConf: 0.77),
            row(2, "well okay", 7, 10, speakerConf: 0.9),
        ])
        XCTAssertEqual(out[1].speakerConf ?? -1, 0.77, accuracy: 1e-9)
    }
}
