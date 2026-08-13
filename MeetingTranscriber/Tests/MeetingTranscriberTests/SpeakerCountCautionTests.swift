import XCTest
@testable import MeetingTranscriber

/// A speaker count that looks like clustering fragmentation gets a CAUTION
/// (audit, 2026-08-13).
///
/// The finding behind it: on a recording verified to hold exactly two people,
/// three of the six engines answered 9, 12 and 15 speakers with automatic
/// counting — and nothing in the app said so. Pinned to 2, every engine returns
/// 2, so the clustering is sound in all of them and only the COUNTING stage
/// fails. The transcript is then split across people who do not exist, which is
/// fabrication rather than degradation.
///
/// The rule is four or more speakers of whom the MEDIAN spoke exactly once.
/// Both halves matter, and the tests below are arranged around exactly that:
/// the count alone would accuse a real 5-person meeting, and the median alone
/// would accuse a correct 2- or 3-speaker clip.
@MainActor
final class SpeakerCountCautionTests: XCTestCase {

    /// `n` speakers with `each` turns apiece.
    private func turns(speakers n: Int, each: Int) -> [SpeakerTurn] {
        var out: [SpeakerTurn] = []
        var t = 0.0
        for _ in 0..<each {
            for s in 1...n {
                out.append(SpeakerTurn(start: t, end: t + 1, id: s, name: "Speaker \(s)"))
                t += 1
            }
        }
        return out
    }

    // MARK: - It fires on the measured failures

    /// The three real results from 2026-08-13, reproduced by shape: many
    /// speakers, one turn each.
    func testTheMeasuredFragmentationCasesAreAllFlagged() {
        for n in [9, 12, 15] {
            XCTAssertNotNil(AudioRecorder.implausibleSpeakerCount(turns(speakers: n, each: 1)),
                            "\(n) speakers with one turn each is the fragmentation signature")
        }
    }

    /// The caution has to be actionable — it is shown instead of an error
    /// precisely because the user can fix it.
    func testTheCautionSaysWhatToChange() {
        let text = AudioRecorder.implausibleSpeakerCount(turns(speakers: 9, each: 1))
        XCTAssertNotNil(text)
        XCTAssertTrue(text!.contains("9 speakers"), "it names the count it is objecting to")
        XCTAssertTrue(text!.lowercased().contains("speaker count"),
                      "it points at the picker beside the record button")
        XCTAssertTrue(text!.contains("Diarization"), "…and at the engine setting")
    }

    // MARK: - It stays silent on the correct results — the half that matters more

    /// A genuine 5-person meeting: five speakers, three turns each. Accusing this
    /// would train the user to ignore the caution, which is worse than not having
    /// one. Both real 5-speaker measurements had a median of 3.
    func testARealFivePersonMeetingIsNotAccused() {
        XCTAssertNil(AudioRecorder.implausibleSpeakerCount(turns(speakers: 5, each: 3)))
    }

    /// THE FALSE-POSITIVE THE COUNT GUARD EXISTS FOR: a correct 3-speaker clip
    /// where each person legitimately has ONE turn. Measured — pyannote on
    /// `Overlap123.wav` returns exactly this, and it is the right answer.
    func testAShortCorrectClipWithOneTurnEachIsNotAccused() {
        for n in [2, 3] {
            XCTAssertNil(AudioRecorder.implausibleSpeakerCount(turns(speakers: n, each: 1)),
                         "\(n) speakers is under the floor — a real answer on a short clip")
        }
    }

    /// Four speakers is the floor, so it is tested from both sides: three with one
    /// turn each is silent, four is flagged. An off-by-one here either accuses
    /// correct 3-speaker clips or misses the smallest real fragmentation.
    func testTheFloorIsExactlyFour() {
        XCTAssertNil(AudioRecorder.implausibleSpeakerCount(turns(speakers: 3, each: 1)))
        XCTAssertNotNil(AudioRecorder.implausibleSpeakerCount(turns(speakers: 4, each: 1)))
    }

    /// Many speakers who each spoke SEVERAL times is a large meeting, not
    /// fragmentation — the rule must not simply mean "lots of speakers".
    func testALargeMeetingWithRealTurnTakingIsNotAccused() {
        XCTAssertNil(AudioRecorder.implausibleSpeakerCount(turns(speakers: 12, each: 4)),
                     "12 people who each speak four times is a big meeting, not a split voice")
    }

    /// Empty and single-speaker inputs are silent rather than crashing on an
    /// empty median.
    func testDegenerateInputsAreSilent() {
        XCTAssertNil(AudioRecorder.implausibleSpeakerCount([]))
        XCTAssertNil(AudioRecorder.implausibleSpeakerCount(turns(speakers: 1, each: 1)))
    }

    /// The median is what is tested, not the mean: one very talkative speaker
    /// must not rescue a set of one-turn speakers around them.
    func testOneTalkativeSpeakerDoesNotMaskTheRest() {
        var t = turns(speakers: 6, each: 1)
        for i in 0..<40 {
            t.append(SpeakerTurn(start: 100 + Double(i), end: 101 + Double(i),
                                 id: 1, name: "Speaker 1"))
        }
        XCTAssertNotNil(AudioRecorder.implausibleSpeakerCount(t),
                        "the mean would be 7.5 here; the median is still 1")
    }
}
