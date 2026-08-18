import XCTest
@testable import MeetingTranscriber

/// "Live only": talker direction while recording, nothing once you stop — the
/// finished transcript is purely the voice engine's (owner, 2026-08-18:
/// *"pokoknya ATND gak akan digunakan ketika stop recording, full diarization
/// model"*).
///
/// The FIRST time-dependent label source. Every other mode is a fixed rule, so
/// what these tests guard is that the schedule really flips — and that nothing
/// else moved with it.
@MainActor
final class PositionLiveOnlyTests: XCTestCase {

    // MARK: - The schedule

    /// While recording it is `both`, verbatim. After Stop it is `pyannote`.
    func testItIsBothWhileRecordingAndVoiceOnlyAfterStop() {
        XCTAssertEqual(PositionSource.atndLiveOnly.effective(recording: true, mossActive: false),
                       .both,
                       "while recording, direction must fill what voice has not covered")
        XCTAssertEqual(PositionSource.atndLiveOnly.effective(recording: false, mossActive: false),
                       .pyannote,
                       "after Stop the position layer must contribute nothing at all")
    }

    /// NO OTHER MODE IS TIME-DEPENDENT. Without this half, folding the schedule in
    /// could silently reschedule the four fixed rules — and `both` in particular,
    /// which is the shipped default.
    func testEveryOtherModeIgnoresTheSchedule() {
        for mode in PositionSource.allCases where mode != .atndLiveOnly {
            for recording in [true, false] {
                XCTAssertEqual(mode.effective(recording: recording, mossActive: false), mode,
                               "\(mode.rawValue) changed with recording=\(recording); "
                               + "only Live only may depend on when it is asked")
            }
        }
    }

    /// While recording it must produce the SAME plan as `both`, not merely a
    /// similar one — it delegates rather than repeating the literal so a future
    /// retune of `both` carries over.
    func testItsLivePlanIsBothsPlan() {
        let ranges: [LabeledRange] = [(start: 0, end: 5, id: 3, name: "Speaker 3")]
        let live = PositionSource.atndLiveOnly.plan(pyannoteRanges: ranges)
        let both = PositionSource.both.plan(pyannoteRanges: ranges)
        XCTAssertEqual(live.displayRanges.map(\.id), both.displayRanges.map(\.id))
        XCTAssertEqual(live.gapFillCoverage?.map(\.id), both.gapFillCoverage?.map(\.id))
        XCTAssertEqual(live.relabelFromPyannote, both.relabelFromPyannote)
    }

    /// After Stop the gap-fill is not merely empty — it is `nil`, which is the
    /// value that means "do not run the position layer at all". Empty coverage
    /// would mean the OPPOSITE: its complement is the whole window, so direction
    /// would tile everything (that is what `Position only` does).
    func testAfterStopTheGapFillIsDisabledNotWidened() {
        let ranges: [LabeledRange] = [(start: 0, end: 5, id: 3, name: "Speaker 3")]
        let plan = PositionSource.atndLiveOnly
            .effective(recording: false, mossActive: false)
            .plan(pyannoteRanges: ranges)
        XCTAssertNil(plan.gapFillCoverage,
                     "empty coverage would make direction tile the whole window — "
                     + "the exact opposite of dropping it")
        XCTAssertEqual(plan.displayRanges.map(\.id), [3])
    }

    // MARK: - What it costs, asserted rather than assumed

    /// ⚠ THE CONSEQUENCE THE OWNER SHOULD SEE. Dropping the position layer means a
    /// stretch the voice engine never labelled has no seat to fall back on. It does
    /// NOT become SPEAKER UNKNOWN — `derivedRows`'s `filled.isEmpty` branch hands
    /// it to the nearest voice turn — but that is a different answer from the one
    /// shown while recording, so the label can CHANGE at Stop.
    ///
    /// Asserted here so the trade is on the record and a future edit that removed
    /// the nearest-turn fallback would fail this rather than quietly reintroducing
    /// UNKNOWN for this mode.
    func testAStretchVoiceNeverLabelledFallsBackToTheNearestVoiceTurnNotUnknown() {
        let recorder = AudioRecorder()
        recorder.positionSource = .atndLiveOnly
        // A finished meeting (state .idle) with one voice turn that does NOT cover
        // the segment's window.
        recorder.liveTurns = [SpeakerTurn(start: 0, end: 5, id: 3, name: "Speaker 3")]
        let seg = AudioRecorder.TranscriptSegment(
            text: "a stretch the engine never turned",
            confirmed: true, window: 40.0...50.0)
        let rows = recorder.derivedRows(for: seg, regions: [])
        XCTAssertFalse(rows.isEmpty)
        for row in rows {
            XCTAssertEqual(row.speaker, "Speaker 3",
                           "with the position layer dropped, an uncovered stretch must "
                           + "take the nearest voice turn — never SPEAKER UNKNOWN")
        }
    }

    // MARK: - The MOSS engine (2026-08-18)

    /// ⚠ THE CHANGE THIS FILE EXISTS TO PIN. `Live only` keeps working under the
    /// MOSS engine, where every FIXED mode is still suppressed.
    ///
    /// The owner recorded with `atnd.position.enabled = 1` and
    /// `atnd.position.source = atndLive` — both correct — and got SPEAKER UNKNOWN
    /// plus no position log at all, because an outer `mossActive ? .pyannote :`
    /// answered before the mode could. The suppression is right for the fixed
    /// modes (their beam spans survive into the finished transcript, alongside
    /// MOSS ids) and wrong for this one, which removes itself at Stop.
    func testLiveOnlySurvivesTheMossEngineWhileEveryFixedModeIsStillSuppressed() {
        XCTAssertEqual(PositionSource.atndLiveOnly.effective(recording: true, mossActive: true),
                       .both,
                       "under MOSS, Live only must still label the meeting as it happens — "
                       + "MOSS emits nothing until its first 30 s chunk lands, so suppressing "
                       + "the beam leaves the meeting with no labels at all")
        XCTAssertEqual(PositionSource.atndLiveOnly.effective(recording: false, mossActive: true),
                       .pyannote,
                       "and it must still drop out at Stop — that is what makes it safe here")

        for mode in PositionSource.allCases where mode != .atndLiveOnly {
            for recording in [true, false] {
                XCTAssertEqual(mode.effective(recording: recording, mossActive: true), .pyannote,
                               "\(mode.rawValue) must stay suppressed under MOSS: its spans "
                               + "would reach the finished transcript")
            }
        }
    }

    /// The POSITIVE half. Without MOSS, nothing is suppressed — otherwise a file
    /// that had simply stopped showing the beam anywhere would pass the test above.
    func testWithoutMossNoModeIsSuppressed() {
        for mode in PositionSource.allCases {
            let live = mode.effective(recording: true, mossActive: false)
            XCTAssertEqual(live, mode == .atndLiveOnly ? .both : mode,
                           "\(mode.rawValue) must be untouched when MOSS is not the engine")
        }
    }

    /// The live caption and the rows read ONE resolver, so they cannot disagree.
    /// `usesPosition` is what gates the caption; asserting it against the same
    /// resolved mode is what pins them together.
    func testTheLiveCaptionAgreesWithTheRowsUnderMoss() {
        let recording = PositionSource.atndLiveOnly.effective(recording: true, mossActive: true)
        XCTAssertTrue(recording.usesPosition,
                      "rows show beam labels while recording under MOSS, so the caption must too")
        let stopped = PositionSource.atndLiveOnly.effective(recording: false, mossActive: true)
        XCTAssertFalse(stopped.usesPosition,
                       "and both must go quiet at Stop")
    }

}
