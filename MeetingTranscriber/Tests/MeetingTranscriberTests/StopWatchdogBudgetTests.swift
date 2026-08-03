import XCTest
@testable import MeetingTranscriber

/// `stopWatchdogSeconds` — the stop overlay must always be willing to wait for
/// the work the session actually scheduled.
///
/// WHY THIS FILE EXISTS. Two bugs on 2026-07-31 were the same shape, and the
/// second was found only because the first taught us the pattern:
///
///   1. the TAIL diarization watchdog was a flat 120 s. In the one mode where
///      the "tail" is the entire recording, a 60-minute meeting was marked
///      "Tail diarization timed out" and its speaker labels thrown away — while
///      the sidecar was working normally.
///   2. the stop budget counted only the CHUNKED full pass. A session that
///      re-labelled with MOSS while transcribing on tail got the 600 s floor for
///      roughly 13 minutes of work, so the overlay would have given up mid-pass.
///
/// Three stop-time passes exist now (chunked full, MOSS re-diarization, and the
/// diarization stop pass). A FOURTH would repeat this a third time, so the rule
/// is pinned rather than remembered: **any scheduled pass must move the budget.**
final class StopWatchdogBudgetTests: XCTestCase {

    private func budget(chunked: Int = 0, moss: Int = 0) -> Double {
        AudioRecorder.stopWatchdogSeconds(chunkedFullPassWindows: chunked,
                                          mossFullPassWindows: moss)
    }

    /// THE INVARIANT: the budget is never LESS than the time the scheduled work
    /// needs. That — not "bigger than the floor" — is the property both bugs
    /// violated.
    ///
    /// The first draft of this test asserted "any pass raises the budget above
    /// the floor" and FAILED honestly on a 1-window pass: 120·1 + 60 + 300 = 480,
    /// which genuinely fits inside the 600 s floor. A small pass needing less
    /// than the floor is correct; a pass needing MORE and not getting it is the
    /// bug. The assertion is written to catch the second without forbidding the
    /// first.
    func testTheBudgetAlwaysCoversTheScheduledWork() {
        XCTAssertEqual(budget(), 600, "no pass scheduled ⇒ the plain floor")

        for windows in [1, 5, 30, 120] {
            let needed = AudioRecorder.fullPassWatchdogSeconds(windowCount: windows)
            XCTAssertGreaterThanOrEqual(budget(chunked: windows), needed,
                                        "\(windows) chunked windows are not covered")
            XCTAssertGreaterThanOrEqual(budget(moss: windows), needed,
                                        "\(windows) MOSS windows are not covered — this WAS the bug")
        }
    }

    /// The floor may not swallow work that exceeds it — the specific shape of
    /// bug #2, where ~13 minutes of MOSS re-labelling sat under a 600 s ceiling.
    func testWorkLargerThanTheFloorIsNotSwallowedByIt() {
        for windows in [10, 30, 120] {
            let needed = AudioRecorder.fullPassWatchdogSeconds(windowCount: windows)
            guard needed > 600 else { continue }
            XCTAssertGreaterThan(budget(moss: windows), 600,
                                 "\(windows) windows need \(needed)s but got the floor")
            XCTAssertGreaterThan(budget(chunked: windows), 600)
        }
    }

    /// Both passes can run in one session, and they contend for the same GPU, so
    /// the budget must reflect BOTH rather than whichever is larger.
    func testTwoPassesCostMoreThanEitherAlone() {
        let both = budget(chunked: 30, moss: 30)
        XCTAssertGreaterThan(both, budget(chunked: 30))
        XCTAssertGreaterThan(both, budget(moss: 30))
    }

    /// Monotonic in each input: more work can never mean less patience.
    func testBudgetNeverShrinksAsWorkGrows() {
        let counts = [0, 1, 10, 60, 240]
        for (a, b) in zip(counts, counts.dropFirst()) {
            XCTAssertLessThanOrEqual(budget(chunked: a), budget(chunked: b))
            XCTAssertLessThanOrEqual(budget(moss: a), budget(moss: b))
        }
    }

    /// A real 60-minute meeting must not be cut off. MOSS re-diarizes in 120 s
    /// windows (30 of them) at roughly 26 s each ≈ 13 minutes — comfortably over
    /// the 600 s floor that used to apply.
    func testASixtyMinuteMossRelabelIsNotCutOff() {
        let windows = AudioRecorder.fullPassWindows(recordingLength: 3600,
                                                    intervalSec: AudioRecorder.mossFullPassWindowSec)
        XCTAssertEqual(windows.count, 30)
        XCTAssertGreaterThan(budget(moss: windows.count), 13 * 60,
                             "the pass takes ~13 minutes; the overlay must outlast it")
    }

    /// Negative or nonsense counts must not shrink the budget below the floor —
    /// a wrong count should degrade to "wait the normal amount", never to "give
    /// up immediately".
    func testNonsenseCountsFallBackToTheFloor() {
        XCTAssertEqual(budget(chunked: -5, moss: -5), 600)
        XCTAssertEqual(budget(chunked: 0, moss: 0), 600)
    }

    /// The floor itself is the pre-existing behaviour and must not regress: a
    /// session with no stop-time pass still waits the 10 minutes it always did.
    func testTheFloorIsUnchangedForAnOrdinarySession() {
        XCTAssertEqual(budget(), 600)
    }
}
