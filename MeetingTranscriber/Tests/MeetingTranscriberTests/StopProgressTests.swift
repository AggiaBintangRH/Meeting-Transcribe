import XCTest
@testable import MeetingTranscriber

/// The stop overlay's progress bar and its estimate.
///
/// ⚠ WHY THIS EXISTS AT ALL. Until 2026-09-04 the stop pass was tail-only and
/// lasted seconds, so a spinner was an honest description of it. The full pass
/// runs for MINUTES — measured 2.1 min for Qwen3 and 10.2 for MOSS on a
/// 60-minute meeting — and the overlay is modal, so this is the screen the user
/// is held on. A wrong number here is not cosmetic: it is the only thing telling
/// them whether to wait or whether the app has hung.
final class StopProgressTests: XCTestCase {

    private func p(_ done: Int, _ total: Int, _ elapsed: Double) -> AudioRecorder.StopProgress {
        AudioRecorder.StopProgress(done: done, total: total, elapsed: elapsed)
    }

    // MARK: - The estimate refuses to guess

    /// 🔴 THE CASE THE DESIGN IS BUILT AROUND. The first window carries the
    /// model's warm-up — measured today at 17.3 s for MOSS's first window
    /// against ~5 s for its later ones — so an estimate drawn from one window
    /// overstates the wait by multiples and then visibly collapses. A bar that
    /// says "about 34 min left" and reaches zero in 10 reads as broken.
    func testNoEstimateBeforeTwoWindowsHaveFinished() {
        XCTAssertNil(p(0, 120, 0).secondsRemaining, "nothing has finished yet")
        XCTAssertNil(p(1, 120, 17.3).secondsRemaining,
                     "one window is warm-up, not a rate — it must abstain")
        XCTAssertNotNil(p(2, 120, 22.0).secondsRemaining,
                        "two windows is the first honest sample")
    }

    /// A finished pass has nothing left to estimate, and must not report "0 s
    /// left" as though it were still counting down.
    func testAFinishedPassOffersNoEstimate() {
        XCTAssertNil(p(120, 120, 300).secondsRemaining)
        XCTAssertEqual(p(120, 120, 300).fraction, 1.0, accuracy: 1e-9)
    }

    /// The rate is MEASURED, not a stored per-model table. `fullPassCostNote`'s
    /// figures were wrong by up to 11x for exactly that reason — a table cannot
    /// know this machine, this recording, or what else is using the GPU now.
    func testTheEstimateIsTheMeasuredRateSoFar() {
        // 10 windows in 50 s = 5 s each; 110 remain.
        let left = p(10, 120, 50).secondsRemaining
        XCTAssertEqual(left ?? -1, 550, accuracy: 0.001)
        // The SAME position on a slower machine must give a bigger number.
        let slower = p(10, 120, 200).secondsRemaining
        XCTAssertEqual(slower ?? -1, 2200, accuracy: 0.001)
    }

    // MARK: - The bar cannot draw nonsense

    /// `fraction` feeds a view width. A value outside 0…1 draws a bar past its
    /// own track, and a negative one is a crash in some layout paths.
    func testTheFractionIsAlwaysDrawable() {
        for (done, total) in [(0, 10), (5, 10), (10, 10), (11, 10), (-1, 10), (3, 0)] {
            let f = p(done, total, 1).fraction
            XCTAssertTrue(f >= 0 && f <= 1, "done \(done) of \(total) gave \(f)")
        }
    }

    /// Windows FINISHED, never the one in flight. Counting the current window as
    /// done would report 100 % while the last one is still running.
    func testTheBarNeverReachesFullWhileWorkRemains() {
        XCTAssertLessThan(p(119, 120, 500).fraction, 1.0)
    }

    // MARK: - The words

    /// Coarse on purpose: nobody watching a ten-minute pass wants a second-by-
    /// second countdown, and the false precision would imply the estimate is
    /// better than it is.
    func testTheDurationReadsLikeSomethingAPersonSays() {
        XCTAssertEqual(OverlayStepRow.durationPhrase(3), "a few seconds")
        XCTAssertEqual(OverlayStepRow.durationPhrase(37), "35 s")
        XCTAssertEqual(OverlayStepRow.durationPhrase(120), "2 min")
        XCTAssertEqual(OverlayStepRow.durationPhrase(220), "3 min 30 s")
        XCTAssertEqual(OverlayStepRow.durationPhrase(1500), "25 min")
    }

    /// It never returns an empty string or a bare number, at any input the
    /// estimate can produce — including the absurd ones a stalled pass would.
    func testEveryDurationSaysAUnit() {
        for s in stride(from: 0.0, through: 7200, by: 7) {
            let phrase = OverlayStepRow.durationPhrase(s)
            XCTAssertFalse(phrase.isEmpty, "\(s)")
            XCTAssertTrue(phrase.contains("s") || phrase.contains("min"), "\(s) -> \(phrase)")
        }
    }

    // MARK: - The monotonic clock

    /// ⚠ A stop pass can run for ten minutes. Timed against a wall clock, an NTP
    /// correction or a DST jump inside that window produces a NEGATIVE estimate,
    /// and "about -2 min left" is worse than no estimate at all.
    func testElapsedIsMonotonicAndNeverNegative() {
        let start = AudioRecorder.monotonicNow()
        XCTAssertGreaterThanOrEqual(AudioRecorder.monotonicSeconds(since: start), 0)
        // A start in the future — the shape a bad caller would produce — clamps
        // to zero rather than becoming a negative elapsed.
        let future = AudioRecorder.monotonicNow() + 60_000_000_000
        XCTAssertEqual(AudioRecorder.monotonicSeconds(since: future), 0, accuracy: 1e-9)
    }
}
