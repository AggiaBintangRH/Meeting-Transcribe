import XCTest
@testable import MeetingTranscriber

/// `tailDiarWatchdogSeconds` — the stop-time TAIL diarization must be given time
/// proportional to what it is actually diarizing.
///
/// THE BUG THIS PINS. The tail watchdog was a flat 120 s, which is generous for
/// the normal tail (one diarization interval, ~30 s of audio) but wrong for the
/// one combination where the "tail" is the WHOLE recording: live labels OFF with
/// "Continue from live labels (tail only)" ON. Nothing clears `chunkAudio` in
/// that mode, so it accumulates from 0 to the end and goes to the sidecar as a
/// single job. A 60-minute meeting cannot be diarized in 120 s, so the watchdog
/// fired, the row was marked "Tail diarization timed out" and the speaker labels
/// were lost — while the sidecar was still working normally.
///
/// The failure was silent in the worst way: nothing was broken, the answer was
/// simply thrown away for being late.
final class TailDiarizationWatchdogTests: XCTestCase {

    private func budget(seconds: Double) -> Double {
        AudioRecorder.tailDiarWatchdogSeconds(sampleCount: Int(seconds * 16_000))
    }

    /// An ordinary tail is well under the floor, so the floor is what applies —
    /// this is the behaviour that was already correct and must not regress.
    func testAnOrdinaryTailKeepsTheGenerousFloor() {
        XCTAssertEqual(budget(seconds: 0), 120)
        XCTAssertEqual(budget(seconds: 30), 120)
        XCTAssertEqual(budget(seconds: 119), 120)
    }

    /// The case that was broken: the whole recording arriving as one "tail".
    func testAWholeRecordingTailGetsRealtimeOrBetter() {
        XCTAssertEqual(budget(seconds: 3600), 3600, accuracy: 1e-6,
                       "a 60-minute tail used to be given 120 s")
        XCTAssertGreaterThan(budget(seconds: 3600), 120)
    }

    /// At least 1x realtime for any length — the same shape `startDiarization`
    /// uses for the full pass (`max(180, recordingElapsed)`).
    func testBudgetIsNeverLessThanRealtime() {
        for seconds in [0.0, 30, 120, 600, 1800, 4027] {
            XCTAssertGreaterThanOrEqual(budget(seconds: seconds), seconds,
                                        "\(seconds)s of audio got less than realtime")
        }
    }

    /// Monotonic: more audio can never mean less time.
    func testMoreAudioNeverMeansLessTime() {
        let lengths = [0.0, 10, 60, 300, 1200, 3600]
        for (a, b) in zip(lengths, lengths.dropFirst()) {
            XCTAssertLessThanOrEqual(budget(seconds: a), budget(seconds: b))
        }
    }

    /// A non-16 kHz rate must still give a sane number — the helper takes the
    /// rate so a caller cannot accidentally read samples as seconds.
    func testRateIsHonouredRatherThanAssumed() {
        XCTAssertEqual(AudioRecorder.tailDiarWatchdogSeconds(sampleCount: 48_000 * 600,
                                                             sampleRate: 48_000),
                       600, accuracy: 1e-6)
    }
}
