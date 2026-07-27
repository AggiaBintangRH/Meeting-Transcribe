import Combine
import XCTest
@testable import MeetingTranscriber

/// The ATND beam follows ANY sound, not only speech. Measured on the owner's
/// array: piano through the room speakers still produced angle/rotation notices
/// with the array's own VAD (`SVAD`) switched ON — that setting does not gate
/// camera tracking. Ungated, music or a slammed door builds a direction cluster
/// for a position where nobody ever spoke.
///
/// So `PositionDiarizer` collects direction only while OUR Silero VAD reports
/// speech. These tests cover both halves of that, because the dangerous failure
/// is not "noise got in" — it is the gate being too tight and swallowing a real
/// speaker, leaving a meeting with no position data at all.
@MainActor
final class PositionSpeechGateTests: XCTestCase {

    /// Feeds beam notices through the shared service the way the socket does.
    private func sendTalking(rotation: Int, times: Int) {
        for _ in 0..<times {
            ATNDBeamService.shared.rawNotices.send(
                (date: .now, notice: .talking(.init(channel: 1, elevation: 45, rotation: rotation))))
        }
    }

    /// A diarizer on a clock we control, so speech and notices can be placed
    /// exactly where the test wants them.
    private func makeDiarizer(gate: Bool, clock: @escaping () -> Double) -> PositionDiarizer {
        let d = PositionDiarizer()
        d.start(tauDeg: 15, smoothingSec: 0.0, mode: .firstCome,
                gateOnSpeech: gate, now: clock)
        return d
    }

    // MARK: - 1. Noise with no speech is not collected

    func testNoiseWithoutSpeechCollectsNothing() {
        var now: Double = 10
        let diarizer = makeDiarizer(gate: true) { now }
        defer { diarizer.stop() }

        // Piano: the array reports a steady direction, our VAD never says speech.
        sendTalking(rotation: 90, times: 40)
        now = 14

        XCTAssertEqual(diarizer.sampleCount(in: 10...14), 0,
                       "Direction with no speech behind it must not become position data.")
        XCTAssertNil(diarizer.label(for: 10...14))
    }

    // MARK: - 2. Real speech IS collected

    func testSpeechIsCollectedNormally() {
        var now: Double = 10
        let diarizer = makeDiarizer(gate: true) { now }
        defer { diarizer.stop() }

        diarizer.noteSpeech(true, at: 10)
        sendTalking(rotation: 90, times: 40)
        now = 10.5

        XCTAssertGreaterThan(diarizer.sampleCount(in: 10...11), 0,
                             "A real talker must still produce position samples.")
    }

    /// The hold window is the whole reason a normal utterance stays continuous:
    /// speech verdicts arrive per audio buffer, notices at 10 Hz, so requiring
    /// them to coincide exactly would punch holes on every inter-word pause.
    func testBriefPauseWithinHoldKeepsCollecting() {
        var now: Double = 10
        let diarizer = makeDiarizer(gate: true) { now }
        defer { diarizer.stop() }

        diarizer.noteSpeech(true, at: 10)
        now = 10.6                       // inside the 1 s hold
        sendTalking(rotation: 90, times: 20)

        XCTAssertGreaterThan(diarizer.sampleCount(in: 10...11), 0,
                             "A pause shorter than the hold must not cut the utterance.")
    }

    func testNoiseLongAfterSpeechIsRejected() {
        var now: Double = 10
        let diarizer = makeDiarizer(gate: true) { now }
        defer { diarizer.stop() }

        diarizer.noteSpeech(true, at: 10)
        now = 30                          // far outside the hold: speech long over
        sendTalking(rotation: 200, times: 40)

        XCTAssertEqual(diarizer.sampleCount(in: 29...31), 0,
                       "Noise well after the last speech must not build a position.")
    }

    // MARK: - 3. VAD off = no verdict to gate on, so no gate

    func testGateDisabledCollectsWithoutAnySpeechVerdict() {
        var now: Double = 10
        let diarizer = makeDiarizer(gate: false) { now }
        defer { diarizer.stop() }

        // No noteSpeech at all — mirrors VAD switched off in Settings.
        sendTalking(rotation: 90, times: 40)
        now = 10.5

        XCTAssertGreaterThan(diarizer.sampleCount(in: 10...11), 0,
                             "With VAD off there is no verdict to gate on; behaviour must be unchanged.")
    }
}
