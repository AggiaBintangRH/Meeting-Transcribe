import XCTest
@testable import MeetingTranscriber

/// The far end's realtime lane is flushed when the far end stops talking
/// (owner, 2026-08-13).
///
/// **The audit that produced this.** `logs/row-order.log` for the reported session
/// contained no `onTranscript` entry at all — the remote lane never emitted a
/// FINAL during recording, so the previous fix (finals become rows) could not
/// fire. Reading the flush sites explains why: the office lane is flushed on an
/// ATND cluster change (`onClusterChange`) AND at the chunk boundary; the remote
/// lane only ever had the boundary. At the owner's 120 s interval that is one
/// final every two minutes, so the far end's speech stayed in one caption card
/// that kept growing — and captions render below the row list, so each new office
/// row appeared above it.
///
/// The office trigger is deliberately not shared: a beam change describes the
/// ROOM. Silence is the one signal that is genuinely the far end's own.
@MainActor
final class RemoteUtteranceGateTests: XCTestCase {

    private let loud: Float = 0.05      // well above remoteSilenceRMS (0.004)
    private let quiet: Float = 0.0

    /// One flush per utterance, at the moment the pause reaches the threshold.
    func testItFiresOnceWhenSpeechIsFollowedByAPause() {
        var gate = AudioRecorder.RemoteUtteranceGate()
        XCTAssertFalse(gate.note(level: loud, duration: 0.5), "still talking")
        XCTAssertFalse(gate.note(level: quiet, duration: 0.5), "0.5 s is not a pause yet")
        XCTAssertTrue(gate.note(level: quiet, duration: 0.5), "1.0 s reached — utterance over")
        XCTAssertFalse(gate.note(level: quiet, duration: 5.0),
                       "one final per utterance, not one per second of quiet")
    }

    /// THE IDLE-CHANNEL CASE, and the reason `pending` exists. A remote channel
    /// carrying nothing must never flush — the owner's own logs are full of
    /// `SKIP remote … near-silent (rms 0.00000)`, and a flush per second there
    /// would be a standing cost for guaranteed-empty text.
    func testAnIdleChannelNeverFlushes() {
        var gate = AudioRecorder.RemoteUtteranceGate()
        for _ in 0..<200 {
            XCTAssertFalse(gate.note(level: quiet, duration: 0.5))
        }
    }

    /// A pause that is interrupted does not count — the far end is still on the
    /// same utterance, and cutting it would split one sentence across two rows.
    func testAnInterruptedPauseResets() {
        var gate = AudioRecorder.RemoteUtteranceGate()
        _ = gate.note(level: loud, duration: 0.5)
        XCTAssertFalse(gate.note(level: quiet, duration: 0.9))
        XCTAssertFalse(gate.note(level: loud, duration: 0.1), "speech resumed")
        XCTAssertFalse(gate.note(level: quiet, duration: 0.9),
                       "the pause starts again from zero")
        XCTAssertTrue(gate.note(level: quiet, duration: 0.2))
    }

    /// Two utterances give two flushes — which is what makes two rows, which is
    /// what lets the far end interleave with the room.
    func testEachUtteranceGetsItsOwnFlush() {
        var gate = AudioRecorder.RemoteUtteranceGate()
        var flushes = 0
        for _ in 0..<3 {
            _ = gate.note(level: loud, duration: 1.0)
            if gate.note(level: quiet, duration: 1.0) { flushes += 1 }
        }
        XCTAssertEqual(flushes, 3)
    }

    /// The threshold is the SAME 1.0 s that decides where one utterance ends when
    /// a chunk is split. "When has someone finished speaking" is one question, and
    /// two constants would be two answers that could drift apart.
    func testTheThresholdIsTheSharedUtterancePause() {
        var gate = AudioRecorder.RemoteUtteranceGate()
        _ = gate.note(level: loud, duration: 0.1)
        XCTAssertFalse(gate.note(level: quiet,
                                 duration: AudioRecorder.utterancePauseSec - 0.01))
        var gate2 = AudioRecorder.RemoteUtteranceGate()
        _ = gate2.note(level: loud, duration: 0.1)
        XCTAssertTrue(gate2.note(level: quiet, duration: AudioRecorder.utterancePauseSec))
    }

    /// The level bar is `remoteSilenceRMS`, the one that already decides a remote
    /// chunk is silent — so "quiet" means the same thing in both places.
    func testTheLevelBarIsTheSharedSilenceThreshold() {
        var gate = AudioRecorder.RemoteUtteranceGate()
        XCTAssertFalse(gate.note(level: AudioRecorder.remoteSilenceRMS, duration: 0.1),
                       "exactly at the bar counts as speech, as it does for chunks")
        XCTAssertTrue(gate.note(level: AudioRecorder.remoteSilenceRMS - 0.001,
                                duration: 1.0))
    }

    /// A new session starts clean, or the first utterance of the next meeting is
    /// judged against the last one's silence.
    func testResetClearsBothHalves() {
        var gate = AudioRecorder.RemoteUtteranceGate()
        _ = gate.note(level: loud, duration: 1.0)
        _ = gate.note(level: quiet, duration: 0.5)
        gate.reset()
        XCTAssertEqual(gate, AudioRecorder.RemoteUtteranceGate())
        XCTAssertFalse(gate.note(level: quiet, duration: 5.0),
                       "nothing pending after a reset")
    }
}
