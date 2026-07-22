import XCTest
@testable import MeetingTranscriber

/// The remote stop-pass branch (2026-07-22): Remote now honours
/// `diarization.continueOnStop` exactly as Office does, reversing phase 4's
/// "remote is ALWAYS a full re-diarization" rule. The motivation was label
/// stability — a full pass re-embeds voices the live passes already enrolled and
/// was observed minting a second profile (R2) for a voice that had matched
/// profile 1 at sim=0.89 seconds earlier. (That similarity collapse is still
/// unexplained; tail mode avoids it structurally rather than by fixing it.)
///
/// Everything decided by `AudioRecorder.remoteStopMode` is pure, so the branch —
/// and, crucially, WHICH branches take the stop gate — is testable without an
/// engine, a sidecar or an Aggregate Device.
final class RemoteStopModeTests: XCTestCase {

    /// One second of 16 kHz audio is the early-out threshold; anything at or
    /// below it is "no tail worth diarizing".
    private let oneSecond = 16_000

    private func mode(finalPass: Bool = true,
                      continueOnStop: Bool = true,
                      remoteStreamActive: Bool = true,
                      hasService: Bool = true,
                      hasRecording: Bool = true,
                      tailSamples: Int = 160_000) -> AudioRecorder.RemoteStopMode {
        AudioRecorder.remoteStopMode(finalPass: finalPass,
                                     continueOnStop: continueOnStop,
                                     remoteStreamActive: remoteStreamActive,
                                     hasDiarizationService: hasService,
                                     hasRemoteRecording: hasRecording,
                                     tailSamples: tailSamples)
    }

    // MARK: - 1. Single-stream sessions are completely unaffected

    /// THE regression bar: with no Remote channel there is no remote pass, no
    /// overlay row and no watchdog — whatever the other settings say.
    func testSingleStreamNeverRunsARemotePass() {
        for finalPass in [true, false] {
            for continueOnStop in [true, false] {
                XCTAssertEqual(mode(finalPass: finalPass,
                                    continueOnStop: continueOnStop,
                                    remoteStreamActive: false),
                               .none,
                               "finalPass=\(finalPass) continueOnStop=\(continueOnStop)")
            }
        }
    }

    // MARK: - 2. The branch itself

    /// The default (continueOnStop on) is now the TAIL, not a full pass — this is
    /// the actual behaviour change.
    func testDefaultsToTailSoLiveLabelsSurviveStop() {
        XCTAssertEqual(mode(), .tail)
    }

    /// continueOnStop off keeps phase 4's behaviour verbatim.
    func testContinueOffStillRunsTheFullPass() {
        XCTAssertEqual(mode(continueOnStop: false), .full)
    }

    /// The full pass needs the Remote WAV; the tail does not (it diarizes the
    /// samples still in memory), so a missing file only blocks the full branch.
    func testFullNeedsTheRemoteWAVButTailDoesNot() {
        XCTAssertEqual(mode(continueOnStop: false, hasRecording: false), .none)
        XCTAssertEqual(mode(continueOnStop: true, hasRecording: false), .tail)
    }

    /// Final pass disabled → no stop pass at all, either mode.
    func testFinalPassOffRunsNothing() {
        XCTAssertEqual(mode(finalPass: false, continueOnStop: true), .none)
        XCTAssertEqual(mode(finalPass: false, continueOnStop: false), .none)
    }

    /// No diarization model loaded → nothing to dispatch to.
    func testNoServiceRunsNothing() {
        XCTAssertEqual(mode(continueOnStop: true, hasService: false), .none)
        XCTAssertEqual(mode(continueOnStop: false, hasService: false), .none)
    }

    /// The `< 1 s tail` early-out, mirroring `diarizeTailChunk`'s. Strictly
    /// greater than 16 000 samples, so exactly one second does NOT dispatch.
    func testShortTailIsSkippedRatherThanDispatched() {
        XCTAssertEqual(mode(tailSamples: 0), .none)
        XCTAssertEqual(mode(tailSamples: oneSecond), .none, "exactly 1s is still skipped")
        XCTAssertEqual(mode(tailSamples: oneSecond + 1), .tail)
    }

    /// A tiny tail must not fall back to a full pass — that would reintroduce the
    /// duplicate-profile failure this change exists to remove.
    func testShortTailDoesNotFallBackToFullPass() {
        XCTAssertNotEqual(mode(tailSamples: 10), .full)
    }

    /// Tail-mode dispatch does not consult the WAV at all, so the full-pass
    /// inputs cannot change the tail decision.
    func testTailDecisionIgnoresFullPassInputs() {
        for hasRecording in [true, false] {
            XCTAssertEqual(mode(continueOnStop: true, hasRecording: hasRecording), .tail)
        }
    }

    // MARK: - 3. The stop gate can never hang

    /// The blocking stop overlay clears only when `remoteFinalDiarDone` is true.
    /// `startRemoteDiarization` sets it false IF AND ONLY IF it returns true (a
    /// pass was dispatched), and returns true exactly when the mode is not
    /// `.none`. So enumerate the whole input space and assert the invariant:
    /// either the gate is never taken, or a job exists that will settle it.
    ///
    /// The four ways a dispatched pass settles the gate are all funnelled through
    /// `completeRemoteDiarization` (idempotent, cancels the watchdog):
    ///   tail: chunk result matching `awaitingRemoteTailWindowStart` · sidecar
    ///         error on the remote stream · temp-WAV write failure · watchdog
    ///   full: final result · sidecar error on the remote stream · missing-WAV
    ///         belt-and-braces branch · watchdog
    func testEveryBranchEitherSkipsTheGateOrHasAJobToSettleIt() {
        var seen = Set<AudioRecorder.RemoteStopMode>()
        for finalPass in [true, false] {
            for continueOnStop in [true, false] {
                for active in [true, false] {
                    for hasService in [true, false] {
                        for hasRecording in [true, false] {
                            for tail in [0, oneSecond, oneSecond + 1, 1_000_000] {
                                let m = mode(finalPass: finalPass,
                                             continueOnStop: continueOnStop,
                                             remoteStreamActive: active,
                                             hasService: hasService,
                                             hasRecording: hasRecording,
                                             tailSamples: tail)
                                seen.insert(m)
                                // Gate taken  <=>  a job was dispatched.
                                let gateTaken = (m != .none)
                                let dispatched = (m == .tail) || (m == .full)
                                XCTAssertEqual(gateTaken, dispatched)
                                // A dispatched full pass always has its WAV; a
                                // dispatched tail always has audio to send. No
                                // branch dispatches with nothing to work on.
                                if m == .full { XCTAssertTrue(hasRecording) }
                                if m == .tail { XCTAssertGreaterThan(tail, oneSecond) }
                                if m != .none {
                                    XCTAssertTrue(finalPass && active && hasService)
                                }
                            }
                        }
                    }
                }
            }
        }
        XCTAssertEqual(seen, [.none, .tail, .full], "all three branches are reachable")
    }
}
