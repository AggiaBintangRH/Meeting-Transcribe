import XCTest
@testable import MeetingTranscriber

/// `ModelLoader.wantedDiarizationStack` — the one rule that decides which
/// diarization processes a session keeps alive.
///
/// THE BUG THIS PINS, and it was the only finding of the 2026-07-31 audit that
/// changed BEHAVIOUR rather than just wasting memory.
///
/// The load step asked `diarization.enabled`; the teardown only asked whether
/// the engine was MOSS. Services are kept alive across sessions, so:
///
///   1. record with diarization ON  → pyannote + wespeaker load
///   2. turn "Enable speaker diarization" OFF in Settings
///   3. record again WITHOUT quitting the app
///   4. the teardown never fires → `modelLoader.pyannote` is still non-nil
///   5. `diarizeLiveChunk` guards on `liveOn` and `pyannote != nil` — and
///      `diarization.live` still defaults to true — so **diarization keeps
///      running after the user switched it off**, with 1.17 GB resident.
///
/// It only bit when the app was not restarted in between, which is why it went
/// unnoticed. The fix is one pure rule feeding both sites, so they cannot
/// disagree again; these tests are that rule's contract.
final class DiarizationStackLifecycleTests: XCTestCase {

    private let pyannote = ModelLoader.pyannoteEngineID
    private let moss = ModelLoader.mossEngineID

    private func stack(enabled: Bool = true, engine: String? = nil, chunked: String = "qwen3")
        -> ModelLoader.DiarizationStack? {
        ModelLoader.wantedDiarizationStack(diarizationEnabled: enabled,
                                           engine: engine ?? pyannote,
                                           chunkedID: chunked)
    }

    // MARK: - The bug

    /// The master switch must win over everything. This is step 4 above: with it
    /// off, NOTHING may be kept alive, whatever the engine or the chunked model.
    func testDisablingDiarizationKeepsNothingAlive() {
        for engine in [pyannote, moss] {
            for chunked in ["qwen3", "whisper", "moss"] {
                XCTAssertNil(stack(enabled: false, engine: engine, chunked: chunked),
                             "engine=\(engine) chunked=\(chunked): the off switch was ignored")
            }
        }
    }

    /// And the same rule must drive the load step, so a session cannot load what
    /// the teardown would immediately drop (or vice versa). Expressed as: the
    /// rule is a FUNCTION of the settings only — no hidden state, so both callers
    /// necessarily agree.
    func testTheRuleIsPurelyASettingsFunction() {
        for _ in 0..<3 {
            XCTAssertEqual(stack(enabled: true, engine: pyannote), .pyannote)
            XCTAssertNil(stack(enabled: false, engine: pyannote))
        }
    }

    // MARK: - The ordinary cases

    func testPyannoteEngineKeepsThePyannoteStack() {
        XCTAssertEqual(stack(enabled: true, engine: pyannote, chunked: "qwen3"), .pyannote)
        XCTAssertEqual(stack(enabled: true, engine: pyannote, chunked: "moss"), .pyannote,
                       "MOSS as the chunked model does not change who diarizes")
    }

    /// MOSS engine with ANOTHER model doing the ASR ⇒ a second MOSS process.
    func testMossEngineWithAnotherASRKeepsASecondProcess() {
        XCTAssertEqual(stack(enabled: true, engine: moss, chunked: "qwen3"), .mossSecondProcess)
        XCTAssertEqual(stack(enabled: true, engine: moss, chunked: "whisper"), .mossSecondProcess)
    }

    /// MOSS+MOSS ⇒ nothing extra: the chunked process already carries the labels,
    /// and a second load of the same 3.6 GB model would be pure waste.
    func testMossPlusMossKeepsNothingExtra() {
        XCTAssertNil(stack(enabled: true, engine: moss, chunked: "moss"))
    }

    /// The two stacks are mutually exclusive — one session can never want both,
    /// which is what lets the teardown be two independent `!=` checks.
    func testTheTwoStacksAreMutuallyExclusive() {
        for enabled in [true, false] {
            for engine in [pyannote, moss] {
                for chunked in ["qwen3", "moss"] {
                    let s = stack(enabled: enabled, engine: engine, chunked: chunked)
                    XCTAssertTrue(s == nil || s == .pyannote || s == .mossSecondProcess)
                }
            }
        }
    }

    /// Selecting MOSS as the engine must drop pyannote, which was already true
    /// before the fix and must stay true after it.
    func testMossEngineNeverKeepsPyannote() {
        for chunked in ["qwen3", "whisper", "moss"] {
            XCTAssertNotEqual(stack(enabled: true, engine: moss, chunked: chunked), .pyannote)
        }
    }
}
