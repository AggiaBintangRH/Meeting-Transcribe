import XCTest
@testable import MeetingTranscriber

/// `ModelLoader.wantedOverlapEngine` — the single rule that decides which
/// overlap-repair sidecar this session keeps alive.
///
/// WHY THESE EXIST. Until 2026-07-31 `overlapRepair` and `dicowRepair` had no
/// `terminate()` call anywhere in the app. Every other persistent service had
/// one; these two were simply missed, and the leak was invisible because both
/// engines are off by default and neither is used during the meeting — they run
/// only at Stop. The measured consequence was a resident, unreachable process:
/// DiCoW is a ~6 GB model in its own interpreter.
///
/// The rule now feeds BOTH the load step and the teardown, so the tests below
/// are equally tests of "does the app start the right one" and "does the app
/// stop the wrong one". A regression in either direction fails here.
final class OverlapEngineLifecycleTests: XCTestCase {

    private let pyannote = ModelLoader.pyannoteEngineID
    private let moss = ModelLoader.mossEngineID
    private let mossformer2 = ModelCatalog.overlapSeparation.id
    private let dicow = ModelCatalog.overlapDicow.id

    /// The ordinary cases: repair on under pyannote keeps exactly the engine the
    /// owner picked, and nothing else.
    func testTheSelectedEngineIsTheOneKept() {
        XCTAssertEqual(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                       engineID: mossformer2,
                                                       diarEngine: pyannote),
                       mossformer2)
        XCTAssertEqual(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                       engineID: dicow,
                                                       diarEngine: pyannote),
                       dicow)
    }

    /// Turning the feature off must drop BOTH — this is the case that used to
    /// leave a sidecar resident until the app quit.
    func testDisablingRepairKeepsNeither() {
        XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: false,
                                                     engineID: mossformer2,
                                                     diarEngine: pyannote))
        XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: false,
                                                     engineID: dicow,
                                                     diarEngine: pyannote))
    }

    /// The MOSS diarization engine cannot run overlap repair at all: both engines
    /// locate their windows from pyannote turns, and a MOSS session produces
    /// none. The load step always knew this; the teardown did not, so switching
    /// to MOSS left a loaded engine running with nothing able to ask it anything.
    func testMossDiarizationEngineKeepsNeitherEvenWhenRepairIsOn() {
        XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                     engineID: mossformer2,
                                                     diarEngine: moss))
        XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                     engineID: dicow,
                                                     diarEngine: moss))
    }

    /// The engines are mutually exclusive, so the rule may never name both. This
    /// is the switching case: picking DiCoW must not leave MossFormer2 wanted,
    /// which is how both ended up resident at once.
    func testTheRuleNeverWantsBothEngines() {
        for engine in [mossformer2, dicow] {
            let wanted = ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                         engineID: engine,
                                                         diarEngine: pyannote)
            let keepsMossformer2 = wanted == mossformer2
            let keepsDicow = wanted == dicow
            XCTAssertFalse(keepsMossformer2 && keepsDicow,
                           "one session may keep at most one overlap engine alive")
            XCTAssertTrue(keepsMossformer2 || keepsDicow,
                          "with repair on under pyannote, exactly one must be kept")
        }
    }

    /// The two ids must stay distinct, because the teardown tells the engines
    /// apart by comparing against them. If they ever collided, one engine would
    /// be torn down every time the other was selected.
    func testTheTwoEngineIDsAreDistinct() {
        XCTAssertNotEqual(mossformer2, dicow)
        XCTAssertEqual(ModelCatalog.overlapEngines.count, 2,
                       "a third engine needs its own teardown branch in ModelLoader.prepare")
    }
}
