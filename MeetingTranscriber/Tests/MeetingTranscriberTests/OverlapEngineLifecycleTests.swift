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
    private let spectral = ModelLoader.spectralEngineID
    private let mossformer2 = ModelCatalog.overlapSeparation.id
    private let dicow = ModelCatalog.overlapDicow.id

    /// The ordinary cases: repair on under pyannote keeps exactly the engine the
    /// owner picked, and nothing else.
    func testTheSelectedEngineIsTheOneKept() {
        XCTAssertEqual(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                       engineID: mossformer2,
                                                       diarEngine: pyannote,
                                                       detectEnabled: false),
                       mossformer2)
        XCTAssertEqual(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                       engineID: dicow,
                                                       diarEngine: pyannote,
                                                       detectEnabled: false),
                       dicow)
    }

    /// Turning the feature off must drop BOTH — this is the case that used to
    /// leave a sidecar resident until the app quit.
    func testDisablingRepairKeepsNeither() {
        XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: false,
                                                     engineID: mossformer2,
                                                     diarEngine: pyannote,
                                                     detectEnabled: true))
        XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: false,
                                                     engineID: dicow,
                                                     diarEngine: pyannote,
                                                     detectEnabled: true))
    }

    /// MOSS and spectral cannot LOCATE overlap in their own turns — MOSS's
    /// segments tile exactly, spectral assigns one label per frame — so with the
    /// detector off there is nothing for either repair engine to work on and both
    /// must be dropped. This was the whole rule until the standalone overlap
    /// detector existed; it is now the detector-off half of it.
    ///
    /// The teardown direction is what makes it matter: switching to MOSS used to
    /// leave a loaded engine running with nothing able to ask it anything.
    func testMossAndSpectralKeepNeitherWhileTheDetectorIsOff() {
        for diar in [moss, spectral] {
            for engine in [mossformer2, dicow] {
                XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                             engineID: engine,
                                                             diarEngine: diar,
                                                             detectEnabled: false),
                             "\(diar) cannot locate overlap without the detector")
            }
        }
    }

    /// THE OTHER HALF, and the reason `detectEnabled` is a parameter at all. The
    /// detector reads the audio directly and needs no turns, so under MOSS and
    /// spectral it is what gives repair somewhere to run. Switching it on must
    /// therefore hand back the engine the owner picked — the same engine, not a
    /// substitute.
    func testTheDetectorGivesMossAndSpectralTheirRepairEngineBack() {
        for diar in [moss, spectral] {
            for engine in [mossformer2, dicow] {
                XCTAssertEqual(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                               engineID: engine,
                                                               diarEngine: diar,
                                                               detectEnabled: true),
                               engine,
                               "\(diar) + detector must keep the engine that was chosen")
            }
        }
    }

    /// The detector can never turn repair ON by itself, under any engine. It
    /// decides WHERE repair may run, never WHETHER — the feature switch does that,
    /// and a detector that quietly started a 6 GB DiCoW would be the substitution
    /// this project forbids.
    func testTheDetectorCannotSwitchRepairOn() {
        for diar in [pyannote, moss, spectral] {
            for engine in [mossformer2, dicow] {
                XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: false,
                                                             engineID: engine,
                                                             diarEngine: diar,
                                                             detectEnabled: true))
            }
        }
    }

    /// pyannote marks overlap in its own turns, so its regions never depended on
    /// the detector. Repair there must be byte-for-byte unaffected by that switch
    /// — the detector is a second opinion under pyannote, never a precondition.
    func testPyannoteRepairIgnoresTheDetectorEntirely() {
        for engine in [mossformer2, dicow] {
            XCTAssertEqual(
                ModelLoader.wantedOverlapEngine(repairEnabled: true, engineID: engine,
                                                diarEngine: pyannote, detectEnabled: false),
                ModelLoader.wantedOverlapEngine(repairEnabled: true, engineID: engine,
                                                diarEngine: pyannote, detectEnabled: true))
        }
    }

    /// The engines are mutually exclusive, so the rule may never name both. This
    /// is the switching case: picking DiCoW must not leave MossFormer2 wanted,
    /// which is how both ended up resident at once.
    func testTheRuleNeverWantsBothEngines() {
        for engine in [mossformer2, dicow] {
            let wanted = ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                         engineID: engine,
                                                         diarEngine: pyannote,
                                                         detectEnabled: false)
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
