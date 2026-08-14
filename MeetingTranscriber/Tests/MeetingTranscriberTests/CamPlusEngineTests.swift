import XCTest
@testable import MeetingTranscriber

/// The CAM++ diarization engine (the sixth, 2026-08-11).
///
/// Mirrors `DiarizenEngineTests` rather than sharing a harness with it, and that
/// is deliberate: those two suites exist to catch the failure mode that produced
/// them both — an engine added by COPYING its predecessor's Swift layer and
/// renaming the class, which shipped `ServiceError.scriptMissing` naming
/// `scripts/nemo/nemo-service.py` from DiariZen and a `configureDiarizen()` that
/// logged `engine=nemo`. A shared harness would be one more thing two engines
/// have in common, when what needs asserting is what they do NOT.
final class CamPlusEngineTests: XCTestCase {

    // MARK: - Routing: its own sidecar, its own log

    /// The script/log pair, both LITERALS. `Config.mossDiarization()` derived its
    /// log name from a model type and a rename silently repointed one process's
    /// stderr into another service's file — while both processes kept working,
    /// which is exactly why it was invisible.
    func testConfigNamesItsOwnSidecarAndLogAsLiterals() {
        XCTAssertEqual(CamPlusService.Config.scriptName, "campplus/campplus-service.py")
        XCTAssertEqual(CamPlusService.Config.logName, "campplus")
    }

    /// The named script must actually exist. Missing, this is a
    /// `ServiceError.scriptMissing` in front of the user at session start.
    func testTheNamedSidecarExistsOnDisk() {
        let scripts = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("scripts")
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: scripts.appendingPathComponent(
                CamPlusService.Config.scriptName).path))
        // The vendored architecture too: without it the checkpoint on disk is
        // unloadable, and on a whole-file engine that failure lands AFTER the
        // meeting with nothing left to re-run.
        for file in ["campplus.py", "pooling_layers.py"] {
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: scripts.appendingPathComponent(
                    "campplus/vendor/wespeaker/models/\(file)").path),
                          "the vendored CAM++ architecture is incomplete: \(file)")
        }
    }

    /// No diarization engine may share a script or a log with another. A shared
    /// log is two writers on one file — the 2026-07-15 mistake.
    func testItSharesNeitherScriptNorLogWithAnotherEngine() {
        let scripts = [CamPlusService.Config.scriptName,
                       SpectralService.Config.scriptName,
                       DiarizenService.Config.scriptName,
                       NemoService.Config.scriptName,
                       PyannoteService.scriptName]
        let logs = [CamPlusService.Config.logName,
                    SpectralService.Config.logName,
                    DiarizenService.Config.logName,
                    NemoService.Config.logName,
                    PyannoteService.logName]
        XCTAssertEqual(Set(scripts).count, scripts.count,
                       "two diarization engines run the same script")
        XCTAssertEqual(Set(logs).count, logs.count,
                       "two diarization engines write the same log")
    }

    /// THE ONE THAT WOULD HAVE CAUGHT THE SHIPPED DEFECT. Every user-facing
    /// string this engine can produce must name ITS OWN files and engine — the
    /// 2026-08-10 DiariZen audit found `scripts/nemo/nemo-service.py` in a
    /// DiariZen error: wrong file, wrong folder, wrong engine, and nothing about
    /// it would look wrong to anyone who had not gone looking.
    func testNoUserFacingErrorNamesAnotherEnginesFiles() {
        let messages = [
            CamPlusService.ServiceError.scriptMissing.errorDescription ?? "",
            CamPlusService.ServiceError.launchFailed("boom").errorDescription ?? "",
            CamPlusService.ServiceError.startupFailed("boom").errorDescription ?? "",
        ]
        for message in messages {
            for foreign in ["nemo", "diarizen", "spectral", "pyannote", "moss"] {
                XCTAssertFalse(message.lowercased().contains(foreign),
                               "a CAM++ error names \(foreign): \(message)")
            }
        }
        XCTAssertTrue(messages[0].contains("scripts/campplus/campplus-service.py"),
                      "the missing-script error must name the file it looked for")
    }

    // MARK: - The engine's place among the rules

    /// It is a BATCH engine: `wantedDiarizationStack` resolves it to its own
    /// stack, and that stack uses the shared identity sidecar — which is what
    /// gives it saved profiles, renaming and `spk` with no new identity code.
    func testTheEngineResolvesToItsOwnStackAndUsesIdentity() {
        let stack = ModelLoader.wantedDiarizationStack(
            diarizationEnabled: true,
            engine: ModelLoader.camPlusEngineID,
            chunkedID: "whisper",
            chunkedEnabled: true)
        XCTAssertEqual(stack, .camPlus)
        XCTAssertEqual(stack?.usesSpeakerIdentity, true)
    }

    /// Diarization off keeps NO stack, whatever the engine says. The 2026-07-31
    /// leak: the load step asked the master switch and the teardown did not.
    func testDiarizationOffKeepsNoStack() {
        XCTAssertNil(ModelLoader.wantedDiarizationStack(
            diarizationEnabled: false,
            engine: ModelLoader.camPlusEngineID,
            chunkedID: "whisper",
            chunkedEnabled: true))
    }

    /// It HONOURS a pinned speaker count — measured, not assumed: the sidecar
    /// reads `num_speakers` and clusters to exactly that many (pinning
    /// `Overlap123.wav` to 2 returns 2).
    ///
    /// ⚠ RE-AIMED 2026-08-14, not deleted. The second half used to assert
    /// DiariZen's `false`, and that was a correct pin of the rule at the time —
    /// its sidecar really did ignore the field. The owner then asked for the
    /// picker to work on every engine and the sidecar was wired, so the half
    /// that still carries the danger is **MOSS**: it has no clustering stage to
    /// bound and no count parameter at all, so a `true` there would light the
    /// picker for a promise nothing anywhere keeps.
    func testItHonoursAPinnedSpeakerCountWhileMossCannot() {
        XCTAssertTrue(ModelLoader.honoursSpeakerCount(
            diarEngine: ModelLoader.camPlusEngineID))
        XCTAssertTrue(ModelLoader.honoursSpeakerCount(
            diarEngine: ModelLoader.diarizenEngineID))
        XCTAssertFalse(ModelLoader.honoursSpeakerCount(
            diarEngine: ModelLoader.mossEngineID))
    }

    /// It CANNOT mark overlap itself, so overlap repair under it needs the
    /// standalone detector — the same side as spectral and NeMo, and the
    /// opposite side from DiariZen.
    ///
    /// Both halves asserted, because this is the rule DiariZen shipped on the
    /// wrong side of: its clustering assigns one label per window, so its turns
    /// never intersect and there are no regions of its own to use.
    func testRepairNeedsTheDetectorUnderThisEngineButNotUnderDiarizen() {
        XCTAssertNil(ModelLoader.wantedOverlapEngine(
            repairEnabled: true, engineID: "mossformer2",
            diarEngine: ModelLoader.camPlusEngineID, detectEnabled: false),
                     "repair must not load under CAM++ with the detector off — "
                     + "there would be nowhere for it to run")
        XCTAssertNotNil(ModelLoader.wantedOverlapEngine(
            repairEnabled: true, engineID: "mossformer2",
            diarEngine: ModelLoader.camPlusEngineID, detectEnabled: true))
        // DiariZen supplies its own regions, so the detector is not required
        // there — without this half, folding the rule in could silently move it.
        XCTAssertNotNil(ModelLoader.wantedOverlapEngine(
            repairEnabled: true, engineID: "mossformer2",
            diarEngine: ModelLoader.diarizenEngineID, detectEnabled: false))
    }

    /// The detector's own sidecar DOES start under this engine — unlike under
    /// DiariZen, where the engine already answers that question. Narrow on
    /// purpose: `wantsOverlapDetect` asks whether the pyannote segmentation
    /// process loads, not whether overlap is detected at all.
    func testTheDetectorSidecarLoadsUnderThisEngine() {
        XCTAssertTrue(ModelLoader.wantsOverlapDetect(
            detectEnabled: true, diarizationEnabled: true,
            diarEngine: ModelLoader.camPlusEngineID))
        XCTAssertFalse(ModelLoader.wantsOverlapDetect(
            detectEnabled: true, diarizationEnabled: true,
            diarEngine: ModelLoader.diarizenEngineID))
    }

    // MARK: - Catalog

    /// The card round-trips through the catalog in BOTH directions, so the tab
    /// cannot show one engine selected while the loader starts another — the
    /// `diarizationEngineValue` indirection that once made the MOSS card look
    /// selected while the engine never switched.
    func testTheCardRoundTripsThroughTheCatalog() {
        let card = ModelCatalog.camPlusDiarization
        XCTAssertEqual(ModelCatalog.diarizationEngineValue(card),
                       ModelLoader.camPlusEngineID)
        XCTAssertEqual(ModelCatalog.diarizationEngine(
            forEngine: ModelLoader.camPlusEngineID).id, card.id)
        XCTAssertTrue(ModelCatalog.diarizationEngines.contains { $0.id == card.id },
                      "the engine exists but is not offered in Settings")
    }

    /// THE CHECKPOINT IS THE APACHE-2.0 ONE, and this is the only thing in Swift
    /// that would notice a swap. The same guard `testTheCheckpointIsTheMitOne`
    /// gives DiariZen, and for the same reason: of the CAM++ weights on the hub
    /// the convenient ones are unlicensed, and this app ships to a paying client
    /// — "no licence declared" is all-rights-reserved, not permissive.
    func testTheCheckpointIsTheApacheLicensedOne() {
        XCTAssertEqual(ModelCatalog.camPlusDiarization.hfRepo,
                       "Wespeaker/wespeaker-voxceleb-campplus-LM")
        XCTAssertTrue(ModelCatalog.camPlusDiarization.badges.contains("Apache 2.0"),
                      "the card must state the licence it is shipped under")
    }
}
