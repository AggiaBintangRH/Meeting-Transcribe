import XCTest
@testable import MeetingTranscriber

/// TWO REALTIME ENGINES, ONE CLIENT (2026-08-11): Nemotron and Parakeet each get
/// their own sidecar and their own log, driven by one `RealtimeASRService` — the
/// chunked side's shape, for the chunked side's reasons.
///
/// The failure these exist for is the one `ChunkedASRSidecarRoutingTests` was
/// written against and the one `Config.mossDiarization()` actually shipped: a
/// script/log pair that has DRIFTED. Both processes keep working, nothing errors,
/// and one engine's stderr lands in the other's log — so the evidence for the
/// next bug is in a file nobody greps. Literals on both sides, and an explicit
/// "these must not be equal", so an accidental re-derivation cannot pass.
final class RealtimeSidecarRoutingTests: XCTestCase {

    private let modelKey = "realtime.model"
    private let languageKey = "realtime.language"
    private let chunkKey = "realtime.chunkMs"

    override func tearDown() {
        for key in [modelKey, languageKey, chunkKey] {
            UserDefaults.standard.removeObject(forKey: key)
        }
        super.tearDown()
    }

    /// Every realtime engine, and the sidecar + log it must resolve to. The log
    /// is named for the SERVICE FOLDER, not for the role — the 2026-07-31 rule,
    /// `scripts/<name>/<name>-service.py` writes `logs/<name>.log`.
    private let expected: [(id: String, config: RealtimeASRService.Config,
                            script: String, log: String)] = [
        ("nemotron", .nemotron(language: "auto", chunkMs: 160, partialMs: RealtimeASRService.defaultPartialMs),
         "nemotron/nemotron-service.py", "nemotron"),
        ("parakeet", .parakeet(language: "auto", partialMs: RealtimeASRService.defaultPartialMs),
         "parakeet/parakeet-service.py", "parakeet"),
        ("funasr", .funasr(language: "auto", partialMs: RealtimeASRService.defaultPartialMs),
         "funasr/funasr-service.py", "funasr"),
    ]

    /// The repo's `scripts/` directory, from this file's own path — the app's
    /// `PythonRuntime.scriptsDir` resolves against the bundle or the launch
    /// directory, neither of which a unit test can rely on.
    private var scriptsDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts")
    }

    func testEachEngineNamesItsOwnSidecarAndLog() {
        for entry in expected {
            XCTAssertEqual(entry.config.scriptName, entry.script,
                           "\(entry.id) resolved to the wrong script")
            XCTAssertEqual(entry.config.logName, entry.log,
                           "\(entry.id) resolved to the wrong log")
            XCTAssertEqual(entry.config.modelID, entry.id)
        }
    }

    /// No two engines may share a script OR a log. A shared script would mean
    /// one of them is not really a separate engine; a shared log would put two
    /// writers on one file — the 2026-07-15 mistake, which is live again the
    /// moment a second process exists.
    func testTheTwoEnginesShareNeitherScriptNorLog() {
        let scripts = expected.map(\.config.scriptName)
        let logs = expected.map(\.config.logName)
        XCTAssertEqual(Set(scripts).count, expected.count,
                       "two realtime engines run the same script")
        XCTAssertEqual(Set(logs).count, expected.count,
                       "two realtime engines write the same log — two writers, one file")
    }

    /// The named file must actually be there. Missing, this is a
    /// `ServiceError.scriptMissing` in front of the user at session start.
    func testEveryNamedSidecarExistsOnDisk() {
        for entry in expected {
            let path = scriptsDir.appendingPathComponent(entry.script).path
            XCTAssertTrue(FileManager.default.fileExists(atPath: path),
                          "\(entry.id) names a script that does not exist: \(path)")
        }
    }

    /// The settings value really selects that engine's sidecar. `Config` is read
    /// at `fromSettings`, so a wrong id here starts the wrong process entirely.
    func testSettingsSelectTheMatchingSidecar() {
        for entry in expected {
            UserDefaults.standard.set(entry.id, forKey: modelKey)
            let config = RealtimeASRService.Config.fromSettings()
            XCTAssertEqual(config.scriptName, entry.script,
                           "realtime.model=\(entry.id) started the wrong script")
            XCTAssertEqual(config.logName, entry.log,
                           "realtime.model=\(entry.id) wrote to the wrong log")
        }
    }

    /// An unknown stored id falls back to a REAL engine whose script exists —
    /// never to a `default:` naming a file that may have been deleted, which is
    /// the trap `ChunkedASRModel.scriptName` and `modelTabStatus` both recorded.
    func testUnknownStoredEngineFallsBackToARealSidecar() {
        UserDefaults.standard.set("some-engine-from-the-future", forKey: modelKey)
        let config = RealtimeASRService.Config.fromSettings()
        XCTAssertEqual(config.modelID, RealtimeASRService.defaultModelID)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: scriptsDir.appendingPathComponent(config.scriptName).path))
        XCTAssertEqual(ModelCatalog.realtimeModel(id: "some-engine-from-the-future").id,
                       ModelCatalog.realtimeModels[0].id,
                       "the catalog must fall back to the FIRST entry, which is the default")
    }

    /// `--chunk-ms` may only reach the sidecar that declares it. Parakeet's
    /// argparse has no such flag, so a leaked one would be an argparse error at
    /// session start — which is why the value is unrepresentable rather than
    /// merely unused.
    func testOnlyNemotronPassesAChunkSize() {
        let nemotron = RealtimeASRService.Config
            .nemotron(language: "en-US", chunkMs: 560, partialMs: 1500)
        XCTAssertEqual(nemotron.processArguments,
                       ["--language", "en-US", "--chunk-ms", "560",
                        "--partial-ms", "1500"])

        let parakeet = RealtimeASRService.Config.parakeet(language: "en", partialMs: 1500)
        XCTAssertNil(parakeet.chunkMs,
                     "Parakeet has no attention context to point a chunk size at")
        XCTAssertEqual(parakeet.processArguments,
                       ["--language", "en", "--partial-ms", "1500"])
        XCTAssertFalse(parakeet.processArguments.contains("--chunk-ms"))

        let funasr = RealtimeASRService.Config.funasr(language: "en", partialMs: 1500)
        XCTAssertNil(funasr.chunkMs,
                     "Fun-ASR has no attention context to point a chunk size at")
        XCTAssertEqual(funasr.processArguments,
                       ["--language", "en", "--partial-ms", "1500"])
        XCTAssertFalse(funasr.processArguments.contains("--chunk-ms"))
    }

    // MARK: - Fun-ASR: the engine whose language is REAL

    /// THE ONE REALTIME ENGINE THAT ACTUALLY USES ITS LANGUAGE SETTING, and the
    /// only one where a wrong code is not merely ignored.
    ///
    /// Parakeet, Granite, Voxtral and MOSS all swallow an unknown code via
    /// `**kwargs`. Fun-ASR's `_map_language()` RAISES a `ValueError` instead, so
    /// an unclamped code does not degrade the caption — it kills the lane for
    /// the rest of the meeting. That is why `realtimeNote` must stay nil here
    /// (there is nothing to apologise for) while `resolveRealtime` must clamp.
    ///
    /// Asserted against Parakeet in the same test on purpose: the two engines
    /// sit on opposite sides of this rule, and a future edit that gave them one
    /// shared answer would be wrong for one of them whichever answer it picked.
    func testFunasrHonoursItsLanguageWhileParakeetDoesNot() {
        XCTAssertNil(Languages.realtimeNote(forModel: RealtimeASRService.funasrModelID),
                     "Fun-ASR's language setting is honoured — a 'this will not "
                     + "change the transcript' note there would be a false warning")
        XCTAssertNotNil(Languages.realtimeNote(forModel: RealtimeASRService.parakeetModelID),
                        "Parakeet's language is inert and MUST say so")
    }

    /// A code Fun-ASR lacks never travels. Its roster is what mainline mlx-audio
    /// can express — English, Chinese, Japanese — not the 30 the model card
    /// advertises, so Indonesian and Malay (two of the Nemotron picker's five
    /// concrete options) are the common case rather than an edge.
    func testACodeFunasrLacksResolvesToAutoAndNeverTravels() {
        UserDefaults.standard.set("funasr", forKey: modelKey)
        for code in ["id", "ms"] {
            UserDefaults.standard.set(code, forKey: languageKey)
            XCTAssertEqual(RealtimeASRService.Config.fromSettings().language, "auto",
                           "\(code) reached the Fun-ASR sidecar, where it would RAISE")
        }
        // And the ones it does have are passed through untouched — without this
        // half, a resolver that clamped EVERYTHING to auto would pass.
        for code in ["en", "zh", "ja"] {
            UserDefaults.standard.set(code, forKey: languageKey)
            XCTAssertEqual(RealtimeASRService.Config.fromSettings().language, code,
                           "\(code) is on Fun-ASR's roster and must reach it")
        }
        // The stored value is NOT rewritten — switching engines must still find
        // the user's own choice.
        UserDefaults.standard.set("id", forKey: languageKey)
        _ = RealtimeASRService.Config.fromSettings()
        XCTAssertEqual(UserDefaults.standard.string(forKey: languageKey), "id")
    }

    /// Fun-ASR speaks BARE ISO CODES, not Nemotron's locale keys. The two live
    /// behind one settings key, so the read boundary is the only thing keeping
    /// `en-US` out of a sidecar that would raise on it.
    func testFunasrGetsABareIsoCodeAndNemotronALocaleKey() {
        UserDefaults.standard.set("en", forKey: languageKey)

        UserDefaults.standard.set("funasr", forKey: modelKey)
        XCTAssertEqual(RealtimeASRService.Config.fromSettings().language, "en")

        UserDefaults.standard.set("nemotron", forKey: modelKey)
        XCTAssertEqual(RealtimeASRService.Config.fromSettings().language, "en-US",
                       "Nemotron takes locale keys — if these two ever agree, one "
                       + "of the sidecars is being handed a code it cannot read")
    }

    /// BOTH engines carry the caption interval — one control, one shared key
    /// (owner, 2026-08-11: *"jangan dibedakan, soalnya sama tentang waktu
    /// interval"*).
    ///
    /// This test asserted the OPPOSITE for a few hours, and the reversal is the
    /// point of keeping it rather than deleting it. The old rule reasoned from a
    /// real measurement — Nemotron is ~9x slower per partial, so its cadence is
    /// not a free choice — but drew the wrong conclusion from it: that is a
    /// reason for the two SIDECARS to honour the number differently (Parakeet
    /// exactly, Nemotron as a floor that `PARTIAL_DUTY` stretches), not a reason
    /// for one engine to lack the control entirely.
    func testBothEnginesPassACaptionInterval() {
        let parakeet = RealtimeASRService.Config.parakeet(language: "auto", partialMs: 750)
        XCTAssertEqual(parakeet.partialMs, 750)
        XCTAssertTrue(parakeet.processArguments.contains("--partial-ms"))

        let nemotron = RealtimeASRService.Config
            .nemotron(language: "auto", chunkMs: 160, partialMs: 750)
        XCTAssertEqual(nemotron.partialMs, 750,
                       "Nemotron must carry the interval too — it is the same "
                       + "control, and withholding it pinned its caption speed "
                       + "to a constant nothing in the UI could reach")
        XCTAssertTrue(nemotron.processArguments.contains("--partial-ms"))
    }

    /// ONE key, so a choice made under one engine survives a switch to the
    /// other. Two keys would make "the same setting" a claim the storage quietly
    /// contradicted — the user would set 2 s under Parakeet, switch, and find
    /// Nemotron back at 1.5 s with nothing explaining it.
    func testTheCaptionIntervalIsOneSharedKeyAcrossEngines() {
        UserDefaults.standard.set(3000, forKey: "realtime.partialMs")
        defer { UserDefaults.standard.removeObject(forKey: "realtime.partialMs") }

        for id in ["nemotron", "parakeet"] {
            UserDefaults.standard.set(id, forKey: modelKey)
            XCTAssertEqual(RealtimeASRService.Config.fromSettings().partialMs, 3000,
                           "\(id) did not read the shared caption-interval key")
        }
    }

    /// A knob that belongs to ONE engine must not make the OTHER engine's config
    /// unequal, because `Config` equality is what decides whether a running
    /// sidecar is torn down and a 2.3 GB model reloaded.
    ///
    /// The set of such knobs SHRANK on 2026-08-11 when the caption interval
    /// became shared, and that is deliberate rather than a hole: `chunkMs` is
    /// still Nemotron's alone and unrepresentable on a Parakeet config, while
    /// the interval is now one control on one key, so moving it SHOULD replace
    /// whichever sidecar is running. The two halves below assert exactly that
    /// difference, so a future edit cannot quietly re-split the interval or
    /// quietly share the chunk size.
    func testAnEngineOwnedKnobCannotRecreateTheOtherEnginesProcess() {
        let parakeetA = RealtimeASRService.Config.parakeet(language: "en", partialMs: 1500)
        let parakeetB = RealtimeASRService.Config.parakeet(language: "en", partialMs: 1500)
        XCTAssertEqual(parakeetA, parakeetB)

        // ENGINE-OWNED: a chunk size cannot even be expressed on a Parakeet
        // config, so the isolation is structural rather than a comparison rule.
        XCTAssertNil(parakeetA.chunkMs)

        // SHARED: both engines carry the interval, and changing it must relaunch
        // whichever one is running — the flag is read once, at startup.
        XCTAssertNotEqual(parakeetA,
                          .parakeet(language: "en", partialMs: 750))
        XCTAssertNotEqual(
            RealtimeASRService.Config.nemotron(language: "en-US", chunkMs: 80,
                                               partialMs: 1500),
            RealtimeASRService.Config.nemotron(language: "en-US", chunkMs: 80,
                                               partialMs: 750))
    }

    /// The language code is FORWARDED to Parakeet even though the model ignores
    /// it — the 2026-07-31 rule, and it is deliberate rather than sloppy. Dropping
    /// it at the Swift boundary would give the user's choice TWO places to vanish
    /// and would leave the sidecar log with no record of what was asked for.
    func testParakeetForwardsTheCodeItWillThenIgnore() {
        UserDefaults.standard.set("parakeet", forKey: modelKey)
        UserDefaults.standard.set("de", forKey: languageKey)
        let config = RealtimeASRService.Config.fromSettings()
        XCTAssertEqual(config.language, "de")
        XCTAssertTrue(config.processArguments.contains("de"))
        XCTAssertNotNil(Languages.realtimeNote(forModel: "parakeet"),
                        "a forwarded-but-ignored code MUST be explained in the UI")
    }

    /// A code the selected engine does not have resolves to auto at the READ
    /// boundary, and never travels. Four of the Nemotron picker's five concrete
    /// options are absent from Parakeet, so this is the common case, not an edge.
    func testACodeParakeetLacksResolvesToAutoAndNeverTravels() {
        UserDefaults.standard.set("parakeet", forKey: modelKey)
        for code in ["id", "ms", "zh", "ja"] {
            UserDefaults.standard.set(code, forKey: languageKey)
            let config = RealtimeASRService.Config.fromSettings()
            XCTAssertEqual(config.language, "auto",
                           "\(code) reached the Parakeet sidecar, which has no such language")
        }
        // And the stored value is NOT rewritten — switching back to Nemotron must
        // still find the user's own choice.
        XCTAssertEqual(UserDefaults.standard.string(forKey: languageKey), "ja")
    }
}
