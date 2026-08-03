import XCTest
@testable import MeetingTranscriber

/// ONE SIDECAR PER ASR MODEL, finished 2026-07-30: the shared
/// `chunked/chunked-asr-service.py` is deleted and each model has its own
/// standalone script and its own log.
///
/// The failure these tests exist for is SILENT and only appears at session start,
/// as "scripts/… not found in the project folder" in front of the user: the routing
/// used to end in `default: defaultScriptName`, and that default pointed at the file
/// that has now been removed. `ChunkedASRModel` therefore REQUIRES `scriptName` and
/// `logName`, so a new model cannot compile without naming its own — but a name can
/// still be a typo or refer to a file someone moved, which is what the existence
/// check below catches.
final class ChunkedASRSidecarRoutingTests: XCTestCase {

    private let modelKey = "chunked.model"

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: modelKey)
        super.tearDown()
    }

    /// Every chunked model, and the sidecar + log it must resolve to.
    private let expected: [(id: String, model: any ChunkedASRModel,
                            script: String, log: String)] = [
        // NAMING CONSISTENCY, 2026-07-31: the log is named for the SERVICE FOLDER,
        // not the role — scripts/<name>/<name>-service.py writes logs/<name>.log,
        // 13/13. The `-asr` suffixes these four used to carry were the last
        // survivors of the shared chunked sidecar. `layout/*` in sidecar-tests.py
        // pins the rule across all 13; this table pins the five chunked models'
        // exact PAIRING, which a set-based check cannot see.
        ("qwen3", Qwen3ASRModel(), "qwen3/qwen3-service.py", "qwen3"),
        ("whisper", WhisperLargeV3Model(), "whisper/whisper-service.py", "whisper"),
        ("granite", GraniteSpeechModel(), "granite/granite-service.py", "granite"),
        ("voxtral", VoxtralMiniModel(), "voxtral/voxtral-service.py", "voxtral"),
        ("moss", MossTranscribeDiarizeModel(), "moss-asr/moss-asr-service.py", "moss-asr"),
    ]

    /// The repo's `scripts/` directory, from this file's own path — the app's
    /// `PythonRuntime.scriptsDir` resolves against the bundle or the launch
    /// directory, neither of which a unit test can rely on.
    private var scriptsDir: URL {
        URL(fileURLWithPath: #filePath)          // …/Tests/MeetingTranscriberTests/<this>
            .deletingLastPathComponent()         // …/Tests/MeetingTranscriberTests
            .deletingLastPathComponent()         // …/Tests
            .deletingLastPathComponent()         // …/MeetingTranscriber
            .deletingLastPathComponent()         // repo root
            .appendingPathComponent("scripts")
    }

    func testEachModelNamesItsOwnSidecarAndLog() {
        for entry in expected {
            XCTAssertEqual(entry.model.scriptName, entry.script,
                           "\(entry.id) resolved to the wrong script")
            XCTAssertEqual(entry.model.logName, entry.log,
                           "\(entry.id) resolved to the wrong log")
        }
    }

    /// No two models may share a script or a log. A shared script would mean the
    /// split was undone for one of them; a shared log would put two writers on one
    /// file, the mistake of 2026-07-15 that splitting the services makes live again.
    func testScriptsAndLogsAreAllDistinct() {
        XCTAssertEqual(Set(expected.map(\.script)).count, expected.count)
        XCTAssertEqual(Set(expected.map(\.log)).count, expected.count)
    }

    /// The named file must actually be there. This is the check that would have
    /// failed while `defaultScriptName` still pointed at the deleted shared
    /// sidecar — and it is the whole reason there is no `default:` any more.
    func testEveryNamedSidecarExistsOnDisk() {
        for entry in expected {
            let path = scriptsDir.appendingPathComponent(entry.script).path
            XCTAssertTrue(FileManager.default.fileExists(atPath: path),
                          "\(entry.id) names a script that does not exist: \(path)")
        }
    }

    /// The deleted shared sidecar must not come back by accident, and nothing may
    /// point at it.
    func testTheSharedSidecarIsGone() {
        let stale = scriptsDir.appendingPathComponent("chunked/chunked-asr-service.py").path
        XCTAssertFalse(FileManager.default.fileExists(atPath: stale))
        for entry in expected {
            XCTAssertFalse(entry.model.scriptName.contains("chunked-asr-service"))
        }
    }

    /// The settings value really does select that model's sidecar — the routing is
    /// read at `Config.fromSettings`, so a wrong id here would start the wrong
    /// process with the right model repo.
    func testSettingsSelectTheMatchingSidecar() {
        for entry in expected {
            UserDefaults.standard.set(entry.id, forKey: modelKey)
            let config = ChunkedASRService.Config.fromSettings()
            XCTAssertEqual(config.scriptName, entry.script,
                           "chunked.model=\(entry.id) started the wrong script")
            XCTAssertEqual(config.logName, entry.log,
                           "chunked.model=\(entry.id) wrote to the wrong log")
        }
    }

    /// An unknown stored id falls back to Qwen3 (the factory's own default), which
    /// must be a script that EXISTS — the fallback is on the model, not on the
    /// script name, which is the difference that matters.
    func testUnknownStoredModelFallsBackToARealSidecar() {
        UserDefaults.standard.set("some-model-from-the-future", forKey: modelKey)
        let config = ChunkedASRService.Config.fromSettings()
        XCTAssertEqual(config.scriptName, Qwen3ASRModel().scriptName)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: scriptsDir.appendingPathComponent(config.scriptName).path))
    }

    /// MOSS's TWO ROLES ARE TWO SERVICES (2026-07-31). MOSS-as-ASR runs
    /// `moss-asr/moss-asr-service.py` → `moss-asr.log`; MOSS-as-diarizer runs
    /// `moss-diar/moss-diar-service.py` → `moss-diar.log`.
    ///
    /// The DIFFERENCE is the assertion that matters. `mossDiarization()` used to
    /// take its log from `MossTranscribeDiarizeModel().logName`; left derived, the
    /// ASR-side rename to "moss-asr" silently repointed this process's stderr into
    /// the ASR log while it still ran the OTHER script — a script/log pair that has
    /// drifted, which is exactly what `Config.logName` forbids, and invisible
    /// because both processes would keep working. Hence the literals below AND the
    /// explicit "these must not be equal" checks: an accidental re-derivation
    /// cannot pass this test.
    func testMossDiarizationConfigUsesItsOwnSidecarAndLog() {
        let config = ChunkedASRService.Config.mossDiarization()
        XCTAssertEqual(config.scriptName, "moss-diar/moss-diar-service.py")
        XCTAssertEqual(config.logName, "moss-diar")

        let asr = MossTranscribeDiarizeModel()
        XCTAssertNotEqual(config.scriptName, asr.scriptName,
                          "the diarization role must not run the ASR role's script")
        XCTAssertNotEqual(config.logName, asr.logName,
                          "the diarization role must not write the ASR role's log")

        // Both named scripts really exist — this is the check that catches the
        // 2026-07-31 split being left half-done in either direction.
        for script in [config.scriptName, asr.scriptName] {
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: scriptsDir.appendingPathComponent(script).path),
                          "\(script) does not exist")
        }
    }
}
