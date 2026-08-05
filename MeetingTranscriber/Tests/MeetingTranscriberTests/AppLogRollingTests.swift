import XCTest
@testable import MeetingTranscriber

/// `PythonRuntime.appendAppLog` — rolling for the APP-OWNED decision logs.
///
/// WHY THIS EXISTS. The 2026-07-31 rolling work protected only the 13 sidecar
/// logs, because those are opened once per process start and `logHandle` could
/// roll them there. The app's own five decision logs were written by five
/// byte-identical open-append-close bodies that never consulted `logRollBytes` at
/// all — and the file that proved it is the same one that motivated rolling in
/// the first place: `position-diarization.log`, measured at **13,187,750 bytes
/// and still growing** on 2026-08-05.
///
/// These write into the REAL data dir, so each case uses a unique name and
/// removes both generations in a `defer`. The oversized file is made with
/// `truncate(atOffset:)` — sparse, so it costs no disk and no time.
final class AppLogRollingTests: XCTestCase {

    private var logsDir: URL {
        PythonRuntime.dataDir.appendingPathComponent("logs")
    }

    /// Unique per case so a parallel or repeated run can never collide with
    /// itself, and so a crash leaves an obviously-disposable name behind.
    private func scratchName() -> String {
        "approllover-test-\(UUID().uuidString.prefix(8))"
    }

    private func remove(_ name: String) {
        try? FileManager.default.removeItem(
            at: logsDir.appendingPathComponent("\(name).log"))
        try? FileManager.default.removeItem(
            at: logsDir.appendingPathComponent("\(name).log.1"))
    }

    /// Under the limit: append, and do not roll. The ordinary case, and the one
    /// that would break if the size check used the wrong comparison.
    func testAnOrdinaryLineJustAppends() throws {
        let name = scratchName()
        defer { remove(name) }

        PythonRuntime.appendAppLog(name: name, message: "first")
        PythonRuntime.appendAppLog(name: name, message: "second")

        let file = logsDir.appendingPathComponent("\(name).log")
        let text = try String(contentsOf: file, encoding: .utf8)
        XCTAssertTrue(text.contains("first"))
        XCTAssertTrue(text.contains("second"))
        XCTAssertEqual(text.split(separator: "\n").count, 2)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: logsDir.appendingPathComponent("\(name).log.1").path),
            "nothing may be rolled below the limit")
    }

    /// At the limit: roll, keeping exactly ONE previous generation. The line that
    /// triggered the roll must land in the FRESH file, not be lost — losing it
    /// would make the roll itself a form of the over-deletion this project treats
    /// as the dangerous direction.
    func testReachingTheLimitRollsAndKeepsTheNewLine() throws {
        let name = scratchName()
        defer { remove(name) }
        let file = logsDir.appendingPathComponent("\(name).log")

        try? FileManager.default.createDirectory(at: logsDir,
                                                 withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: file.path, contents: nil)
        let handle = try FileHandle(forWritingTo: file)
        // Sparse — no bytes are actually written, but the size is real.
        try handle.truncate(atOffset: PythonRuntime.logRollBytes)
        try handle.close()

        PythonRuntime.appendAppLog(name: name, message: "after the roll")

        let rolled = logsDir.appendingPathComponent("\(name).log.1")
        XCTAssertTrue(FileManager.default.fileExists(atPath: rolled.path),
                      "the oversized generation must be kept as .log.1")
        let rolledSize = (try FileManager.default.attributesOfItem(
            atPath: rolled.path)[.size] as? NSNumber)?.uint64Value
        XCTAssertEqual(rolledSize, PythonRuntime.logRollBytes)

        let fresh = try String(contentsOf: file, encoding: .utf8)
        XCTAssertTrue(fresh.contains("after the roll"),
                      "the line that triggered the roll must survive it")
        XCTAssertLessThan(fresh.count, 200, "the fresh generation starts empty")
    }

    /// A SECOND roll replaces `.log.1` rather than accumulating generations —
    /// otherwise "rolled" would just be "unbounded, one file later".
    func testASecondRollReplacesThePreviousGeneration() throws {
        let name = scratchName()
        defer { remove(name) }
        let file = logsDir.appendingPathComponent("\(name).log")
        let rolled = logsDir.appendingPathComponent("\(name).log.1")

        for marker in ["generation one", "generation two"] {
            FileManager.default.createFile(atPath: file.path, contents: nil)
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: PythonRuntime.logRollBytes)
            try handle.close()
            PythonRuntime.appendAppLog(name: name, message: marker)
        }

        XCTAssertTrue(FileManager.default.fileExists(atPath: rolled.path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: logsDir.appendingPathComponent("\(name).log.2").path),
            "only ONE previous generation is kept, by design")
        XCTAssertTrue(try String(contentsOf: file, encoding: .utf8)
                        .contains("generation two"))
    }

    /// Every app-owned log name in one place, so a sixth writer added later has
    /// an obvious home. These deliberately do NOT follow the sidecar rule
    /// (`scripts/<name>/ → logs/<name>.log`): each is a DOMAIN, and one of them
    /// is shared by two engines on purpose.
    func testTheAppOwnedLogNamesAreDomainsNotServices() {
        let appOwned = ["position-diarization", "moss-diarization",
                        "spectral-diarization", "dual-stream",
                        "overlap-repair-decisions"]
        let services = ["whisper", "qwen3", "granite", "voxtral", "moss-asr",
                        "moss-diar", "aligner", "pyannote", "wespeaker",
                        "nemotron", "vad", "dicow", "mossformer2", "spectral"]
        for name in appOwned {
            XCTAssertFalse(services.contains(name),
                           "\(name) collides with a sidecar log — two writers on "
                           + "one file is the 2026-07-15 mistake")
        }
    }
}
