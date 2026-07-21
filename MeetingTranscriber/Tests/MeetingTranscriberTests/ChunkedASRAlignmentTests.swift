import XCTest
@testable import MeetingTranscriber

/// Contract tests for the optional word aligner on the chunked ASR sidecar.
/// Two things must hold: with `align.enabled` off nothing about the sidecar
/// changes, and a `final` payload without the aligner keys still decodes.
final class ChunkedASRAlignmentTests: XCTestCase {

    private let key = "align.enabled"

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: key)
        super.tearDown()
    }

    // MARK: - Config

    func testAlignRepoIsNilWhenDisabled() {
        UserDefaults.standard.set(false, forKey: key)
        XCTAssertNil(ChunkedASRService.Config.fromSettings().alignRepoID)
    }

    func testAlignRepoIsNilWhenUnset() {
        UserDefaults.standard.removeObject(forKey: key)
        XCTAssertNil(ChunkedASRService.Config.fromSettings().alignRepoID)
    }

    func testAlignRepoIsSetWhenEnabled() {
        UserDefaults.standard.set(true, forKey: key)
        XCTAssertEqual(ChunkedASRService.Config.fromSettings().alignRepoID,
                       "mlx-community/Qwen3-ForcedAligner-0.6B-bf16")
    }

    /// Config is Equatable so ModelLoader recreates the sidecar on change —
    /// toggling alignment must therefore produce a different Config.
    func testTogglingAlignmentChangesConfig() {
        UserDefaults.standard.set(false, forKey: key)
        let off = ChunkedASRService.Config.fromSettings()
        UserDefaults.standard.set(true, forKey: key)
        let on = ChunkedASRService.Config.fromSettings()
        XCTAssertNotEqual(off, on)
    }

    /// Mirrors the argument construction in `init` for both settings.
    func testArgumentsOmitAlignModelWhenDisabled() {
        func arguments(for config: ChunkedASRService.Config) -> [String] {
            var arguments = ["--model", config.repoID]
            if config.language != "auto" { arguments += ["--language", config.language] }
            if let repo = config.alignRepoID { arguments += ["--align-model", repo] }
            return arguments
        }
        UserDefaults.standard.set(false, forKey: key)
        XCTAssertFalse(arguments(for: .fromSettings()).contains("--align-model"))
        UserDefaults.standard.set(true, forKey: key)
        XCTAssertTrue(arguments(for: .fromSettings()).contains("--align-model"))
    }

    // MARK: - Decoding

    func testFinalWithoutAlignerKeysStillDecodes() throws {
        let line = #"{"type":"final","text":"hello there"}"#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.type, "final")
        XCTAssertEqual(m.text, "hello there")
        XCTAssertNil(m.words)
        XCTAssertNil(m.dur)
    }

    func testFinalWithAlignerKeysDecodes() throws {
        let line = #"""
        {"type":"final","text":"one-minute break","words":[{"text":"oneminute","start":0.2,"end":0.8,"src":0}],"dur":30.0}
        """#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.text, "one-minute break")
        XCTAssertEqual(m.dur, 30.0)
        XCTAssertEqual(m.words, [ChunkedASRService.AlignedWord(
            text: "oneminute", start: 0.2, end: 0.8, src: 0)])
    }
}
