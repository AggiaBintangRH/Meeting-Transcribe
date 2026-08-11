import XCTest
@testable import MeetingTranscriber

/// Contract tests for Qwen3-ASR's decoding options (Settings → Models → Chunked →
/// "Qwen3 options", added 2026-08-03).
///
/// THE GOVERNING RULE these exist to pin, identical to `WhisperOptionsTests`:
/// adding the options must not change a single transcript until the owner
/// deliberately moves a knob. On this side that means a default `Config` passes
/// the SAME process arguments it passed before the options existed, and every
/// option is absent from UserDefaults until it is set. The sidecar enforces the
/// same rule again on its side (`sidecar-tests.py
/// qwen3/option-defaults-are-todays-behaviour`), and it was additionally PROVED
/// end to end: the pre- and post-change sidecars driven over real audio emitted
/// byte-identical stdout (md5 9955eaaea8328e271d67ae4ca86b4af6).
final class Qwen3OptionsTests: XCTestCase {

    // "qwen3.temperature" and "qwen3.topP" are listed only so the stale-key test
    // can clean up after itself — there are no such options, and there must not
    // be until someone measures them. See testUnmeasuredKnobsAreNeverPassed.
    private let keys = ["qwen3.systemPrompt", "qwen3.repetitionPenalty",
                        "qwen3.repetitionContextSize", "qwen3.temperature",
                        "qwen3.topP", "qwen3.maxTokens",
                        "chunked.model", "chunked.language", "align.enabled"]

    override func setUp() {
        super.setUp()
        keys.forEach { UserDefaults.standard.removeObject(forKey: $0) }
    }

    override func tearDown() {
        keys.forEach { UserDefaults.standard.removeObject(forKey: $0) }
        super.tearDown()
    }

    private func selectQwen3() {
        UserDefaults.standard.set("qwen3", forKey: "chunked.model")
    }

    /// Mirrors the argument construction in `ChunkedASRService.init`, both
    /// per-model lines included — that is what proves at most one contributes.
    private func arguments(for config: ChunkedASRService.Config) -> [String] {
        var arguments = ["--model", config.repoID]
        if config.language != "auto" { arguments += ["--language", config.language] }
        arguments += config.whisper?.processArguments ?? []
        arguments += config.qwen3?.processArguments ?? []
        return arguments
    }

    // MARK: - Defaults are today's behaviour

    /// THE regression guard on this side: with nothing set, the Qwen3 sidecar is
    /// launched with exactly the arguments it was launched with before any of
    /// these options existed.
    func testDefaultsAddNoProcessArguments() {
        selectQwen3()
        let config = ChunkedASRService.Config.fromSettings()
        XCTAssertNotNil(config.qwen3)
        XCTAssertEqual(config.qwen3?.processArguments, [])
        XCTAssertEqual(arguments(for: config),
                       ["--model", "mlx-community/Qwen3-ASR-1.7B-bf16"])
    }

    /// An ABSENT key must yield the documented default, not `integer(forKey:)`'s
    /// 0. A repetition window of 0 instead of 100 would be a real decoding change
    /// produced by a missing key, on a fresh install, with nobody having touched
    /// a setting — the same trap `WhisperOptions` documents for its Doubles,
    /// which applies to this Int just as much.
    func testAbsentKeysYieldQwen3DefaultsNotZero() {
        selectQwen3()
        let options = ChunkedASRService.Config.Qwen3Options.fromSettings()
        XCTAssertEqual(options.systemPrompt, "")
        XCTAssertEqual(options.repetitionPenalty, 0)
        XCTAssertEqual(options.repetitionContextSize, 100)
        XCTAssertEqual(options, .default)
    }

    /// Writing a value that happens to EQUAL the default must still add nothing:
    /// the check is on the value, not on whether the key exists.
    func testExplicitlyDefaultValuesAddNoArguments() {
        selectQwen3()
        let d = UserDefaults.standard
        d.set(0.0, forKey: "qwen3.repetitionPenalty")
        d.set(100, forKey: "qwen3.repetitionContextSize")
        d.set("   ", forKey: "qwen3.systemPrompt")   // whitespace is not a prompt
        XCTAssertEqual(ChunkedASRService.Config.fromSettings().qwen3?.processArguments,
                       [])
    }

    // MARK: - Sentinels

    /// 0 means "off" and must produce no flag at all — the sidecar then leaves
    /// mlx-audio's own `None`. `repetition_penalty`'s real default IS None, not a
    /// number, so sending `--repetition-penalty 0` would be penalty 0: a
    /// different decoder setting, not the default one.
    func testZeroPenaltyProducesNoFlag() {
        selectQwen3()
        UserDefaults.standard.set(0.0, forKey: "qwen3.repetitionPenalty")
        let arguments = ChunkedASRService.Config.fromSettings().qwen3?.processArguments
        XCTAssertEqual(arguments, [])
        XCTAssertFalse(arguments?.contains("--repetition-penalty") ?? true)
    }

    /// The context size is meaningless without a penalty — mlx-audio reads it
    /// only inside `if repetition_penalty` — so it must NEVER be emitted alone.
    /// A flag that reaches the model and changes nothing is exactly the Granite
    /// language picker that shipped doing nothing on 2026-08-01.
    func testContextSizeIsNeverSentWithoutAPenalty() {
        selectQwen3()
        let d = UserDefaults.standard
        d.set(250, forKey: "qwen3.repetitionContextSize")   // moved, but no penalty
        let arguments = ChunkedASRService.Config.fromSettings().qwen3?.processArguments
        XCTAssertEqual(arguments, [])
        XCTAssertFalse(arguments?.contains("--repetition-context-size") ?? true)

        // …and WITH a penalty it does travel.
        d.set(1.2, forKey: "qwen3.repetitionPenalty")
        let withPenalty =
            ChunkedASRService.Config.fromSettings().qwen3?.processArguments ?? []
        XCTAssertTrue(withPenalty.contains("--repetition-context-size"))
        XCTAssertEqual(withPenalty[withPenalty.firstIndex(of: "--repetition-context-size")! + 1],
                       "250")
    }

    /// A penalty at a default window sends the penalty only — the window is at
    /// mlx-audio's own value, so naming it would be noise on the command line.
    func testPenaltyAtDefaultWindowSendsOnlyThePenalty() {
        selectQwen3()
        UserDefaults.standard.set(1.2, forKey: "qwen3.repetitionPenalty")
        XCTAssertEqual(ChunkedASRService.Config.fromSettings().qwen3?.processArguments,
                       ["--repetition-penalty", "1.2"])
    }

    /// Only knobs PROVEN to change the output are exposed. Qwen3's `generate()`
    /// advertises fifteen; the sampling group was never measured (and is inert
    /// while temperature is 0, so decoding is greedy). Asserts no flag can be
    /// produced for one, whatever a stale key holds.
    func testUnmeasuredKnobsAreNeverPassed() {
        selectQwen3()
        let d = UserDefaults.standard
        d.set(0.8, forKey: "qwen3.temperature")
        d.set(0.9, forKey: "qwen3.topP")
        d.set(4096, forKey: "qwen3.maxTokens")
        let config = ChunkedASRService.Config.fromSettings()
        XCTAssertEqual(config.qwen3?.processArguments, [])
        for flag in ["--temperature", "--top-p", "--max-tokens"] {
            XCTAssertFalse(arguments(for: config).contains(flag),
                           "\(flag) must not exist — it was never measured")
        }
    }

    // MARK: - Each knob reaches the sidecar

    func testEachOptionProducesItsFlag() {
        selectQwen3()
        let d = UserDefaults.standard
        d.set("pyannote ATND1061 PREP", forKey: "qwen3.systemPrompt")
        d.set(1.2, forKey: "qwen3.repetitionPenalty")
        d.set(20, forKey: "qwen3.repetitionContextSize")

        let arguments =
            ChunkedASRService.Config.fromSettings().qwen3?.processArguments ?? []
        for flag in ["--system-prompt", "--repetition-penalty",
                     "--repetition-context-size"] {
            XCTAssertTrue(arguments.contains(flag), "missing \(flag) in \(arguments)")
        }
        XCTAssertEqual(arguments[arguments.firstIndex(of: "--system-prompt")! + 1],
                       "pyannote ATND1061 PREP")
        XCTAssertEqual(arguments[arguments.firstIndex(of: "--repetition-penalty")! + 1],
                       "1.2")
    }

    // MARK: - Config identity

    /// Config is Equatable so ModelLoader recreates the sidecar on change. A knob
    /// that did NOT change Config would leave the previous decoder answering for
    /// the rest of the meeting while Settings claims otherwise.
    func testMovingAnyKnobChangesConfig() {
        selectQwen3()
        let before = ChunkedASRService.Config.fromSettings()
        let cases: [(String, Any)] = [
            ("qwen3.systemPrompt", "pyannote"),
            ("qwen3.repetitionPenalty", 1.2),
        ]
        for (key, value) in cases {
            UserDefaults.standard.set(value, forKey: key)
            XCTAssertNotEqual(before, ChunkedASRService.Config.fromSettings(),
                              "\(key) did not change Config")
            UserDefaults.standard.removeObject(forKey: key)
        }
        XCTAssertEqual(before, ChunkedASRService.Config.fromSettings())
    }

    /// The window moves Config only while a penalty makes it reachable — which is
    /// correct, not a gap: with no penalty it cannot alter the sidecar's argument
    /// list, so recreating the process for it would be a restart that changes
    /// nothing.
    func testContextSizeChangesConfigOnlyWithAPenalty() {
        selectQwen3()
        let d = UserDefaults.standard
        let before = ChunkedASRService.Config.fromSettings()
        d.set(250, forKey: "qwen3.repetitionContextSize")
        XCTAssertEqual(before.qwen3?.processArguments,
                       ChunkedASRService.Config.fromSettings().qwen3?.processArguments)

        d.set(1.2, forKey: "qwen3.repetitionPenalty")
        let penalised = ChunkedASRService.Config.fromSettings()
        d.set(20, forKey: "qwen3.repetitionContextSize")
        XCTAssertNotEqual(penalised, ChunkedASRService.Config.fromSettings())
    }

    // MARK: - The other models are untouched

    /// "Only for the Qwen3 script" is structural: no other model can carry these
    /// options, so no other sidecar's argument list can change — even with every
    /// Qwen3 knob moved. Whisper is included deliberately: the two option sets
    /// must be unable to cross, since `--system-prompt` is not an argparse flag
    /// whisper-service.py accepts and would abort the session at startup.
    func testOtherModelsCarryNoQwen3Options() {
        let d = UserDefaults.standard
        d.set("pyannote ATND1061", forKey: "qwen3.systemPrompt")
        d.set(1.2, forKey: "qwen3.repetitionPenalty")
        d.set(250, forKey: "qwen3.repetitionContextSize")
        for model in ["whisper", "voxtral", "granite", "moss"] {
            d.set(model, forKey: "chunked.model")
            let config = ChunkedASRService.Config.fromSettings()
            XCTAssertNil(config.qwen3, "\(model) carries Qwen3 options")
            XCTAssertEqual(arguments(for: config), ["--model", config.repoID],
                           "\(model) argument list changed")
        }
        XCTAssertNil(ChunkedASRService.Config.mossDiarization().qwen3)
    }

    /// …and moving a Qwen3 knob while another model is selected must not recreate
    /// that model's sidecar either.
    func testQwen3KnobsDoNotDisturbAnotherModelsConfig() {
        UserDefaults.standard.set("whisper", forKey: "chunked.model")
        let before = ChunkedASRService.Config.fromSettings()
        UserDefaults.standard.set(1.2, forKey: "qwen3.repetitionPenalty")
        XCTAssertEqual(before, ChunkedASRService.Config.fromSettings())
    }

    /// The two per-model option sets are mutually exclusive by construction: at
    /// most one of the two `processArguments` lines in `init` can contribute.
    func testWhisperAndQwen3OptionsCanNeverBothBePresent() {
        for model in ["whisper", "qwen3", "voxtral", "granite", "moss"] {
            UserDefaults.standard.set(model, forKey: "chunked.model")
            let config = ChunkedASRService.Config.fromSettings()
            XCTAssertFalse(config.whisper != nil && config.qwen3 != nil,
                           "\(model) carries BOTH option sets")
        }
    }
}
