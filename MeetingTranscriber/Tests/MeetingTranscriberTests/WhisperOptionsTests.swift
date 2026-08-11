import XCTest
@testable import MeetingTranscriber

/// Contract tests for Whisper's decoding options (Settings → Models → Chunked →
/// "Whisper options", added 2026-07-29).
///
/// THE GOVERNING RULE these exist to pin: adding the options must not change a
/// single transcript until the owner deliberately moves a knob. On this side
/// that means a default `Config` passes the SAME process arguments it passed
/// before the options existed, and every option is absent from UserDefaults
/// until it is set. The sidecar enforces the same rule again on its side
/// (`sidecar-tests.py whisper/option-defaults-are-todays-behaviour`).
final class WhisperOptionsTests: XCTestCase {

    // "whisper.beamSize" is listed only so the stale-key test can clean up after
    // itself — there is no such option, see testNoBeamSearchOptionIsEverPassed.
    private let keys = ["whisper.initialPrompt", "whisper.beamSize", "whisper.bestOf",
                        "whisper.noSpeechThreshold", "whisper.logprobThreshold",
                        "whisper.compressionThreshold", "whisper.task",
                        "whisper.autoDetectLanguage", "whisper.hallucinationSilenceSec",
                        "chunked.model", "chunked.language", "align.enabled"]

    override func setUp() {
        super.setUp()
        keys.forEach { UserDefaults.standard.removeObject(forKey: $0) }
    }

    override func tearDown() {
        keys.forEach { UserDefaults.standard.removeObject(forKey: $0) }
        super.tearDown()
    }

    private func selectWhisper() {
        UserDefaults.standard.set("whisper", forKey: "chunked.model")
    }

    /// Mirrors the argument construction in `ChunkedASRService.init`.
    private func arguments(for config: ChunkedASRService.Config) -> [String] {
        var arguments = ["--model", config.repoID]
        if config.language != "auto" { arguments += ["--language", config.language] }
        // No --align-model: the forced aligner is its own sidecar since
        // 2026-07-29 and this config knows nothing about it.
        arguments += config.whisper?.processArguments ?? []
        return arguments
    }

    // MARK: - Defaults are today's behaviour

    /// THE regression guard on this side: with nothing set, the Whisper sidecar
    /// is launched with exactly the arguments it was launched with before any of
    /// these options existed.
    func testDefaultsAddNoProcessArguments() {
        selectWhisper()
        let config = ChunkedASRService.Config.fromSettings()
        XCTAssertNotNil(config.whisper)
        XCTAssertEqual(config.whisper?.processArguments, [])
        XCTAssertEqual(arguments(for: config),
                       ["--model", "mlx-community/whisper-large-v3-mlx"])
    }

    /// An ABSENT key must yield the documented default, not `double(forKey:)`'s
    /// 0 — a logprob threshold of 0 instead of -1 would make Whisper retry every
    /// chunk, on a fresh install, with nobody having touched a setting.
    func testAbsentKeysYieldWhisperDefaultsNotZero() {
        selectWhisper()
        let options = ChunkedASRService.Config.WhisperOptions.fromSettings()
        XCTAssertEqual(options.noSpeechThreshold, 0.6)
        XCTAssertEqual(options.logprobThreshold, -1.0)
        XCTAssertEqual(options.compressionThreshold, 2.4)
        XCTAssertEqual(options.task, "transcribe")
        XCTAssertEqual(options.initialPrompt, "")
        XCTAssertFalse(options.autoDetectLanguage)
        XCTAssertEqual(options.bestOf, 0)
        XCTAssertEqual(options.hallucinationSilenceSec, 0)
        XCTAssertEqual(options, .default)
    }

    /// Writing a value that happens to EQUAL the default must still add nothing:
    /// the check is on the value, not on whether the key exists.
    func testExplicitlyDefaultValuesAddNoArguments() {
        selectWhisper()
        let d = UserDefaults.standard
        d.set(0.6, forKey: "whisper.noSpeechThreshold")
        d.set(-1.0, forKey: "whisper.logprobThreshold")
        d.set(2.4, forKey: "whisper.compressionThreshold")
        d.set("transcribe", forKey: "whisper.task")
        d.set("   ", forKey: "whisper.initialPrompt")   // whitespace is not a prompt
        XCTAssertEqual(ChunkedASRService.Config.fromSettings().whisper?.processArguments,
                       [])
    }

    // MARK: - Sentinels

    /// 0 means "off" and must produce no flag at all — the sidecar then leaves
    /// Whisper's own `None`. Sending `--best-of 0` would be a different decoder
    /// setting, not the default one.
    func testZeroSentinelsProduceNoFlags() {
        selectWhisper()
        let d = UserDefaults.standard
        d.set(0, forKey: "whisper.bestOf")
        d.set(0.0, forKey: "whisper.hallucinationSilenceSec")
        let arguments = ChunkedASRService.Config.fromSettings().whisper?.processArguments
        XCTAssertEqual(arguments, [])
        XCTAssertFalse(arguments?.contains("--best-of") ?? true)
        XCTAssertFalse(arguments?.contains("--hallucination-silence-sec") ?? true)
    }

    /// There is no beam-search option and there must not be one: the MLX Whisper
    /// runtime raises NotImplementedError for any beam_size, so the sidecar never
    /// finishes loading (measured 2026-07-29 — `mlx_whisper.transcribe(...,
    /// beam_size=2)` raises immediately at `decoding.py:437`, so the warmup
    /// transcribe fails and the sidecar exits with a load error before the
    /// meeting starts). Asserts the flag can never be produced, whatever a stale
    /// `whisper.beamSize` key holds.
    func testNoBeamSearchOptionIsEverPassed() {
        selectWhisper()
        UserDefaults.standard.set(2, forKey: "whisper.beamSize")
        let config = ChunkedASRService.Config.fromSettings()
        XCTAssertEqual(config.whisper?.processArguments, [])
        XCTAssertFalse(arguments(for: config).contains("--beam-size"))
    }

    // MARK: - Each knob reaches the sidecar

    func testEachOptionProducesItsFlag() {
        selectWhisper()
        let d = UserDefaults.standard
        d.set("pyannote ATND1061", forKey: "whisper.initialPrompt")
        d.set(2, forKey: "whisper.bestOf")
        d.set(0.45, forKey: "whisper.noSpeechThreshold")
        d.set(-1.5, forKey: "whisper.logprobThreshold")
        d.set(2.0, forKey: "whisper.compressionThreshold")
        d.set(2.0, forKey: "whisper.hallucinationSilenceSec")

        let arguments = ChunkedASRService.Config.fromSettings().whisper?.processArguments ?? []
        for flag in ["--initial-prompt", "--best-of",
                     "--no-speech-threshold", "--logprob-threshold",
                     "--compression-threshold", "--hallucination-silence-sec"] {
            XCTAssertTrue(arguments.contains(flag), "missing \(flag) in \(arguments)")
        }
        XCTAssertEqual(arguments[arguments.firstIndex(of: "--initial-prompt")! + 1],
                       "pyannote ATND1061")
        XCTAssertEqual(arguments[arguments.firstIndex(of: "--best-of")! + 1], "2")
    }

    // MARK: - Config identity

    /// Config is Equatable so ModelLoader recreates the sidecar on change. A knob
    /// that did NOT change Config would leave the previous decoder answering for
    /// the rest of the meeting while Settings claims otherwise.
    func testMovingAnyKnobChangesConfig() {
        selectWhisper()
        let before = ChunkedASRService.Config.fromSettings()
        let cases: [(String, Any)] = [
            ("whisper.initialPrompt", "pyannote"),
            ("whisper.bestOf", 5),
            ("whisper.noSpeechThreshold", 0.45),
            ("whisper.logprobThreshold", -1.5),
            ("whisper.compressionThreshold", 2.0),
            ("whisper.hallucinationSilenceSec", 2.0),
        ]
        for (key, value) in cases {
            UserDefaults.standard.set(value, forKey: key)
            XCTAssertNotEqual(before, ChunkedASRService.Config.fromSettings(),
                              "\(key) did not change Config")
            UserDefaults.standard.removeObject(forKey: key)
        }
        XCTAssertEqual(before, ChunkedASRService.Config.fromSettings())
    }

    // MARK: - The two RETIRED knobs may never come back

    /// `whisper.task` and `whisper.autoDetectLanguage` had controls until
    /// 2026-08-06, when the owner removed them and fixed both at their defaults.
    /// A stored value must not outlive the control that set it — someone who once
    /// picked "Translate to English" would otherwise keep receiving a translation
    /// instead of a record of the words said, with nothing in the UI able to
    /// change it back.
    ///
    /// The mirror image of `testEachOptionProducesItsFlag`, and it fails in both
    /// the ways that matter: the flag reappearing on the wire, and the stored
    /// value merely reaching `WhisperOptions` (which would also recreate the
    /// sidecar on a value nobody can see).
    func testRetiredKnobsNeverReachTheSidecar() {
        selectWhisper()
        let d = UserDefaults.standard
        d.set("translate", forKey: "whisper.task")
        d.set(true, forKey: "whisper.autoDetectLanguage")
        defer {
            d.removeObject(forKey: "whisper.task")
            d.removeObject(forKey: "whisper.autoDetectLanguage")
        }
        let config = ChunkedASRService.Config.fromSettings()
        XCTAssertEqual(config.whisper?.task, "transcribe",
                       "a stored task outlived the control that set it")
        XCTAssertEqual(config.whisper?.autoDetectLanguage, false)
        let args = config.whisper?.processArguments ?? []
        XCTAssertFalse(args.contains("--task"), "\(args)")
        XCTAssertFalse(args.contains("--auto-detect-language"), "\(args)")
    }

    /// And they may not recreate the sidecar either: `Config` is Equatable and
    /// drives that, so a key nothing reads must leave it identical.
    func testRetiredKnobsDoNotChangeConfig() {
        selectWhisper()
        let before = ChunkedASRService.Config.fromSettings()
        for (key, value) in [("whisper.task", "translate" as Any),
                             ("whisper.autoDetectLanguage", true as Any)] {
            UserDefaults.standard.set(value, forKey: key)
            XCTAssertEqual(before, ChunkedASRService.Config.fromSettings(),
                           "\(key) still reaches Config")
            UserDefaults.standard.removeObject(forKey: key)
        }
    }

    // MARK: - The other models are untouched

    /// "Only for the Whisper script" is structural: no other model can carry
    /// these options, so no other sidecar's argument list can change — even with
    /// every Whisper knob moved.
    func testOtherModelsCarryNoWhisperOptions() {
        let d = UserDefaults.standard
        d.set("pyannote ATND1061", forKey: "whisper.initialPrompt")
        d.set(5, forKey: "whisper.bestOf")
        d.set(2.0, forKey: "whisper.hallucinationSilenceSec")
        for model in ["qwen3", "voxtral", "granite", "moss"] {
            d.set(model, forKey: "chunked.model")
            let config = ChunkedASRService.Config.fromSettings()
            XCTAssertNil(config.whisper, "\(model) carries Whisper options")
            XCTAssertEqual(arguments(for: config), ["--model", config.repoID],
                           "\(model) argument list changed")
        }
        XCTAssertNil(ChunkedASRService.Config.mossDiarization().whisper)
    }

    /// …and moving a Whisper knob while another model is selected must not
    /// recreate that model's sidecar either.
    func testWhisperKnobsDoNotDisturbAnotherModelsConfig() {
        UserDefaults.standard.set("qwen3", forKey: "chunked.model")
        let before = ChunkedASRService.Config.fromSettings()
        UserDefaults.standard.set(5, forKey: "whisper.bestOf")
        XCTAssertEqual(before, ChunkedASRService.Config.fromSettings())
    }
}
