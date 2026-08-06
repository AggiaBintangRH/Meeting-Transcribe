import XCTest
@testable import MeetingTranscriber

/// `WhisperPreset.bestAccuracy` — the measured preset (2026-08-06).
///
/// The preset's values are identical to the shipped defaults, and that is the
/// MEASUREMENT'S RESULT, not a placeholder: on 67 minutes of real meeting audio
/// every knob either changed nothing at all (best-of, compression, no-speech,
/// logprob −2.0) or deleted real sentences while raising the confidence number
/// (logprob −0.5, hallucination-silence). See `WhisperPreset` for the table.
///
/// THAT AGREEMENT IS EXACTLY WHY THESE TESTS EXIST. Two copies of the same
/// numbers — the preset list and `WhisperOptions`' own defaults — silently drift
/// apart, and the drift would be invisible: a preset button that quietly moves
/// the decoder away from what was measured produces a different transcript with
/// nothing on screen to say so.
final class WhisperPresetTests: XCTestCase {

    private let keys = ["whisper.initialPrompt", "whisper.bestOf",
                        "whisper.noSpeechThreshold", "whisper.logprobThreshold",
                        "whisper.compressionThreshold", "whisper.hallucinationSilenceSec"]

    /// The suite runs in the owner's real preference domain, so every key this
    /// test touches is restored exactly as it was found.
    private func preservingDefaults(_ body: () -> Void) {
        let d = UserDefaults.standard
        let saved = keys.map { ($0, d.object(forKey: $0)) }
        defer {
            for (k, v) in saved {
                if let v { d.set(v, forKey: k) } else { d.removeObject(forKey: k) }
            }
        }
        body()
    }

    /// The preset must produce the SAME decoder configuration the app uses with
    /// nothing stored — i.e. it really is the measured-best set, not a set that
    /// merely looks like it.
    func testThePresetEqualsTheShippedDefaults() {
        preservingDefaults {
            let d = UserDefaults.standard
            d.set("whisper", forKey: "chunked.model")
            for k in keys { d.removeObject(forKey: k) }
            let untouched = ChunkedASRService.Config.fromSettings().whisper

            WhisperPreset.applyBestAccuracy()
            let preset = ChunkedASRService.Config.fromSettings().whisper

            XCTAssertEqual(untouched, preset,
                           "the preset no longer matches the configuration that was measured")
            // And the wire agrees: a preset that sent flags a default run does not
            // would be a different decode however equal the struct looked.
            XCTAssertEqual(preset?.processArguments, [],
                           "the measured-best configuration must send no decoding flags")
        }
    }

    /// Applying it must actually write every key — a preset that silently skipped
    /// one would leave a customised value in force under a "best accuracy" label.
    func testApplyingItOverwritesEveryCustomisedValue() {
        preservingDefaults {
            let d = UserDefaults.standard
            d.set("whisper", forKey: "chunked.model")
            d.set("ATND1061 pyannote", forKey: "whisper.initialPrompt")
            d.set(5, forKey: "whisper.bestOf")
            d.set(0.35, forKey: "whisper.noSpeechThreshold")
            d.set(-0.5, forKey: "whisper.logprobThreshold")
            d.set(3.5, forKey: "whisper.compressionThreshold")
            d.set(2.0, forKey: "whisper.hallucinationSilenceSec")
            XCTAssertFalse(WhisperPreset.bestAccuracyIsActive)

            WhisperPreset.applyBestAccuracy()

            XCTAssertTrue(WhisperPreset.bestAccuracyIsActive)
            XCTAssertEqual(ChunkedASRService.Config.fromSettings().whisper?.processArguments, [],
                           "a customised value survived the preset")
        }
    }

    /// `bestAccuracyIsActive` drives a "in use" badge, so it must fail on ANY one
    /// key being off — otherwise the badge asserts a decode that is not running.
    /// The double comparison also has to tolerate what a Slider really writes.
    func testTheBadgeIsFalseWhenAnySingleKeyDiffers() {
        preservingDefaults {
            let d = UserDefaults.standard
            for (key, _) in WhisperPreset.bestAccuracy {
                WhisperPreset.applyBestAccuracy()
                XCTAssertTrue(WhisperPreset.bestAccuracyIsActive)
                switch key {
                case "whisper.initialPrompt": d.set("x", forKey: key)
                case "whisper.bestOf":        d.set(5, forKey: key)
                default:                      d.set(9.5, forKey: key)
                }
                XCTAssertFalse(WhisperPreset.bestAccuracyIsActive,
                               "\(key) was ignored by the in-use badge")
            }
            // A slider writing 0.6000000000000001 is still the preset.
            WhisperPreset.applyBestAccuracy()
            d.set(0.6000000000000001, forKey: "whisper.noSpeechThreshold")
            XCTAssertTrue(WhisperPreset.bestAccuracyIsActive,
                          "floating-point noise from a Slider must not clear the badge")
        }
    }

    /// The preset covers every knob the Whisper block still exposes. A knob added
    /// later and forgotten here would be left at whatever the user last set while
    /// the button claims the measured configuration is in force.
    func testThePresetCoversEveryExposedKnob() {
        XCTAssertEqual(Set(WhisperPreset.bestAccuracy.map(\.key)), Set(keys),
                       "a Whisper option is exposed in Settings but not covered by the preset")
    }
}
