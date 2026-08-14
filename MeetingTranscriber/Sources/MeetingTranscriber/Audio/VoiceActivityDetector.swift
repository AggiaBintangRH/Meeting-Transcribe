import Foundation

/// Voice activity detection with hysteresis, driven by Settings → Models → VAD.
///
/// Two engines behind one interface:
///  • Silero VAD v6 (Python sidecar) — neural, trained to distinguish speech
///    from music/instruments. Used whenever available.
///  • Heuristic fallback — energy gate + zero-crossing band + syllabic
///    modulation. Used if the sidecar can't start.
///
/// Not thread-safe by design: call `process` from one thread (the audio tap).
final class VoiceActivityDetector {

    struct Config {
        let threshold: Float        // 0.1–0.9; Silero prob cutoff / gate strictness
        let minSilenceMs: Double    // silence needed to switch to silent
        let minSpeechMs: Double     // speech needed to switch to speaking

        static func fromSettings() -> Config {
            let d = UserDefaults.standard
            let threshold = d.object(forKey: "vad.threshold") as? Double ?? 0.5
            let minSilence = d.object(forKey: "vad.minSilenceMs") as? Double ?? 300
            let minSpeech = d.object(forKey: "vad.minSpeechMs") as? Double ?? 250
            return Config(threshold: Float(threshold),
                          minSilenceMs: minSilence,
                          minSpeechMs: minSpeech)
        }
    }

    private let config: Config
    private let silero: SileroVADService?

    // State machine
    private(set) var isSpeaking = false
    private var pendingSpeechMs: Double = 0
    private var pendingSilenceMs: Double = 0

    init(config: Config = .fromSettings(), silero: SileroVADService? = nil) {
        self.config = config
        self.silero = silero
    }

    /// Feed one buffer's features; returns current speaking state.
    /// - Parameters:
    ///   - samples16k: buffer resampled to 16 kHz mono (for Silero)
    ///   - rms/zcr: features of the original buffer (for fallback + noise floor)
    @discardableResult
    func process(samples16k: [Float], rms: Float, zcr: Float, bufferDuration: Double) -> Bool {
        let ms = bufferDuration * 1000

        let voiced: Bool
        if let silero {
            silero.feed(samples16k)
            voiced = silero.latestProbability >= config.threshold
        } else {
            voiced = heuristicVoiced(rms: rms, zcr: zcr, ms: ms)
        }

        applyHysteresis(voiced: voiced, ms: ms)
        return isSpeaking
    }

    // MARK: - Hysteresis (shared by both engines)

    private func applyHysteresis(voiced: Bool, ms: Double) {
        if voiced {
            pendingSilenceMs = 0
            if !isSpeaking {
                pendingSpeechMs += ms
                if pendingSpeechMs >= config.minSpeechMs {
                    isSpeaking = true
                    pendingSpeechMs = 0
                }
            }
        } else {
            pendingSpeechMs = 0
            if isSpeaking {
                pendingSilenceMs += ms
                if pendingSilenceMs >= config.minSilenceMs {
                    isSpeaking = false
                    pendingSilenceMs = 0
                }
            }
        }
    }

    // MARK: - Heuristic fallback engine

    private let zcrMin: Float = 0.015
    private let zcrMax: Float = 0.38
    private var noiseFloor: Float = 0.002

    private var recentRMS: [Float] = []
    private var recentDurations: [Double] = []
    private let modulationWindowSec = 1.0
    private var sustainedSoundMs: Double = 0

    private func heuristicVoiced(rms: Float, zcr: Float, ms: Double) -> Bool {
        trackModulation(rms: rms, duration: ms / 1000)

        let energyOK = rms > gate
        let zcrOK = zcr >= zcrMin && zcr <= zcrMax

        if !energyOK {
            noiseFloor = 0.995 * noiseFloor + 0.005 * rms
        }
        return energyOK && zcrOK && !isSustainedNonSpeech(energyOK: energyOK, ms: ms)
    }

    private var gate: Float {
        let strictness = 2.0 + config.threshold * 8.0
        return max(0.004, noiseFloor * strictness)
    }

    private func trackModulation(rms: Float, duration: Double) {
        recentRMS.append(rms)
        recentDurations.append(duration)
        while recentDurations.reduce(0, +) > modulationWindowSec, recentRMS.count > 1 {
            recentRMS.removeFirst()
            recentDurations.removeFirst()
        }
    }

    /// Speech pulses with syllables; held notes/drones are flat.
    private func isSustainedNonSpeech(energyOK: Bool, ms: Double) -> Bool {
        if energyOK {
            sustainedSoundMs += ms
        } else {
            sustainedSoundMs = 0
            return false
        }
        guard sustainedSoundMs > 800, recentRMS.count >= 8 else { return false }

        let mean = recentRMS.reduce(0, +) / Float(recentRMS.count)
        guard mean > 0 else { return false }
        let variance = recentRMS.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(recentRMS.count)
        return sqrt(variance) / mean < 0.18
    }
}
