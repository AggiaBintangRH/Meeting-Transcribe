import SwiftUI

/// Models → VAD: Silero voice activity detection settings.
struct VADTab: View {
    @AppStorage("vad.enabled")      private var enabled = true
    @AppStorage("vad.threshold")    private var threshold = 0.5
    @AppStorage("vad.minSilenceMs") private var minSilenceMs = 300.0
    @AppStorage("vad.minSpeechMs")  private var minSpeechMs = 250.0

    var body: some View {
        Group {
            ModelCardView(model: ModelCatalog.vad, selected: true)

            SettingToggle(label: "Enable VAD (skip silence before ASR)", isOn: $enabled)

            SettingBlock(title: "Speech threshold — \(String(format: "%.2f", threshold))") {
                Slider(value: $threshold, in: 0.1...0.9, step: 0.05)
                    .tint(Theme.teal)
                Text("Higher = stricter (less noise passes as speech)")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }

            SettingBlock(title: "Min silence to split — \(Int(minSilenceMs)) ms") {
                Slider(value: $minSilenceMs, in: 100...1000, step: 50)
                    .tint(Theme.teal)
            }

            SettingBlock(title: "Min speech duration — \(Int(minSpeechMs)) ms") {
                Slider(value: $minSpeechMs, in: 100...1000, step: 50)
                    .tint(Theme.teal)
            }
        }
    }
}
