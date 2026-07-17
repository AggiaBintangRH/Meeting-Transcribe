import SwiftUI

/// Top-right status chips: RMS meter, ATND, recording state.
struct StatusChipsView: View {
    @ObservedObject var recorder: AudioRecorder

    private var isRecording: Bool { recorder.state == .recording }

    var body: some View {
        HStack(spacing: 10) {
            chip {
                Image(systemName: "waveform")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Theme.teal)
                Text("RMS \(String(format: "%.4f", recorder.rms))")
                    .font(.system(size: 11, weight: .bold).monospaced())
                    .foregroundColor(Theme.textBody)
            }
            vadChip
            chip {
                Image(systemName: "person.2.fill")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(recorder.speakerCount != nil ? Theme.teal : Theme.textDim)
                if recorder.diarizing {
                    Text("ATND …")
                        .font(.system(size: 11, weight: .bold).monospaced())
                        .foregroundColor(Theme.textSecondary)
                } else if let count = recorder.speakerCount {
                    Text("ATND \(count)")
                        .font(.system(size: 11, weight: .heavy).monospaced())
                        .foregroundColor(Theme.textBright)
                } else {
                    Text("ATND –")
                        .font(.system(size: 11, weight: .bold).monospaced())
                        .foregroundColor(Theme.textDim)
                }
            }
            chip {
                Circle()
                    .fill(isRecording ? Theme.red : Theme.teal)
                    .frame(width: 7, height: 7)
                Text(isRecording ? "REC" : "IDLE")
                    .font(.system(size: 11, weight: .heavy).monospaced())
                    .foregroundColor(Theme.textBright)
            }
        }
    }

    /// VAD status: TALKING (voice detected) / SILENT / – when idle or VAD off.
    private var vadChip: some View {
        chip {
            if !isRecording {
                Image(systemName: "person.wave.2")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Theme.textDim)
                Text("VAD –")
                    .font(.system(size: 11, weight: .bold).monospaced())
                    .foregroundColor(Theme.textDim)
            } else if !recorder.vadEnabled {
                Image(systemName: "person.wave.2")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Theme.textDim)
                Text("VAD OFF")
                    .font(.system(size: 11, weight: .bold).monospaced())
                    .foregroundColor(Theme.textDim)
            } else {
                VoiceBars(level: recorder.rms, active: recorder.isSpeaking)
                Text(recorder.isSpeaking ? "TALKING" : "SILENT")
                    .font(.system(size: 11, weight: .heavy).monospaced())
                    .foregroundColor(recorder.isSpeaking ? Theme.teal : Theme.textSubtle)
            }
        }
        .animation(.easeOut(duration: 0.15), value: recorder.isSpeaking)
    }

    private func chip<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 6) { content() }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(RoundedRectangle(cornerRadius: 7).fill(Theme.chip))
    }
}

/// Mini equalizer: 3 bars that rise and fall with the live voice level.
/// Teal + moving while talking; dim + flat when silent.
private struct VoiceBars: View {
    let level: Float   // live RMS
    let active: Bool   // VAD says speech

    /// Per-bar sensitivity so the bars move independently.
    private let gains: [Float] = [22, 40, 30]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<3, id: \.self) { i in
                Capsule()
                    .fill(active ? Theme.teal : Theme.textDim)
                    .frame(width: 2.5, height: height(i))
                    .animation(.easeOut(duration: 0.12), value: level)
            }
        }
        .frame(height: 12, alignment: .center)
    }

    private func height(_ bar: Int) -> CGFloat {
        guard active else { return 3 }
        let h = CGFloat(min(level * gains[bar], 1.0)) * 12
        return max(3, h)
    }
}
