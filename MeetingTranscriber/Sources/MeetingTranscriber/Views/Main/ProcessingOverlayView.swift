import SwiftUI

/// Full-screen dimmed overlay shown after Stop while the last chunk, the
/// diarization final pass and any overlap repair finish. Blocks the controls so
/// a new recording can't start on top of the previous one's tail work.
struct ProcessingOverlayView: View {
    @ObservedObject var recorder: AudioRecorder

    /// Seconds before the escape hatch appears — long enough that a normal stop
    /// never shows it.
    private let escapeHatchDelay: Double = 30

    @State private var showEscapeHatch = false

    var body: some View {
        OverlayCard {
            Text("Finishing transcript…")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(Theme.textPrimary)

            VStack(alignment: .leading, spacing: 12) {
                ForEach(recorder.stopSteps) { step in
                    OverlayStepRow(name: label(step), state: step.state)
                }
            }

            Text("Recording controls unlock when processing completes.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)

            if showEscapeHatch {
                Button(action: { recorder.continueInBackground() }) {
                    Text("Continue in background")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(Theme.textPrimary)
                        .padding(.horizontal, 22)
                        .padding(.vertical, 9)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Theme.chip))
                }
                .buttonStyle(.plain)
            }
        }
        .animation(.easeOut(duration: 0.2), value: showEscapeHatch)
        .task {
            try? await Task.sleep(for: .seconds(escapeHatchDelay))
            guard !Task.isCancelled else { return }
            showEscapeHatch = true
        }
    }

    private func label(_ step: AudioRecorder.StopStep) -> String {
        guard step.id == "repair", let progress = recorder.overlapRepairProgress
        else { return step.name }
        return "\(step.name) \(progress)"
    }
}
