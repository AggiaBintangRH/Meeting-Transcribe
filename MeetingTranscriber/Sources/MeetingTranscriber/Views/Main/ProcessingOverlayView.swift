import SwiftUI

/// Full-screen dimmed overlay shown after Stop while the last chunk, the
/// diarization final pass and any overlap repair finish. Blocks the controls so
/// a new recording can't start on top of the previous one's tail work.
///
/// It OUTLIVES `.processing` when a leg failed, so the red row can be read —
/// see `AudioRecorder.showsStopOverlay`, which is what decides that.
struct ProcessingOverlayView: View {
    @ObservedObject var recorder: AudioRecorder

    /// Seconds before the escape hatch appears — long enough that a normal stop
    /// never shows it.
    private let escapeHatchDelay: Double = 30

    @State private var showEscapeHatch = false

    /// Processing has ENDED and something went red.
    ///
    /// Both halves matter. While the gate is still running a failed leg is not
    /// yet the outcome — the other legs may still finish — so the panel keeps
    /// its spinner and its escape hatch and says nothing final. Only once
    /// `.processing` is over does the red become the answer, and only then is
    /// there anything to close.
    private var finishedWithFailure: Bool {
        recorder.state != .processing && recorder.stopFailed
    }

    var body: some View {
        OverlayCard {
            Text(finishedWithFailure ? "Processing finished with errors"
                                     : "Finishing transcript…")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(finishedWithFailure ? Theme.red : Theme.textPrimary)

            VStack(alignment: .leading, spacing: 12) {
                ForEach(recorder.stopSteps) { step in
                    OverlayStepRow(name: label(step), state: step.state,
                                   progress: step.progress)
                }
            }

            if finishedWithFailure {
                // What the transcript IS, not only what failed. A failed
                // diarization pass still leaves the transcribed text on screen,
                // and a user who reads only the red row has no way to know that.
                Text("The transcript is on screen behind this panel. "
                     + "Whatever the failed step would have added is missing from it.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Button(action: { recorder.dismissStopFailure() }) {
                    Text("Close")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(Theme.textPrimary)
                        .padding(.horizontal, 22)
                        .padding(.vertical, 9)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Theme.chip))
                }
                .buttonStyle(.plain)
            } else {
                Text("Recording controls unlock when processing completes.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }

            if showEscapeHatch, !finishedWithFailure {
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
        .animation(.easeOut(duration: 0.2), value: finishedWithFailure)
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
