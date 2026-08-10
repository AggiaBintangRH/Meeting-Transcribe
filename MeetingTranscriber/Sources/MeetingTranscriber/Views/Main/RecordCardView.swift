import SwiftUI

/// The "Ready to record" card with the big mic button.
struct RecordCardView: View {
    @ObservedObject var recorder: AudioRecorder

    private var isRecording: Bool { recorder.state == .recording }
    private var isPreparing: Bool { recorder.state == .preparing }

    /// A finished meeting is still on screen, so the mic is locked until
    /// `Start Over` clears it (owner, 2026-08-10). Checked BEFORE `state` in both
    /// switches below, because the state at that moment is `.idle` and would
    /// otherwise read "Ready to record" over a mic that cannot be pressed —
    /// a label describing a control's normal behaviour while the control is
    /// inert is worse than no label.
    private var isLocked: Bool { recorder.meetingFinished }

    private var title: String {
        if isLocked { return "Meeting finished" }
        switch recorder.state {
        case .idle:       return "Ready to record"
        case .preparing:  return "Loading models…"
        case .recording:  return "Recording…"
        case .processing: return "Finishing up…"
        }
    }

    private var buttonLabel: String {
        if isLocked { return "START OVER TO RECORD AGAIN" }
        switch recorder.state {
        case .idle:       return "START RECORDING"
        case .preparing:  return "LOADING…"
        case .recording:  return "STOP RECORDING"
        case .processing: return "PROCESSING…"
        }
    }

    var body: some View {
        VStack(spacing: 26) {
            Text(title)
                .font(.system(size: 17, weight: .bold))
                .foregroundColor(Theme.textPrimary)

            Button(action: { recorder.toggle() }) {
                ZStack {
                    Circle()
                        .fill(isRecording ? Theme.red : Theme.teal)
                        .opacity(isPreparing ? 0.5 : 1)
                        .frame(width: 106, height: 106)
                        .shadow(color: (isRecording ? Theme.red : Theme.teal).opacity(isPreparing ? 0.2 : 0.55),
                                radius: 28)
                    if isPreparing {
                        ProgressView()
                            .controlSize(.large)
                            .tint(Theme.recordButtonSpinnerTint)
                    } else {
                        Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                            .font(.system(size: 34, weight: .bold))
                            .foregroundColor(isRecording ? Theme.recordButtonActiveText : Theme.recordButtonIdleText)
                    }
                }
            }
            .buttonStyle(.plain)
            .disabled(isPreparing || recorder.state == .processing || isLocked)
            .opacity(isLocked ? 0.5 : 1)   // the app's one disabled opacity (EnrollmentPromptView)

            Text(buttonLabel)
                .font(.system(size: 12, weight: .heavy))
                .kerning(1.2)
                .foregroundColor(Theme.textPrimary)

            if isRecording, let device = recorder.activeDeviceName {
                Label(device, systemImage: "mic.fill")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textDim)
                    .lineLimit(1)
            }

            if let error = recorder.errorMessage {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.red)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(.vertical, 34)
        .frame(maxWidth: .infinity)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.card))
        .padding(.horizontal, 16)
    }
}
