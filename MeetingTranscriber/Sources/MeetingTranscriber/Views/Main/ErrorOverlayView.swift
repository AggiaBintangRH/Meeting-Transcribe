import SwiftUI

/// The popup for a recording error — a microphone that will not open, a refused
/// configuration, an audio engine that died mid-meeting.
///
/// Those failures used to appear ONLY as a line of small red text at the bottom
/// of the record card, which is where the eye is least likely to be at the
/// moment a recording refuses to start. The owner's complaint on 2026-08-12 was
/// about the stop panel, but the same gap runs through the whole session: an
/// error the app knows about must be put in front of the user, not filed at the
/// edge of a card.
///
/// The card-bottom text is deliberately KEPT. Closing this popup leaves it as
/// the lingering trace, so the reason survives on screen after the modal is
/// gone — dismissing the popup must not be the same thing as forgetting.
struct ErrorOverlayView: View {
    @ObservedObject var recorder: AudioRecorder

    var body: some View {
        OverlayCard {
            Text("Recording error")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(Theme.red)

            // Selectable, because the owner's report of this class of bug began
            // "saya gak bisa screenshot errornya" — the panel closed before the
            // text could be captured. A message that cannot be copied is one
            // that has to be transcribed by hand or re-created on purpose.
            Text(recorder.errorMessage ?? "")
                .font(.system(size: 12))
                .foregroundColor(Theme.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button(action: { recorder.dismissError() }) {
                Text("Close")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                    .padding(.horizontal, 22)
                    .padding(.vertical, 9)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Theme.chip))
            }
            .buttonStyle(.plain)
        }
    }
}
