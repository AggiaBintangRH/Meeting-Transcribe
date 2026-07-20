import SwiftUI

/// Non-blocking name prompt shown DURING recording when the position diarizer
/// is in enrollment mode and a new speaker direction has just appeared.
///
/// It observes the recorder-owned `PositionDiarizer` directly, so recording
/// keeps running underneath — answering only names the cluster. If another
/// speaker is born while the card is open, the diarizer's queue means the card
/// simply re-appears for the next one after Save/Skip.
///
/// Invisible whenever `pendingEnrollment` is nil; ContentView only builds it at
/// all when a diarizer exists (feature on), so it never renders when the
/// position feature is off.
struct EnrollmentPromptView: View {
    @ObservedObject var diarizer: PositionDiarizer
    @State private var name = ""

    var body: some View {
        if let clusterID = diarizer.pendingEnrollment {
            card
                // Re-key on the cluster id so the field clears when the queue
                // advances to the next new speaker.
                .id(clusterID)
                .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "person.line.dotted.person.fill")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(Theme.teal)
                Text("New speaker — who is this?")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                Spacer(minLength: 0)
            }

            Text("A new talker direction just appeared. Name them now, or skip to keep a provisional label. Recording keeps running.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                TextField("Speaker name", text: $name)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12))
                    .frame(width: 200)
                    .onSubmit(save)

                Button(action: save) {
                    Text("Save")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(Theme.selectedTabText)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(RoundedRectangle(cornerRadius: 7).fill(Theme.teal))
                }
                .buttonStyle(.plain)
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .opacity(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.5 : 1)

                Button(action: skip) {
                    Text("Skip")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(Theme.textSecondary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(RoundedRectangle(cornerRadius: 7).fill(Theme.chip))
                }
                .buttonStyle(.plain)

                Spacer(minLength: 0)
            }
        }
        .padding(16)
        .frame(maxWidth: 420, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Theme.card)
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.teal.opacity(0.4), lineWidth: 1))
                .shadow(color: .black.opacity(0.35), radius: 20, y: 8)
        )
        .padding(24)
    }

    private func save() {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        diarizer.resolveEnrollment(name: trimmed)
        name = ""
    }

    private func skip() {
        diarizer.skipEnrollment()
        name = ""
    }
}
