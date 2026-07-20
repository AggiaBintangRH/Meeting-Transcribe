import SwiftUI

/// Right panel: header with status chips + transcript area.
struct MainAreaView: View {
    @ObservedObject var recorder: AudioRecorder

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Rectangle()
                .fill(Theme.divider)
                .frame(height: 1)
            TranscriptView(recorder: recorder)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
    }

    private var topBar: some View {
        HStack(alignment: .center, spacing: 16) {
            // The title gives way when the window narrows; the chips must not,
            // or their text wraps mid-word.
            VStack(alignment: .leading, spacing: 4) {
                Text("Meeting")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                    .lineLimit(1)
                Text("Realtime transcription will appear here.")
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textFaint)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .layoutPriority(0)
            Spacer(minLength: 0)
            StatusChipsView(recorder: recorder)
                .layoutPriority(1)
        }
        .padding(.horizontal, 24)
        .padding(.top, 18)
        .padding(.bottom, 14)
    }
}
