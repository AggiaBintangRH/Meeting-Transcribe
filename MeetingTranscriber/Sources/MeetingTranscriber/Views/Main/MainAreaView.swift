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
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Meeting")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                Text("Realtime transcription will appear here.")
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textFaint)
            }
            Spacer()
            StatusChipsView(recorder: recorder)
        }
        .padding(.horizontal, 24)
        .padding(.top, 18)
        .padding(.bottom, 14)
    }
}
