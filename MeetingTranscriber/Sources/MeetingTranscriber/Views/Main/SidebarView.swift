import SwiftUI

/// Left panel: app title, settings button, record card.
struct SidebarView: View {
    @ObservedObject var recorder: AudioRecorder
    let openSettings: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Spacer()
            RecordCardView(recorder: recorder)
            Spacer()
            Spacer()
            // Under the record card, at the very bottom (owner, 2026-08-10).
            // AFTER the second Spacer rather than between them, so it is pinned to
            // the floor instead of floating in whatever space is left — and so the
            // mic card keeps the exact vertical position it has always had when no
            // meeting is finished, which is every launch.
            MeetingActionsView(recorder: recorder)
        }
        .frame(width: 350)
        .background(Theme.sidebar)
    }

    private var header: some View {
        HStack(spacing: 12) {
            Button(action: openSettings) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                    .frame(width: 36, height: 36)
                    .background(Circle().fill(Theme.logoBackground))
            }
            .buttonStyle(.plain)

            Text("MEETING TRANSCRIBER")
                .font(.system(size: 17, weight: .heavy))
                .kerning(0.5)
                .foregroundColor(Theme.textPrimary)

            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.top, 28)
    }
}
