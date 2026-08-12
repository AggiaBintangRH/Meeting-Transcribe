import SwiftUI

/// One status line in a full-screen overlay: icon · name · failure reason.
/// Shared by the model-loading and stop-processing overlays, which list
/// different work but render it identically.
struct OverlayStepRow: View {
    let name: String
    let state: ModelLoader.ItemState

    var body: some View {
        HStack(spacing: 12) {
            statusIcon
                .frame(width: 18, height: 18)

            VStack(alignment: .leading, spacing: 2) {
                Text(name)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(textColor)
                    .lineLimit(1)
                if case .failed(let message) = state {
                    // Selectable: a sidecar failure arrives as the last three
                    // lines of a Python traceback, which is the one thing worth
                    // pasting somewhere. Not truncated either — it is the only
                    // evidence the user has without opening a log file.
                    Text(message)
                        .font(.system(size: 10))
                        .foregroundColor(Theme.red)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
            Spacer(minLength: 0)
        }
        .animation(.easeOut(duration: 0.2), value: state)
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch state {
        case .pending:
            Circle()
                .stroke(Theme.textFaint, lineWidth: 1.5)
                .frame(width: 12, height: 12)
        case .loading:
            ProgressView()
                .controlSize(.small)
                .tint(Theme.teal)
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(Theme.teal)
        case .failed:
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(Theme.red)
        }
    }

    private var textColor: Color {
        switch state {
        case .pending: return Theme.textDim
        case .loading, .done: return Theme.textPrimary
        case .failed: return Theme.red
        }
    }
}

/// The dimmed card both overlays sit in. The dim layer is hit-testable, so it
/// also swallows every click aimed at the UI behind it.
struct OverlayCard<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        ZStack {
            Theme.overlayDim
                .ignoresSafeArea()

            VStack(spacing: 20) { content }
                .padding(28)
                .frame(width: 420)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Theme.card)
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Theme.overlayBorder, lineWidth: 1)
                        )
                        .shadow(color: Theme.overlayShadow, radius: 30)
                )
        }
        .transition(.opacity)
    }
}
