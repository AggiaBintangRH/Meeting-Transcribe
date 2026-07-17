import SwiftUI

/// Full-screen dimmed overlay shown while models load before recording starts.
struct LoadingOverlayView: View {
    @ObservedObject var loader: ModelLoader

    private var failed: Bool { loader.failureMessage != nil }

    var body: some View {
        ZStack {
            Theme.overlayDim
                .ignoresSafeArea()

            VStack(spacing: 20) {
                Text(failed ? "Model loading failed" : "Loading models")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(failed ? Theme.red : Theme.textPrimary)

                VStack(alignment: .leading, spacing: 12) {
                    ForEach(loader.items) { item in
                        row(item)
                    }
                }

                if failed {
                    Button(action: { loader.dismissFailure() }) {
                        Text("Close")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(Theme.textPrimary)
                            .padding(.horizontal, 22)
                            .padding(.vertical, 9)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Theme.chip))
                    }
                    .buttonStyle(.plain)
                } else {
                    Text("Recording starts automatically when ready.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                }
            }
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

    private func row(_ item: ModelLoader.Item) -> some View {
        HStack(spacing: 12) {
            statusIcon(item.state)
                .frame(width: 18, height: 18)

            VStack(alignment: .leading, spacing: 2) {
                Text(item.name)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(textColor(item.state))
                    .lineLimit(1)
                if case .failed(let message) = item.state {
                    Text(message)
                        .font(.system(size: 10))
                        .foregroundColor(Theme.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .animation(.easeOut(duration: 0.2), value: item.state)
    }

    @ViewBuilder
    private func statusIcon(_ state: ModelLoader.ItemState) -> some View {
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

    private func textColor(_ state: ModelLoader.ItemState) -> Color {
        switch state {
        case .pending: return Theme.textDim
        case .loading, .done: return Theme.textPrimary
        case .failed: return Theme.red
        }
    }
}
