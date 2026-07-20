import SwiftUI

/// Full-screen dimmed overlay shown while models load before recording starts.
struct LoadingOverlayView: View {
    @ObservedObject var loader: ModelLoader

    private var failed: Bool { loader.failureMessage != nil }

    var body: some View {
        OverlayCard {
            Text(failed ? "Model loading failed" : "Loading models")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(failed ? Theme.red : Theme.textPrimary)

            VStack(alignment: .leading, spacing: 12) {
                ForEach(loader.items) { item in
                    OverlayStepRow(name: item.name, state: item.state)
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
    }
}
