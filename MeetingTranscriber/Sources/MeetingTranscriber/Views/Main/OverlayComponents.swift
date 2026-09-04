import SwiftUI

/// One status line in a full-screen overlay: icon · name · failure reason.
/// Shared by the model-loading and stop-processing overlays, which list
/// different work but render it identically.
struct OverlayStepRow: View {
    let name: String
    let state: ModelLoader.ItemState
    /// A long leg's bar. nil for every leg that finishes quickly — and nil draws
    /// NOTHING, never an empty bar: a 0 % bar under a step that will be done in
    /// two seconds invents a wait that is not there.
    var progress: AudioRecorder.StopProgress? = nil

    var body: some View {
        HStack(spacing: 12) {
            statusIcon
                .frame(width: 18, height: 18)

            VStack(alignment: .leading, spacing: 2) {
                Text(name)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(textColor)
                    .lineLimit(1)
                // Selectable: a sidecar failure arrives as the last three lines
                // of a Python traceback, which is the one thing worth pasting
                // somewhere. Not truncated either — it is the only evidence the
                // user has without opening a log file.
                //
                // A skipped step carries a reason too, in the FAINT colour: it
                // is an explanation, not an alarm.
                if case .failed(let message) = state {
                    Text(message)
                        .font(.system(size: 10))
                        .foregroundColor(Theme.red)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                } else if case .skipped(let message) = state {
                    Text(message)
                        .font(.system(size: 10))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                // Only while the leg is actually working. A bar under a finished
                // or failed row describes work that is over.
                if let progress, case .loading = state {
                    progressBar(progress)
                }
            }
            Spacer(minLength: 0)
        }
        .animation(.easeOut(duration: 0.2), value: state)
    }

    /// The bar and its one line of numbers.
    @ViewBuilder
    private func progressBar(_ progress: AudioRecorder.StopProgress) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Theme.chip)
                    Capsule()
                        .fill(Theme.teal)
                        .frame(width: max(0, geo.size.width * progress.fraction))
                }
            }
            .frame(height: 4)
            // ⚠ Animated on the FRACTION, not on the whole value. `elapsed`
            // changes on every report, so animating the value would restart this
            // animation each time and leave the bar permanently mid-glide.
            .animation(.easeOut(duration: 0.25), value: progress.fraction)

            Text(caption(progress))
                .font(.system(size: 10).monospacedDigit())
                .foregroundColor(Theme.textFaint)
        }
        .padding(.top, 3)
        // Sized to the container it really lives in: OverlayCard is 420 wide
        // with 28 of padding, and this row spends 18 on the icon plus 12 of
        // spacing — so 300 keeps the bar clear of the card's edge. 320 was
        // measured against a render 16pt wider than the real card.
        .frame(maxWidth: 300, alignment: .leading)
    }

    /// "7 / 120 windows · 6% · about 3 min 40 s left"
    ///
    /// The estimate is simply ABSENT until there is evidence for one — see
    /// `StopProgress.secondsRemaining`. A row that shows a count and a percentage
    /// for a few seconds and then gains a time is honest; one that shows a
    /// confident wrong number and corrects it downward by minutes is not.
    private func caption(_ progress: AudioRecorder.StopProgress) -> String {
        let unit = progress.total == 1 ? "window" : "windows"
        var parts = ["\(progress.done) / \(progress.total) \(unit)",
                     "\(Int((progress.fraction * 100).rounded()))%"]
        if let left = progress.secondsRemaining {
            parts.append("about \(Self.durationPhrase(left)) left")
        }
        return parts.joined(separator: "  ·  ")
    }

    /// Seconds as something a person reads at a glance.
    ///
    /// Rounded DOWN to the coarser unit as it grows, deliberately: nobody
    /// watching a ten-minute pass wants "8 min 43 s" ticking every second, and
    /// the false precision would imply the estimate is better than it is.
    nonisolated static func durationPhrase(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total < 10 { return "a few seconds" }
        if total < 60 { return "\((total / 5) * 5) s" }
        let minutes = total / 60
        if minutes < 10 {
            let rest = ((total % 60) / 15) * 15
            return rest == 0 ? "\(minutes) min" : "\(minutes) min \(rest) s"
        }
        return "\(minutes) min"
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
        case .skipped:
            // A dash, not a tick and not a cross. Nothing happened here and
            // nothing was supposed to.
            Image(systemName: "minus.circle.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(Theme.textFaint)
        }
    }

    private var textColor: Color {
        switch state {
        case .pending: return Theme.textDim
        case .loading, .done: return Theme.textPrimary
        case .failed: return Theme.red
        case .skipped: return Theme.textDim
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
