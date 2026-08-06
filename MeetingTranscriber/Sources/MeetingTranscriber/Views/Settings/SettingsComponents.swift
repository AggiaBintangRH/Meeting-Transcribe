import SwiftUI

// MARK: - Reusable pieces for settings tabs

/// Model card with radio indicator, spec badges, installed status, hover effect.
struct ModelCardView: View {
    let model: ModelInfo
    let selected: Bool

    @State private var hovered = false

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                .font(.system(size: 15))
                .foregroundColor(selected ? Theme.teal : Theme.textFaint)

            VStack(alignment: .leading, spacing: 6) {
                Text(model.name)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                Text(model.detail)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textDim)
                HStack(spacing: 6) {
                    ForEach(model.badges, id: \.self) { badge in
                        Text(badge)
                            .font(.system(size: 9, weight: .bold).monospaced())
                            .foregroundColor(Theme.textMuted)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 3)
                            .background(RoundedRectangle(cornerRadius: 4).fill(Theme.chip))
                    }
                }
            }
            Spacer()
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Theme.card)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(selected ? Theme.teal.opacity(0.55)
                                : hovered ? Theme.hoverHighlight : .clear,
                                lineWidth: 1)
                )
        )
        .scaleEffect(hovered && !selected ? 1.01 : 1.0)
        .animation(.easeOut(duration: 0.15), value: hovered)
        .onHover { hovered = $0 }
    }

}

/// A sub-tab inside a settings section (Models' model tabs, ATND's tabs).
protocol SettingsSubTab: Hashable, CaseIterable, Identifiable {
    var title: String { get }
    var icon: String { get }
}

extension SettingsSubTab {
    var id: Self { self }
}

/// Pill bar of sub-tabs with a sliding selection indicator.
struct SubTabBar<Tab: SettingsSubTab>: View {
    @Binding var selection: Tab
    /// Unique per bar — two bars must not share one matchedGeometry id.
    let geometryID: String
    let namespace: Namespace.ID

    var body: some View {
        HStack(spacing: 4) {
            ForEach(Array(Tab.allCases)) { t in
                Button(action: {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) { selection = t }
                }) {
                    HStack(spacing: 7) {
                        Image(systemName: t.icon)
                            .font(.system(size: 11, weight: .bold))
                        Text(t.title)
                            .font(.system(size: 12, weight: .bold))
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    .foregroundColor(selection == t ? Theme.selectedTabText : Theme.textMuted)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 9)
                    .frame(maxWidth: .infinity)
                    .background {
                        if selection == t {
                            RoundedRectangle(cornerRadius: 9)
                                .fill(Theme.teal)
                                .matchedGeometryEffect(id: geometryID, in: namespace)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(4)
        .background(RoundedRectangle(cornerRadius: 12).fill(Theme.chip.opacity(0.6)))
    }
}

/// The settings window's spacing, in ONE place.
///
/// Before this there were three competing numbers and the result was visible:
/// the content column used 18, `SettingBlock` added 4 on top of it (so a titled
/// block sat 22 from its neighbour while two plain paragraphs sat at 18), and
/// one tab wrapped its model cards in a VStack of 8. TWO numbers now, and the
/// rule is structural: anything at the top level of a tab is `row` apart,
/// anything INSIDE a `SettingBlock` is `inBlock` apart.
enum SettingsLayout {
    /// Between top-level items in a tab: paragraphs, model cards, blocks.
    static let row: CGFloat = 14
    /// Between a block's title and its controls, and between controls in it.
    static let inBlock: CGFloat = 8
}

/// Download status for the model a tab has SELECTED, shown under its card list.
///
/// It used to be a badge on EVERY card ("INSTALLED" / "NOT DOWNLOADED"), which
/// spent a red chip on models a session was never going to load and made a list
/// of five cards read as four problems. Only the SELECTED model can stop a
/// session from starting, so only that one is reported — and the message says
/// what to DO, which a two-word chip had no room for.
///
/// Modelled on the Aligner tab's row, which had this shape first.
struct ModelInstallStatus: View {
    let model: ModelInfo

    var body: some View {
        let installed = ModelCatalog.isInstalled(model)
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: installed ? "checkmark.circle.fill"
                                        : "exclamationmark.triangle.fill")
                .font(.system(size: 11))
                .foregroundColor(installed ? Theme.teal : Theme.red)
            Text(installed
                 ? "\(model.name) — downloaded."
                 : "\(model.name) is not downloaded. RUN SETUP (download-best-models.sh) "
                   + "before starting a session, or it will fail to start.")
                .font(.system(size: 11))
                .foregroundColor(installed ? Theme.textFaint : Theme.red)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }
}

/// A bordered container for the SELECTED model's own options.
///
/// It exists to answer one question the flat list could not: these controls
/// belong to ONE card in the list below, not to the pipeline. A border and a
/// title that names the model say so.
struct ModelOptionsBox<Content: View>: View {
    let title: String
    let note: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: SettingsLayout.inBlock) {
            Text(title)
                .font(.system(size: 12, weight: .heavy)).kerning(0.3)
                .foregroundColor(Theme.teal)
            Text(note)
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Theme.card.opacity(0.5))
                .overlay(RoundedRectangle(cornerRadius: 10)
                    .stroke(Theme.teal.opacity(0.35), lineWidth: 1))
        )
    }
}

/// Labeled toggle in app style.
struct SettingToggle: View {
    let label: String
    @Binding var isOn: Bool

    var body: some View {
        Toggle(label, isOn: $isOn)
            .font(.system(size: 13, weight: .semibold))
            .foregroundColor(Theme.textPrimary)
            .toggleStyle(.switch)
            .tint(Theme.teal)
    }
}

/// Titled block wrapping any setting control.
struct SettingBlock<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: SettingsLayout.inBlock) {
            Text(title)
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(Theme.textLabel)
            content
        }
    }
}

/// Warning shown when a Remote channel is selected while Voxtral is the chunked
/// model. Measured on this M4, per 30 s chunk: Qwen3 4.3 s, Whisper 4.2 s,
/// Granite 5.6 s, Voxtral 27.0 s — i.e. Voxtral already runs at ~90 % duty for a
/// SINGLE stream, so two streams cannot keep up: chunk N+1 arrives before chunk N
/// finishes and the queue grows without bound.
///
/// Warning only for now — the hard refusal at startup lands with the capture work.
/// Deliberately NOT a silent model fallback: the owner picks models on measured
/// WER and must not have that choice changed behind their back.
struct DualStreamVoxtralWarning: View {
    @AppStorage("chunked.model") private var chunkedModel = "qwen3"
    @AppStorage(MicrophoneSettings.remoteChannelKey) private var remoteChannel = -1

    var body: some View {
        if remoteChannel >= 0 && chunkedModel == "voxtral" {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 11, weight: .bold))
                Text("Voxtral cannot keep up with two streams — it needs about 27 s to transcribe a 30 s chunk (Qwen3 4.3 s, Whisper 4.2 s, Granite 5.6 s), so the second stream would fall permanently behind. Pick another chunked model, or turn the Remote channel off.")
                    .fixedSize(horizontal: false, vertical: true)
            }
            .font(.system(size: 11))
            .foregroundColor(Theme.red)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Theme.red.opacity(0.10))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Theme.red.opacity(0.35), lineWidth: 1)
                    )
            )
        }
    }
}

/// Language dropdown listing exactly one model's languages.
///
/// There is no default list any more: the caller states which model's roster to
/// show (`Languages.pickerEntries(forModel:)` for a chunked model,
/// `Languages.realtime` for Nemotron). One shared list was wrong in both
/// directions at once — see the `Languages` doc comment.
struct LanguagePicker: View {
    @Binding var selection: String
    let entries: [Languages.Entry]

    var body: some View {
        Picker("", selection: $selection) {
            ForEach(entries, id: \.code) { lang in
                Text(lang.name).tag(lang.code)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .tint(Theme.teal)
        .frame(width: 220, alignment: .leading)
    }
}
