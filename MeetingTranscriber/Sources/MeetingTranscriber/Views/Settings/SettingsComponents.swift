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

            installedBadge
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

    private var installedBadge: some View {
        let installed = ModelCatalog.isInstalled(model)
        return Text(installed ? "INSTALLED" : "NOT DOWNLOADED")
            .font(.system(size: 9, weight: .heavy))
            .kerning(0.5)
            .foregroundColor(installed ? Theme.teal : Theme.red)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 5)
                    .fill((installed ? Theme.teal : Theme.red).opacity(0.12))
            )
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
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(Theme.textLabel)
            content
        }
        .padding(.top, 4)
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
