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

/// Language dropdown fed by Languages.all.
struct LanguagePicker: View {
    @Binding var selection: String

    var body: some View {
        Picker("", selection: $selection) {
            ForEach(Languages.all, id: \.code) { lang in
                Text(lang.name).tag(lang.code)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .tint(Theme.teal)
        .frame(width: 220, alignment: .leading)
    }
}
