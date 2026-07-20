import SwiftUI

/// Settings window: left rail with sections, "Models" section holds 3 sub-tabs.
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss

    // Main sections (left rail) — room to grow (General, Export, …)
    enum Section: String, CaseIterable {
        case models = "Models"
        case microphone = "Microphone"
        case atnd = "ATND"

        var icon: String {
            switch self {
            case .models: return "cpu"
            case .microphone: return "mic.fill"
            case .atnd: return "antenna.radiowaves.left.and.right"
            }
        }

        /// Header line under the section title (sections with sub-tabs use theirs).
        var subtitle: String {
            switch self {
            case .models, .atnd: return ""
            case .microphone: return "Input device and channel selection"
            }
        }
    }

    // Sub-tabs inside Models
    enum ModelTab: String, CaseIterable, SettingsSubTab {
        case realtime = "Realtime"
        case chunked = "Chunked"
        case vad = "VAD"
        case diarization = "Diarization"
        case overlap = "Overlap"

        var title: String { rawValue }

        var icon: String {
            switch self {
            case .realtime: return "bolt.fill"
            case .chunked: return "doc.text.fill"
            case .vad: return "waveform.and.mic"
            case .diarization: return "person.2.fill"
            case .overlap: return "person.wave.2.fill"
            }
        }

        var subtitle: String {
            switch self {
            case .realtime: return "Live captions while recording"
            case .chunked: return "Accurate final transcript"
            case .vad: return "Voice activity detection"
            case .diarization: return "Speaker identification — who spoke when"
            case .overlap: return "Overlap recovery — separate simultaneous speech"
            }
        }
    }

    // Sub-tabs inside ATND
    enum ATNDSubTab: String, CaseIterable, SettingsSubTab {
        case connection = "Connection"
        case command = "Command"
        case position = "Position ID"

        var title: String { rawValue }

        var icon: String {
            switch self {
            case .connection: return "network"
            case .command: return "terminal.fill"
            case .position: return "person.line.dotted.person.fill"
            }
        }

        var subtitle: String {
            switch self {
            case .connection: return "ATND1061 beamforming array — beam and talker data over IP"
            case .command: return "Send IP control commands to the array"
            case .position: return "Speaker identity from talker direction — replaces voice-based labels while the array is connected"
            }
        }
    }

    @State private var section: Section = .models
    @State private var modelTab: ModelTab = .realtime
    @State private var atndTab: ATNDSubTab = .connection
    @Namespace private var tabIndicator

    var body: some View {
        HStack(spacing: 0) {
            leftRail
            Rectangle().fill(Theme.divider).frame(width: 1)
            content
        }
        .frame(minWidth: 860, idealWidth: 920, maxWidth: .infinity,
               minHeight: 540, idealHeight: 620, maxHeight: .infinity)
        .background(Theme.sidebar)
    }

    // MARK: Left rail

    private var leftRail: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("SETTINGS")
                .font(.system(size: 13, weight: .heavy))
                .kerning(1.2)
                .foregroundColor(Theme.textPrimary)
                .padding(.bottom, 14)

            ForEach(Section.allCases, id: \.self) { s in
                Button(action: {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.85)) { section = s }
                }) {
                    HStack(spacing: 10) {
                        Image(systemName: s.icon)
                            .font(.system(size: 12, weight: .bold))
                            .frame(width: 16)
                        Text(s.rawValue)
                            .font(.system(size: 13, weight: .bold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.85)
                        Spacer(minLength: 0)
                    }
                    .foregroundColor(section == s ? Theme.selectedTabText : Theme.textSecondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(section == s ? Theme.selectedTabBackground : .clear)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }

            Spacer()

            Button(action: { dismiss() }) {
                HStack(spacing: 8) {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .bold))
                    Text("Close")
                        .font(.system(size: 12, weight: .bold))
                }
                .foregroundColor(Theme.textSubtle)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(RoundedRectangle(cornerRadius: 8).fill(Theme.chip))
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .frame(width: 200)
    }

    // MARK: Content

    private var content: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 3) {
                Text(section.rawValue)
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                Text(headerSubtitle)
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textDim)
                    .animation(.none, value: modelTab)
                    .animation(.none, value: atndTab)
            }
            .padding(.horizontal, 24)
            .padding(.top, 20)
            .padding(.bottom, 14)

            if section == .models {
                SubTabBar(selection: $modelTab, geometryID: "activeModelTab", namespace: tabIndicator)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
            } else if section == .atnd {
                SubTabBar(selection: $atndTab, geometryID: "activeATNDTab", namespace: tabIndicator)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
            }

            Rectangle().fill(Theme.divider).frame(height: 1)

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    switch section {
                    case .models:
                        switch modelTab {
                        case .realtime:    RealtimeModelTab()
                        case .chunked:     ChunkedModelTab()
                        case .vad:         VADTab()
                        case .diarization: DiarizationTab()
                        case .overlap:     OverlapTab()
                        }
                    case .microphone:
                        MicrophoneTab()
                    case .atnd:
                        switch atndTab {
                        case .connection: ATNDConnectionTab()
                        case .command:    ATNDCommandTab()
                        case .position:   ATNDPositionTab()
                        }
                    }
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
                .id("\(section)-\(modelTab)-\(atndTab)") // re-render for transition
                .transition(.opacity.combined(with: .move(edge: .trailing)))
            }
            .animation(.spring(response: 0.32, dampingFraction: 0.86), value: modelTab)
            .animation(.spring(response: 0.32, dampingFraction: 0.86), value: atndTab)
            .animation(.spring(response: 0.32, dampingFraction: 0.86), value: section)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
    }

    /// Sections with sub-tabs show the active sub-tab's subtitle.
    private var headerSubtitle: String {
        switch section {
        case .models: return modelTab.subtitle
        case .atnd:   return atndTab.subtitle
        default:      return section.subtitle
        }
    }
}
