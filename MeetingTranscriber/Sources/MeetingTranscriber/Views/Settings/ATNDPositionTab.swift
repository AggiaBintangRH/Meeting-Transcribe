import SwiftUI

/// Settings → ATND → Position ID: derive speaker identity from the ATND1061's
/// talker-direction stream instead of (or alongside) voice-based pyannote
/// labels.
///
/// The controls here only write UserDefaults — the recorder reads them once at
/// `beginCapture()`. Nothing on this tab talks to the array directly; the beam
/// stream is owned by `ATNDBeamService` and consumed by the recorder's
/// `PositionDiarizer`.
struct ATNDPositionTab: View {
    @AppStorage("atnd.position.enabled")     private var enabled = false
    @AppStorage("atnd.position.tauPreset")   private var tauPreset = "normal"
    @AppStorage("atnd.position.tauDeg")      private var tauDeg = 15.0
    @AppStorage("atnd.position.smoothingMs") private var smoothingMs = 400.0
    @AppStorage("atnd.position.mode")        private var mode = "firstcome"
    @AppStorage("atnd.position.source")      private var source = PositionSource.both.rawValue

    private enum TauPreset: String, CaseIterable {
        case tight, normal, loose, custom

        var label: String {
            switch self {
            case .tight:  return "Tight"
            case .normal: return "Normal"
            case .loose:  return "Loose"
            case .custom: return "Custom"
            }
        }

        /// The degree value a preset fills in. `custom` fills nothing.
        var degrees: Double? {
            switch self {
            case .tight:  return 10
            case .normal: return 15
            case .loose:  return 25
            case .custom: return nil
            }
        }
    }

    var body: some View {
        Group {
            SettingToggle(label: "Use talker direction for speaker labels", isOn: $enabled)

            // Which layer the transcript's labels are DRAWN from. Both layers keep
            // running in every mode — this only changes what you see, so it's safe
            // to flip between recordings while calibrating a room.
            SettingBlock(title: "Label source") {
                Picker("", selection: $source) {
                    Text("Both").tag(PositionSource.both.rawValue)
                    Text("Position only").tag(PositionSource.atnd.rawValue)
                    Text("Voice only").tag(PositionSource.pyannote.rawValue)
                    Text("Position timing").tag(PositionSource.atndTimingPyannoteIdentity.rawValue)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 480, alignment: .leading)
                .disabled(!enabled)
                VStack(alignment: .leading, spacing: 6) {
                    sourceRow("Both", "Voice labels win wherever pyannote has a turn; talker direction fills only the stretches it left uncovered.")
                    sourceRow("Position only", "Every row is labeled by talker direction. Voice diarization still runs underneath — its labels just aren't shown.")
                    sourceRow("Voice only", "Pure voice labels, no direction at all. Effectively the same as turning this feature off; it's here so you can A/B the two layers on one recording.")
                    sourceRow("Position timing", "Direction decides where each row starts and ends; voice decides the name. Rows keep voice speaker numbers wherever voice had a turn, so overlap repair and saved speaker profiles still apply to them — unlike \"Position only\".")
                }
                Text("With \"Both\", if voice diarization decides two seats are the same person it will merge them into one speaker — voice wins, so the direction split disappears. That is expected, not a bug: switch to \"Position only\" to see the seats.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Cluster tightness") {
                Picker("", selection: Binding(
                    get: { TauPreset(rawValue: tauPreset) ?? .normal },
                    set: { preset in
                        tauPreset = preset.rawValue
                        if let deg = preset.degrees { tauDeg = deg }
                    }
                )) {
                    ForEach(TauPreset.allCases, id: \.self) { preset in
                        Text(preset.label).tag(preset)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(!enabled)
                Text("How close two talker bearings must be to count as the same speaker.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }

            SettingBlock(title: "Merge threshold — \(Int(tauDeg))°") {
                Slider(value: $tauDeg, in: 5...45, step: 1)
                    .tint(Theme.teal)
                    .disabled(!enabled)
                    .onChange(of: tauDeg) { _, newValue in
                        // Only demote to "custom" when the value doesn't match a
                        // preset. Guarding this way stops a preset tap (which sets
                        // tauDeg) from immediately snapping the picker to Custom.
                        if !TauPreset.allCases.contains(where: { $0.degrees == newValue }) {
                            tauPreset = TauPreset.custom.rawValue
                        }
                    }
                Text("Smaller keeps nearby talkers apart; larger merges them into one speaker.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }

            SettingBlock(title: "Direction smoothing — \(Int(smoothingMs)) ms") {
                Slider(value: $smoothingMs, in: 100...1000, step: 50)
                    .tint(Theme.teal)
                    .disabled(!enabled)
                Text("How long a direction is averaged before it counts — longer is steadier but slower to react.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }

            SettingBlock(title: "Naming") {
                Picker("", selection: $mode) {
                    Text("First-come").tag("firstcome")
                    Text("Enrollment").tag("enrollment")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 320, alignment: .leading)
                .disabled(!enabled)
                Text("First-come auto-labels speakers \"Speaker 1, 2, 3…\" as they first talk (rename them later in the transcript). Enrollment pops a prompt asking for each new speaker's name the moment they first talk.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "What to expect") {
                VStack(alignment: .leading, spacing: 8) {
                    expectRow("This needs the ATND array connected and listening. Whenever it's down — including if it drops mid-recording — labeling falls back to voice-based pyannote, so a single transcript can end up mixing position and voice labels.")
                    expectRow("The array reports only one talker direction at a time, so genuinely simultaneous speech still can't be split into two texts — same hard limit as everywhere else.")
                    expectRow("Two people sitting at nearly the same bearing from the array can merge into one speaker. Raise the merge threshold only if you're over-splitting; lower it if distinct people are collapsing together.")
                    expectRow("Someone who gets up and changes seats will be treated as a new speaker from their new direction.")
                    expectRow("These thresholds are reasoned starting points, NOT tuned on real meetings. Expect to adjust them on your own recordings.")
                }
            }
        }
    }

    /// One "Mode — what it does" line under the source picker.
    @ViewBuilder
    private func sourceRow(_ name: String, _ text: String) -> some View {
        Text("\(name) — ").bold().font(.system(size: 11)).foregroundColor(Theme.textDim)
            + Text(text).font(.system(size: 11)).foregroundColor(Theme.textFaint)
        // (the concatenation is one Text, so it wraps as a single paragraph)
    }

    @ViewBuilder
    private func expectRow(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "info.circle.fill")
                .font(.system(size: 10))
                .foregroundColor(Theme.textMuted)
                .padding(.top, 2)
            Text(text)
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
