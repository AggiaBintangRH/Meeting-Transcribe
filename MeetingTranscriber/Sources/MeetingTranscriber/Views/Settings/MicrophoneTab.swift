import SwiftUI

/// Settings → Microphone: input device dropdown + per-channel role assignment.
///
/// Two roles share one device: **Office** (the room, e.g. the ATND array) and an
/// optional **Remote** (the conferencing app's audio arriving on a loopback
/// input). Both must come from the SAME device — see `MicrophoneSettings` for why
/// two engines/clocks are not an option.
///
/// The three rules are enforced by the controls themselves, not just described:
/// Office is a radio (exactly one, always); Remote is a radio with an explicit
/// "Off" (at most one); the Office channel is simply absent from the Remote row,
/// so they can never collide; and the whole Remote block only exists when the
/// device has 2+ channels, which settles the single-channel-device case by
/// construction.
struct MicrophoneTab: View {
    @AppStorage("mic.deviceUID") private var deviceUID = ""
    @AppStorage("mic.channel")   private var channel = 0
    /// -1 = no Remote stream (single-stream, today's behaviour).
    @AppStorage(MicrophoneSettings.remoteChannelKey) private var remoteChannel = -1

    @State private var devices: [AudioInputDevice] = []

    private var selectedDevice: AudioInputDevice? {
        devices.first { $0.uid == deviceUID }
    }

    var body: some View {
        Group {
            // Device dropdown
            SettingBlock(title: "Input device") {
                HStack(spacing: 10) {
                    Picker("", selection: $deviceUID) {
                        Text("Select a microphone…").tag("")
                        ForEach(devices) { device in
                            Text("\(device.name)  (\(device.channelCount) ch)")
                                .tag(device.uid)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .tint(Theme.teal)
                    .frame(minWidth: 180, maxWidth: 400, alignment: .leading)

                    Button(action: refresh) {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(Theme.textSecondary)
                            .frame(width: 28, height: 28)
                            .background(RoundedRectangle(cornerRadius: 7).fill(Theme.chip))
                    }
                    .buttonStyle(.plain)
                    .help("Rescan input devices")
                }
            }

            // Channel checkboxes — one per channel the device actually has
            if let device = selectedDevice {
                SettingBlock(title: "Office channel — only one can be active") {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 10)],
                              alignment: .leading, spacing: 10) {
                        ForEach(0..<device.channelCount, id: \.self) { ch in
                            channelCheckbox(ch, role: .office)
                        }
                    }
                    Text("The people in the room. Recording uses channel \(channel + 1) of \(device.name).")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .transition(.opacity.combined(with: .move(edge: .top)))

                // Remote is only meaningful when there is a second channel to put it on.
                if device.channelCount >= 2 {
                    SettingBlock(title: "Remote channel — optional, for people joining online") {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 10)],
                                  alignment: .leading, spacing: 10) {
                            remoteOffCheckbox()
                            // The Office channel is not offered here at all, so the
                            // two roles cannot land on the same channel.
                            ForEach((0..<device.channelCount).filter { $0 != channel }, id: \.self) { ch in
                                channelCheckbox(ch, role: .remote)
                            }
                        }
                        Text(remoteCaption)
                            .font(.system(size: 11))
                            .foregroundColor(Theme.textFaint)
                            .fixedSize(horizontal: false, vertical: true)

                        DualStreamVoxtralWarning()
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            } else if !devices.isEmpty {
                Text("Choose a microphone to see its channels.")
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textDim)
            } else {
                Text("No input devices found.")
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textDim)
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.85), value: deviceUID)
        .animation(.spring(response: 0.3, dampingFraction: 0.85), value: remoteChannel)
        .onAppear { refresh() }
        .onChange(of: deviceUID) { _, _ in
            clampChannel()
        }
        .onChange(of: channel) { _, _ in
            // Office just moved onto the channel Remote was using — drop Remote
            // rather than shuffle it somewhere the user did not pick.
            if remoteChannel == channel { remoteChannel = -1 }
        }
    }

    /// Explains the Aggregate Device requirement, because "one device, two
    /// channels" is the part that is not obvious — a loopback input is a separate
    /// macOS device until the user combines it with the array in Audio MIDI Setup.
    private var remoteCaption: String {
        if remoteChannel < 0 {
            return "Off — one stream, as before. Both streams have to come from this one device, so a hybrid meeting needs an Aggregate Device (Audio MIDI Setup) combining the room array with the loopback input carrying the conferencing app's audio."
        }
        return "Channel \(remoteChannel + 1) carries the online participants — normally a loopback input combined with the room array into one Aggregate Device (Audio MIDI Setup). Keeping both on one device keeps them on one clock, so the two streams stay aligned over a long meeting."
    }

    // MARK: - Components

    /// Which stream a channel tile assigns. Office is required; Remote is optional.
    private enum ChannelRole {
        case office, remote

        var accent: Color { self == .office ? Theme.teal : Theme.remoteRole }
        var border: Color { self == .office ? Theme.toggleOnBorder : Theme.remoteRoleBorder }
    }

    private func channelCheckbox(_ ch: Int, role: ChannelRole) -> some View {
        let isOn = role == .office ? channel == ch : remoteChannel == ch
        return Button(action: {
            switch role {
            case .office: channel = ch                              // radio: one is always on
            case .remote: remoteChannel = isOn ? -1 : ch            // radio + tap-to-clear
            }
        }) {
            tile(label: "Channel \(ch + 1)", isOn: isOn, role: role)
        }
        .buttonStyle(.plain)
    }

    /// Explicit "no Remote stream" tile — Remote must be switchable back off, and
    /// an always-visible Off reads better than "tap the active one again".
    private func remoteOffCheckbox() -> some View {
        Button(action: { remoteChannel = -1 }) {
            tile(label: "Off", isOn: remoteChannel < 0, role: .remote)
        }
        .buttonStyle(.plain)
    }

    private func tile(label: String, isOn: Bool, role: ChannelRole) -> some View {
        HStack(spacing: 7) {
            Image(systemName: isOn ? "checkmark.square.fill" : "square")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(isOn ? role.accent : Theme.textFaint)
            Text(label)
                .font(.system(size: 12, weight: .semibold))
                .lineLimit(1)
                .foregroundColor(isOn ? Theme.textPrimary : Theme.textMuted)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Theme.card)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isOn ? role.border : .clear, lineWidth: 1)
                )
        )
    }

    // MARK: - Logic

    private func refresh() {
        devices = AudioDeviceManager.inputDevices()
        // Auto-select the only device, or keep a valid saved selection
        if selectedDevice == nil {
            if let first = devices.first, devices.count == 1 {
                deviceUID = first.uid
            } else if !devices.contains(where: { $0.uid == deviceUID }) {
                deviceUID = ""
            }
        }
        clampChannel()
    }

    /// Keep both roles valid for the selected device (reset if out of range).
    /// Mirrors `MicrophoneSettings.resolve` — an invalid Remote goes back to off
    /// rather than being relocated to a channel the user never chose.
    private func clampChannel() {
        guard let device = selectedDevice else { return }
        if channel >= device.channelCount || channel < 0 { channel = 0 }
        if remoteChannel >= device.channelCount || remoteChannel == channel { remoteChannel = -1 }
    }
}
