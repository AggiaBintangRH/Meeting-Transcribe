import SwiftUI

/// Settings → Microphone: input device dropdown + single-select channel checkboxes.
struct MicrophoneTab: View {
    @AppStorage("mic.deviceUID") private var deviceUID = ""
    @AppStorage("mic.channel")   private var channel = 0

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
                SettingBlock(title: "Channel — only one can be active") {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 10)],
                              alignment: .leading, spacing: 10) {
                        ForEach(0..<device.channelCount, id: \.self) { ch in
                            channelCheckbox(ch)
                        }
                    }
                    Text("Recording uses channel \(channel + 1) of \(device.name).")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
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
        .onAppear { refresh() }
        .onChange(of: deviceUID) { _, _ in
            clampChannel()
        }
    }

    // MARK: - Components

    private func channelCheckbox(_ ch: Int) -> some View {
        let isOn = channel == ch
        return Button(action: { channel = ch }) { // radio behavior: only one active
            HStack(spacing: 7) {
                Image(systemName: isOn ? "checkmark.square.fill" : "square")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(isOn ? Theme.teal : Theme.textFaint)
                Text("Channel \(ch + 1)")
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
                            .stroke(isOn ? Theme.toggleOnBorder : .clear, lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
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

    /// Keep the channel valid for the selected device (reset if out of range).
    private func clampChannel() {
        guard let device = selectedDevice else { return }
        if channel >= device.channelCount || channel < 0 { channel = 0 }
    }
}
