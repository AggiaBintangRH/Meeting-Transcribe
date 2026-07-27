import SwiftUI

/// Settings → ATND → Connection: how to reach the Audio-Technica ATND1061
/// beamforming array microphone (IP Control spec v9.0).
///
/// Order matters and mirrors the device, not just the layout. TCP comes first
/// because the array's multicast address, its port, and the three notification
/// switches that decide whether anything is broadcast at all are all held on
/// the array and only reachable over TCP (`g_network` / `s_network`). Until
/// that read happens, the multicast group below is a guess at the factory
/// default — which is why it stays locked.
///
/// Settings only: nothing here connects yet, so the tab stores values and
/// deliberately claims no live status.
struct ATNDConnectionTab: View {
    @AppStorage("atnd.enabled")          private var enabled = false
    @AppStorage("atnd.deviceIP")         private var deviceIP = ""
    @AppStorage("atnd.controlPort")      private var controlPort = 17300
    @AppStorage("atnd.multicastGroup")   private var multicastGroup = "239.0.0.100"
    @AppStorage("atnd.multicastPort")    private var multicastPort = 17000
    @AppStorage("atnd.interfaceIP")      private var interfaceIP = ""

    @ObservedObject private var control = ATNDControlService.shared

    /// What Auto currently resolves to, shown so the field is not a blank box
    /// the user has to guess at. Recomputed when the device IP changes, since
    /// that is what the routing lookup is based on.
    @State private var detectedInterface: String?

    private var tcpConnected: Bool { control.isConnected }

    private func refreshDetectedInterface() {
        detectedInterface = ATNDBeamService.localAddressRouting(toDeviceAt: deviceIP)
    }

    var body: some View {
        Group {
            Text("Audio-Technica ATND1061 beamforming array. The array reports which of its 6 beams is speaking, and each beam's level, over the network.")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

            SettingToggle(label: "Use ATND1061 beam data", isOn: $enabled)

            // MARK: 1 — TCP

            SettingBlock(title: "1 · TCP control connection") {
                HStack(spacing: 10) {
                    field("Device IP", text: $deviceIP, width: 150, placeholder: "192.168.0.10")
                    field("Port", value: $controlPort, width: 80)
                }
                .disabled(!enabled || control.state == .connecting)

                HStack(spacing: 10) {
                    connectButton
                    statusChip
                    Spacer(minLength: 0)
                }

                Text("The array's address on your network. Its factory default control port is 17300. This connection carries commands both ways — it is how the app learns the array's multicast settings and switches its notifications on.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // MARK: 2 — UDP, gated behind TCP

            SettingBlock(title: "2 · UDP multicast notifications") {
                HStack(spacing: 10) {
                    field("Group", text: $multicastGroup, width: 150)
                    field("Port", value: $multicastPort, width: 80)
                }
                .disabled(!tcpConnected)

                HStack(spacing: 8) {
                    // Placeholder carries the resolved address, so leaving the
                    // field empty still SHOWS which interface will be used —
                    // an empty box gave no way to tell Auto from broken.
                    field("Interface", text: $interfaceIP, width: 150,
                          placeholder: detectedInterface.map { "Auto — \($0)" } ?? "Auto")
                        .disabled(!enabled)

                    if interfaceIP.isEmpty, let detected = detectedInterface {
                        Button("Use \(detected)") { interfaceIP = detected }
                            .buttonStyle(.link)
                            .font(.system(size: 11))
                            .disabled(!enabled)
                    }
                }

                Text(tcpConnected
                     ? "The group and port belong to the array, not to this app — these must match what the array is set to, or nothing arrives. Shown are the factory defaults, 239.0.0.100 : 17000 — these two are still typed in by hand, not read back from the array. What the Command tab does read from the array (g_network) are the three notification switches that decide whether it broadcasts at all."
                     : "Locked until the TCP connection above is up: the group and port live on the array and are only meaningful once it answers. Shown are the factory defaults, 239.0.0.100 : 17000.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text(interfaceIP.isEmpty
                     ? (detectedInterface.map {
                          "Interface is THIS MAC's address, not the array's — do not put the Device IP here. Auto resolved it to \($0) by asking the routing table which interface reaches the array, and the multicast group is joined there. Type an address only to override that."
                       } ?? "Interface is THIS MAC's address, not the array's — do not put the Device IP here. Auto cannot resolve it yet: enter the array's Device IP above and it will appear. Until then the app binds every interface, which can silently receive nothing on a Mac with more than one network.")
                     : "Overriding Auto with \(interfaceIP). This must be THIS MAC's own address on the array's network — never the array's Device IP. Clear the field to go back to automatic detection.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "What this does today") {
                Text("Connect opens the TCP link and holds it, which proves the array is reachable at that address. Once it is up, the Command tab reads and changes the array's own settings over it. No notifications are received yet, and recording is unaffected — beam data is not part of the transcript yet.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("The array's audio is a separate matter from this tab: its analog output is already mixed down to one channel, so beam-per-speaker audio requires the Dante model and a Dante input device, chosen in the Microphone tab.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear { refreshDetectedInterface() }
        // The lookup is derived from the device address, so re-resolve whenever
        // it is edited — including while it is still being typed and is not yet
        // a valid address, which simply yields nil.
        .onChange(of: deviceIP) { refreshDetectedInterface() }
    }

    // MARK: - Components

    private var connectButton: some View {
        let connecting = control.state == .connecting
        let busy = tcpConnected || connecting || control.isRetrying
        return Button(action: {
            if busy {
                control.disconnect()
            } else {
                control.connect(host: deviceIP, port: controlPort)
            }
        }) {
            HStack(spacing: 7) {
                if connecting {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.7)
                        .frame(width: 11, height: 11)
                } else {
                    Image(systemName: busy ? "xmark.circle.fill" : "bolt.horizontal.circle.fill")
                        .font(.system(size: 11, weight: .bold))
                }
                Text(buttonTitle)
                    .font(.system(size: 12, weight: .bold))
            }
            .foregroundColor(busy ? Theme.textSecondary : Theme.selectedTabText)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(busy ? Theme.chip : Theme.teal)
            )
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.5)
    }

    private var buttonTitle: String {
        if control.state == .connecting { return "Cancel" }
        if tcpConnected                 { return "Disconnect" }
        if control.isRetrying           { return "Stop" }
        return "Connect"
    }

    private var statusChip: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(statusColor)
                .frame(width: 7, height: 7)
            Text(statusText)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(Theme.textSecondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(RoundedRectangle(cornerRadius: 7).fill(Theme.card))
    }

    private var statusText: String {
        switch control.state {
        case .disconnected:      return "Not connected"
        case .connecting:        return "Connecting…"
        case .connected:         return "Connected — array is reachable"
        case .failed(let error):
            return control.isRetrying ? "\(error) Reconnecting every 5s…" : error
        }
    }

    private var statusColor: Color {
        switch control.state {
        case .connected:    return Theme.teal
        case .connecting:   return Theme.textMuted
        // Amber, not red: still trying, not given up.
        case .failed:       return control.isRetrying ? Theme.amber : Theme.red
        case .disconnected: return Theme.textFaint
        }
    }

    private func field(_ label: String, text: Binding<String>,
                       width: CGFloat, placeholder: String = "") -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(Theme.textMuted)
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12).monospaced())
                .frame(width: width)
        }
    }

    private func field(_ label: String, value: Binding<Int>, width: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(Theme.textMuted)
            TextField("", value: value, format: .number.grouping(.never))
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12).monospaced())
                .frame(width: width)
        }
    }
}
