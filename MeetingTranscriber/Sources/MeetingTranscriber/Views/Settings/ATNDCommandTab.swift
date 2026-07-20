import SwiftUI

/// Settings → ATND → Command: send IP control commands to the ATND1061.
///
/// Every row shows three separate things, and keeping them separate is the
/// whole design:
///
/// - the **control**, which is what you want;
/// - the **send result** (ACK/NAK), which is what the array said about the
///   message;
/// - the **device value**, which is what the array is actually set to.
///
/// They are not the same. An ACK only means the array accepted the message —
/// so a toggle glows from the device value alone, never from what was last
/// sent, and every action re-reads the array afterwards to find out what
/// really happened.
struct ATNDCommandTab: View {
    @ObservedObject private var control = ATNDControlService.shared
    @ObservedObject private var model = ATNDCommandModel.shared

    /// Locally-held choices, live only while a Set is in flight. Once the array
    /// has answered and been re-read, the device value takes over again — which
    /// is what makes a refused change snap back on its own.
    @State private var roomTypePending: Int?
    @State private var beamPending: Int?
    @State private var weightPending: Int?

    @State private var cameraText = ""
    @State private var levelText = ""
    @State private var cameraError: String?
    @State private var levelError: String?

    private var connected: Bool { control.isConnected }

    var body: some View {
        Group {
            if !connected { disconnectedBanner }

            header

            SettingBlock(title: "Device") {
                deviceIDRow
            }

            SettingBlock(title: "Mute") {
                glowRow("Device mute", "Mutes the whole array (s_mute).",
                        row: model.deviceMute) { await model.setDeviceMute($0) }
                ForEach(ATNDCommands.OutputMute.channels, id: \.self) { channel in
                    glowRow(ATNDCommands.OutputMute.name(channel),
                            "Mutes output channel \(channel) only (s_output_mute).",
                            row: model.outputMute[channel]) {
                        await model.setOutputMute(channel: channel, muted: $0)
                    }
                }
            }

            SettingBlock(title: "Audio processing") {
                glowRow("Automatic gain control", "Levels quiet and loud talkers (s_agc).",
                        row: model.agc) { await model.setAGC($0) }
                aecRow
            }

            SettingBlock(title: "Notifications") {
                Text("The three switches the array's own broadcast depends on. All three live in one setting (s_network), so each change sends only that field and leaves the rest — including the IP address — untouched.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                glowRow("Notification", "Talker position and status broadcast.",
                        row: model.notification) { await model.setNotification($0) }
                glowRow("Audio level notification", "Per-beam level broadcast.",
                        row: model.audioLevelNotification) { await model.setAudioLevelNotification($0) }
                glowRow("Camera control notification", "Camera control broadcast.",
                        row: model.cameraNotification) { await model.setCameraNotification($0) }
            }

            roomAndBeam
            smartMixBlock
            intervalBlock
            vadBlock
        }
        .task(id: control.state) {
            if connected {
                await model.refreshAll()
            } else {
                // Nothing on screen may outlive the link it came from.
                model.reset()
                cameraText = ""
                levelText = ""
            }
        }
        .onChange(of: model.cameraInterval.device) { _, new in
            if case .value(let ms) = new { cameraText = String(ms) }
        }
        .onChange(of: model.levelInterval.device) { _, new in
            if case .value(let ms) = new { levelText = String(ms) }
        }
    }

    // MARK: - Blocks

    private var roomAndBeam: some View {
        SettingBlock(title: "Room & beams") {
            segmentedRow("Room type", "How reverberant the room is (s_roomtype).",
                         row: model.roomType,
                         options: Array(ATNDCommands.RoomType.range).map { ($0, ATNDCommands.RoomType.name($0)) },
                         pending: $roomTypePending) { await model.setRoomType($0) }

            segmentedRow("Beam sensitivity", "How readily a beam locks onto a talker (s_audio_system).",
                         row: model.beamSensitivity,
                         options: Array(ATNDCommands.AudioSystem.beamSensitivityRange).map { ($0, ATNDCommands.beamSensitivityName($0)) },
                         pending: $beamPending) { await model.setBeamSensitivity($0) }
        }
    }

    private var smartMixBlock: some View {
        SettingBlock(title: "Gain share weight") {
            HStack(spacing: 10) {
                Text("Beam")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(Theme.textMuted)
                Picker("", selection: Binding(
                    get: { model.smartMixChannel },
                    set: { channel in Task { await model.selectSmartMixChannel(channel) } }
                )) {
                    ForEach(ATNDCommands.SmartMix.channelRange, id: \.self) { channel in
                        Text("\(channel + 1)").tag(channel)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 240)
                .disabled(!connected || model.refreshing)
                Spacer(minLength: 0)
            }

            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Weight")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(Theme.textPrimary)
                    Text("Gain Share weight for this beam, −15.0 to +15.0 dB in 0.5 dB steps (s_smart_mix). The array stores it as 0–60.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                sendChip(model.smartMixWeight.send)
                deviceChip(model.smartMixWeight.device) { raw in
                    String(format: "%+.1f dB", ATNDCommands.smartMixWeightDB(raw))
                }
            }

            let weight = weightPending ?? deviceInt(model.smartMixWeight.device)
            HStack(spacing: 10) {
                Stepper(value: Binding(
                    get: { weight ?? 30 },
                    set: { value in
                        weightPending = value
                        Task {
                            await model.setSmartMixWeight(value)
                            weightPending = nil
                        }
                    }
                ), in: ATNDCommands.SmartMix.weightRange) {
                    Text(weight.map { String(format: "%+.1f dB", ATNDCommands.smartMixWeightDB($0)) } ?? "—")
                        .font(.system(size: 12, weight: .semibold).monospaced())
                        .foregroundColor(Theme.textBody)
                        .frame(width: 70, alignment: .leading)
                }
                .disabled(!rowUsable(model.smartMixWeight))
                Spacer(minLength: 0)
            }

            note(model.smartMixWeight.note)
        }
    }

    private var intervalBlock: some View {
        SettingBlock(title: "Notification intervals") {
            intervalRow("Talker position interval",
                        "How often the array reports talker position, 100–300000 ms (s_camera_control_interval). Its own default is 100 ms.",
                        row: model.cameraInterval, text: $cameraText, error: $cameraError) {
                await model.setCameraInterval($0)
            }
            intervalRow("Level meter interval",
                        "How often the array reports beam levels, 100–300000 ms (s_level_meter_interval).",
                        row: model.levelInterval, text: $levelText, error: $levelError) {
                await model.setLevelInterval($0)
            }
        }
    }

    private var vadBlock: some View {
        SettingBlock(title: "Voice activity detection") {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("VAD")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(Theme.textPrimary)
                    Text("Switches the array's voice activity detection on or off (SVAD).")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                }
                Spacer(minLength: 8)
                sendChip(model.vadSend)
                plainButton("Turn On")  { await model.sendVAD(true) }
                plainButton("Turn Off") { await model.sendVAD(false) }
            }

            Text("The array does not report VAD state (no read command) — only the send result is shown.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var disconnectedBanner: some View {
        HStack(spacing: 9) {
            Image(systemName: "bolt.horizontal.circle")
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(Theme.amber)
            Text("Not connected — open the Connection tab and connect first.")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(Theme.textSecondary)
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 9).fill(Theme.card))
    }

    private var header: some View {
        HStack(spacing: 10) {
            Button(action: { Task { await model.refreshAll() } }) {
                HStack(spacing: 7) {
                    if model.refreshing {
                        ProgressView()
                            .controlSize(.small)
                            .scaleEffect(0.7)
                            .frame(width: 11, height: 11)
                    } else {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 11, weight: .bold))
                    }
                    Text("Refresh")
                        .font(.system(size: 12, weight: .bold))
                }
                .foregroundColor(Theme.selectedTabText)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(RoundedRectangle(cornerRadius: 8).fill(Theme.teal))
            }
            .buttonStyle(.plain)
            .disabled(!connected || model.refreshing)
            .opacity(connected && !model.refreshing ? 1 : 0.5)

            Text(model.lastRefresh.map { "Last refreshed \(clock($0))" } ?? "Not read yet")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
            Spacer(minLength: 0)
        }
    }

    private var deviceIDRow: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Device ID")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Text("The array's own ID, 00–FF (g_deviceid). Read-only here.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }
            Spacer(minLength: 8)
            deviceChip(model.deviceID.device) { $0 }
        }
    }

    /// AEC is deliberately the odd one out.
    ///
    /// The spec's AEC table and the spec's own AEC example contradict each other
    /// (12 fields vs 11, and an "AEC Enable" of 2 that no reading allows), so
    /// only field 0 is trustworthy — and only when the live answer actually
    /// reads 0 or 1. When it does not, the raw answer is shown as-is and the
    /// toggle stays disabled: an unverifiable index is a wrong write to real
    /// hardware, and no control at all beats that.
    @ViewBuilder
    private var aecRow: some View {
        glowRow("Acoustic echo cancellation", "Cancels far-end echo (s_aec_general).",
                row: model.aec) { await model.setAEC($0) }

        if case .unreadable(let why) = model.aec.device {
            VStack(alignment: .leading, spacing: 4) {
                Text("This array's AEC answer does not match the layout in the spec — \(why). The control is disabled rather than guessing which field to write.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.amber)
                    .fixedSize(horizontal: false, vertical: true)
                Text(model.aecRaw.isEmpty ? "(no parameters)" : model.aecRaw)
                    .font(.system(size: 11).monospaced())
                    .foregroundColor(Theme.textDim)
                    .textSelection(.enabled)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 6)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Theme.chip))
            }
        }
    }

    // MARK: - Row shapes

    /// A power button that glows from the device's own value and nothing else.
    @ViewBuilder
    private func glowRow(_ title: String, _ subtitle: String,
                         row: Row<Bool>,
                         action: @escaping (Bool) async -> Void) -> some View {
        let isOn = row.device == .value(true)
        let usable = rowUsable(row)

        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Text(subtitle)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            sendChip(row.send)
            deviceChip(row.device) { $0 ? "On" : "Off" }

            Button(action: { Task { await action(!isOn) } }) {
                Image(systemName: "power")
                    .font(.system(size: 13, weight: .heavy))
                    .foregroundColor(isOn ? Theme.selectedTabText : Theme.textMuted)
                    .frame(width: 34, height: 34)
                    .background(Circle().fill(isOn ? Theme.teal : Theme.chip))
                    // The glow is the device's state made visible. It must never
                    // be driven by an optimistic local value.
                    .shadow(color: isOn ? Theme.teal.opacity(0.6) : .clear, radius: 8)
            }
            .buttonStyle(.plain)
            .disabled(!usable)
            .opacity(usable ? 1 : 0.45)
        }
        note(row.note)
    }

    @ViewBuilder
    private func segmentedRow(_ title: String, _ subtitle: String,
                              row: Row<Int>,
                              options: [(Int, String)],
                              pending: Binding<Int?>,
                              action: @escaping (Int) async -> Void) -> some View {
        let device = deviceInt(row.device)
        let usable = rowUsable(row)

        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Text(subtitle)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            sendChip(row.send)
            deviceChip(row.device) { value in
                options.first { $0.0 == value }?.1 ?? String(value)
            }
        }

        // Selection falls back to the device value the moment nothing is in
        // flight, so a refused change snaps back with no extra bookkeeping.
        Picker("", selection: Binding(
            get: { pending.wrappedValue ?? device ?? options[0].0 },
            set: { value in
                pending.wrappedValue = value
                Task {
                    await action(value)
                    pending.wrappedValue = nil
                }
            }
        )) {
            ForEach(options, id: \.0) { option in
                Text(option.1).tag(option.0)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(maxWidth: 320, alignment: .leading)
        .disabled(!usable)
        .opacity(device == nil ? 0.5 : 1)

        note(row.note)
    }

    @ViewBuilder
    private func intervalRow(_ title: String, _ subtitle: String,
                             row: Row<Int>,
                             text: Binding<String>,
                             error: Binding<String?>,
                             action: @escaping (Int) async -> Void) -> some View {
        let usable = rowUsable(row)

        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Text(subtitle)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            sendChip(row.send)
            deviceChip(row.device) { "\($0) ms" }
        }

        HStack(spacing: 8) {
            TextField("ms", text: text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12).monospaced())
                .frame(width: 90)
                .disabled(!usable)

            ForEach([100, 200, 500, 1000], id: \.self) { preset in
                // Presets fill the field; they do not send. Nothing here reaches
                // the array until Apply is pressed.
                Button(action: { text.wrappedValue = String(preset); error.wrappedValue = nil }) {
                    Text(preset == 1000 ? "1 s" : "\(preset) ms")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(Theme.textSecondary)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Theme.chip))
                }
                .buttonStyle(.plain)
                .disabled(!usable)
            }

            Button(action: {
                guard let ms = Int(text.wrappedValue.trimmingCharacters(in: .whitespaces)),
                      ATNDCommands.intervalRange.contains(ms) else {
                    error.wrappedValue = "Enter a whole number of milliseconds between 100 and 300000."
                    return
                }
                error.wrappedValue = nil
                Task { await action(ms) }
            }) {
                Text("Apply")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(Theme.selectedTabText)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(RoundedRectangle(cornerRadius: 7).fill(Theme.teal))
            }
            .buttonStyle(.plain)
            .disabled(!usable)
            .opacity(usable ? 1 : 0.5)

            Spacer(minLength: 0)
        }

        if let message = error.wrappedValue {
            Text(message)
                .font(.system(size: 11))
                .foregroundColor(Theme.red)
        }
        note(row.note)
    }

    // MARK: - Chips

    private func sendChip(_ status: SendStatus) -> some View {
        Group {
            switch status {
            case .idle:
                EmptyView()
            case .sending:
                chip {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.55)
                        .frame(width: 9, height: 9)
                    Text("Sending…").foregroundColor(Theme.textSecondary)
                }
            case .ack(let at):
                chip {
                    Image(systemName: "checkmark").foregroundColor(Theme.teal)
                    Text("ACK \(clock(at))").foregroundColor(Theme.textSecondary)
                }
            case .nak(let code, let at):
                chip {
                    Image(systemName: "xmark").foregroundColor(Theme.red)
                    Text("NAK \(code) · \(ATNDCommands.nakText(code)) \(clock(at))")
                        .foregroundColor(Theme.textSecondary)
                }
            case .timeout:
                chip {
                    Image(systemName: "clock").foregroundColor(Theme.amber)
                    Text("No reply").foregroundColor(Theme.textSecondary)
                }
            case .failed(let why):
                chip {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundColor(Theme.red)
                    Text(why).foregroundColor(Theme.textSecondary).lineLimit(1)
                }
            }
        }
        .font(.system(size: 10, weight: .semibold))
    }

    /// What the array says it is — the only thing a glow is ever allowed to
    /// come from.
    private func deviceChip<V>(_ value: DeviceValue<V>,
                               describe: (V) -> String) -> some View {
        Group {
            switch value {
            case .unknown:
                chip { Text("—").foregroundColor(Theme.textFaint) }
            case .querying:
                chip {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.55)
                        .frame(width: 9, height: 9)
                    Text("Reading…").foregroundColor(Theme.textFaint)
                }
            case .value(let v):
                chip {
                    Text(describe(v))
                        .foregroundColor(Theme.textBright)
                        .monospacedDigit()
                }
            case .unreadable(let why):
                chip {
                    Text("unreadable")
                        .foregroundColor(Theme.amber)
                }
                .help(why)
            }
        }
        .font(.system(size: 10, weight: .semibold))
        .frame(minWidth: 62, alignment: .trailing)
    }

    private func chip<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 5) { content() }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(RoundedRectangle(cornerRadius: 6).fill(Theme.card))
    }

    @ViewBuilder
    private func note(_ text: String?) -> some View {
        if let text {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.circle.fill")
                    .font(.system(size: 10))
                    .foregroundColor(Theme.amber)
                Text(text)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.amber)
            }
        }
    }

    private func plainButton(_ title: String, action: @escaping () async -> Void) -> some View {
        Button(action: { Task { await action() } }) {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(Theme.textSecondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(RoundedRectangle(cornerRadius: 7).fill(Theme.chip))
        }
        .buttonStyle(.plain)
        .disabled(!connected || model.vadSend == .sending)
        .opacity(connected ? 1 : 0.5)
    }

    // MARK: - Helpers

    /// A row can be driven only when the link is up and the array's own value is
    /// known. Unknown, unreadable, mid-read or mid-send all mean hands off.
    private func rowUsable<V>(_ row: Row<V>) -> Bool {
        guard connected, row.send != .sending else { return false }
        if case .value = row.device { return true }
        return false
    }

    private func deviceInt(_ value: DeviceValue<Int>) -> Int? {
        if case .value(let v) = value { return v }
        return nil
    }

    private func clock(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: date)
    }
}
