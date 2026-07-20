import Foundation
import Combine

/// What the array says one of its settings is.
///
/// The point of this type is that "we have not read it" and "it is off" are
/// different things, and the UI must never blur them. A glow that came from
/// what the app last *sent* would be a lie the moment the array disagreed.
enum DeviceValue<V: Equatable>: Equatable {
    /// Never read, or read attempt abandoned. Shows as "—".
    case unknown
    case querying
    /// Read from the array, this session.
    case value(V)
    /// The array answered, but not with something this app can trust.
    case unreadable(String)
}

/// What happened to the last thing this row sent. Independent of `DeviceValue`
/// on purpose: an ACK means "the array accepted the message", not "the setting
/// now has the value you wanted" — only a re-read can say that.
enum SendStatus: Equatable {
    case idle
    case sending
    case ack(Date)
    case nak(code: String, Date)
    case timeout
    case failed(String)
}

struct Row<V: Equatable>: Equatable {
    var device: DeviceValue<V> = .unknown
    var send: SendStatus = .idle
    /// Set when the verify-read came back disagreeing with what was sent.
    var note: String?
}

/// Backs Settings → ATND → Command.
///
/// Every action follows the same shape: **send → record the ACK/NAK → re-read
/// from the array → show what the array said.** Nothing is ever displayed
/// optimistically.
@MainActor
final class ATNDCommandModel: ObservableObject {
    static let shared = ATNDCommandModel()

    private let control = ATNDControlService.shared

    // MARK: Rows

    @Published var deviceID = Row<String>()
    @Published var deviceMute = Row<Bool>()
    /// Index 0 Analog Out, 1 Auto Mix (`g_output_mute` channel numbering).
    @Published var outputMute: [Row<Bool>] = [Row<Bool>(), Row<Bool>()]
    @Published var agc = Row<Bool>()
    @Published var aec = Row<Bool>()

    /// The three `g_network` switches. One fetch feeds all three — they live in
    /// the same answer, so reading them separately would be three times the
    /// traffic for the same bytes.
    @Published var notification = Row<Bool>()
    @Published var audioLevelNotification = Row<Bool>()
    @Published var cameraNotification = Row<Bool>()

    @Published var roomType = Row<Int>()
    @Published var beamSensitivity = Row<Int>()
    @Published var smartMixWeight = Row<Int>()
    /// Which beam `smartMixWeight` is about — Gain Share is per beam channel.
    @Published var smartMixChannel = 0

    @Published var cameraInterval = Row<Int>()
    @Published var levelInterval = Row<Int>()

    /// VAD has no read command in the spec (SVAD exists, GVAD does not), so it
    /// gets a send status and no device value. There is nothing to glow from.
    @Published var vadSend: SendStatus = .idle

    // MARK: AEC gate
    //
    // The spec contradicts itself about the AEC field layout (see
    // ATNDCommands.AEC). Only field 0 is trustworthy, and only when the live
    // answer actually reads 0 or 1. If it does not, the raw answer is shown
    // read-only and the toggle stays disabled — rather than the app guessing at
    // a write to real hardware.

    /// The whole `g_aec_general` answer as it came in, for the read-only case.
    @Published private(set) var aecRaw = ""
    /// Answer width, so a Set matches whatever the array actually speaks.
    @Published private(set) var aecFieldCount = 0
    var aecUsable: Bool { if case .value = aec.device { return true }; return false }

    // MARK: Sweep

    @Published private(set) var refreshing = false
    @Published private(set) var lastRefresh: Date?

    private init() {}

    // MARK: - Reading

    /// Reads every row, in one sequential pass.
    ///
    /// Sequential is not a compromise: the transport serializes requests anyway
    /// (the protocol has no request IDs), so a "parallel" sweep would queue up
    /// identically while making failures harder to attribute.
    func refreshAll() async {
        guard control.isConnected, !refreshing else { return }
        refreshing = true
        defer { refreshing = false }

        do {
            try await readDeviceID()
            try await readMute()
            try await readOutputMute(0)
            try await readOutputMute(1)
            try await readAGC()
            try await readAEC()
            try await readNetwork()
            try await readRoomType()
            try await readAudioSystem()
            try await readSmartMix(smartMixChannel)
            try await readCameraInterval()
            try await readLevelInterval()
            lastRefresh = Date()
        } catch {
            // A link error means the rest of the sweep never happened. Those
            // rows stay `.unknown` — which is the truth — instead of inheriting
            // a stale value or a fake failure.
        }
    }

    /// Clears everything back to "never read".
    ///
    /// Called the moment the link is not up. A value from a previous session is
    /// worse than no value: the array may have been changed by someone else, or
    /// swapped for a different unit entirely.
    func reset() {
        deviceID = Row<String>()
        deviceMute = Row<Bool>()
        outputMute = [Row<Bool>(), Row<Bool>()]
        agc = Row<Bool>()
        aec = Row<Bool>()
        aecRaw = ""
        aecFieldCount = 0
        notification = Row<Bool>()
        audioLevelNotification = Row<Bool>()
        cameraNotification = Row<Bool>()
        roomType = Row<Int>()
        beamSensitivity = Row<Int>()
        smartMixWeight = Row<Int>()
        cameraInterval = Row<Int>()
        levelInterval = Row<Int>()
        vadSend = .idle
        lastRefresh = nil
    }

    private func readDeviceID() async throws {
        try await read(\.deviceID, command: "g_deviceid") { fields in
            guard let raw = fields.first, ATNDCommands.isDeviceID(raw) else {
                return .unreadable("not a device ID")
            }
            return .value(raw)
        }
    }

    private func readMute() async throws {
        try await read(\.deviceMute, command: "g_mute") { single($0) }
    }

    private func readOutputMute(_ channel: Int) async throws {
        try await read(\.outputMute[channel], command: "g_output_mute", params: String(channel)) { fields in
            guard fields.count == ATNDCommands.OutputMute.fieldCount else {
                return .unreadable("unexpected \(fields.count) fields")
            }
            // The answer echoes the channel. If it echoes a different one, the
            // reply is not about the channel that was asked for.
            guard fields[ATNDCommands.OutputMute.channel] == String(channel) else {
                return .unreadable("answered about channel \(fields[ATNDCommands.OutputMute.channel])")
            }
            guard let muted = ATNDCommands.boolField(fields[ATNDCommands.OutputMute.muted]) else {
                return .unreadable("unexpected value")
            }
            return .value(muted)
        }
    }

    private func readAGC() async throws {
        try await read(\.agc, command: "g_agc") { fields in
            guard fields.count == ATNDCommands.AGC.fieldCount else {
                return .unreadable("unexpected \(fields.count) fields")
            }
            guard let on = ATNDCommands.boolField(fields[ATNDCommands.AGC.enable]) else {
                return .unreadable("unexpected value")
            }
            return .value(on)
        }
    }

    private func readAEC() async throws {
        aec.device = .querying
        do {
            let fields = try await control.get("g_aec_general")
            aecRaw = fields.joined(separator: ",")
            aecFieldCount = fields.count
            // The gate: field 0 must actually read 0 or 1. The spec's own
            // example says "2" here, which no reading of the table allows — so
            // if that is what the array sends, this app does not pretend to
            // understand it.
            guard let enable = fields.first.flatMap(ATNDCommands.boolField) else {
                aec.device = .unreadable(fields.isEmpty
                                         ? "empty answer"
                                         : "AEC Enable reads ‘\(fields[0])’, not 0 or 1")
                return
            }
            aec.device = .value(enable)
        } catch {
            aecRaw = ""
            aecFieldCount = 0
            try failRead(&aec.device, error)
        }
    }

    private func readNetwork() async throws {
        notification.device = .querying
        audioLevelNotification.device = .querying
        cameraNotification.device = .querying
        do {
            let fields = try await control.get("g_network")
            // The gate: g_network's Get layout is 21 fields and the three
            // switches sit at fixed indices in it. A different width means this
            // is not the layout this code was written against, and the indices
            // below would be pointing at whatever happened to be there.
            guard fields.count == ATNDCommands.NetworkGet.fieldCount else {
                setNetwork(.unreadable("unexpected \(fields.count) fields"))
                return
            }
            guard let n = ATNDCommands.boolField(fields[ATNDCommands.NetworkGet.notification]),
                  let a = ATNDCommands.boolField(fields[ATNDCommands.NetworkGet.audioLevelNotification]),
                  let c = ATNDCommands.boolField(fields[ATNDCommands.NetworkGet.cameraControlNotification]) else {
                setNetwork(.unreadable("unexpected values"))
                return
            }
            notification.device = .value(n)
            audioLevelNotification.device = .value(a)
            cameraNotification.device = .value(c)
        } catch {
            if isLinkError(error) {
                setNetwork(.unknown)
                throw error
            }
            setNetwork(.unreadable(shortReason(error)))
        }
    }

    private func setNetwork(_ value: DeviceValue<Bool>) {
        notification.device = value
        audioLevelNotification.device = value
        cameraNotification.device = value
    }

    private func readRoomType() async throws {
        try await read(\.roomType, command: "g_roomtype") { fields in
            guard let raw = fields.first, let v = Int(raw),
                  ATNDCommands.RoomType.range.contains(v) else {
                return .unreadable("unexpected value")
            }
            return .value(v)
        }
    }

    private func readAudioSystem() async throws {
        try await read(\.beamSensitivity, command: "g_audio_system") { fields in
            guard fields.count == ATNDCommands.AudioSystem.fieldCount else {
                return .unreadable("unexpected \(fields.count) fields")
            }
            guard let v = Int(fields[ATNDCommands.AudioSystem.beamSensitivity]),
                  ATNDCommands.AudioSystem.beamSensitivityRange.contains(v) else {
                return .unreadable("unexpected value")
            }
            return .value(v)
        }
    }

    private func readSmartMix(_ channel: Int) async throws {
        try await read(\.smartMixWeight, command: "g_smart_mix", params: String(channel)) { fields in
            guard fields.count == ATNDCommands.SmartMix.fieldCount else {
                return .unreadable("unexpected \(fields.count) fields")
            }
            guard fields[ATNDCommands.SmartMix.inputChannel] == String(channel) else {
                return .unreadable("answered about beam \(fields[ATNDCommands.SmartMix.inputChannel])")
            }
            guard let v = Int(fields[ATNDCommands.SmartMix.weight]),
                  ATNDCommands.SmartMix.weightRange.contains(v) else {
                return .unreadable("unexpected value")
            }
            return .value(v)
        }
    }

    private func readCameraInterval() async throws {
        try await read(\.cameraInterval, command: "g_camera_control_interval") { interval($0) }
    }

    private func readLevelInterval() async throws {
        try await read(\.levelInterval, command: "g_level_meter_interval") { interval($0) }
    }

    // MARK: - Writing

    func setDeviceMute(_ on: Bool) async {
        await send(\.deviceMute, command: "s_mute", params: on ? "1" : "0", expecting: on) {
            try await self.readMute()
        }
    }

    func setOutputMute(channel: Int, muted: Bool) async {
        await send(\.outputMute[channel], command: "s_output_mute",
                   params: ATNDCommands.outputMuteSetPayload(channel: channel, muted: muted),
                   expecting: muted) {
            try await self.readOutputMute(channel)
        }
    }

    func setAGC(_ on: Bool) async {
        await send(\.agc, command: "s_agc",
                   params: ATNDCommands.agcSetPayload(enable: on), expecting: on) {
            try await self.readAGC()
        }
    }

    func setAEC(_ on: Bool) async {
        // Only reachable while the gate is open, i.e. the array's own answer
        // parsed — so `aecFieldCount` is the width the array actually speaks,
        // not a width the spec claims.
        guard aecUsable, aecFieldCount > 0 else { return }
        await send(\.aec, command: "s_aec_general",
                   params: ATNDCommands.aecSetPayload(enable: on, fieldCount: aecFieldCount),
                   expecting: on) {
            try await self.readAEC()
        }
    }

    func setNotification(_ on: Bool) async {
        await send(\.notification, command: "s_network",
                   params: ATNDCommands.networkSetPayload(notification: on), expecting: on) {
            try await self.readNetwork()
        }
    }

    func setAudioLevelNotification(_ on: Bool) async {
        await send(\.audioLevelNotification, command: "s_network",
                   params: ATNDCommands.networkSetPayload(audioLevel: on), expecting: on) {
            try await self.readNetwork()
        }
    }

    func setCameraNotification(_ on: Bool) async {
        await send(\.cameraNotification, command: "s_network",
                   params: ATNDCommands.networkSetPayload(camera: on), expecting: on) {
            try await self.readNetwork()
        }
    }

    func setRoomType(_ value: Int) async {
        await send(\.roomType, command: "s_roomtype", params: String(value), expecting: value) {
            try await self.readRoomType()
        }
    }

    func setBeamSensitivity(_ value: Int) async {
        await send(\.beamSensitivity, command: "s_audio_system",
                   params: ATNDCommands.audioSystemSetPayload(beamSensitivity: value),
                   expecting: value) {
            try await self.readAudioSystem()
        }
    }

    func selectSmartMixChannel(_ channel: Int) async {
        smartMixChannel = channel
        smartMixWeight = Row<Int>()
        guard control.isConnected else { return }
        try? await readSmartMix(channel)
    }

    func setSmartMixWeight(_ weight: Int) async {
        let channel = smartMixChannel
        await send(\.smartMixWeight, command: "s_smart_mix",
                   params: ATNDCommands.smartMixSetPayload(channel: channel, weight: weight),
                   expecting: weight) {
            try await self.readSmartMix(channel)
        }
    }

    func setCameraInterval(_ ms: Int) async {
        await send(\.cameraInterval, command: "s_camera_control_interval",
                   params: String(ms), expecting: ms) {
            try await self.readCameraInterval()
        }
    }

    func setLevelInterval(_ ms: Int) async {
        await send(\.levelInterval, command: "s_level_meter_interval",
                   params: String(ms), expecting: ms) {
            try await self.readLevelInterval()
        }
    }

    /// SVAD is Set-only — the spec has no matching Get, so there is no device
    /// value to update afterwards and nothing to verify against.
    func sendVAD(_ on: Bool) async {
        vadSend = .sending
        do {
            vadSend = status(for: try await control.set("SVAD", params: on ? "1" : "0"))
        } catch {
            vadSend = status(forThrown: error)
        }
    }

    // MARK: - Machinery

    private func read<V>(_ row: ReferenceWritableKeyPath<ATNDCommandModel, Row<V>>,
                         command: String,
                         params: String = "",
                         parse: ([String]) -> FieldRead<V>) async throws {
        self[keyPath: row].device = .querying
        do {
            switch parse(try await control.get(command, params: params)) {
            case .value(let value):     self[keyPath: row].device = .value(value)
            case .unreadable(let why):  self[keyPath: row].device = .unreadable(why)
            }
        } catch {
            try failRead(&self[keyPath: row].device, error)
        }
    }

    /// A link error abandons the read (and the sweep); anything the array itself
    /// said is that row's own problem and does not stop the others.
    private func failRead<V>(_ slot: inout DeviceValue<V>, _ error: Error) throws {
        if isLinkError(error) {
            slot = .unknown
            throw error
        }
        slot = .unreadable(shortReason(error))
    }

    private func send<V: Equatable>(_ row: ReferenceWritableKeyPath<ATNDCommandModel, Row<V>>,
                                    command: String,
                                    params: String,
                                    expecting wanted: V,
                                    verify: @escaping () async throws -> Void) async {
        self[keyPath: row].send = .sending
        self[keyPath: row].note = nil
        do {
            self[keyPath: row].send = status(for: try await control.set(command, params: params))
        } catch {
            self[keyPath: row].send = status(forThrown: error)
            return
        }
        // Re-read whatever the reply was. A NAK especially: the array refused
        // something, and the only way to know where that leaves the setting is
        // to ask it.
        do { try await verify() } catch { return }

        if case .value(let actual) = self[keyPath: row].device, actual != wanted {
            self[keyPath: row].note = "The array kept its previous value."
        }
    }

    private func status(for reply: ATNDReply) -> SendStatus {
        switch reply {
        case .ack:             return .ack(Date())
        case .nak(let code):   return .nak(code: code, Date())
        case .answer:          return .failed("the array sent an answer where an ACK belongs")
        }
    }

    private func status(forThrown error: Error) -> SendStatus {
        guard let e = error as? ATNDError else { return .failed("\(error)") }
        switch e {
        case .timeout:            return .timeout
        case .linkDropped:        return .failed("the link dropped")
        case .notConnected:       return .failed("not connected")
        case .malformed(let why): return .failed(why)
        case .nak(let code):      return .nak(code: code, Date())
        }
    }

    private func isLinkError(_ error: Error) -> Bool {
        guard let e = error as? ATNDError else { return true }
        switch e {
        case .timeout, .linkDropped, .notConnected: return true
        case .nak, .malformed:                      return false
        }
    }

    private func shortReason(_ error: Error) -> String {
        guard let e = error as? ATNDError else { return "\(error)" }
        switch e {
        case .nak(let code):      return "NAK \(code) — \(ATNDCommands.nakText(code))"
        case .malformed(let why): return why
        default:                  return "\(e)"
        }
    }
}

// MARK: - Small field parsers

/// What one answer's fields turned out to be. Not `Result`: the failure here is
/// a sentence for the user, not an error to propagate.
enum FieldRead<V> {
    case value(V)
    case unreadable(String)
}

private func single(_ fields: [String]) -> FieldRead<Bool> {
    guard fields.count == 1, let v = ATNDCommands.boolField(fields[0]) else {
        return .unreadable("unexpected value")
    }
    return .value(v)
}

private func interval(_ fields: [String]) -> FieldRead<Int> {
    guard fields.count == 1, let v = Int(fields[0]),
          ATNDCommands.intervalRange.contains(v) else {
        return .unreadable("unexpected value")
    }
    return .value(v)
}
