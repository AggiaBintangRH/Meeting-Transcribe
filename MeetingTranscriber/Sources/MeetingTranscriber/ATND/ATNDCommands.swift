import Foundation

/// The ATND1061 IP Control protocol (spec v9.0) as pure data — no socket, no
/// state, no main actor. Everything here is a string in / string out, so the
/// framing and the field layouts can be tested without hardware.
///
/// The field indices below are the single most dangerous part of this feature:
/// a wrong index does not fail loudly, it writes the wrong setting to real
/// hardware. Every constant therefore cites the spec table it came from, and
/// the ones the spec contradicts itself about are marked as such rather than
/// guessed at.
enum ATNDCommands {

    // MARK: - Framing (spec 2.2, Tables 2-2 … 2-7)

    /// Single CR. The spec says 0x0d and means it — not CRLF.
    static let terminator: UInt8 = 0x0d

    /// "The message string length including carriage return codes is greater
    /// than the upper limit" is NAK 01 (Table 2-6); the limit is 287 bytes.
    static let maxMessageBytes = 287

    enum Handshake: String {
        /// Set — the array answers ACK or NAK.
        case set = "S"
        /// Get — the array answers with a setting-status Answer.
        case get = "O"
    }

    /// `<command> <handshake> <model_id> <unit_no> <continue> [params]` (Table 2-2).
    ///
    /// Model ID and Unit No are "Not used" in every host→array command in the
    /// spec — they are the literal strings `0000` and `00`, not this array's
    /// device ID. Only the array's *replies* carry a real device ID.
    static func action(command: String, handshake: Handshake, params: String = "") -> String {
        // The trailing space is present in every one of the spec's own examples,
        // including the ones with no parameters (`g_network␣O␣0000␣00␣NC␣↲`).
        "\(command) \(handshake.rawValue) 0000 00 NC \(params)"
    }

    static func frame(_ line: String) -> Data {
        var data = Data(line.utf8)
        data.append(terminator)
        return data
    }

    // MARK: - Reply parsing

    enum ParseResult {
        /// A well-formed reply to the command we asked about.
        case reply(ATNDReply)
        /// Well-formed, but about a different command — not ours to consume.
        case mismatch
        case malformed(String)
    }

    /// Classifies one CR-stripped reply line against the command in flight.
    ///
    /// The protocol has no request IDs, so the command name is the only link
    /// between a reply and its request. Anything that does not match is left
    /// alone rather than guessed at.
    static func parse(_ line: String, expecting command: String) -> ParseResult {
        let tokens = line.split(separator: " ").map(String.init)
        guard let head = tokens.first else { return .malformed("empty reply") }
        guard head == command else { return .mismatch }

        // ACK (Table 2-4) / NAK (Table 2-5)
        if tokens.count >= 2, tokens[1] == "ACK" { return .reply(.ack) }
        if tokens.count >= 2, tokens[1] == "NAK" {
            guard tokens.count >= 3 else { return .malformed("NAK without an error code") }
            return .reply(.nak(code: tokens[2]))
        }

        // Setting status return (Table 2-7):
        // <command> <model_id> <unit_no> <continue> <params>
        guard tokens.count >= 4 else { return .malformed("truncated answer") }
        guard tokens[1] == "0000" else { return .malformed("unexpected Model ID \(tokens[1])") }
        // Unit No is the array's Device ID, 00–FF (Table 2-7) — NOT always 00.
        // A device with a non-zero ID is normal and must not be rejected.
        guard isDeviceID(tokens[2]) else { return .malformed("unexpected Unit No \(tokens[2])") }
        // Split replies (CS/CM/CE) are a real part of the spec that nothing here
        // asks for. Treating them as malformed is honest: this code cannot
        // reassemble them, so it must not pretend a fragment is the whole value.
        guard tokens[3] == "NC" else { return .malformed("split reply (\(tokens[3])) is not supported") }

        guard tokens.count >= 5 else { return .reply(.answer(fields: [])) }
        // Rejoin: a parameter list never contains spaces, but rejoining is
        // cheaper than trusting that.
        let params = tokens[4...].joined(separator: " ")
        return .reply(.answer(fields: params.components(separatedBy: ",")))
    }

    /// Two hex digits, per the Unit No / Device ID domain "00 to FF".
    static func isDeviceID(_ text: String) -> Bool {
        text.count == 2 && text.allSatisfy(\.isHexDigit)
    }

    /// Table 2-6.
    static func nakText(_ code: String) -> String {
        switch code {
        case "01": return "syntax error"
        case "02": return "invalid command"
        case "03": return "split error"
        case "04": return "parameter out of range"
        case "05": return "transmission timeout"
        case "90": return "busy"
        case "92": return "busy (save mode)"
        case "93": return "busy (extension)"
        case "99": return "other error"
        default:   return "error \(code)"
        }
    }

    // MARK: - Field layouts
    //
    // Sparse Sets are safe because the spec says so in section 2: "When you send
    // a command from the host, you can omit its parameters. To omit them,
    // specify no data by separating with commas". So every builder below emits
    // a full-width skeleton of empty fields and fills in only what is meant to
    // change — reserved fields stay untouched on the device.

    /// `g_network` answer — 21 fields (Table 4-72).
    ///
    /// The Get and Set layouts DIFFER: the MAC address exists only in the Get
    /// answer (index 4) and pushes every later field one place right. Reading
    /// an answer with the Set indices, or writing a Set with these, is off by
    /// one from index 4 onward.
    enum NetworkGet {
        static let fieldCount = 21
        static let ipConfigMode = 0
        static let ipAddress = 1
        static let subnetMask = 2
        static let gateway = 3
        static let macAddress = 4        // Get only
        static let allowDiscovery = 5
        static let port = 6
        static let notification = 7
        static let audioLevelNotification = 8
        static let multicastAddress = 9
        static let multicastPort = 10
        // 11–19 Reserved (9 fields)
        static let cameraControlNotification = 20
    }

    /// `s_network` parameters — 20 fields, no MAC (Table 4-70).
    enum NetworkSet {
        static let fieldCount = 20
        static let ipConfigMode = 0
        static let ipAddress = 1
        static let subnetMask = 2
        static let gateway = 3
        static let allowDiscovery = 4
        static let port = 5
        static let notification = 6
        static let audioLevelNotification = 7
        static let multicastAddress = 8
        static let multicastPort = 9
        // 10–18 Reserved (9 fields)
        static let cameraControlNotification = 19
    }

    /// The three notification switches, and nothing else. Everything the array
    /// would need a reboot for (IP, port, multicast) is deliberately not
    /// writable from here.
    static func networkSetPayload(notification: Bool? = nil,
                                  audioLevel: Bool? = nil,
                                  camera: Bool? = nil) -> String {
        var fields = [String](repeating: "", count: NetworkSet.fieldCount)
        if let notification { fields[NetworkSet.notification] = flag(notification) }
        if let audioLevel   { fields[NetworkSet.audioLevelNotification] = flag(audioLevel) }
        if let camera       { fields[NetworkSet.cameraControlNotification] = flag(camera) }
        return fields.joined(separator: ",")
    }

    /// `s_agc` / `g_agc` — 3 fields, same layout both ways (Tables 4-37, 4-39).
    enum AGC {
        static let fieldCount = 3
        static let enable = 0           // 0 Off, 1 On
        // 1 Reserved
        static let targetLevel = 2      // -10 … 10 dB
    }

    static func agcSetPayload(enable: Bool) -> String {
        var fields = [String](repeating: "", count: AGC.fieldCount)
        fields[AGC.enable] = flag(enable)
        return fields.joined(separator: ",")
    }

    /// `s_aec_general` / `g_aec_general` — Table 4-34 lists 12 fields.
    ///
    /// **The spec contradicts itself here.** Table 4-34's own example is
    /// `2,1,,,0,,20,20,1,1,` — eleven fields, with a field 0 of "2" that is
    /// outside AEC Enable's documented 0/1 domain. So the table and the example
    /// cannot both be right, and only field 0 is positionally trustworthy (both
    /// readings agree it is AEC Enable and it comes first).
    ///
    /// Consequence, deliberately: only the Enable toggle is offered, and only
    /// when the live answer's field 0 actually reads 0 or 1. The NC Enable and
    /// NLP fields are NOT exposed — their indices are unverifiable, and an
    /// unverifiable index is a wrong write to real hardware.
    enum AEC {
        static let tableFieldCount = 12    // Table 4-34
        static let enable = 0              // 0 Off, 1 On — the only trustworthy index
    }

    /// Width-agnostic on purpose: the payload is built to match whatever width
    /// the array actually answered with, since the spec's two claims disagree.
    static func aecSetPayload(enable: Bool, fieldCount: Int) -> String {
        var fields = [String](repeating: "", count: max(fieldCount, AEC.enable + 1))
        fields[AEC.enable] = flag(enable)
        return fields.joined(separator: ",")
    }

    /// `s_audio_system` / `g_audio_system` — 9 fields, same both ways
    /// (Tables 4-120, 4-122).
    enum AudioSystem {
        static let fieldCount = 9
        // 0–3 Reserved
        static let tx6 = 4                 // 0 Automix, 1 Priority 5
        static let beamSensitivity = 5     // 0 Low, 1 Mid, 2 High
        static let autoAttenuation = 6     // 0 Disabled, 1 Enabled
        static let attenuationLevel = 7    // 0 … 28
        static let holdTime = 8            // 0 … 100 (0–10 s, 0.1 step)
        static let beamSensitivityRange = 0...2
    }

    static func audioSystemSetPayload(beamSensitivity: Int) -> String {
        var fields = [String](repeating: "", count: AudioSystem.fieldCount)
        fields[AudioSystem.beamSensitivity] = String(beamSensitivity)
        return fields.joined(separator: ",")
    }

    /// `s_smart_mix` / `g_smart_mix` — 7 fields, same both ways
    /// (Tables 4-31, 4-33). Per beam channel: the Get takes the channel as its
    /// parameter and the answer echoes it back.
    enum SmartMix {
        static let fieldCount = 7
        static let inputChannel = 0        // 0 … 5 = Beam Channel 1 … 6
        // 1 Reserved
        static let weight = 2              // 0 … 60 = -15.0 … +15.0 dB, 0.5 steps
        // 3–6 Reserved
        static let channelRange = 0...5
        static let weightRange = 0...60
    }

    /// Weight 0…60 maps onto -15.0…+15.0 dB in 0.5 dB steps (Table 4-31).
    static func smartMixWeightDB(_ raw: Int) -> Double { Double(raw) * 0.5 - 15.0 }

    static func smartMixSetPayload(channel: Int, weight: Int) -> String {
        var fields = [String](repeating: "", count: SmartMix.fieldCount)
        fields[SmartMix.inputChannel] = String(channel)
        fields[SmartMix.weight] = String(weight)
        return fields.joined(separator: ",")
    }

    /// `s_output_mute` / `g_output_mute` — `<channel>,<mute>` (Tables 4-58, 4-59, 4-60).
    /// The Get takes the channel and the answer echoes it, so a reply can and
    /// must be checked against the channel that was asked for.
    enum OutputMute {
        static let fieldCount = 2
        static let channel = 0             // 0 Analog Out, 1 Auto Mix
        static let muted = 1               // 0 Not muted, 1 Muted
        static let channels = [0, 1]
        static func name(_ channel: Int) -> String {
            channel == 0 ? "Analog output" : "Auto mix output"
        }
    }

    static func outputMuteSetPayload(channel: Int, muted: Bool) -> String {
        "\(channel),\(flag(muted))"
    }

    /// `s_roomtype` / `g_roomtype` — one field (Tables 4-130, 4-131, 4-132).
    enum RoomType {
        static let range = 0...2
        static func name(_ value: Int) -> String {
            switch value {
            case 0:  return "Dry"
            case 1:  return "Live"
            case 2:  return "Reverberant"
            default: return "?"
            }
        }
    }

    static func beamSensitivityName(_ value: Int) -> String {
        switch value {
        case 0:  return "Low"
        case 1:  return "Mid"
        case 2:  return "High"
        default: return "?"
        }
    }

    /// `s_camera_control_interval` and `s_level_meter_interval` share this
    /// domain: a plain millisecond integer, 100 … 300000.
    static let intervalRange = 100...300_000

    /// `s_mute` / `g_mute` — one field, 0 Not muted / 1 Muted (Tables 4-124, 4-126).
    /// `SVAD` — one field, 0/1. **Set-only: the spec has no GVAD**, so this
    /// value can never be read back from the array.

    private static func flag(_ on: Bool) -> String { on ? "1" : "0" }

    /// A single 0/1 field, rejecting anything else. Used everywhere a toggle
    /// reads its state, so an out-of-domain value shows as unreadable rather
    /// than silently as "Off".
    static func boolField(_ text: String) -> Bool? {
        switch text {
        case "0": return false
        case "1": return true
        default:  return nil
        }
    }
}
