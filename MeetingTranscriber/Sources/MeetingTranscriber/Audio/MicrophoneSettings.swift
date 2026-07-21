import Foundation

/// Reads the microphone configuration saved by Settings → Microphone.
/// Single source of truth for which device + channel(s) to record from.
///
/// Dual-stream (Office + Remote) capture deliberately takes BOTH channels from
/// ONE device. A hybrid meeting has the room array on one channel and the
/// conferencing app's audio (via a loopback/virtual input) on another; the owner
/// combines them in Audio MIDI Setup as an **Aggregate Device**, which puts both
/// physical inputs under a single drift-corrected clock.
///
/// Two independent `AVAudioEngine`s were considered and rejected: separate device
/// clocks drift by seconds over a long meeting, and the app's entire timeline —
/// chunk windows, diarization windows, the ATND boundary timeline, overlap
/// regions — hangs off one accumulated clock (`recordingElapsed += frames / rate`).
/// Two clocks means two timelines that silently disagree. Do not "simplify" this
/// into a second engine.
final class MicrophoneSettings {

    let device: AudioInputDevice?   // nil → system default input
    let channel: Int                // Office stream. 0-based, clamped to the device's channels
    /// Remote stream (conferencing audio), or nil for today's single-stream
    /// behaviour. Never equal to `channel`.
    let remoteChannel: Int?

    private init(device: AudioInputDevice?, channel: Int, remoteChannel: Int?) {
        self.device = device
        self.channel = channel
        self.remoteChannel = remoteChannel
    }

    /// UserDefaults key for the Remote channel. Absent or `-1` means single-stream.
    /// `-1` (not simply "absent") is needed because `@AppStorage` always writes a
    /// value once the user has touched the control — it needs an off state.
    static let remoteChannelKey = "mic.remoteChannel"

    /// Resolve the persisted selection against currently connected devices.
    static func current() -> MicrophoneSettings {
        let defaults = UserDefaults.standard
        let uid = defaults.string(forKey: "mic.deviceUID") ?? ""
        let savedChannel = defaults.integer(forKey: "mic.channel")
        // `integer(forKey:)` cannot distinguish an absent key from 0, and 0 is a
        // valid channel — so probe for existence first.
        let savedRemote = defaults.object(forKey: remoteChannelKey) == nil
            ? nil
            : defaults.integer(forKey: remoteChannelKey)

        let device = AudioDeviceManager.inputDevices().first { $0.uid == uid }
        let resolved = resolve(savedChannel: savedChannel,
                               savedRemote: savedRemote,
                               // nil → unknown device; the office channel is clamped
                               // again against the actual format at capture time.
                               channelCount: device?.channelCount)
        return MicrophoneSettings(device: device,
                                  channel: resolved.channel,
                                  remoteChannel: resolved.remoteChannel)
    }

    /// Pure resolution of the stored values against a device's channel count.
    /// Split out from `current()` so it is unit-testable without UserDefaults or
    /// real audio hardware.
    ///
    /// - Parameters:
    ///   - savedChannel: raw `mic.channel`.
    ///   - savedRemote: raw `mic.remoteChannel`, or nil when the key is absent.
    ///   - channelCount: the selected device's channel count, or nil if no device
    ///     matched (system default — count unknown until the engine is running).
    ///
    /// Remote resolves to nil (single-stream) rather than being relocated whenever
    /// it is not usable: absent, `-1`, out of range for the device, or equal to the
    /// Office channel. **Rejecting is the chosen behaviour** — silently moving the
    /// Remote stream to another channel would record whatever happens to be there,
    /// and the user would never be told. Falling back to single-stream is the
    /// behaviour they already understand.
    static func resolve(savedChannel: Int,
                        savedRemote: Int?,
                        channelCount: Int?) -> (channel: Int, remoteChannel: Int?) {
        let channel: Int
        if let channelCount, channelCount > 0 {
            channel = min(max(savedChannel, 0), channelCount - 1)
        } else {
            channel = max(savedChannel, 0)
        }

        guard let savedRemote, savedRemote >= 0 else {
            return (channel, nil)            // absent or -1 → single-stream, as today
        }
        if savedRemote == channel {
            return (channel, nil)            // Office and Remote cannot be one channel
        }
        if let channelCount, savedRemote >= channelCount {
            return (channel, nil)            // device changed under a stale selection
        }
        return (channel, savedRemote)
    }
}
