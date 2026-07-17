import Foundation

/// Reads the microphone configuration saved by Settings → Microphone.
/// Single source of truth for which device + channel to record from.
final class MicrophoneSettings {

    let device: AudioInputDevice?   // nil → system default input
    let channel: Int                // 0-based, clamped to the device's channels

    private init(device: AudioInputDevice?, channel: Int) {
        self.device = device
        self.channel = channel
    }

    /// Resolve the persisted selection against currently connected devices.
    static func current() -> MicrophoneSettings {
        let defaults = UserDefaults.standard
        let uid = defaults.string(forKey: "mic.deviceUID") ?? ""
        let savedChannel = defaults.integer(forKey: "mic.channel")

        let device = AudioDeviceManager.inputDevices().first { $0.uid == uid }
        let channel: Int
        if let device {
            channel = min(max(savedChannel, 0), device.channelCount - 1)
        } else {
            channel = max(savedChannel, 0) // clamped again against the actual format later
        }
        return MicrophoneSettings(device: device, channel: channel)
    }
}
