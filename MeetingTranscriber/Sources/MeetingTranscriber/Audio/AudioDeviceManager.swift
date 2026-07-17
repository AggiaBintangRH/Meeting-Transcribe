import AVFoundation
import CoreAudio
import Foundation

/// An audio input device (microphone) with its input channel count.
struct AudioInputDevice: Identifiable, Hashable {
    let id: AudioDeviceID
    let uid: String
    let name: String
    let channelCount: Int
}

/// Enumerates macOS audio input devices via CoreAudio.
enum AudioDeviceManager {

    static func inputDevices() -> [AudioInputDevice] {
        allDeviceIDs().compactMap { deviceID in
            let channels = inputChannelCount(deviceID)
            guard channels > 0 else { return nil } // input-capable devices only
            return AudioInputDevice(
                id: deviceID,
                uid: stringProperty(deviceID, kAudioDevicePropertyDeviceUID) ?? "\(deviceID)",
                name: stringProperty(deviceID, kAudioObjectPropertyName) ?? "Unknown device",
                channelCount: channels
            )
        }
    }

    /// Route an AVAudioEngine's input to a specific device (must be called before engine.start()).
    @discardableResult
    static func setInputDevice(_ deviceID: AudioDeviceID, on engine: AVAudioEngine) -> Bool {
        guard let audioUnit = engine.inputNode.audioUnit else { return false }
        var id = deviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &id,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        return status == noErr
    }

    // MARK: - CoreAudio plumbing

    private static func allDeviceIDs() -> [AudioDeviceID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                             &address, 0, nil, &size) == noErr else { return [] }
        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                         &address, 0, nil, &size, &ids) == noErr else { return [] }
        return ids
    }

    /// Number of input channels (sum over all input streams).
    private static func inputChannelCount(_ deviceID: AudioDeviceID) -> Int {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &size) == noErr,
              size > 0 else { return 0 }

        let listPtr = UnsafeMutableRawPointer.allocate(byteCount: Int(size),
                                                       alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { listPtr.deallocate() }
        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, listPtr) == noErr else { return 0 }

        let bufferList = UnsafeMutableAudioBufferListPointer(listPtr.assumingMemoryBound(to: AudioBufferList.self))
        return bufferList.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    private static func stringProperty(_ deviceID: AudioDeviceID,
                                       _ selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size = UInt32(MemoryLayout<CFString?>.size)
        var value: CFString? = nil
        let status = withUnsafeMutablePointer(to: &value) { ptr in
            AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, ptr)
        }
        guard status == noErr, let cfString = value else { return nil }
        return cfString as String
    }
}
