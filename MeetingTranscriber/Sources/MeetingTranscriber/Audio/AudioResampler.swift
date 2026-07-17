import AVFoundation

/// Converts audio buffers to 16 kHz mono Float32 — the format Silero VAD
/// and all ASR models expect. Stateful (keeps converter context between
/// buffers), so use one instance per recording session.
final class AudioResampler {

    private let converter: AVAudioConverter
    private let outFormat: AVAudioFormat

    init?(inputFormat: AVAudioFormat, outputRate: Double = 16_000) {
        guard let out = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                      sampleRate: outputRate,
                                      channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: inputFormat, to: out)
        else { return nil }
        outFormat = out
        converter = conv
    }

    /// Resample one buffer; returns 16 kHz mono samples (may be empty).
    func resample(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let ratio = outFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
        guard capacity > 0,
              let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity)
        else { return [] }

        var served = false
        let status = converter.convert(to: out, error: nil) { _, outStatus in
            if served {
                outStatus.pointee = .noDataNow
                return nil
            }
            served = true
            outStatus.pointee = .haveData
            return buffer
        }

        guard status != .error, let channel = out.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: channel[0], count: Int(out.frameLength)))
    }
}
