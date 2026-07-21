import AVFoundation

/// Pure audio-buffer math: channel extraction and RMS.
/// Stateless — safe to call from the audio render thread.
final class AudioBufferProcessor {

    private init() {}

    /// Copy one channel out of a multi-channel buffer into a new mono buffer.
    /// Returns nil if the channel index is invalid or allocation fails.
    static func extractChannel(_ buffer: AVAudioPCMBuffer, channel: Int) -> AVAudioPCMBuffer? {
        guard let source = buffer.floatChannelData,
              channel >= 0, channel < Int(buffer.format.channelCount),
              let monoFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                             sampleRate: buffer.format.sampleRate,
                                             channels: 1,
                                             interleaved: false),
              let mono = AVAudioPCMBuffer(pcmFormat: monoFormat,
                                          frameCapacity: buffer.frameLength)
        else { return nil }

        mono.frameLength = buffer.frameLength
        if let dest = mono.floatChannelData {
            dest[0].update(from: source[channel], count: Int(buffer.frameLength))
        }
        return mono
    }

    /// Root-mean-square level of a (mono) buffer's first channel. 0.0 = silence.
    static func rms(_ buffer: AVAudioPCMBuffer) -> Float {
        guard let data = buffer.floatChannelData?[0] else { return 0 }
        let n = Int(buffer.frameLength)
        guard n > 0 else { return 0 }
        var sum: Float = 0
        for i in 0..<n { sum += data[i] * data[i] }
        return sqrt(sum / Float(n))
    }

    /// Root-mean-square level of already-extracted mono samples. Same math as
    /// the buffer overload, for callers that hold a plain `[Float]` (the 16 kHz
    /// accumulation buffers) rather than an `AVAudioPCMBuffer`.
    static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for s in samples { sum += s * s }
        return sqrt(sum / Float(samples.count))
    }

    /// Zero-crossing rate (0–1, crossings per sample) of a mono buffer.
    /// Voiced speech ≈ 0.02–0.12, fricatives up to ~0.35,
    /// low tones/bass < 0.02, hiss/cymbals > 0.4.
    static func zeroCrossingRate(_ buffer: AVAudioPCMBuffer) -> Float {
        guard let data = buffer.floatChannelData?[0] else { return 0 }
        let n = Int(buffer.frameLength)
        guard n > 1 else { return 0 }
        var crossings = 0
        for i in 1..<n where (data[i - 1] < 0) != (data[i] < 0) {
            crossings += 1
        }
        return Float(crossings) / Float(n - 1)
    }
}
