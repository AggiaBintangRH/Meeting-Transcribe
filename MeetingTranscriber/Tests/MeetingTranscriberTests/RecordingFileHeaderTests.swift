import AVFoundation
import XCTest
@testable import MeetingTranscriber

/// The PREMISE behind `AudioRecorder.finalizeRecordingFiles` and the quit hooks
/// in `AppDelegate`: an `AVAudioFile` opened for writing leaves the WAV `data`
/// chunk size at **0** until the object is RELEASED.
///
/// WHY PIN A FRAMEWORK'S BEHAVIOUR. The fix is worth nothing if this stops being
/// true, and it would stop being true SILENTLY — recordings would simply start
/// surviving a hard kill, and nobody would notice that the hooks had become
/// decoration. The reverse matters more: if a future macOS wrote the size only at
/// some other moment, the hooks would keep running while no longer rescuing
/// anything, and the failure would look exactly like today's — a whole meeting
/// reading as 0.0 seconds.
///
/// Measured 2026-08-05 on the owner's machine before any of this was written:
/// **13 recordings, ~1.7 GB**, one holding 34.1 minutes of real speech, all with
/// a `data` size of 0 over a file full of samples. Recoverable with
/// `scripts/tools/repair-wav-header.py`.
///
/// These write only into `FileManager.default.temporaryDirectory` and remove
/// what they create.
final class RecordingFileHeaderTests: XCTestCase {

    /// Same shape the recorder writes: mono float32 at 44.1 kHz.
    private let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: 44_100.0,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
    ]

    private func makeBuffer(frames: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
        let format = try XCTUnwrap(AVAudioFormat(settings: settings))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format,
                                                    frameCapacity: frames))
        buffer.frameLength = frames
        // Real non-zero samples, so a reader cannot "succeed" on silence.
        for i in 0..<Int(frames) {
            buffer.floatChannelData?[0][i] = sinf(Float(i) * 0.01) * 0.5
        }
        return buffer
    }

    /// Walk the RIFF chunks and return the `data` chunk's DECLARED size — the one
    /// field this whole fix is about. Read from the bytes rather than through a
    /// decoder, because a decoder is exactly the thing that mis-reports it.
    private func declaredDataSize(_ url: URL) throws -> UInt32? {
        let bytes = try Data(contentsOf: url)
        guard bytes.count > 12,
              bytes[0..<4].elementsEqual(Array("RIFF".utf8)),
              bytes[8..<12].elementsEqual(Array("WAVE".utf8)) else { return nil }
        var offset = 12
        while offset + 8 <= bytes.count {
            let id = String(decoding: bytes[offset..<offset + 4], as: UTF8.self)
            let size = bytes[(offset + 4)..<(offset + 8)]
                .reversed().reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
            if id == "data" { return size }
            offset += 8 + Int(size) + Int(size & 1)
        }
        return nil
    }

    /// THE TRAP ITSELF: while the file object is alive, the header says zero.
    /// This is what a recording looks like when the app is killed mid-meeting.
    func testAnOpenRecordingFileStillDeclaresZeroBytes() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mt-open-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let buffer = try makeBuffer(frames: 4_410)
        var file: AVAudioFile? = try AVAudioFile(forWriting: url, settings: settings)
        try file?.write(from: buffer)

        XCTAssertEqual(try declaredDataSize(url), 0,
                       "if this ever stops being 0, the quit hooks in AppDelegate "
                       + "are no longer rescuing anything and this test is the only "
                       + "thing that would say so")
        // Some bytes really are on disk already — the audio is present, only the
        // count is missing. That asymmetry is why the repair tool can work at all.
        let onDisk = try FileManager.default
            .attributesOfItem(atPath: url.path)[.size] as? NSNumber
        XCTAssertGreaterThan(onDisk?.intValue ?? 0, 4_410 * 4,
                             "the samples must already be written while the header "
                             + "still says zero — that is the whole premise")
        file = nil
    }

    /// RELEASING is what writes the size. `stop()` relies on exactly this, and so
    /// does `finalizeRecordingFiles`, which does nothing else.
    func testReleasingTheFileWritesTheRealSize() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mt-closed-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let frames: AVAudioFrameCount = 4_410
        var file: AVAudioFile? = try AVAudioFile(forWriting: url, settings: settings)
        try file?.write(from: try makeBuffer(frames: frames))
        file = nil                      // the only step that matters

        XCTAssertEqual(try declaredDataSize(url), frames * 4,
                       "mono float32: 4 bytes per frame")
        // …and a reader now agrees, which is what every stage of this app does.
        let reopened = try AVAudioFile(forReading: url)
        XCTAssertEqual(reopened.length, AVAudioFramePosition(frames))
    }

    /// `active` must mean "there are unflushed files", never "a recorder exists" —
    /// the quit hook does nothing when it is nil, so a stale non-nil value would
    /// send it closing files that are already closed, and a stale nil would let a
    /// live recording be lost. It starts nil, which is the safe direction.
    @MainActor func testTheActiveRecorderStartsUnset() {
        // Constructing a recorder must not register it: registration happens only
        // once `AVAudioEngine.start()` has really succeeded.
        _ = AudioRecorder()
        XCTAssertNil(AudioRecorder.active,
                     "a recorder that never started capture has no open files")
    }
}
