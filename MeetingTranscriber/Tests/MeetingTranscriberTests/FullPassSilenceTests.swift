import XCTest
@testable import MeetingTranscriber

/// `fullPassWindowsAtSilence` — the stop-time pass must cut where the LIVE pass
/// cuts, i.e. at silence, not on a stopwatch.
///
/// WHY. The live boundary is
///     `(chunkElapsed >= interval && !speaking) || chunkElapsed >= interval * 1.5`
/// so a chunk is only closed once the VAD reports silence, which is what the
/// Chunked tab means by "cut at silence, never mid-word". The full pass first
/// shipped with UNIFORM boundaries, which can slice a word in half — and a word
/// split across two windows is mis-transcribed at both ends. A re-transcription
/// that damages boundaries the live pass got right is worse than none, so these
/// pin the rule rather than the implementation.
final class FullPassSilenceTests: XCTestCase {

    private let frame = AudioRecorder.silenceFrameSec   // 0.02

    /// Build an energy profile: `speech` everywhere except the given quiet spans.
    private func profile(seconds: Double, quiet: [ClosedRange<Double>],
                         loud: Float = 0.20) -> [Float] {
        let count = Int(seconds / frame)
        return (0..<count).map { i in
            let t = Double(i) * frame
            return quiet.contains(where: { $0.contains(t) }) ? 0.0005 : loud
        }
    }

    private func windows(_ energies: [Float], length: Double, interval: Double)
        -> [ClosedRange<Double>] {
        AudioRecorder.fullPassWindowsAtSilence(recordingLength: length,
                                               intervalSec: interval,
                                               energies: energies)
    }

    // MARK: - The point of the feature

    /// A pause just after the target: the boundary must move to it rather than
    /// landing on the stopwatch mid-word.
    func testBoundaryMovesForwardToTheNearestPause() {
        // Speech throughout except a gap at 32.0–33.0 s; interval 30 s.
        let energies = profile(seconds: 70, quiet: [32.0...33.0])
        let w = windows(energies, length: 70, interval: 30)
        XCTAssertGreaterThanOrEqual(w[0].upperBound, 32.0,
                                    "the cut must wait for the pause, not fire at 30 s")
        XCTAssertLessThanOrEqual(w[0].upperBound, 33.0 + frame,
                                 "and it must take the FIRST pause, not a later one")
        XCTAssertNotEqual(w[0].upperBound, 30.0, accuracy: 1e-9,
                          "a 30.0 s cut is the uniform behaviour this replaces")
    }

    /// Continuous speech with no pause at all ⇒ the live rule's second clause:
    /// cut at 1.5x the interval, never later.
    func testContinuousSpeechFallsBackToTheOnePointFiveCap() {
        let energies = profile(seconds: 200, quiet: [])
        let w = windows(energies, length: 200, interval: 30)
        XCTAssertEqual(w[0].upperBound, 45.0, accuracy: frame,
                       "no silence in range ⇒ the hard cap, exactly as the live path does")
    }

    /// The cut never moves EARLIER than the interval — shortening a window would
    /// hand the model less context than it was going to get.
    func testBoundaryNeverMovesEarlierThanTheInterval() {
        // A pause well BEFORE the target must be ignored.
        let energies = profile(seconds: 90, quiet: [10.0...12.0, 34.0...35.0])
        let w = windows(energies, length: 90, interval: 30)
        XCTAssertGreaterThanOrEqual(w[0].upperBound, 30.0)
    }

    // MARK: - Invariants that must hold for any profile

    /// Windows must tile the recording: start at 0, end at the end, no gaps and
    /// no overlaps. `replaceOfficeSegments` deletes by time overlap, so a gap
    /// would leave a stretch of the meeting with no text at all.
    func testWindowsTileTheRecordingWithoutGaps() {
        for quiet in [[], [32.0...33.0], [31.0...31.5, 64.0...65.0]] {
            let energies = profile(seconds: 100, quiet: quiet)
            let w = windows(energies, length: 100, interval: 30)
            XCTAssertEqual(w.first?.lowerBound, 0, "must start at the beginning")
            XCTAssertEqual(w.last?.upperBound ?? 0, 100, accuracy: 1e-6,
                           "must reach the end of the recording")
            for (a, b) in zip(w, w.dropFirst()) {
                XCTAssertEqual(a.upperBound, b.lowerBound, accuracy: 1e-9,
                               "gap or overlap between windows")
            }
        }
    }

    /// Every window must be non-empty and forward-going, whatever the audio —
    /// a zero-length window would spin the driver loop forever.
    func testWindowsAlwaysAdvance() {
        for quiet in [[], [0.0...100.0], [29.0...31.0]] {
            let energies = profile(seconds: 100, quiet: quiet)
            for w in windows(energies, length: 100, interval: 30) {
                XCTAssertGreaterThan(w.upperBound, w.lowerBound)
            }
        }
    }

    /// Digital silence throughout must not produce a degenerate plan — the
    /// threshold is relative to the recording's own speech level, so a silent
    /// file must still tile rather than divide by zero or return nothing.
    func testAllSilentRecordingStillProducesUsableWindows() {
        let energies = [Float](repeating: 0, count: Int(100 / frame))
        let w = windows(energies, length: 100, interval: 30)
        XCTAssertFalse(w.isEmpty)
        XCTAssertEqual(w.last?.upperBound ?? 0, 100, accuracy: 1e-6)
    }

    /// An unreadable recording gives no profile, and the caller then falls back
    /// to uniform windows. Pinned so the fallback is not quietly removed.
    func testEmptyProfileYieldsNoWindowsSoTheCallerCanFallBack() {
        XCTAssertTrue(windows([], length: 100, interval: 30).isEmpty)
        XCTAssertFalse(AudioRecorder.fullPassWindows(recordingLength: 100,
                                                     intervalSec: 30).isEmpty,
                       "the uniform fallback must still work")
    }

    /// A missing file must return nil rather than throwing or hanging.
    func testEnergyProfileOfAMissingFileIsNil() {
        let url = URL(fileURLWithPath: "/nonexistent/\(UUID().uuidString).wav")
        XCTAssertNil(AudioRecorder.fullPassEnergyProfile(from: url))
    }
}
