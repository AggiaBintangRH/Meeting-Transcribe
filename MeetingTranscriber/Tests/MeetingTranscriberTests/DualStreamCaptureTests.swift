import XCTest
@testable import MeetingTranscriber

/// Unit tests for phase 2 of dual-stream capture — the parts of `beginCapture`'s
/// Remote path that are reachable without audio hardware.
///
/// Phase 1 (`DualStreamSettingsTests`) covers resolution against the *stored*
/// device's channel count. This file covers the second, independent check:
/// re-validating that choice against the format the engine actually delivers.
/// The two are separate because an **Aggregate Device** can present a different
/// layout at capture time than it advertised in the device list — the single
/// biggest unknown in the feature, and the reason this phase exists at all.
///
/// What is NOT testable here (no hardware in CI): the tap callback itself,
/// channel-index ordering across an aggregate's sub-devices, drift-corrected
/// buffer sizes, and whether the written Remote WAV actually contains the
/// conferencing audio. Those are on-device checks.
final class DualStreamCaptureTests: XCTestCase {

    // MARK: - 1. Nil remote ⇒ the added code is inert

    /// The regression bar for phase 2: with no Remote configured, the resolver
    /// returns nil, so no second file is created and the tap's Remote block is
    /// skipped entirely — the callback does exactly what it did before.
    func testNilRemoteStaysNil() {
        XCTAssertNil(AudioRecorder.resolveRemoteChannel(nil,
                                                        officeChannel: 0,
                                                        liveChannelCount: 8))
    }

    /// A negative stored value can only arrive from a stale/garbage default;
    /// it is off, never an index.
    func testNegativeRemoteIsOff() {
        XCTAssertNil(AudioRecorder.resolveRemoteChannel(-1, officeChannel: 0, liveChannelCount: 8))
        XCTAssertNil(AudioRecorder.resolveRemoteChannel(-7, officeChannel: 0, liveChannelCount: 8))
    }

    // MARK: - 2. A valid remote survives the live-format clamp

    func testValidRemoteSurvives() {
        XCTAssertEqual(AudioRecorder.resolveRemoteChannel(1, officeChannel: 0, liveChannelCount: 2), 1)
        XCTAssertEqual(AudioRecorder.resolveRemoteChannel(2, officeChannel: 5, liveChannelCount: 8), 2)
        // Last channel of the live format is in range (the bound is exclusive).
        XCTAssertEqual(AudioRecorder.resolveRemoteChannel(7, officeChannel: 0, liveChannelCount: 8), 7)
    }

    // MARK: - 3. The live format is the authority

    /// The scenario this whole re-check exists for: the aggregate device delivers
    /// fewer channels than the saved selection assumed. Degrade to single-stream
    /// (reject, don't relocate) rather than fail the recording — capturing the
    /// room is far better than capturing nothing.
    func testRemotePastLiveChannelCountDegrades() {
        XCTAssertNil(AudioRecorder.resolveRemoteChannel(6, officeChannel: 0, liveChannelCount: 2))
        XCTAssertNil(AudioRecorder.resolveRemoteChannel(2, officeChannel: 0, liveChannelCount: 2),
                     "The bound is exclusive — channel 2 does not exist on a 2-channel format")
    }

    /// A mono live format can never carry a Remote stream.
    func testMonoLiveFormatHasNoRemote() {
        XCTAssertNil(AudioRecorder.resolveRemoteChannel(1, officeChannel: 0, liveChannelCount: 1))
    }

    /// Collision is checked against the office channel AFTER its own live-format
    /// clamp (`min(mic.channel, channelCount - 1)`), so a stale office value that
    /// clamps onto the remote channel cannot leave both streams reading one channel.
    func testCollisionWithClampedOfficeChannelDegrades() {
        // mic.channel 99 on a 2-channel format clamps to 1; remote 1 now collides.
        let office = min(99, 2 - 1)
        XCTAssertEqual(office, 1)
        XCTAssertNil(AudioRecorder.resolveRemoteChannel(1, officeChannel: office, liveChannelCount: 2))
    }

    // MARK: - 4. Filename derivation

    /// The Remote name is derived from the Office URL, never re-formatted from
    /// `Date()`, so the pair can never end up with two different stamps.
    func testRemoteURLDerivation() {
        let office = URL(fileURLWithPath: "/tmp/recordings/meeting-2026-07-21T10-30-00Z.wav")
        let remote = AudioRecorder.remoteURL(forOffice: office)
        XCTAssertEqual(remote.lastPathComponent, "meeting-2026-07-21T10-30-00Z-remote.wav")
        XCTAssertEqual(remote.deletingLastPathComponent(), office.deletingLastPathComponent(),
                       "Both files live in the same recordings directory")
    }

    /// Dots inside the stamp must not be mistaken for the extension boundary —
    /// only the trailing `.wav` is an extension.
    func testRemoteURLKeepsDottedStampIntact() {
        let office = URL(fileURLWithPath: "/tmp/recordings/meeting-2026-07-21T10-30-00.123Z.wav")
        XCTAssertEqual(AudioRecorder.remoteURL(forOffice: office).lastPathComponent,
                       "meeting-2026-07-21T10-30-00.123Z-remote.wav")
    }

    /// Derivation is a pure function of the office URL: same input, same output.
    func testRemoteURLIsDeterministic() {
        let office = URL(fileURLWithPath: "/tmp/recordings/meeting-x.wav")
        XCTAssertEqual(AudioRecorder.remoteURL(forOffice: office),
                       AudioRecorder.remoteURL(forOffice: office))
    }

    // MARK: - 5. Phase 1 → phase 2 hand-off

    /// End-to-end over the two pure stages, as `beginCapture` chains them: a
    /// selection valid for the advertised device but not for the live format must
    /// fall out at the second stage, not the first.
    func testSettingsResolutionThenLiveFormatClamp() {
        // Device advertised 8 channels; the aggregate actually delivers 2.
        let settings = MicrophoneSettings.resolve(savedChannel: 0, savedRemote: 5, channelCount: 8)
        XCTAssertEqual(settings.remoteChannel, 5, "Valid against the advertised count")

        let office = min(settings.channel, 2 - 1)
        XCTAssertNil(AudioRecorder.resolveRemoteChannel(settings.remoteChannel,
                                                        officeChannel: office,
                                                        liveChannelCount: 2),
                     "…but rejected once the live format turns out to be 2-channel")
    }
}
