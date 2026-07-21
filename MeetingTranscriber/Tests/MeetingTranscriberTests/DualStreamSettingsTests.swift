import XCTest
@testable import MeetingTranscriber

/// Unit tests for dual-stream (Office + Remote) settings resolution.
///
/// `MicrophoneSettings.resolve` is the whole decision — it is split out of
/// `current()` precisely so it can be exercised without UserDefaults or real audio
/// hardware. `savedRemote == nil` models an absent `mic.remoteChannel` key;
/// `channelCount == nil` models "no device matched" (system default).
///
/// The regression bar for the entire dual-stream feature is section 1: with no
/// Remote channel configured, resolution must produce exactly what it produced
/// before this feature existed — the same office channel and `nil` remote, so
/// every downstream consumer sees today's single-stream world.
final class DualStreamSettingsTests: XCTestCase {

    // MARK: - 1. Unset ⇒ single-stream, byte-identical to pre-feature behaviour

    /// Key absent ⇒ nil. This is the state of every existing installation, and it
    /// must not become "remote = channel 0" just because `integer(forKey:)`
    /// returns 0 for a missing key (which is why `current()` probes `object(forKey:)`).
    func testAbsentKeyIsSingleStream() {
        let r = MicrophoneSettings.resolve(savedChannel: 3, savedRemote: nil, channelCount: 8)
        XCTAssertEqual(r.channel, 3, "Office channel must be untouched by the remote logic")
        XCTAssertNil(r.remoteChannel, "An absent key means single-stream")
    }

    /// `-1` is the explicit "off" written by the UI once the user has touched the
    /// control — it must resolve identically to the key never having existed.
    func testMinusOneIsSingleStream() {
        let absent = MicrophoneSettings.resolve(savedChannel: 3, savedRemote: nil, channelCount: 8)
        let off    = MicrophoneSettings.resolve(savedChannel: 3, savedRemote: -1, channelCount: 8)
        XCTAssertEqual(off.channel, absent.channel)
        XCTAssertNil(off.remoteChannel)
    }

    /// The office channel keeps its old clamping in every remote state: unset,
    /// off, and set. Nothing about the office stream changed.
    func testOfficeChannelClampingUnchanged() {
        for remote: Int? in [nil, -1, 2] {
            let r = MicrophoneSettings.resolve(savedChannel: 99, savedRemote: remote, channelCount: 8)
            XCTAssertEqual(r.channel, 7, "Out-of-range office channel clamps to the last channel")
        }
        for remote: Int? in [nil, -1, 2] {
            let r = MicrophoneSettings.resolve(savedChannel: -5, savedRemote: remote, channelCount: 8)
            XCTAssertEqual(r.channel, 0, "Negative office channel clamps to 0")
        }
        // No device matched → count unknown → the office channel is passed through
        // (clamped later against the real format), exactly as before.
        let unknown = MicrophoneSettings.resolve(savedChannel: 99, savedRemote: nil, channelCount: nil)
        XCTAssertEqual(unknown.channel, 99)
    }

    // MARK: - 2. A valid remote resolves

    func testValidRemoteResolves() {
        let r = MicrophoneSettings.resolve(savedChannel: 0, savedRemote: 1, channelCount: 2)
        XCTAssertEqual(r.channel, 0)
        XCTAssertEqual(r.remoteChannel, 1)
    }

    func testValidRemoteBelowOfficeResolves() {
        let r = MicrophoneSettings.resolve(savedChannel: 5, savedRemote: 2, channelCount: 8)
        XCTAssertEqual(r.channel, 5)
        XCTAssertEqual(r.remoteChannel, 2)
    }

    // MARK: - 3. Invalid remotes fall back to single-stream

    /// Device swapped for one with fewer channels under a stored selection.
    /// Chosen behaviour: reset to nil (single-stream), NOT clamp to the last
    /// channel — clamping would silently record whatever happens to be on that
    /// channel while the user still believes Remote points where they put it.
    func testRemoteBeyondChannelCountResetsToNil() {
        let r = MicrophoneSettings.resolve(savedChannel: 0, savedRemote: 6, channelCount: 2)
        XCTAssertEqual(r.channel, 0, "The office channel survives")
        XCTAssertNil(r.remoteChannel)
    }

    /// A 1-channel device can never carry a Remote stream; it falls out of the
    /// range check without a special case.
    func testSingleChannelDeviceHasNoRemote() {
        let r = MicrophoneSettings.resolve(savedChannel: 0, savedRemote: 1, channelCount: 1)
        XCTAssertEqual(r.channel, 0)
        XCTAssertNil(r.remoteChannel)
    }

    /// Chosen behaviour for remote == office: REJECT (resolve to nil), so the two
    /// roles can never read one channel. The UI prevents this from being set at
    /// all, so reaching here means a stale value — falling back to the
    /// single-stream behaviour the user already understands.
    func testRemoteEqualToOfficeIsRejected() {
        let r = MicrophoneSettings.resolve(savedChannel: 4, savedRemote: 4, channelCount: 8)
        XCTAssertEqual(r.channel, 4)
        XCTAssertNil(r.remoteChannel)
    }

    /// The collision is checked against the CLAMPED office channel, not the raw
    /// stored one — otherwise a stale office value could clamp onto the remote
    /// channel and both streams would read the same audio.
    func testRemoteCollidesWithClampedOfficeChannel() {
        // savedChannel 99 clamps to 7 on an 8-channel device; remote 7 now collides.
        let r = MicrophoneSettings.resolve(savedChannel: 99, savedRemote: 7, channelCount: 8)
        XCTAssertEqual(r.channel, 7)
        XCTAssertNil(r.remoteChannel)
    }

    /// Negative values other than -1 are garbage, and are treated as off rather
    /// than as an index.
    func testNegativeRemoteIsOff() {
        let r = MicrophoneSettings.resolve(savedChannel: 0, savedRemote: -7, channelCount: 8)
        XCTAssertNil(r.remoteChannel)
    }

    // MARK: - 4. Unknown device (system default)

    /// With no device matched there is no channel count to validate against, so a
    /// non-colliding remote is kept and validated later against the live format.
    /// Only the checks that do not need the count still apply.
    func testUnknownDeviceKeepsNonCollidingRemote() {
        let kept = MicrophoneSettings.resolve(savedChannel: 0, savedRemote: 1, channelCount: nil)
        XCTAssertEqual(kept.remoteChannel, 1)

        let collision = MicrophoneSettings.resolve(savedChannel: 1, savedRemote: 1, channelCount: nil)
        XCTAssertNil(collision.remoteChannel)
    }
}
