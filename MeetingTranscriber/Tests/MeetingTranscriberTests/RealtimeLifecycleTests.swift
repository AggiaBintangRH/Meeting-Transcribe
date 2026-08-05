import XCTest
@testable import MeetingTranscriber

/// `ModelLoader.wantsRealtime` — the single rule that decides whether this
/// session keeps the REALTIME (Nemotron) sidecar alive.
///
/// WHY THESE EXIST. Until 2026-08-05 the realtime sidecar was loaded by
/// `realtime.enabled` and stopped by nothing: the only `terminate()` in the file
/// sat inside the load step, to replace the process when its settings changed.
/// Services are kept alive across sessions, so switching realtime captions off
/// left a **1.70 GB** process resident until the app quit, with nothing able to
/// ask it anything. It was invisible because the BEHAVIOUR was already right —
/// `AudioRecorder` reads `nemotronASR?.office` only when the same flag is on — so
/// only the memory was wrong, and memory does not show up in a transcript.
///
/// This was the last persistent service without a want/teardown pair. The rule
/// now feeds BOTH `buildSteps` and the teardown in `loadAll`, so these are
/// equally tests of "does the app start it" and "does the app stop it".
final class RealtimeLifecycleTests: XCTestCase {

    /// Both directions of the rule itself. Thin on purpose — the value of the
    /// function is that ONE of it exists, not that it computes something clever.
    func testTheFlagDecidesInBothDirections() {
        XCTAssertTrue(ModelLoader.wantsRealtime(realtimeEnabled: true))
        XCTAssertFalse(ModelLoader.wantsRealtime(realtimeEnabled: false),
                       "realtime off must mean the sidecar is not kept — this is "
                       + "the case that used to leak 1.70 GB")
    }

    /// THE LOAD-BEARING ONE. The teardown terminates the process when
    /// `wantsRealtime` is false, and detaches only the remote lane when
    /// `wantsRemoteRealtime` is false. Those two must never both apply to a live
    /// process: wanting a remote LANE on a sidecar this session is about to
    /// terminate is incoherent, and would mean the `else if` at the call site
    /// silently dropped a case that mattered.
    ///
    /// Checked across the whole matrix rather than at one point, because the
    /// coupling lives in `wantsRemoteRealtime`'s `&& realtimeEnabled` and a future
    /// edit could remove it without touching this file.
    func testRemoteRealtimeIsNeverWantedWhenRealtimeItselfIsNot() {
        for channel in [nil, 0, 1, 7] as [Int?] {
            for realtime in [true, false] {
                let keepsProcess = ModelLoader.wantsRealtime(realtimeEnabled: realtime)
                let wantsRemoteLane = ModelLoader.wantsRemoteRealtime(
                    remoteChannel: channel, realtimeEnabled: realtime)
                if wantsRemoteLane {
                    XCTAssertTrue(keepsProcess,
                                  "a remote lane was wanted on a sidecar this session "
                                  + "terminates (channel=\(String(describing: channel)), "
                                  + "realtime=\(realtime))")
                }
            }
        }
    }

    /// A Remote channel alone must not resurrect realtime. The dual-stream work
    /// added a second lane to the SAME process, so "there is a Remote channel" is
    /// not by itself a reason to keep a sidecar the user switched off.
    func testARemoteChannelDoesNotKeepTheSidecarAliveOnItsOwn() {
        XCTAssertFalse(ModelLoader.wantsRealtime(realtimeEnabled: false))
        XCTAssertFalse(ModelLoader.wantsRemoteRealtime(remoteChannel: 1,
                                                       realtimeEnabled: false))
    }
}
