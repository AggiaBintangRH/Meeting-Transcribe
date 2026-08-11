import XCTest
@testable import MeetingTranscriber

/// `ModelLoader.wantsRealtime` — the single rule that decides whether this
/// session keeps the REALTIME sidecar alive (Nemotron or Parakeet, per
/// `realtime.model`; the rule is about the process, not about which engine runs
/// in it, which is why a second engine did not change its signature).
///
/// WHY THESE EXIST. Until 2026-08-05 the realtime sidecar was loaded by
/// `realtime.enabled` and stopped by nothing: the only `terminate()` in the file
/// sat inside the load step, to replace the process when its settings changed.
/// Services are kept alive across sessions, so switching realtime captions off
/// left a **1.70 GB** process resident until the app quit, with nothing able to
/// ask it anything. It was invisible because the BEHAVIOUR was already right —
/// `AudioRecorder` reads `realtimeASR?.office` only when the same flag is on — so
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

    // MARK: - The engine SWITCH (2026-08-11)

    /// AN ABSENT `realtime.model` MUST PRODUCE EXACTLY THE NEMOTRON CONFIG.
    ///
    /// This is the first-launch-after-update case, and it is the one that fails
    /// invisibly: `ModelLoader.load` reuses a running sidecar only when the new
    /// `Config` is EQUAL to the live one, so a default that is a hair off would
    /// terminate and reload a perfectly good 1.70 GB process — every session, for
    /// no reason, with nothing in the UI saying why. The same governing rule the
    /// Whisper options change was built around: defaults reproduce today's
    /// behaviour exactly.
    func testAnAbsentModelKeyIsExactlyTheNemotronConfig() {
        let d = UserDefaults.standard
        let savedModel = d.object(forKey: "realtime.model")
        let savedLanguage = d.object(forKey: "realtime.language")
        let savedChunk = d.object(forKey: "realtime.chunkMs")
        defer {
            for (key, value) in [("realtime.model", savedModel),
                                 ("realtime.language", savedLanguage),
                                 ("realtime.chunkMs", savedChunk)] {
                if let value { d.set(value, forKey: key) } else { d.removeObject(forKey: key) }
            }
        }
        for key in ["realtime.model", "realtime.language", "realtime.chunkMs"] {
            d.removeObject(forKey: key)
        }
        // chunkMs is the PINNED 1120, not the old 160 default — and 1120 really
        // is "what has always run". The 160 never reached the model: the sidecar
        // dropped it into a temp-WAV fallback that cannot carry an attention
        // context, and 1120 is byte-identical to running with none at all. So
        // pinning it changed the stored number and NOT the behaviour.
        XCTAssertEqual(RealtimeASRService.Config.fromSettings(),
                       .nemotron(language: "auto",
                                 chunkMs: RealtimeASRService.pinnedChunkMs,
                                 partialMs: RealtimeASRService.defaultPartialMs),
                       "with no stored settings the realtime config must be the one "
                       + "that has always run — otherwise the first launch after an "
                       + "update silently reloads the sidecar")
    }

    /// THE ENGINE SWITCH REALLY REPLACES THE PROCESS. `Config` equality is the
    /// whole mechanism: `ModelLoader.load` returns early when `existing.config ==
    /// config`, so an engine id that did not participate in equality would leave
    /// the OLD model transcribing while Settings showed the new one selected —
    /// the substitution this project refuses everywhere else.
    func testSwitchingEngineChangesTheConfigAndSoReplacesTheProcess() {
        let nemotron = RealtimeASRService.Config.nemotron(language: "auto", chunkMs: 160, partialMs: RealtimeASRService.defaultPartialMs)
        let parakeet = RealtimeASRService.Config.parakeet(language: "auto", partialMs: RealtimeASRService.defaultPartialMs)
        XCTAssertNotEqual(nemotron, parakeet,
                          "the two engines' configs compare equal — a switch would "
                          + "reuse the running sidecar and keep the old model")
        XCTAssertNotEqual(nemotron.scriptName, parakeet.scriptName)
        XCTAssertNotEqual(nemotron.logName, parakeet.logName)
    }

    /// A NEMOTRON KNOB MOVED WHILE PARAKEET IS SELECTED MUST NOT RECREATE THE
    /// PARAKEET PROCESS. `chunkMs` is nil under Parakeet — structurally, not by
    /// convention — so the value cannot enter its `Config` and cannot make it
    /// unequal. Without this, changing a control that is not even VISIBLE under
    /// the selected engine would tear down and reload a 2.3 GB sidecar.
    func testANemotronKnobDoesNotRecreateTheParakeetProcess() {
        let d = UserDefaults.standard
        let savedModel = d.object(forKey: "realtime.model")
        let savedChunk = d.object(forKey: "realtime.chunkMs")
        defer {
            if let savedModel { d.set(savedModel, forKey: "realtime.model") }
            else { d.removeObject(forKey: "realtime.model") }
            if let savedChunk { d.set(savedChunk, forKey: "realtime.chunkMs") }
            else { d.removeObject(forKey: "realtime.chunkMs") }
        }
        d.set("parakeet", forKey: "realtime.model")
        d.set(160, forKey: "realtime.chunkMs")
        let before = RealtimeASRService.Config.fromSettings()
        d.set(1120, forKey: "realtime.chunkMs")
        XCTAssertEqual(before, RealtimeASRService.Config.fromSettings(),
                       "moving Nemotron's chunk size changed the PARAKEET config")
    }
}
