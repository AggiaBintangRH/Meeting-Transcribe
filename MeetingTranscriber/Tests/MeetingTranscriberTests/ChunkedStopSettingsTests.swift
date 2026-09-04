import AVFoundation
import XCTest
@testable import MeetingTranscriber

/// The stop-time chunked-ASR pass (owner-requested 2026-08-03):
/// `chunked.finalPass` + `chunked.continueOnStop`, mirroring the Diarization
/// tab's pair in shape but not in default.
///
/// Same discipline as `RemoteStopModeTests`: everything that decides WHICH legs
/// of the stop gate are taken is a pure function, so the branch is testable
/// without an engine, a sidecar or a recording.
///
/// The regression bar for this whole file is requirement #1 — with both keys
/// ABSENT the stop path must be exactly what it was before this change — and the
/// dangerous corner is the unconfirmed-segment sweep, which is asserted in BOTH
/// directions because deleting the tail leaves no trace anywhere.
final class ChunkedStopSettingsTests: XCTestCase {

    private func mode(finalPass: Bool = true,
                      continueOnStop: Bool = true,
                      hasChunkedModel: Bool = true,
                      hasRecording: Bool = true,
                      model: String = "qwen3") -> AudioRecorder.ChunkedStopMode {
        AudioRecorder.chunkedStopMode(finalPass: finalPass,
                                      continueOnStop: continueOnStop,
                                      hasChunkedModel: hasChunkedModel,
                                      hasRecording: hasRecording,
                                      chunkedModelID: model)
    }

    private func plan(_ m: AudioRecorder.ChunkedStopMode) -> AudioRecorder.ChunkedStopPlan {
        AudioRecorder.chunkedStopPlan(m)
    }

    // MARK: - 1. Absent keys reproduce today's behaviour

    /// The exact read `stop()` performs. Both keys absent → the tail pass, i.e.
    /// the branch this code has always taken.
    func testAbsentKeysDefaultToTheTailPass() {
        let defaults = UserDefaults.standard
        let finalPass = defaults.object(forKey: "chunked.finalPass") as? Bool ?? true
        let continueOnStop = defaults.object(forKey: "chunked.continueOnStop") as? Bool ?? true
        // The defaults themselves, so a flipped `?? false` cannot pass silently.
        XCTAssertTrue(nil as Bool? ?? true, "finalPass defaults to true")
        XCTAssertTrue(nil as Bool? ?? true, "continueOnStop defaults to true")
        XCTAssertEqual(mode(finalPass: finalPass, continueOnStop: continueOnStop), .tail)
    }

    /// The mirrored-but-opposite default, pinned so the pair cannot be "made
    /// consistent" by someone who has not read why. Diarization's tail toggle is
    /// FALSE by default (a stop pass re-diarizes everything); the chunked one is
    /// TRUE (the pass covers only the tail), because TRUE is what the app does.
    func testChunkedTailDefaultIsTheOppositeOfDiarizationsOnPurpose() {
        let diarizationDefault = nil as Bool? ?? false     // diarization.continueOnStop
        let chunkedDefault = nil as Bool? ?? true          // chunked.continueOnStop
        XCTAssertNotEqual(diarizationDefault, chunkedDefault)
    }

    /// The default plan IS today's stop path, item by item: queue the tail
    /// window, stash its align audio, FLUSH the sidecar, do not settle here, and
    /// sweep the realtime leftovers afterwards. No full pass.
    func testDefaultPlanIsTodaysStopPathItemByItem() {
        XCTAssertEqual(plan(mode()),
                       AudioRecorder.ChunkedStopPlan(queuesTailWindow: true,
                                                     stashesAlignAudio: true,
                                                     flushesSidecar: true,
                                                     settlesImmediately: false,
                                                     sweepsUnconfirmedTail: true,
                                                     runsFullPass: false))
    }

    /// The pre-existing "no chunked model" branch, unchanged in behaviour and now
    /// expressed as `.none`: settle right here, queue nothing, flush nothing.
    func testNoChunkedModelStillSettlesImmediatelyAndQueuesNothing() {
        let p = plan(mode(hasChunkedModel: false))
        XCTAssertTrue(p.settlesImmediately)
        XCTAssertFalse(p.queuesTailWindow)
        XCTAssertFalse(p.flushesSidecar)
        XCTAssertFalse(p.runsFullPass)
        // …whatever the two new settings say — no model means no pass, full stop.
        for finalPass in [true, false] {
            for tail in [true, false] {
                XCTAssertEqual(mode(finalPass: finalPass, continueOnStop: tail,
                                    hasChunkedModel: false), AudioRecorder.ChunkedStopMode.none)
            }
        }
    }

    // MARK: - 2. finalPass OFF

    /// finalPass OFF settles the leg with no window queued, no align audio
    /// stashed and no FLUSH — nothing is coming, so the gate has to complete in
    /// `stop()` itself.
    func testFinalPassOffSettlesWithoutQueueingOrStashingOrFlushing() {
        for tail in [true, false] {
            let p = plan(mode(finalPass: false, continueOnStop: tail))
            XCTAssertEqual(mode(finalPass: false, continueOnStop: tail),
                           AudioRecorder.ChunkedStopMode.none, "tail=\(tail)")
            XCTAssertTrue(p.settlesImmediately, "tail=\(tail)")
            XCTAssertFalse(p.queuesTailWindow, "tail=\(tail)")
            XCTAssertFalse(p.stashesAlignAudio, "tail=\(tail)")
            XCTAssertFalse(p.flushesSidecar, "tail=\(tail)")
            XCTAssertFalse(p.runsFullPass, "tail=\(tail)")
        }
    }

    /// THE dangerous assertion, in BOTH directions. With no pass the realtime
    /// tail is the only text that audio will ever have, so the sweep must be off;
    /// with a pass it must stay on, or a pre-Stop fragment survives as an orphan
    /// SPEAKER UNKNOWN row — the bug the sweep was added for.
    func testTheUnconfirmedSweepFollowsWhetherAPassActuallyRan() {
        XCTAssertFalse(plan(mode(finalPass: false)).sweepsUnconfirmedTail,
                       "finalPass off must NOT delete the realtime tail")
        XCTAssertFalse(plan(mode(hasChunkedModel: false)).sweepsUnconfirmedTail,
                       "no chunked model means no authoritative text either")
        XCTAssertTrue(plan(mode()).sweepsUnconfirmedTail,
                      "the tail pass is authoritative — the sweep stays on")
        XCTAssertTrue(plan(mode(continueOnStop: false)).sweepsUnconfirmedTail,
                      "the full pass replaces every confirmed segment — sweep stays on")
    }

    /// Exactly one mode skips the sweep, and it is the one where no ASR ran.
    func testOnlyTheNoPassModeSkipsTheSweep() {
        for m in [AudioRecorder.ChunkedStopMode.none, .tail, .full] {
            XCTAssertEqual(plan(m).sweepsUnconfirmedTail, m != .none, "\(m)")
        }
    }

    // MARK: - 3. The full pass

    func testTailOffAsksForTheFullPass() {
        XCTAssertEqual(mode(continueOnStop: false), .full)
        let p = plan(.full)
        XCTAssertTrue(p.runsFullPass)
        // The full pass covers the tail itself, so the sidecar's live buffer is
        // deliberately NOT flushed: a FLUSH would emit a `final` that pops a
        // window from an empty queue and append a windowless segment.
        XCTAssertFalse(p.flushesSidecar)
        XCTAssertFalse(p.queuesTailWindow)
        XCTAssertFalse(p.settlesImmediately)
    }

    /// A uniform partition of the whole recording, in order, with no gaps and no
    /// overlaps — 0…30, 30…60, … and a final partial window.
    func testFullPassEnumeratesTheWholeRecordingInOrder() {
        let windows = AudioRecorder.fullPassWindows(recordingLength: 74.9, intervalSec: 30)
        XCTAssertEqual(windows.count, 3)
        XCTAssertEqual(windows[0].lowerBound, 0, accuracy: 1e-9)
        XCTAssertEqual(windows[0].upperBound, 30, accuracy: 1e-9)
        XCTAssertEqual(windows[1].lowerBound, 30, accuracy: 1e-9)
        XCTAssertEqual(windows[2].upperBound, 74.9, accuracy: 1e-9)
        for (a, b) in zip(windows, windows.dropFirst()) {
            XCTAssertEqual(a.upperBound, b.lowerBound, accuracy: 1e-9, "no gap, no overlap")
        }
    }

    /// A 60-minute meeting at each offered interval. These counts are what the
    /// watchdog budget and the "(7/120)" progress label are computed from.
    func testFullPassWindowCountsForARealMeetingLength() {
        XCTAssertEqual(AudioRecorder.fullPassWindows(recordingLength: 3600, intervalSec: 15).count, 240)
        XCTAssertEqual(AudioRecorder.fullPassWindows(recordingLength: 3600, intervalSec: 30).count, 120)
        XCTAssertEqual(AudioRecorder.fullPassWindows(recordingLength: 3600, intervalSec: 60).count, 60)
        XCTAssertEqual(AudioRecorder.fullPassWindows(recordingLength: 3600, intervalSec: 120).count, 30)
    }

    /// An exact multiple must not produce a zero-length trailing window, and a
    /// sub-quarter-second sliver is dropped rather than costing a round trip.
    func testFullPassDropsSliversAndDegenerateInputs() {
        XCTAssertEqual(AudioRecorder.fullPassWindows(recordingLength: 60, intervalSec: 30).count, 2)
        let withSliver = AudioRecorder.fullPassWindows(recordingLength: 60.1, intervalSec: 30)
        XCTAssertEqual(withSliver.count, 2, "0.1 s tail is not worth a window")
        XCTAssertTrue(AudioRecorder.fullPassWindows(recordingLength: 0, intervalSec: 30).isEmpty)
        XCTAssertTrue(AudioRecorder.fullPassWindows(recordingLength: 0.1, intervalSec: 30).isEmpty)
        XCTAssertTrue(AudioRecorder.fullPassWindows(recordingLength: 100, intervalSec: 0).isEmpty)
        XCTAssertTrue(AudioRecorder.fullPassWindows(recordingLength: -5, intervalSec: 30).isEmpty)
    }

    /// The watchdog must scale with the work, or it fires in the middle of a
    /// healthy pass and reports minutes of real transcription as a timeout.
    func testWatchdogBudgetScalesWithTheWindowCount() {
        let short = AudioRecorder.fullPassWatchdogSeconds(windowCount: 3)
        let long = AudioRecorder.fullPassWatchdogSeconds(windowCount: 120)
        XCTAssertGreaterThan(long, short)
        XCTAssertGreaterThan(long, 600, "must exceed the unscaled stop watchdog")
        // Never zero, whatever it is handed.
        XCTAssertGreaterThan(AudioRecorder.fullPassWatchdogSeconds(windowCount: 0), 0)
    }

    // MARK: - 4. Refusals

    /// ⚠ VOXTRAL IS NOW THE ONLY REFUSAL. MOSS was lifted 2026-08-18 — the
    /// message cited a 2048-token cap raised to 5120 on 2026-08-05, and a 360 s
    /// pass this code has never made (it cuts at `chunked.intervalSec`, ceiling
    /// 120 s). What actually blocked it was the FILE-TRANSCRIBE frame discarding
    /// MOSS's speaker segments; that frame carries them now.
    ///
    /// Voxtral's refusal is untouched and is about TIME, which no cap change can
    /// move: ~27 s per 30 s chunk is ~54 minutes for a 60-minute meeting.
    func testOnlyVoxtralRefusesTheFullPass() {
        XCTAssertNotNil(AudioRecorder.chunkedFullPassRefusalMessage(chunkedModelID: "voxtral"))
        // The mid-recording hole: a refused full pass degrades to the tail,
        // never to "no pass at all", so no audio is left untranscribed.
        XCTAssertEqual(mode(continueOnStop: false, model: "voxtral"), .tail)
        XCTAssertEqual(mode(continueOnStop: true, model: "voxtral"), .tail)
    }

    /// The half that would have caught the lift being done by deletion rather
    /// than by decision: MOSS must now REACH the full pass, not merely stop
    /// being refused.
    func testMossReachesTheFullPassNow() {
        XCTAssertNil(AudioRecorder.chunkedFullPassRefusalMessage(chunkedModelID: "moss"),
                     "the token-cap reason expired twice over — see chunkedFullPassRefusalMessage")
        XCTAssertEqual(mode(continueOnStop: false, model: "moss"), .full,
                       "stop pass on + continue off must re-transcribe the whole recording")
        XCTAssertEqual(mode(continueOnStop: true, model: "moss"), .tail,
                       "and continue-on-stop must still mean tail, exactly as before")
    }

    func testEveryOtherModelMayRunTheFullPass() {
        for id in ["qwen3", "whisper", "granite", "moss"] {
            XCTAssertNil(AudioRecorder.chunkedFullPassRefusalMessage(chunkedModelID: id), id)
            XCTAssertEqual(mode(continueOnStop: false, model: id), .full, id)
        }
    }

    // MARK: - The windows must COVER the recording

    /// Every second of the recording lands in exactly one window.
    ///
    /// ⚠ NOTHING ASSERTED THIS BEFORE, and it is the property the whole mode
    /// rests on: `replaceOfficeSegments` DELETES the live chunk text a window
    /// overlaps and puts the new text in its place. A gap between two windows is
    /// therefore not a gap in the re-transcription — it is **speech deleted from
    /// the transcript**, with the live text that covered it removed by the
    /// windows on either side. That is this project's worst failure direction and
    /// it would leave no trace outside the log.
    ///
    /// Driven across adversarial profiles rather than one tidy case, because the
    /// silence search is what can move a boundary: all-quiet (every candidate
    /// matches, so cuts land at the target), all-loud (nothing matches, so every
    /// cut falls back to the 1.5x cap), and a real alternating pattern.
    func testTheWindowsTileTheRecordingWithNoGapAndNoOverlap() {
        let frame = AudioRecorder.silenceFrameSec
        let length = 400.0
        let frames = Int(length / frame) + 8

        let profiles: [(String, [Float])] = [
            ("all quiet", [Float](repeating: 0.0001, count: frames)),
            ("all loud", [Float](repeating: 0.5, count: frames)),
            ("alternating 3 s speech / 1 s pause",
             (0..<frames).map { Float(Int(Double($0) * frame) % 4 == 3 ? 0.0001 : 0.5) }),
        ]

        for interval in [30.0, 60.0, 120.0] {
            for (name, energies) in profiles {
                let windows = AudioRecorder.fullPassWindowsAtSilence(
                    recordingLength: length, intervalSec: interval, energies: energies)
                let label = "\(name) @ \(Int(interval))s"
                XCTAssertFalse(windows.isEmpty, label)
                XCTAssertEqual(windows.first?.lowerBound ?? -1, 0.0, accuracy: 1e-9,
                               "\(label): the pass must start at the beginning of the recording")
                // The tail may legitimately be dropped when it is under
                // `minTailSec` — a quarter second no model can use — and nothing
                // more than that.
                XCTAssertEqual(windows.last?.upperBound ?? -1, length, accuracy: 0.25,
                               "\(label): audio after the last window is deleted, not kept")
                for (a, b) in zip(windows, windows.dropFirst()) {
                    XCTAssertEqual(a.upperBound, b.lowerBound, accuracy: 1e-9,
                                   "\(label): a gap or an overlap between windows — a GAP is "
                                   + "deleted speech, because the windows either side remove "
                                   + "the live text that covered it")
                }
            }
        }
    }

    // MARK: - The scope of the stop pass (restored 2026-09-04)

    /// A FRESH INSTALL re-transcribes the WHOLE recording, not the tail.
    ///
    /// ⚠ THIS ASSERTS THE THING THE OWNER REPORTED AS BROKEN. From 2026-08-06 the
    /// mode was pinned to `.tail` by a hard-coded `let chunkedTailOnly = true` in
    /// `stop()`, and their report was precise: "it not remove the chunk it use
    /// the chunked text instead re transcribe at start to stop". Tail-only
    /// re-transcribes roughly one chunk and leaves every earlier row exactly as
    /// the live pass wrote it — which is what they saw.
    ///
    /// Written against `ShippedDefaults` rather than a literal `.full`, so
    /// changing the shipped scope moves the assertion with it rather than
    /// leaving a test that pins a value nobody ships.
    func testAFreshInstallReTranscribesTheWholeRecording() {
        let shipped = mode(finalPass: true,
                           continueOnStop: !ShippedDefaults.chunkedFullPassAtStop)
        XCTAssertEqual(shipped, .full,
                       "the shipped stop pass must cover the whole recording — tail-only "
                       + "leaves the live chunk text in place, which is the defect reported")
        XCTAssertTrue(plan(shipped).runsFullPass)
        XCTAssertFalse(plan(shipped).queuesTailWindow,
                       "a full pass must not ALSO queue the tail window — that audio is "
                       + "already inside the last full-pass window and would be transcribed twice")
    }

    /// Both scopes stay reachable, and only the scope changes between them.
    /// Without this the fix could silently become "always full", which is the
    /// same defect as "always tail" pointing the other way — and Voxtral has
    /// nowhere to go if the tail is unreachable.
    func testTheScopeIsAChoiceAndNothingElseMovesWithIt() {
        XCTAssertEqual(mode(continueOnStop: true), .tail)
        XCTAssertEqual(mode(continueOnStop: false), .full)
        for m in [AudioRecorder.ChunkedStopMode.tail, .full] {
            XCTAssertTrue(plan(m).sweepsUnconfirmedTail,
                          "\(m): a pass DID run, so the leftover realtime text is superseded")
            XCTAssertFalse(plan(m).settlesImmediately, "\(m): something asynchronous is coming")
        }
    }

    /// The refusal names the model, says what to do about it, and points at the
    /// setting by the words the user actually sees — the shape every other
    /// refusal in this app follows.
    func testRefusalsSayWhatToDoAboutIt() {
        for id in ["voxtral"] {
            let message = AudioRecorder.chunkedFullPassRefusalMessage(chunkedModelID: id) ?? ""
            XCTAssertTrue(message.contains(AudioRecorder.fullPassToggleLabel),
                          "\(id): the refusal must point at the toggle by the words on it")
            XCTAssertTrue(message.contains("chunked model"), id)
        }
        // Every model the picker offers has a cost line for the Settings copy.
        for id in ["qwen3", "whisper", "granite", "voxtral", "moss"] {
            XCTAssertFalse(AudioRecorder.fullPassCostNote(chunkedModelID: id).isEmpty, id)
        }
    }

    /// A missing recording cannot reach the full pass — there would be nothing to
    /// re-transcribe — and degrades to the tail rather than to nothing.
    func testFullPassNeedsTheRecordingAndDegradesToTheTailWithoutIt() {
        XCTAssertEqual(mode(continueOnStop: false, hasRecording: false), .tail)
        XCTAssertEqual(mode(continueOnStop: true, hasRecording: false), .tail)
    }

    // MARK: - 5. The stop gate can never hang

    /// The whole input space: every mode either settles the chunk leg in `stop()`
    /// itself, or hands it to something that will settle it — a queued window
    /// drained by `onChunkTranscript`/`onChunkError`/`chunkWatchdog`, or the full
    /// pass, which holds `chunkedBusy` and settles on completion, on failure and
    /// on its own scaled watchdog.
    func testEveryBranchEitherSettlesHereOrHasAJobToSettleIt() {
        var seen = Set<AudioRecorder.ChunkedStopMode>()
        for finalPass in [true, false] {
            for tail in [true, false] {
                for hasModel in [true, false] {
                    for hasRecording in [true, false] {
                        for id in ["qwen3", "whisper", "granite", "voxtral", "moss"] {
                            let m = mode(finalPass: finalPass, continueOnStop: tail,
                                         hasChunkedModel: hasModel,
                                         hasRecording: hasRecording, model: id)
                            seen.insert(m)
                            let p = plan(m)
                            let asynchronous = p.flushesSidecar || p.runsFullPass
                            XCTAssertNotEqual(p.settlesImmediately, asynchronous,
                                              "exactly one of the two must be true (\(m))")
                            // A queued window exists if and only if something will
                            // drain it.
                            XCTAssertEqual(p.queuesTailWindow, p.flushesSidecar)
                            // Nothing is stashed for the aligner without a window
                            // to attach it to, or the buffer leaks for the session.
                            if p.stashesAlignAudio { XCTAssertTrue(p.queuesTailWindow) }
                            if m == .full {
                                XCTAssertTrue(hasRecording && hasModel && finalPass && !tail)
                                XCTAssertNil(AudioRecorder.chunkedFullPassRefusalMessage(chunkedModelID: id))
                            }
                        }
                    }
                }
            }
        }
        XCTAssertEqual(seen, [.none, .tail, .full], "all three branches are reachable")
    }

    // MARK: - 6. Reading a window out of the recorded WAV

    /// The full pass writes one temp WAV per window and must delete it on BOTH
    /// the success and the error path. `writeTempWAV` is the only thing that
    /// creates one, so this pins the invariant the loop depends on: the file is
    /// really there while the request runs, and removing it really removes it.
    func testTempWindowWAVsAreWrittenAndFullyRemoved() throws {
        let samples = (0..<16_000).map { sinf(Float($0) * 0.01) }
        let url = try XCTUnwrap(AudioRecorder.writeTempWAV(samples: samples, prefix: "full-pass"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        try? FileManager.default.removeItem(at: url)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path),
                       "the error path's cleanup must leave nothing behind")
    }

    /// A window really is read back as the 16 kHz mono audio the sidecars expect,
    /// and out-of-range or empty windows return nil rather than a silent
    /// zero-length buffer that would be transcribed as a hallucination.
    func testLoadWindowReturnsSixteenKilohertzMonoOrNil() throws {
        // 4 s of 16 kHz mono, written the same way a recording is.
        let samples = (0..<64_000).map { sinf(Float($0) * 0.02) }
        let source = try XCTUnwrap(AudioRecorder.writeTempWAV(samples: samples, prefix: "full-pass-src"))
        defer { try? FileManager.default.removeItem(at: source) }

        // Exact, not approximate — `AVAudioFile.read` returns fewer frames than
        // asked for (32 000 requested comes back as 31 104), so a single read
        // would silently shorten every window by ~3 %. The loop in
        // `loadWindow16k` exists for this, and only an exact count catches it.
        let window = try XCTUnwrap(AudioRecorder.loadWindow16k(from: source, start: 1.0, end: 3.0))
        XCTAssertEqual(window.count, 32_000, "2 s at 16 kHz, whole")

        // Past the end, and a zero-length window.
        XCTAssertNil(AudioRecorder.loadWindow16k(from: source, start: 10, end: 12))
        XCTAssertNil(AudioRecorder.loadWindow16k(from: source, start: 2, end: 2))
        XCTAssertNil(AudioRecorder.loadWindow16k(
            from: FileManager.default.temporaryDirectory
                .appendingPathComponent("no-such-recording.wav"),
            start: 0, end: 1))

        // The last window is clamped to the file rather than padded with silence —
        // padding would feed the model digital silence, which MOSS answers as a
        // chatbot and Whisper answers with a canned caption.
        let tail = try XCTUnwrap(AudioRecorder.loadWindow16k(from: source, start: 3.5, end: 8.0))
        XCTAssertEqual(tail.count, 8_000, "0.5 s of real audio, not 4.5 s padded with silence")
    }
}
