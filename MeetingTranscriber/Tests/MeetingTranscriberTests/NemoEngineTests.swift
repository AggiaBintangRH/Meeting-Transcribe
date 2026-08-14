import XCTest
@testable import MeetingTranscriber

/// The NEMO diarization engine (2026-08-07) — the FOURTH diarizer, and the second
/// whole-file batch one after spectral.
///
/// These MIRROR `SpectralEngineTests` deliberately rather than sharing a base
/// class: the point of each is that THIS engine's wiring is right, and a shared
/// harness would let a missing NeMo case pass on spectral's behalf. What they pin,
/// and why each is a real failure rather than tidiness:
///
///  1. **The four-way stack rule.** `wantedDiarizationStack` feeds BOTH the load
///     step and the teardown. Three separate leaks in this codebase (pyannote, the
///     two overlap engines, Nemotron) were all a load rule and a teardown rule
///     computed apart, and NeMo is the most expensive one yet to leave resident:
///     its peak RSS scales with the audio it was last given — 13.33 GB for a
///     67-minute recording.
///  2. **The detector gate.** NME-SC assigns exactly one label per instant, so its
///     turns never intersect and `overlapRegions()` is empty by construction. Like
///     MOSS and spectral it therefore gets repair only through the standalone
///     detector — and pyannote must be untouched in both directions.
///  3. **The live-dispatch guard**, including the silent half: the buffer must not
///     accumulate at ~230 MB/hour for a pass that reads the recording FILE.
///  4. **Literal script and log names, differing per engine**, and the OWN
///     interpreter. `Config.mossDiarization()` once DERIVED its `logName` from a
///     model type; a rename then repointed one process's stderr into another's log
///     while it still ran the other script — invisible, because both processes
///     keep working.
final class NemoEngineTests: XCTestCase {

    private let pyannote = ModelLoader.pyannoteEngineID
    private let moss = ModelLoader.mossEngineID
    private let spectral = ModelLoader.spectralEngineID
    private let nemo = ModelLoader.nemoEngineID

    // MARK: - 1. The four-way stack rule

    private func stack(enabled: Bool = true, engine: String, chunked: String = "qwen3",
                       chunkedOn: Bool = true) -> ModelLoader.DiarizationStack? {
        ModelLoader.wantedDiarizationStack(diarizationEnabled: enabled,
                                           engine: engine, chunkedID: chunked,
                                           chunkedEnabled: chunkedOn)
    }

    /// NeMo selects ITS OWN stack, and — the direction that makes this able to fail
    /// — no other engine selects it. Asserted across chunked models too, because
    /// the MOSS branch is the only one that reads `chunkedID` and a mis-placed case
    /// could inherit that sensitivity.
    func testNemoSelectsItsOwnStackAndNoOtherEngineDoes() {
        for chunked in ["qwen3", "whisper", "moss"] {
            XCTAssertEqual(stack(engine: nemo, chunked: chunked), .nemo,
                           "chunked=\(chunked): nemo must select the nemo stack")
            for other in [pyannote, spectral, moss] {
                XCTAssertNotEqual(stack(engine: other, chunked: chunked), .nemo,
                                  "chunked=\(chunked): \(other) must never select the "
                                  + "nemo stack")
            }
        }
    }

    /// The master switch still wins over all FOUR engines — the 2026-07-31 bug's
    /// teardown-that-never-fired, extended to the new engine.
    func testDisablingDiarizationKeepsNothingAliveUnderNemoToo() {
        for engine in [pyannote, spectral, nemo, moss] {
            XCTAssertNil(stack(enabled: false, engine: engine),
                         "engine=\(engine): the off switch was ignored")
        }
    }

    /// Adding a fourth case must not have turned the fall-through into a hole that
    /// loads nothing for a stale or garbage stored value.
    func testAnUnknownEngineStillResolvesToPyannote() {
        XCTAssertEqual(stack(engine: "nemoish"), .pyannote)
        XCTAssertEqual(stack(engine: "NEMO"), .pyannote, "the value is matched exactly")
        XCTAssertEqual(stack(engine: ""), .pyannote)
    }

    /// The chunked master switch must not disturb this engine: it borrows nothing
    /// from the chunked process, so its stack is the same either way. Without this
    /// the MOSS coupling could quietly widen into every engine.
    func testTheChunkedSwitchCannotChangeTheNemoStack() {
        for chunkedID in ["qwen3", "whisper", "moss"] {
            XCTAssertEqual(stack(engine: nemo, chunked: chunkedID, chunkedOn: true),
                           stack(engine: nemo, chunked: chunkedID, chunkedOn: false),
                           "chunked=\(chunkedID) changed the nemo stack")
        }
    }

    /// NeMo emits run-local labels and needs WeSpeaker to say who those people are
    /// — the same path spectral and pyannote take, which is exactly why this engine
    /// needed NO new id space and keeps profiles, renaming and `spk`.
    func testTheNemoStackUsesSpeakerIdentity() {
        XCTAssertTrue(ModelLoader.DiarizationStack.nemo.usesSpeakerIdentity,
                      "without the embedder a NeMo turn has no way to be named")
    }

    /// The catalog id and the setting value must round-trip. This is the MOSS trap
    /// (`moss-diar` the id, `moss` the setting) generalised: binding a card to
    /// `model.id` stored a value nothing ever matched, so the card looked selected
    /// while the engine never switched. NeMo uses ONE string for both, like
    /// spectral — asserted rather than assumed.
    func testTheNemoCardRoundTripsThroughItsSettingValue() {
        XCTAssertEqual(ModelCatalog.diarizationEngineValue(ModelCatalog.nemoDiarization),
                       nemo)
        XCTAssertEqual(ModelCatalog.diarizationEngine(forEngine: nemo).id,
                       ModelCatalog.nemoDiarization.id)
        XCTAssertEqual(ModelCatalog.nemoDiarization.id, nemo,
                       "catalog id and setting value are deliberately the same string here")
        // And it really is in the list the tab renders — a card that exists but is
        // never offered is a setting nothing can select.
        XCTAssertTrue(ModelCatalog.diarizationEngines.contains { $0.id == nemo })
    }

    /// The MOSS⟺MOSS biconditional must treat NeMo exactly as it treats pyannote
    /// and spectral: offered beside any non-MOSS ASR, never beside MOSS. This
    /// needed no code change, which is precisely why it is worth asserting.
    func testNemoIsOfferedWhereverPyannoteAndSpectralAre() {
        XCTAssertFalse(ModelLoader.diarizationEngineIsSelectable(nemo, chunkedID: "moss"))
        for other in ["qwen3", "whisper", "granite", "voxtral"] {
            XCTAssertTrue(ModelLoader.diarizationEngineIsSelectable(nemo, chunkedID: other))
        }
    }

    // MARK: - 2. The detector gate

    /// Neither repair engine is kept alive under NeMo WITHOUT THE DETECTOR — and,
    /// the other direction, both still are under pyannote, so this cannot pass by
    /// refusing everything.
    func testNemoKeepsNeitherOverlapEngineWithoutTheDetector() {
        for engineID in [ModelCatalog.overlapSeparation.id, ModelCatalog.overlapDicow.id] {
            XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                         engineID: engineID,
                                                         diarEngine: nemo,
                                                         detectEnabled: false),
                         "\(engineID) was kept alive for an engine that cannot produce "
                         + "overlap regions at all")
            XCTAssertEqual(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                           engineID: engineID,
                                                           diarEngine: pyannote,
                                                           detectEnabled: false),
                           engineID,
                           "\(engineID) must still load under pyannote — the NeMo "
                           + "exclusion must not have widened into every engine")
        }
    }

    /// THE OTHER HALF, and the reason `detectEnabled` is a parameter at all:
    /// switching the detector on hands back the engine the owner picked — the same
    /// engine, never a substitute.
    func testTheDetectorGivesNemoItsRepairEngineBack() {
        for engineID in [ModelCatalog.overlapSeparation.id, ModelCatalog.overlapDicow.id] {
            XCTAssertEqual(ModelLoader.wantedOverlapEngine(repairEnabled: true,
                                                           engineID: engineID,
                                                           diarEngine: nemo,
                                                           detectEnabled: true),
                           engineID)
        }
    }

    /// The detector decides WHERE repair may run, never WHETHER. A detector that
    /// quietly started a 6 GB DiCoW would be the substitution this project forbids.
    func testTheDetectorCannotSwitchRepairOnUnderNemo() {
        for engineID in [ModelCatalog.overlapSeparation.id, ModelCatalog.overlapDicow.id] {
            XCTAssertNil(ModelLoader.wantedOverlapEngine(repairEnabled: false,
                                                         engineID: engineID,
                                                         diarEngine: nemo,
                                                         detectEnabled: true))
        }
    }

    // MARK: - 3. The live-dispatch guard

    /// A live office window under NeMo is dropped, not queued: no dispatch, no temp
    /// WAV registered, and — the silent half — no buffer left to grow.
    @MainActor
    func testNemoNeverDispatchesAnOfficeLiveChunk() {
        let recorder = AudioRecorder()
        recorder.nemoDiarizationActive = true
        recorder.chunkAudio = [Float](repeating: 0.2, count: 480_000)  // 30 s at 16 kHz

        recorder.diarizeLiveChunk(windowStart: 0)

        XCTAssertTrue(recorder.chunkAudio.isEmpty,
                      "the office buffer must be dropped, not accumulated for a pass that "
                      + "reads the recording file instead (~230 MB/hour otherwise)")
        XCTAssertTrue(recorder.chunkFileByWindow.isEmpty,
                      "nothing may be handed to a sidecar that has no chunk branch")
    }

    /// The remote twin, discriminating in BOTH directions: under NeMo the buffer is
    /// dropped, while the ordinary early-out (no remote stream) deliberately
    /// RETAINS it — so a guard left out, or one that swallowed every session, fails
    /// here.
    @MainActor
    func testNemoDropsTheRemoteDiarizationBufferButOtherEnginesKeepIt() {
        let underNemo = AudioRecorder()
        underNemo.nemoDiarizationActive = true
        underNemo.remoteDiarAudio = [Float](repeating: 0.2, count: 480_000)
        underNemo.diarizeRemoteLiveChunk(windowStart: 0)
        XCTAssertTrue(underNemo.remoteDiarAudio.isEmpty)

        let underPyannote = AudioRecorder()
        underPyannote.remoteDiarAudio = [Float](repeating: 0.2, count: 480_000)
        underPyannote.diarizeRemoteLiveChunk(windowStart: 0)
        XCTAssertEqual(underPyannote.remoteDiarAudio.count, 480_000,
                       "the pre-existing early-out keeps its buffer — the NeMo guard "
                       + "must not have changed behaviour for the other engines")
    }

    /// Repair is skipped under NeMo wherever the decision is made — here, the
    /// overlay's row list. Services outlive a session, so one left over from a
    /// previous pyannote meeting would otherwise put a row on the overlay for work
    /// that cannot happen.
    @MainActor
    func testNemoListsNoOverlapRepairRowWithoutADetector() {
        let recorder = AudioRecorder()
        recorder.nemoDiarizationActive = true
        XCTAssertTrue(recorder.usesDetectedRegionsForRepair,
                      "NeMo must take its regions from the detector, not from turns")
        XCTAssertFalse(recorder.overlapRepairWillRun)
    }

    /// The skip is LOGGED under the right engine's name. A two-way `?:` silently
    /// mislabelled the third engine, and a log line naming the wrong engine sends
    /// the next debugging session to the wrong page of Settings.
    @MainActor
    func testTheSkipLogNamesTheEngineThatIsActuallyRunning() {
        let underNemo = AudioRecorder()
        underNemo.nemoDiarizationActive = true
        XCTAssertEqual(underNemo.batchEngineNameForLog, "NEMO")

        let underSpectral = AudioRecorder()
        underSpectral.spectralDiarizationActive = true
        XCTAssertEqual(underSpectral.batchEngineNameForLog, "SPECTRAL")

        let underMoss = AudioRecorder()
        underMoss.mossDiarizationActive = true
        XCTAssertEqual(underMoss.batchEngineNameForLog, "MOSS")
    }

    // MARK: - The stop pass, and the tail that does not exist

    /// The office rule, all three inputs both ways — the SAME shared pure function
    /// spectral uses, called with NeMo's flag. NEITHER `continueOnStop` NOR
    /// `finalPass` is in the signature, and both absences are deliberate: there is
    /// no tail to continue from, and the stop pass IS the labels here.
    ///
    /// THE DANGEROUS DIRECTION IS A PASS THAT DOES NOT RUN, since this engine has
    /// no live labels to fall back on: a meeting would simply end with no speakers.
    func testTheOfficePassNeedsTheEngineTheServiceAndTheRecording() {
        func runs(active: Bool = true, service: Bool = true, recording: Bool = true) -> Bool {
            AudioRecorder.runsBatchOfficePass(batchActive: active,
                                              hasService: service, hasRecording: recording, finalPass: true)
        }
        XCTAssertTrue(runs())
        XCTAssertFalse(runs(active: false), "a pyannote session must not take this branch")
        XCTAssertFalse(runs(service: false))
        XCTAssertFalse(runs(recording: false))
    }

    /// `diarization.finalPass` is SHARED with pyannote, and a NeMo session must be
    /// unaffected by whatever pyannote left in it — that value has no control under
    /// this engine, so honouring it would strand the user with an unlabelled
    /// meeting and nothing to change. Absent from the signature is what guarantees
    /// it; this asserts the guarantee rather than trusting the shape.
    func testAStoredFinalPassCannotReachTheNemoPass() {
        let d = UserDefaults.standard
        let saved = d.object(forKey: "diarization.finalPass")
        defer {
            if let saved { d.set(saved, forKey: "diarization.finalPass") }
            else { d.removeObject(forKey: "diarization.finalPass") }
        }
        for stored in [true, false] {
            d.set(stored, forKey: "diarization.finalPass")
            XCTAssertTrue(AudioRecorder.runsBatchOfficePass(batchActive: true,
                                                            hasService: true,
                                                            hasRecording: true, finalPass: true),
                          "a stored finalPass=\(stored) reached the NeMo pass")
        }
    }

    /// ⚠ RE-AIMED 2026-08-14 — `supportsTail` still matters, but in a different
    /// branch. `continueOnStop` used to be read whenever the stop pass ran, so
    /// this asserted that NeMo fell through to `.full` while pyannote got
    /// `.tail` from the same inputs. The owner then made a stop pass
    /// unconditionally a full pass (*"pas On mah diulang diarize dari awal sampai
    /// akhir"*), which is a stronger version of the same guarantee: with the stop
    /// pass ON, NeMo gets its full pass and `supportsTail` cannot affect it at
    /// all.
    ///
    /// Both directions are still asserted, and the discriminating half moved to
    /// the stop-pass-OFF branch — the only place a tail is now decided.
    func testRemoteTailIsUnavailableToNemoButUnchangedForPyannote() {
        func mode(supportsTail: Bool, finalPass: Bool) -> AudioRecorder.RemoteStopMode {
            AudioRecorder.remoteStopMode(finalPass: finalPass, continueOnStop: true,
                                         remoteStreamActive: true, hasDiarizationService: true,
                                         hasRemoteRecording: true, tailSamples: 480_000,
                                         supportsTail: supportsTail)
        }
        for supportsTail in [true, false] {
            XCTAssertEqual(mode(supportsTail: supportsTail, finalPass: true), .full,
                           "a stop pass is a FULL pass for every engine — "
                           + "supportsTail=\(supportsTail) must not change that")
        }
        XCTAssertEqual(mode(supportsTail: false, finalPass: false), .none,
                       "with the stop pass off and no tail available there is "
                       + "nothing this engine can run")
        XCTAssertEqual(mode(supportsTail: true, finalPass: false), .tail,
                       "pyannote CAN continue from its live labels, which is what "
                       + "makes this parameter the thing that changed the answer")
    }

    /// NeMo adds NO startup refusal, and that is a decision rather than an
    /// omission: it runs once after Stop, so a second stream can only be waited
    /// for — the same evidence that allows spectral + dual-stream. Voxtral's
    /// refusal is about duty cycle DURING recording and must survive.
    func testDualStreamIsAllowedWithNemoWhileVoxtralIsStillRefused() {
        for chunked in ["qwen3", "whisper", "granite", "moss"] {
            XCTAssertNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                                chunkedModelID: chunked),
                         "chunked=\(chunked): a dual-stream NeMo session was refused")
            XCTAssertNil(AudioRecorder.mossRefusalMessage(chunkedModelID: chunked,
                                                          diarizationEngine: nemo,
                                                          remoteChannel: 1),
                         "chunked=\(chunked): the MOSS-engine rule captured NeMo")
        }
        XCTAssertNotNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                               chunkedModelID: "voxtral"),
                        "Voxtral + Remote must still be refused — its duty cycle is the "
                        + "reason that rule exists and NeMo does not repeal it")
    }

    // MARK: - 4. Literal names, and its OWN interpreter

    /// The repo's `scripts/` directory, from this file's own path — the app's
    /// `PythonRuntime.scriptsDir` resolves against the bundle or the launch
    /// directory, neither of which a unit test can rely on.
    private var scriptsDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // …/Tests/MeetingTranscriberTests
            .deletingLastPathComponent()   // …/Tests
            .deletingLastPathComponent()   // …/MeetingTranscriber
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("scripts")
    }

    /// Both names are LITERALS and both differ from every other diarizer's. The
    /// failure this forbids is the `Config.mossDiarization()` one: a derived
    /// `logName` silently repointing one process's stderr into another's log while
    /// it still runs the other script — a drifted script/log pair, invisible
    /// because both processes keep working.
    func testNemoConfigNamesItsOwnSidecarAndLogAsLiterals() {
        XCTAssertEqual(NemoService.Config.scriptName, "nemo/nemo-service.py")
        XCTAssertEqual(NemoService.Config.logName, "nemo")

        let others: [(engine: String, script: String, log: String)] = [
            ("pyannote", PyannoteService.scriptName, PyannoteService.logName),
            ("spectral", SpectralService.Config.scriptName, SpectralService.Config.logName),
            ("wespeaker", WeSpeakerService.Config.scriptName, WeSpeakerService.Config.logName),
            ("moss-diar", ChunkedASRService.Config.mossDiarization().scriptName,
                          ChunkedASRService.Config.mossDiarization().logName),
        ]
        for other in others {
            XCTAssertNotEqual(NemoService.Config.scriptName, other.script,
                              "nemo must not run \(other.engine)'s script")
            XCTAssertNotEqual(NemoService.Config.logName, other.log,
                              "nemo must not write \(other.engine)'s log — two writers "
                              + "on one file is the 2026-07-15 mistake")
        }

        // The named script really exists. This is the half that catches a rename
        // applied to disk but not to Swift; for a diarizer it surfaces loudly as
        // `ServiceError.scriptMissing` at session start, in front of the user.
        for script in [NemoService.Config.scriptName] + others.map(\.script) {
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: scriptsDir.appendingPathComponent(script).path),
                          "\(script) does not exist")
        }
    }

    /// The log basename obeys the exceptionless rule
    /// `scripts/<name>/<name>-service.py` → `logs/<name>.log`, derived here rather
    /// than restated, so a rename of either half fails. This is also the Swift half
    /// of what makes `layout/log-name-matches-folder` pass now that `scripts/nemo/`
    /// exists.
    func testTheLogBasenameIsDerivedFromTheServiceFolder() {
        let folder = NemoService.Config.scriptName.split(separator: "/").first.map(String.init)
        XCTAssertEqual(folder, NemoService.Config.logName)
        XCTAssertEqual(NemoService.Config.scriptName, "\(folder ?? "")/\(folder ?? "")-service.py")
    }

    /// The app's own decision log is a DIFFERENT file from the sidecar's stderr,
    /// and for this engine that matters more than for any other: NeMo logs to
    /// stdout, so its sidecar redirects every non-protocol writer into
    /// `logs/nemo.log`, which is therefore busy. Two writers on one file is the
    /// 2026-07-15 mistake.
    func testTheSwiftDecisionLogIsNotTheSidecarLog() {
        XCTAssertNotEqual("nemo-diarization", NemoService.Config.logName)
    }

    /// It runs in its OWN interpreter, and the name is a literal on the Swift side
    /// because `PythonRuntime.command(forScript:venvName:)` returns nil for a
    /// missing one — deliberately, with NO fallback to the main `.venv`, since
    /// running NeMo there fails deep inside the model with a far worse error.
    func testNemoDeclaresItsOwnVenvAndSaysSoWhenItIsMissing() {
        XCTAssertEqual(NemoService.Config.venvName, ".venv-nemo")
        let message = NemoService.ServiceError.venvMissing.errorDescription ?? ""
        XCTAssertTrue(message.contains(".venv-nemo"), message)
        XCTAssertTrue(message.contains("download-best-models.sh"),
                      "a setup error must name the thing that fixes it: \(message)")
    }

    /// The install check must look for BOTH local checkpoints. NeMo's are two flat
    /// `.nemo` archives, not an HF-hub cache, so the generic `models--…` path would
    /// always report "not downloaded" — and a check that passes with one file
    /// missing fails deep inside the pipeline at Stop instead, the worst moment.
    func testTheInstallCheckLooksForBothLocalCheckpoints() {
        let dir = ModelCatalog.modelsDir.appendingPathComponent("nemo")
        let both = ["titanet_large.nemo", "vad_multilingual_marblenet.nemo"].allSatisfy {
            FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
        XCTAssertEqual(ModelCatalog.isInstalled(ModelCatalog.nemoDiarization), both,
                       "the install check disagrees with the files on disk at \(dir.path)")
        // And it is NOT answering through the HF-hub branch, which would look for
        // models/hub/models--nemo and always say no.
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: ModelCatalog.modelsDir.appendingPathComponent("hub/models--nemo").path),
            "if this ever exists, the special case above is no longer what is being tested")
    }
}
