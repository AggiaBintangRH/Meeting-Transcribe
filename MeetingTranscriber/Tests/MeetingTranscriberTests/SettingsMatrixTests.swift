import XCTest
@testable import MeetingTranscriber

/// FULL-MATRIX AUDIT of the settings surface (2026-08-06, owner-requested).
///
/// Every other test here pins ONE rule. This file asks a different question: with
/// all the rules pointed at the same configuration, do they AGREE? That is where
/// this project's expensive bugs have lived — a load rule and a teardown rule
/// computed separately (three leaks), a UI rule and a loader rule that disagreed
/// (the MOSS card), a setting read in two places that could disagree about one
/// meeting (the three locked diarization keys).
///
/// The sweep is exhaustive over the settings that interact: 5 chunked models × 4
/// diarization engines × 2 diarization on/off × 2 chunked on/off × 2 detector ×
/// 2 repair × 2 aligner × 2 realtime = 2560 configurations (1920 before NeMo
/// became the fourth engine, 2026-08-07). Every assertion below runs against every
/// one of them.
///
/// It is deliberately built from the PURE static rules, not from a live loader:
/// those functions are what `buildSteps` and the teardown both read, so agreement
/// here is agreement in the app.
final class SettingsMatrixTests: XCTestCase {

    private let chunkedIDs = ["qwen3", "whisper", "granite", "voxtral", "moss"]
    private var engines: [String] {
        [ModelLoader.pyannoteEngineID, ModelLoader.spectralEngineID,
         ModelLoader.nemoEngineID, ModelLoader.diarizenEngineID,
         ModelLoader.mossEngineID]
    }

    /// One point of the matrix, and everything the rules say about it.
    private struct Config {
        let chunkedID: String, engine: String
        let diarOn: Bool, chunkedOn: Bool, detectOn: Bool
        let repairOn: Bool, alignOn: Bool, realtimeOn: Bool

        var stack: ModelLoader.DiarizationStack? {
            ModelLoader.wantedDiarizationStack(diarizationEnabled: diarOn, engine: engine,
                                               chunkedID: chunkedID, chunkedEnabled: chunkedOn)
        }
        var wantsChunked: Bool { ModelLoader.wantsChunked(chunkedEnabled: chunkedOn) }
        var wantsAligner: Bool {
            ModelLoader.wantsAligner(alignEnabled: alignOn, chunkedID: chunkedID,
                                     chunkedEnabled: chunkedOn)
        }
        var wantsDetect: Bool {
            ModelLoader.wantsOverlapDetect(detectEnabled: detectOn, diarizationEnabled: diarOn,
                                           diarEngine: engine)
        }
        var repairEngine: String? {
            ModelLoader.wantedOverlapEngine(repairEnabled: repairOn,
                                            engineID: ModelCatalog.overlapSeparation.id,
                                            diarEngine: engine, detectEnabled: detectOn)
        }
        var wantsRealtime: Bool { ModelLoader.wantsRealtime(realtimeEnabled: realtimeOn) }
        /// Whether Settings can even express this pairing — the biconditional.
        var reachableFromUI: Bool {
            ModelLoader.diarizationEngineIsSelectable(engine, chunkedID: chunkedID)
        }
        var label: String {
            "chunked=\(chunkedID) engine=\(engine) diar=\(diarOn) chunkedOn=\(chunkedOn) "
            + "detect=\(detectOn) repair=\(repairOn) align=\(alignOn) rt=\(realtimeOn)"
        }
    }

    private func sweep(_ body: (Config) -> Void) {
        for c in chunkedIDs { for e in engines {
            for diar in [true, false] { for chunkedOn in [true, false] {
                for detect in [true, false] { for repair in [true, false] {
                    for align in [true, false] { for rt in [true, false] {
                        body(Config(chunkedID: c, engine: e, diarOn: diar, chunkedOn: chunkedOn,
                                    detectOn: detect, repairOn: repair, alignOn: align,
                                    realtimeOn: rt))
                    }}}}}}}}
    }

    func testTheMatrixIsTheSizeItClaims() {
        var n = 0
        sweep { _ in n += 1 }
        // 5 chunked models x FIVE engines (DiariZen joined 2026-08-10) x six flags.
        XCTAssertEqual(n, 5 * 5 * 2 * 2 * 2 * 2 * 2 * 2)
    }

    // MARK: - 1. Nothing is loaded that has nothing to do

    /// Diarization off must keep NO diarization stack, whatever else is set. This
    /// is the 2026-07-31 leak — the load step asked the master switch and the
    /// teardown did not, so diarization kept running after it was switched off.
    func testDiarizationOffLoadsNoStackAnywhereInTheMatrix() {
        sweep { c in
            guard !c.diarOn else { return }
            XCTAssertNil(c.stack, c.label)
            XCTAssertFalse(c.wantsDetect, "the detector marks rows; there are none — \(c.label)")
        }
    }

    /// The aligner splits chunked segments. No chunked pass, or a model that
    /// attributes its own text, means there is nothing to split.
    func testTheAlignerIsNeverWantedWithoutSomethingToAlign() {
        sweep { c in
            guard c.wantsAligner else { return }
            XCTAssertTrue(c.wantsChunked, "aligner wanted with no chunked pass — \(c.label)")
            XCTAssertNotEqual(c.chunkedID, "moss", "aligner wanted for MOSS — \(c.label)")
            XCTAssertTrue(c.alignOn)
        }
    }

    /// Repair needs somewhere to run, and the engines split TWO ways — not into
    /// "pyannote" and "the rest", which is how this read until 2026-08-10.
    ///
    /// Two engines mark overlap in their OWN turns, so repair may load with the
    /// detector off:
    ///   * **pyannote** — its segmentation model has always done this.
    ///   * **DiariZen** — measured, not assumed: its powerset head predicts
    ///     two-speaker frames directly (11 classes = 1 silence + 4 singles +
    ///     6 PAIRS), and its turns intersect on real audio (11 pairs / 13.30 s on
    ///     `recordings/Overlap123.wav`). It was grouped with the batch engines when
    ///     it landed, purely by resemblance, and that cost it repair unless a
    ///     redundant 32 MB detector was switched on.
    ///
    /// The other three genuinely cannot, and the reason is structural rather than
    /// incidental: MOSS's segments tile exactly, spectral's Viterbi and NeMo's
    /// NME-SC each assign exactly one label per instant. `overlapRegions()` is
    /// empty under them however the audio sounds.
    ///
    /// Written as an explicit SET rather than `!= pyannote` so a sixth engine
    /// cannot inherit either answer by default — it has to be classified.
    func testRepairIsNeverWantedWithoutASourceOfRegions() {
        let marksItsOwnOverlap: Set = [ModelLoader.pyannoteEngineID,
                                       ModelLoader.diarizenEngineID]
        sweep { c in
            guard c.repairEngine != nil else { return }
            XCTAssertTrue(c.repairOn, c.label)
            if !marksItsOwnOverlap.contains(c.engine) {
                XCTAssertTrue(c.detectOn,
                              "repair loaded under \(c.engine) with no detector — \(c.label)")
            }
        }
    }

    /// The other direction, and it is the one that makes the split above provable:
    /// under an engine that marks its own overlap, the detector must NOT be able to
    /// decide whether repair loads. Without this, moving an engine into
    /// `marksItsOwnOverlap` above would weaken the test rather than restate it —
    /// the `if` would simply stop running for that engine and nothing would check
    /// what happens instead.
    func testEnginesThatMarkTheirOwnOverlapDoNotNeedTheDetector() {
        for engine in [ModelLoader.pyannoteEngineID, ModelLoader.diarizenEngineID] {
            let withDetector = ModelLoader.wantedOverlapEngine(
                repairEnabled: true, engineID: ModelCatalog.overlapSeparation.id,
                diarEngine: engine, detectEnabled: true)
            let without = ModelLoader.wantedOverlapEngine(
                repairEnabled: true, engineID: ModelCatalog.overlapSeparation.id,
                diarEngine: engine, detectEnabled: false)
            XCTAssertNotNil(without,
                            "\(engine) finds its own overlap regions, so the detector "
                            + "must not gate repair")
            XCTAssertEqual(withDetector, without,
                           "the detector must not change WHETHER repair loads under "
                           + "\(engine) — it is a second opinion, never a switch")
        }
    }

    /// WHICH ENGINES ACT ON A PINNED SPEAKER COUNT, asserted in both directions.
    ///
    /// The main-window SPK control writes one key that every engine's pass reads,
    /// so the only thing stopping it becoming a lie is this split. Measured
    /// 2026-08-10, auto vs pinned on the same two files: spectral returned **13
    /// speakers on a 3-person clip** and 3 when pinned, while pyannote and NeMo
    /// were unchanged because both already counted correctly — all three honour
    /// the number, which is what puts them on this side.
    ///
    /// DiariZen is the half that matters: `diarizen-service.py` never reads
    /// `num_speakers`, so if it ever appeared here the chip would light up and
    /// promise something no sidecar does.
    func testOnlyTheEnginesThatReadTheCountAreSaidToHonourIt() {
        for engine in [ModelLoader.pyannoteEngineID, ModelLoader.spectralEngineID,
                       ModelLoader.nemoEngineID] {
            XCTAssertTrue(ModelLoader.honoursSpeakerCount(diarEngine: engine),
                          "\(engine)'s sidecar reads num_speakers and must be honoured")
        }
        for engine in [ModelLoader.diarizenEngineID, ModelLoader.mossEngineID] {
            XCTAssertFalse(ModelLoader.honoursSpeakerCount(diarEngine: engine),
                           "\(engine) does not read num_speakers — saying it does would "
                           + "make the SPK control promise what no pass delivers")
        }
    }

    /// PYANNOTE SENDS THE COUNT ONLY ON ITS FULL STOP PASS, and the SPK chip must
    /// say so rather than lighting up for a session that discards the number.
    ///
    /// Found by the 2026-08-10 audit tracing UI → dispatch → wire → sidecar: the
    /// tail pass is a `chunk` job and `diarizeChunk` carries no `num_speakers`.
    /// That omission is CORRECT — a few seconds of tail need not contain everyone,
    /// so forcing the meeting's count onto it would make it invent people — which
    /// is exactly why the honest fix is in the chip, not in the wire.
    ///
    /// The dangerous leg is `finalPass = true, continueOnStop = true`: that toggle
    /// is visible only while the stop pass is OFF, so a stored `true` can outlive
    /// the control that set it — the value-outliving-its-control shape.
    func testPyannoteOnlyGetsTheCountOnAFullStopPass() {
        func reaches(_ finalPass: Bool, _ continueOnStop: Bool) -> Bool {
            ModelLoader.speakerCountReachesEngine(diarEngine: ModelLoader.pyannoteEngineID,
                                                  diarizationEnabled: true,
                                                  finalPass: finalPass,
                                                  continueOnStop: continueOnStop)
        }
        XCTAssertTrue(reaches(true, false), "the full stop pass is the one that sends it")
        XCTAssertFalse(reaches(true, true), "a TAIL pass is a chunk job and carries no count")
        XCTAssertFalse(reaches(false, false), "no stop pass at all means no count")
        XCTAssertFalse(reaches(false, true))

        // The BATCH engines ignore both keys by design — their stop pass IS the
        // labels — so neither setting may take the count away from them. Without
        // this half, folding the pyannote rule in could silently disable spectral,
        // which is the ONE engine the count measurably helps.
        for engine in [ModelLoader.spectralEngineID, ModelLoader.nemoEngineID] {
            for finalPass in [true, false] {
                for tail in [true, false] {
                    XCTAssertTrue(
                        ModelLoader.speakerCountReachesEngine(diarEngine: engine,
                                                              diarizationEnabled: true,
                                                              finalPass: finalPass,
                                                              continueOnStop: tail),
                        "\(engine) must keep the count whatever finalPass=\(finalPass) "
                        + "continueOnStop=\(tail) say — it reads neither")
                }
            }
        }

        // And diarization off means nothing is counted, for anyone.
        XCTAssertFalse(ModelLoader.speakerCountReachesEngine(
            diarEngine: ModelLoader.spectralEngineID, diarizationEnabled: false,
            finalPass: true, continueOnStop: false))
    }

    /// THE REMOTE STREAM IS NEVER TOLD THE ROOM'S HEADCOUNT.
    ///
    /// `diarNumSpeakers` describes the room — the chip sits beside the room's RMS
    /// meter and its own doc says "how many people are in the room". Nothing asks
    /// how many people are on the far end, and the two streams are separate
    /// identity spaces precisely because they hold different people.
    ///
    /// All four remote passes sent it anyway until the 2026-08-11 audit, and a
    /// pinned count is an EXACT constraint on three of the four engines (pyannote
    /// and spectral pass `num_speakers=`, NeMo turns it into
    /// `oracle_num_speakers`), so a 5-person room with one caller on the line split
    /// that caller into five remote profiles.
    ///
    /// WHICH DISPATCH SITE USES WHICH CONSTANT is pinned in
    /// `layout/remote-passes-never-send-the-room-count`, both directions — a Swift
    /// test cannot see a call site. This pins the INTENT: an edit that decides to
    /// forward the room's count after all has to come here and argue with the
    /// reason, rather than changing one line in four files.
    func testTheRemotePassesAlwaysAskForAutomaticCounting() {
        XCTAssertEqual(AudioRecorder.remoteNumSpeakers, 0,
                       "0 = auto. Any other value would be this app asserting a "
                       + "headcount for people it has never been told about.")
    }

    /// The control and the passes are ONE value, not two readers that agree.
    /// `AudioRecorder.diarNumSpeakers` must read the same key the chip writes —
    /// eight dispatch sites go through it, and a stale constant there would send
    /// `auto` to every engine while the UI showed a number.
    @MainActor
    func testTheSpeakerCountControlAndThePassesShareOneValue() {
        let d = UserDefaults.standard
        let key = "diarization.numSpeakers"
        let saved = d.object(forKey: key)
        defer {
            if let saved { d.set(saved, forKey: key) } else { d.removeObject(forKey: key) }
        }

        d.removeObject(forKey: key)
        XCTAssertEqual(AudioRecorder.diarNumSpeakers, 0, "absent must mean auto")

        d.set(7, forKey: key)
        XCTAssertEqual(AudioRecorder.diarNumSpeakers, 7)

        // A negative can only arrive from a corrupt domain, and it must not reach a
        // sidecar: `num_speakers` is compared `> 0` there, so a negative would read
        // as auto anyway — clamped here so the value the app believes and the value
        // the sidecar acts on are the same number.
        d.set(-3, forKey: key)
        XCTAssertEqual(AudioRecorder.diarNumSpeakers, 0)
    }

    // MARK: - 2. Nothing that IS needed is left out

    /// Every stack that exists must carry identity: `SpeakerTurn` has no
    /// representation for an unidentified turn, so a stack without the embedder is
    /// a state no display path can render.
    func testEveryLoadedStackUsesSpeakerIdentity() {
        sweep { c in
            guard let stack = c.stack else { return }
            XCTAssertTrue(stack.usesSpeakerIdentity, c.label)
        }
    }

    /// THE COUPLING THAT WOULD FAIL SILENTLY. Diarization on must ALWAYS produce a
    /// stack — a meeting that asked for speakers and loads nothing to find them
    /// ends with no speakers and no error anywhere.
    func testDiarizationOnAlwaysProducesAStack() {
        sweep { c in
            guard c.diarOn else { return }
            XCTAssertNotNil(c.stack, "diarization on but nothing loaded — \(c.label)")
        }
    }

    /// And under the MOSS engine, the stack must name a process that this session
    /// really starts: `.mossOwnASR` borrows the chunked sidecar, so it is only
    /// legal while there IS one.
    func testMossOwnASRIsOnlyChosenWhenTheChunkedProcessExists() {
        sweep { c in
            guard c.stack == .mossOwnASR else { return }
            XCTAssertTrue(c.wantsChunked, "MOSS borrows a process that is not loaded — \(c.label)")
            XCTAssertEqual(c.chunkedID, "moss", c.label)
        }
    }

    // MARK: - 3. The UI and the loader agree

    /// Every pairing Settings can express must be one the loader supports, and the
    /// engine must be one the rule still offers. This is the check that would have
    /// caught an engine left STORED while its card was hidden.
    func testEveryUIReachablePairingLoadsSomethingCoherent() {
        sweep { c in
            guard c.reachableFromUI, c.diarOn, c.chunkedOn else { return }
            XCTAssertNotNil(c.stack, c.label)
            if c.engine == ModelLoader.mossEngineID {
                XCTAssertEqual(c.chunkedID, "moss",
                               "UI offered MOSS diarization beside another ASR — \(c.label)")
                XCTAssertEqual(c.stack, .mossOwnASR, c.label)
            }
        }
    }

    /// The fallback the UI applies when a stored engine stops being offered must
    /// itself be offered — otherwise correcting the setting strands it again.
    func testTheFallbackIsAlwaysSelectable() {
        for chunkedID in chunkedIDs {
            let fallback = ModelLoader.fallbackDiarizationEngine(chunkedID: chunkedID)
            XCTAssertTrue(ModelLoader.diarizationEngineIsSelectable(fallback, chunkedID: chunkedID),
                          "chunked=\(chunkedID) falls back to an engine it does not offer")
        }
    }

    // MARK: - 4. Startup refusals are complete and reachable

    /// A refusal must name a control the user can actually reach. Voxtral + Remote
    /// and Voxtral + MOSS both name the Chunked tab; the chunked-off one names
    /// Chunked and Microphone. All three are live tabs.
    func testEveryRefusalNamesAReachableTab() {
        let refusals: [String?] = [
            AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1, chunkedModelID: "voxtral"),
            AudioRecorder.mossRefusalMessage(chunkedModelID: "voxtral",
                                             diarizationEngine: ModelLoader.mossEngineID,
                                             remoteChannel: nil),
            AudioRecorder.chunkedOffRefusalMessage(remoteChannel: 1, chunkedEnabled: false),
        ]
        for r in refusals {
            let m = try? XCTUnwrap(r)
            XCTAssertNotNil(m)
            XCTAssertTrue(m?.contains("Settings →") ?? false || m?.contains("Models →") ?? false,
                          "a refusal that names no control: \(m ?? "nil")")
        }
    }

    /// No UI-reachable single-stream configuration may be refused. A refusal the
    /// user cannot avoid is a dead end, and the only refusals left are about
    /// Voxtral's duty cycle and a Remote channel — never about a plain meeting.
    func testNoPlainSingleStreamConfigurationIsRefused() {
        sweep { c in
            guard c.reachableFromUI else { return }
            XCTAssertNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: nil,
                                                                chunkedModelID: c.chunkedID),
                         c.label)
            XCTAssertNil(AudioRecorder.chunkedOffRefusalMessage(remoteChannel: nil,
                                                                chunkedEnabled: c.chunkedOn),
                         c.label)
            XCTAssertNil(AudioRecorder.mossRefusalMessage(chunkedModelID: c.chunkedID,
                                                          diarizationEngine: c.engine,
                                                          remoteChannel: nil),
                         "Voxtral+MOSS is unreachable from the UI now — \(c.label)")
        }
    }

    // MARK: - 5. Every model a tab can select is one the loader knows

    /// A card the user can pick must resolve to a real catalog entry. `chunkedModel(id:)`
    /// and friends fall back rather than crash, so a typo'd id would silently load
    /// the DEFAULT model and the transcript's provenance would be a lie.
    func testEverySelectableCardResolvesToItself() {
        for m in ModelCatalog.chunked {
            XCTAssertEqual(ModelCatalog.chunkedModel(id: m.id).id, m.id, m.name)
        }
        for m in ModelCatalog.overlapEngines {
            XCTAssertEqual(ModelCatalog.overlapEngine(id: m.id).id, m.id, m.name)
        }
        for m in ModelCatalog.overlapDetectors {
            XCTAssertEqual(ModelCatalog.overlapDetector(id: m.id).id, m.id, m.name)
        }
        for m in ModelCatalog.diarizationEngines {
            let value = ModelCatalog.diarizationEngineValue(m)
            XCTAssertTrue(self.engines.contains(value),
                          "\(m.name) maps to engine value '\(value)', which no rule matches")
        }
    }

    // MARK: - 6. Every remaining setting really reaches its sidecar
    //
    // The matrix above covers the rules that decide WHAT LOADS. These cover the
    // other half — the values that ride into a running process. It is the same
    // failure the Granite and Voxtral language pickers shipped with: a control
    // whose value never arrives, invisible because nothing errors.

    /// Writes each key, checks the effect, restores the previous value.
    private func withKey(_ key: String, _ value: Any, _ body: () -> Void) {
        let d = UserDefaults.standard
        let saved = d.object(forKey: key)
        d.set(value, forKey: key)
        body()
        if let saved { d.set(saved, forKey: key) } else { d.removeObject(forKey: key) }
    }

    /// `Config` is Equatable and that is what makes `ModelLoader` replace a running
    /// sidecar. A setting that does NOT change it leaves the previous process
    /// answering for the rest of the meeting while Settings claims otherwise.
    func testEveryRealtimeSettingChangesTheRealtimeConfig() {
        let before = NemotronASRService.Config.fromSettings()
        for (key, value) in [("realtime.language", "id" as Any),
                             ("realtime.chunkMs", 320 as Any)] {
            withKey(key, value) {
                XCTAssertNotEqual(before, NemotronASRService.Config.fromSettings(),
                                  "\(key) never reaches the realtime sidecar")
            }
        }
        XCTAssertEqual(before, NemotronASRService.Config.fromSettings(),
                       "a key was not restored")
    }

    /// The VAD is built fresh per session rather than kept as a process, so there
    /// is no Equatable Config to compare — the fields are checked directly.
    func testEveryVADSettingReachesTheVAD() {
        withKey("vad.threshold", 0.9) {
            XCTAssertEqual(VoiceActivityDetector.Config.fromSettings().threshold, 0.9,
                           accuracy: 0.0001)
        }
        withKey("vad.minSilenceMs", 999.0) {
            XCTAssertEqual(VoiceActivityDetector.Config.fromSettings().minSilenceMs, 999.0,
                           accuracy: 0.0001)
        }
        withKey("vad.minSpeechMs", 888.0) {
            XCTAssertEqual(VoiceActivityDetector.Config.fromSettings().minSpeechMs, 888.0,
                           accuracy: 0.0001)
        }
    }

    /// The chunked language and the two prompts are the settings most likely to be
    /// dropped silently at the Swift boundary, because each is optional on the
    /// wire — an omitted flag looks exactly like a default.
    func testTheChunkedLanguageAndPromptsReachTheirSidecar() {
        withKey("chunked.model", "whisper") {
            withKey("chunked.language", "fr") {
                XCTAssertEqual(ChunkedASRService.Config.fromSettings().language, "fr",
                               "the picked language never reaches the chunked sidecar")
            }
            withKey("whisper.initialPrompt", "ATND1061 pyannote") {
                let args = ChunkedASRService.Config.fromSettings().whisper?.processArguments ?? []
                XCTAssertTrue(args.contains("--initial-prompt"), "\(args)")
            }
        }
        withKey("chunked.model", "qwen3") {
            withKey("qwen3.systemPrompt", "PREP framework") {
                let args = ChunkedASRService.Config.fromSettings().qwen3?.processArguments ?? []
                XCTAssertTrue(args.contains("--system-prompt"), "\(args)")
            }
        }
    }

    /// The pyannote clustering threshold is sent per job rather than as a launch
    /// flag, so no `Config` guards it — this is the only check that it is read at
    /// all, and it is the one setting left that changes a transcript's speakers.
    func testTheClusterThresholdIsReadFromSettings() {
        withKey("diarization.clusterThreshold", 0.75) {
            XCTAssertEqual(PyannoteService.clusterThreshold(), 0.75, accuracy: 0.0001)
        }
        withKey("diarization.clusterThreshold", 0.45) {
            XCTAssertEqual(PyannoteService.clusterThreshold(), 0.45, accuracy: 0.0001)
        }
    }

    /// Each chunked model must name its OWN sidecar and log. Routing became a
    /// protocol requirement precisely so a new model cannot fall through to a
    /// deleted file, and two models sharing one log is the 2026-07-15 mistake.
    ///
    /// Driven through the real factory — the same switch the app uses — so a model
    /// added to the catalog but forgotten there fails here rather than silently
    /// resolving to Qwen3 via the `default:` arm.
    func testEveryChunkedModelNamesItsOwnScriptAndLog() {
        let d = UserDefaults.standard
        let saved = d.string(forKey: "chunked.model")
        defer {
            if let saved { d.set(saved, forKey: "chunked.model") }
            else { d.removeObject(forKey: "chunked.model") }
        }
        var scripts = Set<String>(), logs = Set<String>()
        for m in ModelCatalog.chunked {
            d.set(m.id, forKey: "chunked.model")
            let model = ChunkedASRModelFactory.fromSettings()
            XCTAssertEqual(model.info.id, m.id,
                           "the factory resolved \(m.id) to \(model.info.id) — a catalog entry "
                           + "with no case falls through to the default model")
            XCTAssertTrue(scripts.insert(model.scriptName).inserted,
                          "\(m.name) shares a sidecar script with another model")
            XCTAssertTrue(logs.insert(model.logName).inserted,
                          "\(m.name) shares a log with another model — the 2026-07-15 mistake")
        }
    }
}
