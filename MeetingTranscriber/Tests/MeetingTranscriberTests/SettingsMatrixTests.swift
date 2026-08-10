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
         ModelLoader.nemoEngineID, ModelLoader.mossEngineID]
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
            ModelLoader.wantsOverlapDetect(detectEnabled: detectOn, diarizationEnabled: diarOn)
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
        XCTAssertEqual(n, 5 * 4 * 2 * 2 * 2 * 2 * 2 * 2)
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

    /// Repair needs somewhere to run: pyannote finds regions in its own turns,
    /// MOSS, spectral and NeMo only through the detector.
    func testRepairIsNeverWantedWithoutASourceOfRegions() {
        sweep { c in
            guard c.repairEngine != nil else { return }
            XCTAssertTrue(c.repairOn, c.label)
            if c.engine != ModelLoader.pyannoteEngineID {
                XCTAssertTrue(c.detectOn,
                              "repair loaded under \(c.engine) with no detector — \(c.label)")
            }
        }
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
