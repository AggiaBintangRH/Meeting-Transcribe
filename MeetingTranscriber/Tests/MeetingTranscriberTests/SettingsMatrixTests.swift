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

    /// ⚠ BOTH DERIVED FROM THE CATALOG, not listed. They were hand-written until
    /// 2026-09-02, when a seventh diarization engine and a sixth chunked model
    /// were added and the 1920-configuration sweep went on sweeping the old six
    /// and five — saying nothing at all about the new ones while staying green.
    /// A list beside a catalog only ever drifts one way, and this file exists
    /// precisely to ask whether the rules AGREE about a configuration; a
    /// configuration it never visits is one it cannot answer for.
    private var chunkedIDs: [String] { ModelCatalog.chunked.map(\.id) }
    private var engines: [String] {
        ModelCatalog.diarizationEngines.map(ModelCatalog.diarizationEngineValue)
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
        // Chunked models x engines x six boolean flags, both counts DERIVED from
        // the catalog like the lists they sweep. A literal here would have to be
        // edited every time a model or engine lands, and forgetting is silent —
        // the assertion still passes for the OLD size while the sweep visits more.
        XCTAssertEqual(n, chunkedIDs.count * engines.count * 2 * 2 * 2 * 2 * 2 * 2)
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

    /// The Aligner TAB must name the model the loader will actually run.
    ///
    /// This is the load-bearing half of the 2026-08-12 change: the tab showed a
    /// 1.2 GB Qwen3 card under MOSS, for a process `wantsAligner` has refused to
    /// start since the aligner became its own sidecar. Nothing failed, nothing
    /// was empty — the UI simply described a session that was not happening.
    ///
    /// Swept rather than spot-checked, so the two rules cannot drift apart for
    /// some future chunked model: wherever the loader refuses the aligner BECAUSE
    /// OF THE MODEL, the tab must offer the built-in card, and wherever it agrees
    /// to load it the tab must offer Qwen3.
    func testTheAlignerTabNamesWhicheverModelTheLoaderWillRun() {
        for chunkedID in chunkedIDs {
            let offered = ModelCatalog.wordAligners(forChunkedModel: chunkedID)
            XCTAssertEqual(offered.count, 1, "one aligner per model — \(chunkedID)")
            // Asked with everything else ON, so the only reason left to refuse is
            // the chunked model itself — the same question the tab asks.
            let loaderWillRun = ModelLoader.wantsAligner(alignEnabled: true,
                                                         chunkedID: chunkedID,
                                                         chunkedEnabled: true)
            XCTAssertEqual(offered[0].id,
                           loaderWillRun ? ModelCatalog.wordAligner.id
                                         : ModelCatalog.wordAlignerMoss.id,
                           "the tab and the loader disagree about \(chunkedID) — a card "
                           + "for a model that never loads is the UI describing a "
                           + "session that is not happening")
        }
    }

    /// "The aligner does not load" has TWO causes and the rail must not treat
    /// them alike (owner, 2026-08-12: *"ini masih gak di atas aligner tabnya"*).
    ///
    /// Under MOSS the attribution HAPPENS — MOSS emits a timed segment per
    /// speaker — so filing the tab under "NOT USED BY YOUR MODELS" tells the user
    /// the job is not being done when it is. With the chunked pass off, nothing
    /// happens at all and the heading is correct. This is the same call the
    /// DiariZen detector got on 2026-08-10, after the opposite was tried and
    /// reverted the same day.
    func testMossIsBuiltInWhileNoChunkedPassIsGenuinelyNotUsed() {
        // MOSS: the loader still refuses the aligner…
        XCTAssertFalse(ModelLoader.wantsAligner(alignEnabled: true, chunkedID: "moss",
                                                chunkedEnabled: true))
        // …but the work is happening, so the rail must NOT dim the tab.
        XCTAssertTrue(ModelLoader.alignmentIsBuiltIn(chunkedID: "moss",
                                                     chunkedEnabled: true))

        // No chunked pass: nothing happens, whatever the model is. BOTH cases,
        // because a rule reading only the model id would call this one built-in
        // and promise attribution that never runs.
        for id in chunkedIDs {
            XCTAssertFalse(ModelLoader.alignmentIsBuiltIn(chunkedID: id,
                                                          chunkedEnabled: false),
                           "with no chunked pass nothing is attributed — \(id)")
        }

        // And every OTHER model is neither: the aligner really loads there, so a
        // "built in" badge would deny a 1.2 GB model that is running.
        for id in chunkedIDs where id != "moss" {
            XCTAssertFalse(ModelLoader.alignmentIsBuiltIn(chunkedID: id,
                                                          chunkedEnabled: true), id)
            XCTAssertTrue(ModelLoader.wantsAligner(alignEnabled: true, chunkedID: id,
                                                   chunkedEnabled: true), id)
        }
    }

    /// The two rules must PARTITION the reasons, with no configuration falling in
    /// both. A tab cannot be simultaneously "not used" and "built in", and this is
    /// the sweep that would catch a future model landing in both.
    func testNoConfigurationIsBothBuiltInAndUnwanted() {
        for id in chunkedIDs {
            for chunkedOn in [true, false] {
                let wanted = ModelLoader.wantsAligner(alignEnabled: true, chunkedID: id,
                                                      chunkedEnabled: chunkedOn)
                let builtIn = ModelLoader.alignmentIsBuiltIn(chunkedID: id,
                                                             chunkedEnabled: chunkedOn)
                XCTAssertFalse(wanted && builtIn,
                               "\(id)/chunked=\(chunkedOn): the aligner cannot both "
                               + "load and be redundant")
            }
        }
    }

    /// A SEVENTH engine must break something.
    ///
    /// The MOSS notice's engine list went stale twice — it missed DiariZen until
    /// 2026-08-10 and CAM++ until the 2026-08-13 audit — each time telling the
    /// user they had fewer choices than they did, while the card list beside it
    /// read the catalog and was right both times. The list is derived now, and
    /// this is what stops the derivation quietly losing an engine instead:
    /// `diarizationEngineShortName` returns nil for anything it does not know,
    /// so a new engine drops out of the sentence silently unless something
    /// asserts otherwise. This does.
    func testEveryDiarizationEngineHasAShortName() {
        for engine in ModelCatalog.diarizationEngines {
            let value = ModelCatalog.diarizationEngineValue(engine)
            XCTAssertNotNil(ModelCatalog.diarizationEngineShortName(value),
                            "\(engine.name) has no short name, so it would vanish "
                            + "from every sentence that lists the engines — the "
                            + "exact defect this replaced")
        }
    }

    /// …and the sentence really names them all, in the form a reader expects.
    func testTheMossNoticeListsEveryNonMossEngine() {
        let sentence = ModelCatalog.diarizationEnginesWithoutMoss
        let expected = ModelCatalog.diarizationEngines
            .map { ModelCatalog.diarizationEngineValue($0) }
            .filter { $0 != ModelLoader.mossEngineID }

        XCTAssertEqual(expected.count, ModelCatalog.diarizationEngines.count - 1,
                       "every engine but MOSS — if this "
                       + "changes, the count below is what proves the sentence kept up")
        for value in expected {
            let name = ModelCatalog.diarizationEngineShortName(value) ?? "?"
            XCTAssertTrue(sentence.contains(name),
                          "the MOSS notice omits \(name): \"\(sentence)\"")
        }
        XCTAssertFalse(sentence.contains("MOSS"),
                       "…and must not offer MOSS as the way out of MOSS")
        XCTAssertTrue(sentence.contains(" or "),
                      "read as prose, not as a comma-separated dump")
    }

    /// The pure overlap rule and the recorder's own property must agree, engine
    /// by engine.
    ///
    /// `ModelLoader.marksItsOwnOverlap` exists because a Settings tab cannot see
    /// the recorder's per-session `*Active` flags — and the moment there are two
    /// expressions of one fact, they drift. That is not hypothetical here: the
    /// Detect overlap tab spent from the NeMo release until the 2026-08-13 audit
    /// telling users only "MOSS and spectral" were affected while the recorder
    /// had four engines in that set.
    @MainActor
    func testTheOverlapRuleAgreesWithTheRecorder() {
        for engine in ModelCatalog.diarizationEngines {
            let value = ModelCatalog.diarizationEngineValue(engine)
            let r = AudioRecorder()
            r.mossDiarizationActive = value == ModelLoader.mossEngineID
            r.spectralDiarizationActive = value == ModelLoader.spectralEngineID
            r.nemoDiarizationActive = value == ModelLoader.nemoEngineID
            r.camPlusDiarizationActive = value == ModelLoader.camPlusEngineID
            r.diarizenDiarizationActive = value == ModelLoader.diarizenEngineID

            XCTAssertEqual(r.usesDetectedRegionsForRepair,
                           !ModelLoader.marksItsOwnOverlap(diarEngine: value),
                           "\(value): the tab's rule and the recorder's disagree — "
                           + "one of them is lying to somebody")
        }
    }

    /// …and the sentence built from it names every one of them.
    func testTheDetectorNoteNamesEveryEngineThatCannotMarkOverlap() {
        let sentence = ModelCatalog.diarizationEnginesWithoutOwnOverlap
        let expected = ModelCatalog.diarizationEngines
            .map { ModelCatalog.diarizationEngineValue($0) }
            .filter { !ModelLoader.marksItsOwnOverlap(diarEngine: $0) }

        // DERIVED: everything except the two that mark their own overlap
        // (pyannote and DiariZen). A literal was 4 until VibeVoice landed and
        // would have needed hand-editing for every engine after it.
        XCTAssertEqual(expected.count, ModelCatalog.diarizationEngines.count - 2,
                       "every engine but pyannote and DiariZen cannot mark overlap")
        for value in expected {
            let name = ModelCatalog.diarizationEngineShortName(value) ?? "?"
            XCTAssertTrue(sentence.contains(name),
                          "the detector note omits \(name): \"\(sentence)\"")
        }
        // The two that DO mark it must stay out, or the note would send users to
        // a detector that has nothing to add for them.
        for value in [ModelLoader.pyannoteEngineID, ModelLoader.diarizenEngineID] {
            let name = ModelCatalog.diarizationEngineShortName(value) ?? "?"
            XCTAssertFalse(sentence.contains(name),
                           "\(name) marks its own overlap and must not be listed")
        }
    }

    /// The built-in card must NOT claim word timestamps, because MOSS does not
    /// produce them: it returns `{start, end, speaker, text}` per SEGMENT and its
    /// own sidecar docstring says "No `conf` and no `words`, ever".
    ///
    /// What makes the aligner redundant is that its PURPOSE is already served —
    /// word times exist to split one chunk between speakers at the exact word,
    /// and MOSS emits a separate timed segment per speaker to begin with. Both
    /// directions asserted: the Qwen3 card must still make the claim, or this
    /// test would pass on a catalog that had stopped describing either.
    func testTheBuiltInAlignerCardDoesNotClaimWordTimestamps() {
        XCTAssertFalse(ModelCatalog.wordAlignerMoss.badges.contains("word timestamps"),
                       "MOSS reports no word times — claiming them here would be the "
                       + "fabrication direction this project ranks worst")
        XCTAssertTrue(ModelCatalog.wordAligner.badges.contains("word timestamps"),
                      "and the real aligner must still claim them, or this test is "
                      + "asserting nothing")
    }

    /// A built-in card still has to point at something real: the built-in aligner
    /// and the MOSS chunked entry are the same weights, so they must agree about
    /// whether they are installed. A card reporting "not installed" beside a
    /// chunked model reporting "installed" would send the user to download a
    /// model they already have.
    func testTheBuiltInAlignerSharesMossInstallState() {
        let moss = ModelCatalog.chunked.first { $0.id == "moss" }
        XCTAssertNotNil(moss)
        XCTAssertEqual(ModelCatalog.wordAlignerMoss.hfRepo, moss?.hfRepo)
        XCTAssertEqual(ModelCatalog.isInstalled(ModelCatalog.wordAlignerMoss),
                       ModelCatalog.isInstalled(moss!))
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
    /// ⚠ DiariZen MOVED SIDES on 2026-08-14 (owner: the picker must work on every
    /// engine). Its sidecar now narrows `min_speakers`/`max_speakers` per job —
    /// instance bounds, because that pipeline overrides `__call__` and takes no
    /// `num_speakers=` kwarg. Measured one-directional: it obeys a SMALLER count
    /// and ignores a larger one, so a wrong number there can only merge people,
    /// never invent one.
    ///
    /// **MOSS is now the half that matters**, and it is a permanent `false` rather
    /// than an unfinished one: speaker labels come out of a 0.9B LM as `[Sxx]`
    /// tags, there is no clustering stage to bound, and steering it by prompt was
    /// measured inert (2026-07-31). Five of six on this side; the sixth cannot
    /// join without a mechanism that does not exist.
    func testOnlyTheEnginesThatReadTheCountAreSaidToHonourIt() {
        for engine in [ModelLoader.pyannoteEngineID, ModelLoader.spectralEngineID,
                       ModelLoader.nemoEngineID, ModelLoader.camPlusEngineID,
                       ModelLoader.diarizenEngineID] {
            XCTAssertTrue(ModelLoader.honoursSpeakerCount(diarEngine: engine),
                          "\(engine)'s sidecar reads num_speakers and must be honoured")
        }
        XCTAssertFalse(ModelLoader.honoursSpeakerCount(diarEngine: ModelLoader.mossEngineID),
                       "MOSS has no count mechanism at all — saying it does would "
                       + "make the SPK control promise what no pass delivers")
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
    /// ⚠ RE-AIMED 2026-08-14. This used to assert a THIRD state — `finalPass` on
    /// WITH the tail on, which sent no count — and called it "the dangerous leg",
    /// because that toggle was visible only while the stop pass was OFF and a
    /// stored `true` could outlive the control that set it. The owner hit exactly
    /// that and settled it at the source: *"pas On mah diulang diarize dari awal
    /// sampai akhir."* A stop pass is now unconditionally a FULL pass, so the
    /// state no longer exists and `continueOnStop` is no longer a parameter.
    ///
    /// The rule that survives is the one that was always the real one: a count
    /// reaches pyannote when a full pass runs, and never on a `chunk` job.
    func testPyannoteOnlyGetsTheCountOnAFullStopPass() {
        func reaches(_ finalPass: Bool) -> Bool {
            ModelLoader.speakerCountReachesEngine(diarEngine: ModelLoader.pyannoteEngineID,
                                                  diarizationEnabled: true,
                                                  finalPass: finalPass)
        }
        XCTAssertTrue(reaches(true), "the stop pass is a full pass and sends it")
        XCTAssertFalse(reaches(false),
                       "no stop pass means live windows and at most a tail, both "
                       + "`chunk` jobs that carry no count")

        // The BATCH engines ignore `finalPass` for this — their pass IS the labels
        // — so it may not take the count away from them. Without this half,
        // folding the pyannote rule in could silently disable spectral, which is
        // the ONE engine the count measurably helps.
        for engine in [ModelLoader.spectralEngineID, ModelLoader.nemoEngineID,
                       ModelLoader.camPlusEngineID, ModelLoader.diarizenEngineID] {
            for finalPass in [true, false] {
                XCTAssertTrue(
                    ModelLoader.speakerCountReachesEngine(diarEngine: engine,
                                                          diarizationEnabled: true,
                                                          finalPass: finalPass),
                    "\(engine) must keep the count whatever finalPass=\(finalPass) "
                    + "says — it does not read it")
            }
        }

        // And diarization off means nothing is counted, for anyone.
        XCTAssertFalse(ModelLoader.speakerCountReachesEngine(
            diarEngine: ModelLoader.spectralEngineID, diarizationEnabled: false,
            finalPass: true))
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
    /// `@MainActor` on the test below is required rather than stylistic:
    /// `remoteNumSpeakers` is main-actor isolated, and `XCTAssertEqual` takes its
    /// arguments as nonisolated autoclosures — a warning today and an ERROR in
    /// the Swift 6 language mode.
    ///
    /// RE-AIMED 2026-08-13, when the owner asked for a Remote picker. It used to
    /// assert `remoteNumSpeakers == 0` — "remote is always automatic" — which was
    /// true of the pinned constant but was never the invariant worth pinning.
    ///
    /// **The invariant is that the two counts are DIFFERENT VALUES.** The 🔴 bug
    /// this test was written for was the ROOM's headcount reaching the remote
    /// passes: a 5-person room with one caller split that caller into five
    /// profiles. A number the user typed FOR the far end is a different thing
    /// entirely; a number typed for the room leaking across is the defect. So the
    /// test now asserts they cannot be the same reader, which is what would let
    /// the leak return.
    @MainActor
    func testTheRoomsCountCannotReachTheRemotePasses() {
        let d = UserDefaults.standard
        let office = "diarization.numSpeakers"
        let remote = "diarization.remoteNumSpeakers"
        let savedOffice = d.object(forKey: office)
        let savedRemote = d.object(forKey: remote)
        defer {
            savedOffice.map { d.set($0, forKey: office) } ?? d.removeObject(forKey: office)
            savedRemote.map { d.set($0, forKey: remote) } ?? d.removeObject(forKey: remote)
        }

        // The room is pinned to 5; the far end is left on Auto — the owner's own
        // failing case, and the one a shared reader would get wrong.
        d.set(5, forKey: office)
        d.removeObject(forKey: remote)
        XCTAssertEqual(AudioRecorder.diarNumSpeakers, 5)
        XCTAssertEqual(AudioRecorder.remoteNumSpeakers, 0,
                       "the room's 5 reached the far end — this is the bug that "
                       + "split one caller into five Remote Speaker profiles")

        // …and the far end can carry its own number without disturbing the room.
        d.set(2, forKey: remote)
        XCTAssertEqual(AudioRecorder.remoteNumSpeakers, 2)
        XCTAssertEqual(AudioRecorder.diarNumSpeakers, 5,
                       "and it must not leak back the other way either")
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

    /// Removes a key, checks the effect, restores the previous value.
    ///
    /// ⚠ This suite runs in the OWNER'S REAL preference domain, so the restore is
    /// not tidiness — without it a test would leave the app's own setting deleted.
    private func withoutKey(_ key: String, _ body: () -> Void) {
        let d = UserDefaults.standard
        let saved = d.object(forKey: key)
        d.removeObject(forKey: key)
        body()
        if let saved { d.set(saved, forKey: key) }
    }

    /// `Config` is Equatable and that is what makes `ModelLoader` replace a running
    /// sidecar. A setting that does NOT change it leaves the previous process
    /// answering for the rest of the meeting while Settings claims otherwise.
    func testEveryRealtimeSettingChangesTheRealtimeConfig() {
        withKey("realtime.model", "nemotron") {
            let before = RealtimeASRService.Config.fromSettings()
            for (key, value) in [("realtime.language", "id" as Any),
                                 ("realtime.partialMs", 3000 as Any)] {
                withKey(key, value) {
                    XCTAssertNotEqual(before, RealtimeASRService.Config.fromSettings(),
                                      "\(key) never reaches the realtime sidecar")
                }
            }

            // AND THE OPPOSITE, for the key whose control was removed on
            // 2026-08-11. `realtime.chunkMs` must NOT reach the sidecar: the
            // picker is gone and the value is pinned, so a number left behind by
            // an older build — or written by hand — must not be able to decide
            // behaviour that nothing in the UI can change back. That is the rule
            // the 2026-08-06 pass established for six other retired keys, and
            // this is the assertion that keeps it true for the seventh.
            withKey("realtime.chunkMs", 320 as Any) {
                XCTAssertEqual(before, RealtimeASRService.Config.fromSettings(),
                               "a stored realtime.chunkMs still reaches the sidecar — "
                               + "its control was removed, so the value must be inert")
            }
            XCTAssertEqual(RealtimeASRService.Config.fromSettings().chunkMs,
                           RealtimeASRService.pinnedChunkMs,
                           "the pinned chunk size must still be SENT — removing the "
                           + "control must not silently drop the capability")

            XCTAssertEqual(before, RealtimeASRService.Config.fromSettings(),
                           "a key was not restored")
        }
    }

    /// The ENGINE is a setting too, and the one whose omission would be worst: a
    /// `realtime.model` that did not change the Config would leave the previously
    /// loaded model transcribing while the tab showed the other one selected.
    ///
    /// Checked over EVERY PAIR rather than one, because that is what the third
    /// engine changed: with two, "they differ" and "each is distinct" are the
    /// same statement; with three a newcomer can collide with either incumbent,
    /// and only one of those collisions would be caught by a fixed pair.
    func testTheRealtimeEngineItselfChangesTheConfig() {
        var configs: [(String, RealtimeASRService.Config)] = []
        for m in ModelCatalog.realtimeModels {
            withKey("realtime.model", m.id) {
                let config = RealtimeASRService.Config.fromSettings()
                XCTAssertEqual(config.modelID, m.id,
                               "realtime.model never reaches the realtime sidecar")
                configs.append((m.id, config))
            }
        }
        for i in configs.indices {
            for j in configs.indices where j > i {
                XCTAssertNotEqual(configs[i].1, configs[j].1,
                                  "\(configs[i].0) and \(configs[j].0) compare equal — "
                                  + "switching between them would reuse the running "
                                  + "sidecar and keep the old model transcribing")
            }
        }
    }

    /// THE NEGATIVE DIRECTION, and it is the half a matrix sweep cannot see.
    /// Under Parakeet the chunk-size block is not even shown, and its value must
    /// be unable to enter the Config — otherwise moving an invisible control
    /// tears down and reloads a 2.3 GB sidecar. Structural: `chunkMs` is nil in a
    /// Parakeet config, so there is no value to differ.
    func testTheChunkSizeCannotReachTheParakeetConfig() {
        for engine in ["parakeet", "funasr"] {
            withKey("realtime.model", engine) {
                withKey("realtime.chunkMs", 160) {
                    let before = RealtimeASRService.Config.fromSettings()
                    XCTAssertNil(before.chunkMs, engine)
                    withKey("realtime.chunkMs", 1120) {
                        XCTAssertEqual(before, RealtimeASRService.Config.fromSettings(),
                                       "a Nemotron-only knob changed the \(engine) config")
                    }
                }
            }
        }
    }

    /// Every realtime engine in the catalog must round-trip through the lookup —
    /// the same shape the chunked/diarization/detector rows above assert, and the
    /// reason `realtimeModel(id:)` has a fallback rather than a `default:`.
    func testEveryRealtimeEngineRoundTripsThroughTheCatalog() {
        for m in ModelCatalog.realtimeModels {
            XCTAssertEqual(ModelCatalog.realtimeModel(id: m.id).id, m.id, m.name)
            withKey("realtime.model", m.id) {
                XCTAssertEqual(RealtimeASRService.Config.fromSettings().modelID, m.id,
                               "\(m.name) is offered but no Config names it")
            }
        }
        XCTAssertEqual(ModelCatalog.realtimeModels.first?.id,
                       RealtimeASRService.defaultModelID,
                       "the FIRST catalog entry is the default — the two must agree, "
                       + "or an unknown stored id resolves to a model the loader "
                       + "would not have started")
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

    /// The pause that ends an utterance, and therefore starts a ROW.
    ///
    /// ⚠ THIS PINS A RANGE, NOT THE NUMBER, and the range is the measurement.
    /// The lower bound is why the value moved: 300 ms is an ordinary pause
    /// INSIDE a sentence, so the live transcript cut a new row roughly every
    /// three seconds (measured: 120 rows / 18.0 per minute over the 43-minute
    /// 7-person meeting, median utterance 2.9 s).
    ///
    /// The UPPER bound is the one that matters more, because breaching it is
    /// silent. Every realtime sidecar caps its utterance buffer at
    /// `MAX_BUFFER = 60 s` and trims from the FRONT, so audio past that ceiling
    /// is discarded with nothing in the transcript to show for it. Measured
    /// longest utterance: 27.3 s at 600 ms, but **74.4 s at 1200 ms** — past
    /// the cap, i.e. real speech dropped. A value above ~800 ms must not be
    /// adopted without re-measuring that column, and this assertion is what
    /// makes someone re-read the reason before changing it.
    func testTheUtterancePauseSitsBetweenAChoppyRowAndALostBuffer() {
        XCTAssertGreaterThan(ShippedDefaults.vadMinSilenceMs, 300.0,
                             "300 ms is a within-sentence pause: at that value one "
                             + "sentence becomes several rows and the live "
                             + "transcript reads as choppy rather than as turns")
        XCTAssertLessThanOrEqual(ShippedDefaults.vadMinSilenceMs, 800.0,
                                 "above ~800 ms the longest measured utterance "
                                 + "approaches the sidecars' 60 s MAX_BUFFER, which "
                                 + "trims from the FRONT — speech would be dropped "
                                 + "with no trace anywhere but the audio file")
    }

    /// The value the VAD really receives is the shipped one, not a literal that
    /// happens to match it today. Without this the two could drift apart and the
    /// slider would show one number while the state machine used another.
    func testAnAbsentPauseKeyIsExactlyTheShippedDefault() {
        withoutKey("vad.minSilenceMs") {
            XCTAssertEqual(VoiceActivityDetector.Config.fromSettings().minSilenceMs,
                           ShippedDefaults.vadMinSilenceMs, accuracy: 0.0001)
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
