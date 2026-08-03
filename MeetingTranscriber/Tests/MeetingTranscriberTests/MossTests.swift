import XCTest
@testable import MeetingTranscriber

/// MOSS-Transcribe-Diarize as the diarization engine.
///
/// What is worth pinning here is not "does the model work" — that is the model's
/// business and the owner's call. It is the set of claims the APP makes on the
/// model's behalf, every one of which is a claim the model itself never made:
///
///   * a speaker label is valid inside ONE chunk (so ids must not collide across
///     chunks, and the display name must say which chunk it is);
///   * these labels are not voice profiles (so they must reach neither profile
///     store, nor the position diarizer, nor the rename path);
///   * the model attributed the text itself (so the rows must be its segments,
///     not a re-guess of them);
///   * one configuration cannot keep up in real time (so it is refused, loudly,
///     before any 3.6 GB load).
@MainActor
final class MossTests: XCTestCase {

    private func segment(_ speaker: String, _ start: Double, _ end: Double,
                         _ text: String) -> ChunkedASRService.MossSegment {
        ChunkedASRService.MossSegment(start: start, end: end, speaker: speaker, text: text)
    }

    // MARK: - Wire decoding contract

    /// `segments` is additive: a MOSS payload carries it, and every payload that
    /// predates MOSS (i.e. every MLX sidecar's, past and present) must still
    /// decode with it absent. Same rule the aligner keys and `conf` follow.
    func testMessageDecodesWithSegments() throws {
        let json = """
        {"type":"final","text":"Hello there. Hi.","segments":[
          {"start":0.0,"end":4.1,"speaker":"S01","text":"Hello there."},
          {"start":4.4,"end":5.2,"speaker":"S02","text":"Hi."}]}
        """
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(json.utf8))
        XCTAssertEqual(m.type, "final")
        XCTAssertEqual(m.text, "Hello there. Hi.")
        XCTAssertEqual(m.segments?.count, 2)
        XCTAssertEqual(m.segments?.first,
                       ChunkedASRService.MossSegment(start: 0.0, end: 4.1,
                                                     speaker: "S01", text: "Hello there."))
        XCTAssertNil(m.conf, "MOSS reports no confidence — absent stays absent")
    }

    func testMessageDecodesWithoutSegments() throws {
        let json = #"{"type":"final","text":"Hello there."}"#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(json.utf8))
        XCTAssertEqual(m.text, "Hello there.")
        XCTAssertNil(m.segments,
                     "An MLX sidecar's final has no segments key and must still decode")
    }

    /// An empty chunk (silence-gated, or a chunk the model returned nothing for)
    /// still sends a well-formed final, so the window FIFO always drains.
    func testMessageDecodesEmptySegments() throws {
        let json = #"{"type":"final","text":"","segments":[]}"#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(json.utf8))
        XCTAssertEqual(m.segments?.isEmpty, true)
    }

    // MARK: - Segments → turns

    /// Chunk-local seconds become recording time, and the ids/names carry the
    /// chunk they came from.
    func testSegmentsMapToAbsoluteTurns() {
        let turns = AudioRecorder.mossTurns(
            from: [segment("S01", 0.0, 4.1, "Hello there."),
                   segment("S02", 4.4, 5.2, "Hi.")],
            window: 30.0...60.0, chunkIndex: 7)

        XCTAssertEqual(turns.count, 2)
        XCTAssertEqual(turns[0].start, 30.0, accuracy: 1e-9)
        XCTAssertEqual(turns[0].end, 34.1, accuracy: 1e-9)
        XCTAssertEqual(turns[1].start, 34.4, accuracy: 1e-9)
        XCTAssertEqual(turns[1].end, 35.2, accuracy: 1e-9)
        XCTAssertEqual(turns[0].id, AudioRecorder.mossIDBase + 701)
        XCTAssertEqual(turns[1].id, AudioRecorder.mossIDBase + 702)
        XCTAssertEqual(turns[0].name, "Speaker 1")
        XCTAssertEqual(turns[1].name, "Speaker 2")
        XCTAssertNil(turns[0].conf, "MOSS scores no match, so there is nothing to report")
    }

    /// The core honesty property, and it now lives ENTIRELY in the id.
    ///
    /// The same raw label in two chunks is still two DIFFERENT speakers as far as
    /// the app is concerned, because the model made no claim that they are the
    /// same person. Since 2026-07-31 the NAME no longer carries that — the owner
    /// chose the plain `Speaker N` label knowing both chunks would read alike —
    /// so this test asserts the split survives where it still exists. If the id
    /// ever stops embedding the chunk index, `coalesceAdjacentSameSpeaker` will
    /// merge two unrelated people into one row and nothing else will object.
    func testSameLabelInTwoChunksGetsDifferentIDsButNowTheSameName() {
        let a = AudioRecorder.mossTurns(from: [segment("S01", 0, 5, "one")],
                                        window: 0...30, chunkIndex: 0)
        let b = AudioRecorder.mossTurns(from: [segment("S01", 0, 5, "two")],
                                        window: 30...60, chunkIndex: 1)
        XCTAssertNotEqual(a[0].id, b[0].id,
                          "the id is the ONLY thing keeping two chunks' speakers apart")
        XCTAssertEqual(a[0].name, "Speaker 1")
        XCTAssertEqual(b[0].name, "Speaker 1",
                       "accepted cost of the plain label: identical text, different people")
    }

    /// Times are clamped INTO the window. A segment claiming to end after its own
    /// chunk would overlap the next chunk's rows and break the transcript's
    /// otherwise strictly increasing timeline.
    func testTimesAreClampedIntoTheWindow() {
        let turns = AudioRecorder.mossTurns(
            from: [segment("S01", -2.0, 99.0, "runaway")],
            window: 30.0...60.0, chunkIndex: 0)
        XCTAssertEqual(turns[0].start, 30.0, accuracy: 1e-9)
        XCTAssertEqual(turns[0].end, 60.0, accuracy: 1e-9)
    }

    /// A speaker span with no words is nothing the transcript can show.
    func testEmptyTextSegmentsAreDropped() {
        let turns = AudioRecorder.mossTurns(
            from: [segment("S01", 0, 1, "   "), segment("S02", 1, 2, "real")],
            window: 0...30, chunkIndex: 0)
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].name, "Speaker 2")
    }

    func testSpeakerIndexParsing() {
        XCTAssertEqual(AudioRecorder.mossSpeakerIndex("S01"), 1)
        XCTAssertEqual(AudioRecorder.mossSpeakerIndex("S12"), 12)
        XCTAssertEqual(AudioRecorder.mossSpeakerIndex("nonsense"), 0,
                       "A malformed label degrades to one shared id, never a crash")
    }

    /// The clamp that keeps one chunk's block from running into the next one's.
    func testLocalIndexIsClampedBelowTheChunkStride() {
        let ceiling = AudioRecorder.mossWireID(chunkIndex: 0, localIndex: 500)
        XCTAssertEqual(ceiling, AudioRecorder.mossIDBase + 99)
        XCTAssertLessThan(ceiling, AudioRecorder.mossWireID(chunkIndex: 1, localIndex: 0))
    }

    // MARK: - Id-space disjointness

    /// MOSS ids sit above everything else and belong to no store.
    func testMossIDsAreAboveEveryOtherSpace() {
        XCTAssertGreaterThan(AudioRecorder.mossIDBase, PositionDiarizer.positionIDBase)
        XCTAssertGreaterThan(PositionDiarizer.positionIDBase, AudioRecorder.remoteIDBase)
        XCTAssertGreaterThan(AudioRecorder.remoteIDBase, 0)
    }

    func testMossIDsAreNotProfileIDs() {
        let id = AudioRecorder.mossWireID(chunkIndex: 3, localIndex: 1)
        XCTAssertFalse(SpeakerProfileStore.isProfileID(id),
                       "A MOSS label is backed by no voice profile — neither store may see it")
        XCTAssertTrue(SpeakerProfileStore.isProfileID(AudioRecorder.remoteIDBase + 1))
        XCTAssertTrue(SpeakerProfileStore.isProfileID(1))
    }

    /// A MOSS id is ALSO `>= positionIDBase`, which is exactly why `renameSpeaker`
    /// has to test for it FIRST: without the early return it would fall into the
    /// position branch and rename cluster `id - 100_000`, which is either absent
    /// or somebody else's. The guard is a predicate on the id, so it is checked
    /// here as one — calling `renameSpeaker` for real would only prove the absence
    /// of an effect.
    func testMossIDsWouldOtherwiseFallIntoThePositionBranch() {
        let id = AudioRecorder.mossWireID(chunkIndex: 0, localIndex: 1)
        XCTAssertGreaterThanOrEqual(id, PositionDiarizer.positionIDBase,
                                    "This is the hazard the early return exists for")
        XCTAssertFalse(id < AudioRecorder.mossIDBase,
                       "…and this is the guard that catches it")
        XCTAssertTrue(PositionDiarizer.positionIDBase + 1 < AudioRecorder.mossIDBase,
                      "A genuine position id still reaches the position branch")
    }

    /// Turns from MOSS are non-office and non-remote in the existing predicates'
    /// eyes, which is why they must live in `mossTurns` rather than `liveTurns`.
    func testMossTurnsAreRejectedByBothSpacePredicates() {
        let turns = AudioRecorder.mossTurns(from: [segment("S01", 0, 1, "hi")],
                                            window: 0...30, chunkIndex: 0)
        XCTAssertEqual(AudioRecorder.nonOfficeIDs(in: turns), turns.map(\.id))
        XCTAssertEqual(AudioRecorder.nonRemoteIDs(in: turns), turns.map(\.id))
    }

    // MARK: - Process count

    /// The primary mode is ONE process: MOSS in both roles means one 3.6 GB load
    /// and one forward pass per chunk, not two.
    func testNeedsSecondMossProcess() {
        XCTAssertFalse(ModelLoader.needsSecondMossProcess(chunkedID: "moss", engine: "moss"),
                       "One sidecar already carries both the text and the segments")
        XCTAssertTrue(ModelLoader.needsSecondMossProcess(chunkedID: "whisper", engine: "moss"))
        XCTAssertTrue(ModelLoader.needsSecondMossProcess(chunkedID: "qwen3", engine: "moss"))
        XCTAssertFalse(ModelLoader.needsSecondMossProcess(chunkedID: "moss", engine: "pyannote"),
                       "MOSS as plain ASR under pyannote loads no diarization MOSS")
        XCTAssertFalse(ModelLoader.needsSecondMossProcess(chunkedID: "whisper", engine: "pyannote"))
    }

    // MARK: - Catalog id vs setting value

    /// The engine card's catalog id and the `diarization.engine` value it stores
    /// are DIFFERENT strings for MOSS, and both have to be: the chunked entry
    /// already owns the catalog id "moss", while the loader and the recorder read
    /// the setting as a plain "pyannote" | "moss" choice. Binding the picker to
    /// `m.id` instead of to this mapping stores "moss-diar", which nothing
    /// matches — the engine would appear selected and never actually switch.
    func testDiarizationEngineValueMapsCatalogIDToSettingValue() {
        XCTAssertEqual(ModelCatalog.mossDiarization.id, "moss-diar")
        XCTAssertEqual(ModelLoader.mossEngineID, "moss")
        XCTAssertNotEqual(ModelCatalog.mossDiarization.id, ModelLoader.mossEngineID,
                          "If these ever become equal, the mapping below is what kept them apart")

        XCTAssertEqual(ModelCatalog.diarizationEngineValue(ModelCatalog.mossDiarization),
                       ModelLoader.mossEngineID)
        XCTAssertEqual(ModelCatalog.diarizationEngineValue(ModelCatalog.diarization),
                       ModelLoader.pyannoteEngineID)
        XCTAssertEqual(ModelCatalog.diarization.id, ModelLoader.pyannoteEngineID,
                       "pyannote's id and engine value coincide — only MOSS diverges")
    }

    /// Round-trips, so the tab can go both ways without a second mapping.
    func testDiarizationEngineRoundTrips() {
        for model in ModelCatalog.diarizationEngines {
            let value = ModelCatalog.diarizationEngineValue(model)
            XCTAssertEqual(ModelCatalog.diarizationEngine(forEngine: value).id, model.id)
        }
        XCTAssertEqual(ModelCatalog.diarizationEngine(forEngine: "nonsense").id,
                       ModelCatalog.diarization.id,
                       "An unknown value falls back to pyannote — the safe default")
    }

    /// Both roles are the same weights on disk, so one download serves both and
    /// the install check cannot disagree between the two cards.
    func testBothMossRolesShareOneRepo() {
        XCTAssertEqual(ModelCatalog.mossDiarization.hfRepo,
                       ModelCatalog.chunkedModel(id: "moss").hfRepo)
        XCTAssertEqual(ModelCatalog.mossDiarization.hfRepo,
                       "OpenMOSS-Team/MOSS-Transcribe-Diarize")
    }

    // MARK: - Startup refusal

    func testRefusesVoxtralWithMossDiarization() {
        let message = AudioRecorder.mossRefusalMessage(chunkedModelID: "voxtral",
                                                       diarizationEngine: "moss",
                                                       remoteChannel: nil)
        XCTAssertNotNil(message)
        XCTAssertTrue(message?.contains("27 s") == true, "The message names the numbers")
        XCTAssertTrue(message?.contains("Settings → Models → Chunked") == true,
                      "…and says which setting to change")
    }

    func testRefusalMentionsRemoteWhenOneIsConfigured() {
        let message = AudioRecorder.mossRefusalMessage(chunkedModelID: "voxtral",
                                                       diarizationEngine: "moss",
                                                       remoteChannel: 1)
        XCTAssertTrue(message?.contains("Remote stream doubles") == true)
    }

    /// Everything else starts. In particular MOSS in both roles is never refused,
    /// with or without a Remote channel: it is one process, and 2 × 6.4 s of a
    /// 30 s budget is 43 % duty.
    func testAllowedMossConfigurations() {
        XCTAssertNil(AudioRecorder.mossRefusalMessage(chunkedModelID: "moss",
                                                      diarizationEngine: "moss",
                                                      remoteChannel: nil))
        XCTAssertNil(AudioRecorder.mossRefusalMessage(chunkedModelID: "moss",
                                                      diarizationEngine: "moss",
                                                      remoteChannel: 1))
        XCTAssertNil(AudioRecorder.mossRefusalMessage(chunkedModelID: "whisper",
                                                      diarizationEngine: "moss",
                                                      remoteChannel: 1))
        XCTAssertNil(AudioRecorder.mossRefusalMessage(chunkedModelID: "moss",
                                                      diarizationEngine: "pyannote",
                                                      remoteChannel: nil))
    }

    /// Voxtral is only refused BY THIS RULE when MOSS is the engine — the
    /// pre-existing dual-stream refusal is a separate rule and is not disturbed.
    func testVoxtralUnderPyannoteIsNotThisRulesBusiness() {
        XCTAssertNil(AudioRecorder.mossRefusalMessage(chunkedModelID: "voxtral",
                                                      diarizationEngine: "pyannote",
                                                      remoteChannel: 1))
        XCTAssertNotNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                               chunkedModelID: "voxtral"),
                        "The dual-stream refusal still owns that case, unchanged")
    }

    // MARK: - Pinned rows (MOSS in both roles)

    /// N segments become N confirmed rows with the model's own spans and labels —
    /// no `assignSentences` re-guessing an attribution the model already made.
    func testPinnedSegmentsBecomeOneRowEach() {
        let recorder = AudioRecorder()
        recorder.mossDiarizationActive = true
        recorder.mossIsChunkedModel = true
        recorder.mossIncomingSegments = [segment("S01", 0.0, 4.1, "Hello there."),
                                         segment("S02", 4.4, 5.2, "Hi."),
                                         segment("S01", 5.5, 9.0, "How are you?")]
        XCTAssertTrue(recorder.applyMossChunk(window: 0.0...30.0))
        recorder.rebuildDisplayRows()

        XCTAssertEqual(recorder.segments.count, 3)
        XCTAssertTrue(recorder.segments.allSatisfy(\.confirmed))
        // Guarded: everything below indexes the rows, and a collapsed list should
        // fail this one assertion rather than trap the whole test binary.
        guard recorder.displayRows.count == 3 else {
            return XCTFail("expected 3 rows, got \(recorder.displayRows.map(\.text))")
        }
        XCTAssertEqual(recorder.displayRows.map(\.text),
                       ["Hello there.", "Hi.", "How are you?"])
        XCTAssertEqual(recorder.displayRows.map(\.speaker),
                       ["Speaker 1", "Speaker 2", "Speaker 1"])
        XCTAssertEqual(recorder.displayRows[0].start, 0.0)
        XCTAssertEqual(recorder.displayRows[0].end, 4.1)
        XCTAssertEqual(recorder.displayRows[2].start, 5.5)
        XCTAssertEqual(recorder.displayRows[2].end, 9.0)
        XCTAssertTrue(recorder.displayRows.allSatisfy { $0.speakerConf == nil },
                      "MOSS scores no speaker match — nil means not measured")
        XCTAssertTrue(recorder.displayRows.allSatisfy { $0.asrConf == nil },
                      "…and reports no transcript confidence either")
        XCTAssertTrue(recorder.displayRows.allSatisfy { !$0.overlapped },
                      "MOSS makes no overlap claim, so no row is tagged as one")
        XCTAssertTrue(recorder.liveTurns.isEmpty,
                      "Office-only state stays pure pyannote — MOSS turns live apart")
        XCTAssertEqual(recorder.mossTurns.count, 3)
    }

    /// The chunk boundary is a hard wall: the same voice either side of it has two
    /// different ids, so `coalesceAdjacentSameSpeaker` cannot weld the rows into
    /// one person. Continuity is unknown, and the transcript says so.
    func testRowsDoNotCoalesceAcrossChunks() {
        let recorder = AudioRecorder()
        recorder.mossDiarizationActive = true
        recorder.mossIsChunkedModel = true

        recorder.mossIncomingSegments = [segment("S01", 0.0, 20.0, "First half.")]
        XCTAssertTrue(recorder.applyMossChunk(window: 0.0...30.0))
        recorder.mossIncomingSegments = [segment("S01", 0.0, 20.0, "Second half.")]
        XCTAssertTrue(recorder.applyMossChunk(window: 30.0...60.0))
        recorder.rebuildDisplayRows()

        // Since the labels became plain `Speaker N` (owner, 2026-07-31) these two
        // rows are IDENTICAL on screen, so this assertion is the last thing
        // standing between two unrelated people and one merged row.
        XCTAssertEqual(recorder.displayRows.count, 2,
                       "two chunks must stay two rows even though both now render "
                       + "as the same 'Speaker 1' — only the id keeps them apart")
        XCTAssertEqual(recorder.displayRows.map(\.speaker), ["Speaker 1", "Speaker 1"])
        XCTAssertNotEqual(recorder.displayRows[0].speakerID, recorder.displayRows[1].speakerID)
    }

    /// Under the pyannote engine — including MOSS as a plain chunked recogniser —
    /// none of the above applies: the segments are ignored and the ordinary path
    /// runs untouched.
    func testSegmentsAreIgnoredUnderThePyannoteEngine() {
        let recorder = AudioRecorder()
        recorder.mossDiarizationActive = false
        recorder.mossIsChunkedModel = true
        recorder.mossIncomingSegments = [segment("S01", 0.0, 4.0, "Hello there.")]

        XCTAssertFalse(recorder.applyMossChunk(window: 0.0...30.0),
                       "The chunked callback must fall through to its normal append")
        XCTAssertTrue(recorder.segments.isEmpty)
        XCTAssertTrue(recorder.mossTurns.isEmpty)
    }

    /// The display reads exactly one turn collection, chosen by the engine.
    func testDisplayTurnsFollowTheEngine() {
        let recorder = AudioRecorder()
        recorder.liveTurns = [SpeakerTurn(start: 0, end: 5, id: 1, name: "Speaker 1")]
        recorder.mossTurns = AudioRecorder.mossTurns(from: [segment("S01", 0, 5, "hi")],
                                                     window: 0...30, chunkIndex: 0)

        recorder.mossDiarizationActive = false
        XCTAssertEqual(recorder.displayTurns.map(\.id), [1])
        recorder.mossDiarizationActive = true
        XCTAssertEqual(recorder.displayTurns.map(\.id), recorder.mossTurns.map(\.id))
    }
}
