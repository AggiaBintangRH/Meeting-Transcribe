import XCTest
@testable import MeetingTranscriber

/// `chunked.enabled` — the master switch added 2026-08-06, and the two rules it
/// had to be coupled to.
///
/// WHY IT NEEDED ITS OWN SWITCH. The Chunked tab already carried
/// `chunked.finalPass` ("Run a transcription pass at stop"), which READS like an
/// on/off and is not one: it governs only the extra pass after Stop, and with it
/// off chunked ASR still runs all meeting and still produces the transcript. The
/// rail's SWITCHED OFF group would therefore have been lying about the model that
/// was writing the whole transcript.
///
/// The dangerous direction here is not memory, it is a MEETING WITH NO SPEAKERS:
/// under the MOSS engine the diarizer borrows its segments from the chunked
/// process, so switching that process off has to hand MOSS its own — see
/// `testMossGetsItsOwnProcessWhenTheChunkedPassIsOff`, which is the case that
/// would fail silently.
final class ChunkedMasterSwitchTests: XCTestCase {

    private let moss = ModelLoader.mossEngineID
    private let pyannote = ModelLoader.pyannoteEngineID
    private let spectral = ModelLoader.spectralEngineID
    private let nemo = ModelLoader.nemoEngineID

    // MARK: - The switch itself

    func testTheSwitchIsTheWholeRule() {
        XCTAssertTrue(ModelLoader.wantsChunked(chunkedEnabled: true))
        XCTAssertFalse(ModelLoader.wantsChunked(chunkedEnabled: false))
    }

    // MARK: - Coupling 1: MOSS must not be left without a process

    /// THE CASE THAT WOULD HAVE FAILED SILENTLY. In MOSS+MOSS mode one sidecar
    /// serves both roles, so `needsSecondMossProcess` is false — correct, until
    /// the chunked pass is switched off and that sidecar no longer exists. Without
    /// this coupling the MOSS engine would load nothing at all and the meeting
    /// would end with no speakers and no error.
    func testMossGetsItsOwnProcessWhenTheChunkedPassIsOff() {
        XCTAssertFalse(ModelLoader.needsSecondMossProcess(chunkedID: "moss", engine: moss,
                                                          chunkedEnabled: true),
                       "MOSS+MOSS with the pass ON must still share one process")
        XCTAssertTrue(ModelLoader.needsSecondMossProcess(chunkedID: "moss", engine: moss,
                                                         chunkedEnabled: false),
                      "with no chunked process there is nothing to borrow segments from")
    }

    /// And the stack rule agrees, since that is what `buildSteps` and the teardown
    /// both read. A `.mossOwnASR` stack with the chunked pass off would name a
    /// process this session never starts.
    func testTheStackNeverSaysMossOwnASRWithTheChunkedPassOff() {
        XCTAssertEqual(ModelLoader.wantedDiarizationStack(diarizationEnabled: true,
                                                          engine: moss, chunkedID: "moss",
                                                          chunkedEnabled: true),
                       .mossOwnASR)
        XCTAssertEqual(ModelLoader.wantedDiarizationStack(diarizationEnabled: true,
                                                          engine: moss, chunkedID: "moss",
                                                          chunkedEnabled: false),
                       .mossSecondProcess)
    }

    /// The switch may not disturb the other two engines: they never borrowed
    /// anything from the chunked process, so their stack is the same either way.
    /// Without this the coupling above could quietly widen into every engine.
    func testTheOtherEnginesAreUnaffectedByTheSwitch() {
        for engine in [pyannote, spectral, nemo] {
            for chunkedID in ["qwen3", "whisper", "moss"] {
                XCTAssertEqual(
                    ModelLoader.wantedDiarizationStack(diarizationEnabled: true, engine: engine,
                                                       chunkedID: chunkedID, chunkedEnabled: true),
                    ModelLoader.wantedDiarizationStack(diarizationEnabled: true, engine: engine,
                                                       chunkedID: chunkedID, chunkedEnabled: false),
                    "engine=\(engine) chunked=\(chunkedID) changed with the chunked switch")
            }
        }
    }

    /// Diarization's own master switch still wins over all of it — the 2026-07-31
    /// rule must not have been weakened by adding a second input.
    func testDiarizationOffStillKeepsNothing() {
        for chunkedOn in [true, false] {
            for engine in [pyannote, moss, spectral, nemo] {
                XCTAssertNil(ModelLoader.wantedDiarizationStack(diarizationEnabled: false,
                                                                engine: engine, chunkedID: "moss",
                                                                chunkedEnabled: chunkedOn))
            }
        }
    }

    // MARK: - Coupling 2: the aligner has nothing left to align

    /// The aligner splits CHUNKED segments into words. With no chunked pass there
    /// are none, so loading 1.2 GB for it is the waste every want-rule here exists
    /// to prevent — and the Aligner tab is filed under "not used by your models"
    /// rather than "switched off", because its own toggle cannot rescue it.
    func testTheAlignerIsNotWantedWithoutAChunkedPass() {
        for chunkedID in ["qwen3", "whisper"] {
            XCTAssertTrue(ModelLoader.wantsAligner(alignEnabled: true, chunkedID: chunkedID,
                                                   chunkedEnabled: true))
            XCTAssertFalse(ModelLoader.wantsAligner(alignEnabled: true, chunkedID: chunkedID,
                                                    chunkedEnabled: false))
        }
        // The MOSS exclusion is independent and must survive: it is about MOSS
        // returning its own times, not about whether a pass runs.
        XCTAssertFalse(ModelLoader.wantsAligner(alignEnabled: true, chunkedID: "moss",
                                                chunkedEnabled: true))
    }

    // MARK: - Coupling 3: a Remote stream has no other source of text

    /// The remote stream is transcribed ONLY by the chunked pass, and the
    /// pre-existing `remoteWanted` guard drops it silently when no chunked sidecar
    /// exists. Silent is the failure direction this project refuses, so the
    /// combination is declined at startup — before any model loads.
    func testARemoteChannelWithTheChunkedPassOffIsRefused() {
        XCTAssertNotNil(AudioRecorder.chunkedOffRefusalMessage(remoteChannel: 1,
                                                               chunkedEnabled: false))
        // …and the three cases that must still start normally. Single-stream with
        // the pass off is a legitimate choice (realtime captions only) and must
        // NOT be refused — that is what stops this widening into a general ban.
        XCTAssertNil(AudioRecorder.chunkedOffRefusalMessage(remoteChannel: nil,
                                                            chunkedEnabled: false))
        XCTAssertNil(AudioRecorder.chunkedOffRefusalMessage(remoteChannel: 1,
                                                            chunkedEnabled: true))
        XCTAssertNil(AudioRecorder.chunkedOffRefusalMessage(remoteChannel: nil,
                                                            chunkedEnabled: true))
    }

    /// The message must name both ways out, since the two settings live on
    /// different tabs and the user is looking at neither.
    func testTheRefusalNamesBothWaysOut() {
        let m = AudioRecorder.chunkedOffRefusalMessage(remoteChannel: 2,
                                                       chunkedEnabled: false) ?? ""
        XCTAssertTrue(m.contains("Models → Chunked"), m)
        XCTAssertTrue(m.contains("Microphone"), m)
    }

    // MARK: - MOSS diarization is only OFFERED beside the MOSS chunked model

    /// Owner, 2026-08-06: MOSS chunked ⟺ MOSS diarization, BOTH directions. The
    /// rule is read by three places that must agree — the card filter,
    /// `selectChunkedModel` and `DiarizationTab.onAppear` — and disagreement
    /// leaves an engine STORED with no card on screen selecting it.
    func testTheMossPairingIsExclusiveInBothDirections() {
        let spectralID = ModelLoader.spectralEngineID
        // With MOSS as the ASR, MOSS is the ONLY diarizer offered.
        XCTAssertTrue(ModelLoader.diarizationEngineIsSelectable(moss, chunkedID: "moss"))
        XCTAssertFalse(ModelLoader.diarizationEngineIsSelectable(pyannote, chunkedID: "moss"))
        XCTAssertFalse(ModelLoader.diarizationEngineIsSelectable(spectralID, chunkedID: "moss"))
        XCTAssertFalse(ModelLoader.diarizationEngineIsSelectable(nemo, chunkedID: "moss"))
        // With any other ASR, MOSS is the one NOT offered.
        for other in ["qwen3", "whisper", "granite", "voxtral"] {
            XCTAssertFalse(ModelLoader.diarizationEngineIsSelectable(moss, chunkedID: other))
            XCTAssertTrue(ModelLoader.diarizationEngineIsSelectable(pyannote, chunkedID: other))
            XCTAssertTrue(ModelLoader.diarizationEngineIsSelectable(spectralID, chunkedID: other))
            XCTAssertTrue(ModelLoader.diarizationEngineIsSelectable(nemo, chunkedID: other))
        }
    }

    /// Every chunked model must leave at least one engine selectable, and the
    /// fallback must be one of them. A rule that offered nothing, or fell back to
    /// an engine it had just hidden, would leave Settings unusable — and the
    /// fallback is written separately from the filter, so nothing else pairs them.
    func testTheFallbackEngineIsAlwaysOneTheRuleStillOffers() {
        for chunkedID in ["moss", "qwen3", "whisper", "granite", "voxtral"] {
            let offered = [pyannote, spectral, nemo, moss].filter {
                ModelLoader.diarizationEngineIsSelectable($0, chunkedID: chunkedID)
            }
            XCTAssertFalse(offered.isEmpty, "\(chunkedID) offers no diarization engine at all")
            XCTAssertTrue(offered.contains(ModelLoader.fallbackDiarizationEngine(chunkedID: chunkedID)),
                          "\(chunkedID) falls back to an engine it does not offer")
        }
    }

    /// It is a UI-availability rule ONLY. The loader still supports MOSS as the
    /// diarizer beside another ASR — a second process — and must keep doing so,
    /// or relaxing the one-line UI rule later would restore a card that no longer
    /// works. This is the test that stops the two being conflated.
    func testTheLoaderStillSupportsMossDiarizationBesideAnotherASR() {
        XCTAssertTrue(ModelLoader.needsSecondMossProcess(chunkedID: "whisper", engine: moss,
                                                         chunkedEnabled: true))
        XCTAssertEqual(ModelLoader.wantedDiarizationStack(diarizationEnabled: true,
                                                          engine: moss, chunkedID: "whisper",
                                                          chunkedEnabled: true),
                       .mossSecondProcess)
    }

    // MARK: - The stop gate

    /// With no chunked sidecar the stop plan must SETTLE the gate here and must
    /// NOT sweep the realtime tail — that tail is the only text the audio will
    /// ever have. This state predates the switch (it was the "no chunked model"
    /// early-out), which is why the switch needed no new stop-gate branch; pinned
    /// so a future edit cannot quietly make the switch hang the overlay or delete
    /// the user's only transcript.
    func testNoChunkedModelSettlesTheGateAndKeepsTheRealtimeTail() {
        for finalPass in [true, false] {
            for tailOnly in [true, false] {
                let mode = AudioRecorder.chunkedStopMode(
                    finalPass: finalPass, continueOnStop: tailOnly,
                    hasChunkedModel: false, hasRecording: true, chunkedModelID: "qwen3")
                let plan = AudioRecorder.chunkedStopPlan(mode)
                XCTAssertTrue(plan.settlesImmediately,
                              "the stop gate would hang with no chunked sidecar")
                XCTAssertFalse(plan.sweepsUnconfirmedTail,
                               "the realtime tail is the only transcript — it may not be swept")
                XCTAssertFalse(plan.runsFullPass)
                XCTAssertFalse(plan.flushesSidecar)
                XCTAssertFalse(plan.queuesTailWindow)
            }
        }
    }
}
