import XCTest
@testable import MeetingTranscriber

/// Unit tests for phase 3 of dual-stream capture — transcribing the Remote
/// stream and rendering its rows.
///
/// Everything reachable without hardware is a pure function here on purpose:
/// the row merge, the silence gate and the Voxtral refusal are the three
/// decisions this phase makes, and none of them needs an engine, a sidecar or a
/// device to be exercised. The plumbing around them (the tap's remote resampler,
/// the `-2` file-transcribe round trip, the stop gate's timing) is not testable
/// without the Aggregate Device and is listed as an on-device check instead.
@MainActor
final class DualStreamRemoteTests: XCTestCase {

    // MARK: - Helpers

    private func office(_ start: Double?, _ text: String) -> AudioRecorder.SpeakerUtterance {
        AudioRecorder.SpeakerUtterance(id: "office-\(text)", speaker: "Speaker 1",
                                       speakerID: 1, start: start, end: start.map { $0 + 1 },
                                       text: text, confirmed: true)
    }

    private func remoteSegment(_ start: Double, _ end: Double, _ text: String)
        -> AudioRecorder.RemoteSegment {
        AudioRecorder.RemoteSegment(text: text, window: start...end)
    }

    /// A constant-amplitude signal — RMS equals the amplitude exactly, so a test
    /// can sit precisely on either side of the gate's threshold.
    private func tone(amplitude: Float, count: Int) -> [Float] {
        (0..<count).map { $0.isMultiple(of: 2) ? amplitude : -amplitude }
    }

    // MARK: - 1. The unset path is inert (the regression bar)

    /// With no Remote channel configured nothing ever appends a `RemoteSegment`,
    /// so `rebuildDisplayRows` merges against an empty remote list. That merge
    /// must return the office rows completely untouched — same rows, same order,
    /// same count — or every single-stream recording would be affected by a
    /// feature it is not using.
    func testNoRemoteRowsLeavesOfficeRowsUntouched() {
        let rows = [office(3, "c"), office(1, "a"), office(2, "b")]  // deliberately unsorted
        let merged = AudioRecorder.mergeRowsByStartTime(office: rows, remote: [])
        XCTAssertEqual(merged, rows, "Office order is preserved verbatim, NOT re-sorted")
    }

    /// The other half of the same guarantee: with no remote segments there are no
    /// remote rows to merge in the first place.
    func testEmptyRemoteSegmentsProduceNoRows() {
        XCTAssertTrue(AudioRecorder.remoteRows([]).isEmpty)
    }

    /// A single-stream session cannot even reach the refusal.
    func testNoRemoteChannelNeverRefuses() {
        for model in ["qwen3", "whisper", "granite", "voxtral"] {
            XCTAssertNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: nil,
                                                                chunkedModelID: model))
        }
    }

    // MARK: - 2. Office + Remote row merge ordering

    func testRowsMergeSortedByStartTime() {
        let officeRows = [office(0, "a"), office(10, "b"), office(20, "c")]
        let remoteRows = AudioRecorder.remoteRows([remoteSegment(5, 8, "r1"),
                                                   remoteSegment(15, 18, "r2")])
        let merged = AudioRecorder.mergeRowsByStartTime(office: officeRows, remote: remoteRows)
        XCTAssertEqual(merged.map(\.text), ["a", "r1", "b", "r2", "c"])
        XCTAssertEqual(merged.map(\.start), [0, 5, 10, 15, 20])
    }

    /// A tie puts Office first — the room is the primary record, and a stable,
    /// stated rule beats whichever order a sort happened to produce.
    func testTiedStartsPutOfficeFirst() {
        let merged = AudioRecorder.mergeRowsByStartTime(
            office: [office(5, "office")],
            remote: AudioRecorder.remoteRows([remoteSegment(5, 6, "remote")]))
        XCTAssertEqual(merged.map(\.text), ["office", "remote"])
    }

    /// A row with no start time (an unconfirmed realtime segment with no window
    /// yet) counts as +∞, so remote rows go ahead of it — but it keeps its place
    /// among the office rows, because office order is never re-sorted.
    func testRowsWithoutStartCountAsInfinity() {
        let merged = AudioRecorder.mergeRowsByStartTime(
            office: [office(nil, "pending"), office(30, "late")],
            remote: AudioRecorder.remoteRows([remoteSegment(10, 12, "remote")]))
        XCTAssertEqual(merged.map(\.text), ["remote", "pending", "late"])
    }

    /// This is a merge walk, not a sort of the combined list: office rows are
    /// emitted in exactly the order they arrived, even when that order is not
    /// chronological. Separation-debug rows depend on this — they carry real
    /// windows but are pinned to the end of the transcript on purpose.
    func testOfficeOrderIsNeverResorted() {
        let officeRows = [office(20, "pinned-debug-style"), office(1, "early")]
        let merged = AudioRecorder.mergeRowsByStartTime(
            office: officeRows,
            remote: AudioRecorder.remoteRows([remoteSegment(5, 6, "remote")]))
        XCTAssertEqual(merged.map(\.text), ["remote", "pinned-debug-style", "early"],
                       "Remote is placed before the first later-starting office row; "
                       + "office rows keep their own order")
    }

    /// Remote segments are appended when their transcription lands. `remoteRows`
    /// sorts them chronologically so the merge never has to trust append order.
    func testRemoteRowsAreSortedChronologically() {
        let remote = AudioRecorder.remoteRows([remoteSegment(9, 10, "second"),
                                               remoteSegment(1, 2, "first")])
        XCTAssertEqual(remote.map(\.text), ["first", "second"])
        let merged = AudioRecorder.mergeRowsByStartTime(office: [office(5, "office")],
                                                        remote: remote)
        XCTAssertEqual(merged.map(\.text), ["first", "office", "second"])
    }

    /// All-remote and all-office transcripts both survive the walk.
    func testMergeHandlesOneSidedInputs() {
        let remote = AudioRecorder.remoteRows([remoteSegment(1, 2, "r")])
        XCTAssertEqual(AudioRecorder.mergeRowsByStartTime(office: [], remote: remote).map(\.text),
                       ["r"])
        XCTAssertEqual(AudioRecorder.mergeRowsByStartTime(office: [office(1, "o")], remote: [])
                        .map(\.text), ["o"])
    }

    // MARK: - 3. Remote rows carry the Remote label space

    func testRemoteRowsAreLabelledAndFlagged() {
        let rows = AudioRecorder.remoteRows([remoteSegment(4, 9, "hello from the call")])
        XCTAssertEqual(rows.count, 1)
        let row = rows[0]
        XCTAssertEqual(row.speaker, "Remote Speaker - Speaker Unknown")
        XCTAssertEqual(AudioRecorder.remoteSpeakerLabel, "Remote Speaker - Speaker Unknown")
        XCTAssertTrue(row.isRemote, "The UI needs the flag to keep the label spaces distinct")
        XCTAssertNil(row.speakerID, "No id ⇒ no rename ⇒ nothing can reach SpeakerProfileStore")
        XCTAssertEqual(row.start, 4)
        XCTAssertEqual(row.end, 9)
        XCTAssertTrue(row.confirmed)
        XCTAssertFalse(row.overlapped, "Remote never enters the office overlap path")
    }

    /// An empty (or whitespace-only) transcript renders nothing rather than an
    /// empty amber card.
    func testBlankRemoteSegmentsAreDropped() {
        XCTAssertTrue(AudioRecorder.remoteRows([remoteSegment(0, 1, "   \n ")]).isEmpty)
    }

    /// Phase 4 groundwork: the id spaces are chosen so one integer comparison
    /// separates them. Nothing uses `remoteIDBase` yet — this pins the ranges so
    /// they cannot be chosen to collide later.
    func testRemoteIDBaseCannotCollideWithPositionIDs() {
        XCTAssertLessThan(AudioRecorder.remoteIDBase, PositionDiarizer.positionIDBase)
        XCTAssertGreaterThan(AudioRecorder.remoteIDBase, 0)
    }

    // MARK: - 4. Silence gate

    /// The point of the gate: an idle conferencing channel is digital silence,
    /// and transcribing it every interval would burn GPU budget for nothing.
    func testDigitalSilenceIsSkipped() {
        let reason = AudioRecorder.remoteChunkSkipReason([Float](repeating: 0, count: 480_000))
        XCTAssertNotNil(reason)
        XCTAssertTrue(reason?.contains("near-silent") == true, "Got: \(reason ?? "nil")")
    }

    /// Ordinary speech level passes.
    func testSpeechLevelChunkIsTranscribed() {
        XCTAssertNil(AudioRecorder.remoteChunkSkipReason(tone(amplitude: 0.05, count: 480_000)))
    }

    /// The threshold itself is inclusive, and a hair below it is rejected — the
    /// exact boundary, since RMS of a constant-amplitude signal IS the amplitude.
    func testThresholdBoundary() {
        let n = 16_000
        XCTAssertNil(AudioRecorder.remoteChunkSkipReason(
            tone(amplitude: AudioRecorder.remoteSilenceRMS, count: n)),
            "At the threshold the chunk is transcribed")
        XCTAssertNotNil(AudioRecorder.remoteChunkSkipReason(
            tone(amplitude: AudioRecorder.remoteSilenceRMS * 0.5, count: n)),
            "Half the threshold is silence")
    }

    /// Erring low is deliberate: quiet-but-real speech must still be transcribed.
    func testQuietSpeechIsNotMistakenForSilence() {
        XCTAssertNil(AudioRecorder.remoteChunkSkipReason(tone(amplitude: 0.01, count: 32_000)))
    }

    /// Too short to be worth a round trip, regardless of level.
    func testVeryShortChunkIsSkipped() {
        let reason = AudioRecorder.remoteChunkSkipReason(tone(amplitude: 0.5, count: 7_999))
        XCTAssertNotNil(reason)
        XCTAssertTrue(reason?.contains("under 0.5s") == true, "Got: \(reason ?? "nil")")
        XCTAssertNil(AudioRecorder.remoteChunkSkipReason(tone(amplitude: 0.5, count: 8_000)),
                     "Exactly 0.5s is long enough")
    }

    func testEmptyChunkIsSkipped() {
        XCTAssertNotNil(AudioRecorder.remoteChunkSkipReason([]))
    }

    // MARK: - 5. Voxtral startup refusal

    /// Voxtral needs ~27 s per 30 s chunk — ~90 % duty for ONE stream, so two
    /// streams grow the sidecar queue without bound. Refuse at startup.
    func testRemotePlusVoxtralRefuses() {
        let message = AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                             chunkedModelID: "voxtral")
        XCTAssertNotNil(message)
        XCTAssertTrue(message?.contains("Voxtral") == true)
        XCTAssertTrue(message?.contains("Remote") == true,
                      "The message must name both halves of the conflict")
    }

    /// Every other chunked model is fine with two streams (4.3 / 4.2 / 5.6 s per
    /// 30 s chunk ⇒ ~29 % duty for Qwen3 even doubled).
    func testRemotePlusOtherModelsStartNormally() {
        for model in ["qwen3", "whisper", "granite"] {
            XCTAssertNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                                chunkedModelID: model),
                         "\(model) must not be refused")
        }
    }

    /// Channel 0 is a valid Remote channel — the predicate keys off nil-ness, not
    /// truthiness, so a remote on channel 0 is still refused with Voxtral.
    func testRemoteChannelZeroStillRefuses() {
        XCTAssertNotNil(AudioRecorder.dualStreamRefusalMessage(remoteChannel: 0,
                                                               chunkedModelID: "voxtral"))
    }

    /// It is a refusal, never a silent substitution: the message tells the user
    /// what to change instead of quietly picking a different model for them.
    func testRefusalTellsTheUserWhatToChange() {
        let message = AudioRecorder.dualStreamRefusalMessage(remoteChannel: 1,
                                                             chunkedModelID: "voxtral") ?? ""
        XCTAssertTrue(message.contains("Settings"), "Got: \(message)")
    }
}
