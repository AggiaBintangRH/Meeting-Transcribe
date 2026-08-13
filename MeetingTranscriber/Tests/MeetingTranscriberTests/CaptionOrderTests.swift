import XCTest
@testable import MeetingTranscriber

/// The two LIVE caption cards are ordered by who started speaking first
/// (owner, 2026-08-13).
///
/// The report: *"masih pas realtime remote terus dibawah office"* — during
/// recording the far end's caption sat under the room's, always. That was
/// literal: the view emitted the office card then the remote card, with the
/// order written into the layout and justified as *the room is the primary
/// record*. A fine rule for a TIE and wrong for everything else — the owner
/// watched the far end speak into a card pinned beneath a room caption the room
/// had not produced.
///
/// This is a SEPARATE bug from the row-ordering one fixed the same day. That was
/// confirmed rows and character-position times; this is the two provisional cards
/// below them, which never went through the merge at all.
@MainActor
final class CaptionOrderTests: XCTestCase {

    // MARK: - The rule

    /// THE REPORTED CASE: the far end started first, so it is drawn first.
    func testWhoeverStartedFirstIsDrawnFirst() {
        XCTAssertTrue(AudioRecorder.remoteCaptionComesFirst(officeStartedAt: 12.0,
                                                            remoteStartedAt: 4.0))
        XCTAssertFalse(AudioRecorder.remoteCaptionComesFirst(officeStartedAt: 4.0,
                                                             remoteStartedAt: 12.0))
    }

    /// An exact tie keeps Office on top — the one case the old fixed order got
    /// right, and worth pinning so a future "simplification" to `<=` does not
    /// silently reverse the room and the call whenever they start together.
    func testAnExactTieKeepsTheRoomFirst() {
        XCTAssertFalse(AudioRecorder.remoteCaptionComesFirst(officeStartedAt: 7.5,
                                                             remoteStartedAt: 7.5))
    }

    /// A card with no timing claim never displaces one that has one, in either
    /// direction. Both halves matter: nil-remote must not jump the queue, and
    /// nil-office must not hold the far end down — which is the original bug in
    /// miniature, since the office caption is nil exactly when the room is silent.
    func testACardWithoutATimeNeverDisplacesOneWithATime() {
        XCTAssertTrue(AudioRecorder.remoteCaptionComesFirst(officeStartedAt: nil,
                                                            remoteStartedAt: 3.0),
                      "the room has said nothing — the far end must not sit under it")
        XCTAssertFalse(AudioRecorder.remoteCaptionComesFirst(officeStartedAt: 3.0,
                                                             remoteStartedAt: nil))
        XCTAssertFalse(AudioRecorder.remoteCaptionComesFirst(officeStartedAt: nil,
                                                             remoteStartedAt: nil))
    }

    // MARK: - The timestamps the rule reads

    /// `startedAt` is when the utterance BEGAN, not when it was last updated. A
    /// caption is rewritten on every partial, so tracking the latest write would
    /// make the two cards swap places mid-sentence as each speaker typed on.
    func testTheRemoteStartTimeIsTheStartNotTheLatestUpdate() {
        var caption = AudioRecorder.RemoteCaption()
        caption.update(to: "so the", at: 4.0)
        caption.update(to: "so the budget", at: 9.0)
        caption.update(to: "so the budget is approved", at: 14.0)
        XCTAssertEqual(caption.startedAt, 4.0, "still the first partial's time")
    }

    /// Committing ends the utterance, so the next one starts a new clock rather
    /// than inheriting the old card's place in the order.
    func testCommittingClearsTheStartSoTheNextUtteranceIsItsOwn() {
        var caption = AudioRecorder.RemoteCaption()
        caption.update(to: "first", at: 4.0)
        caption.commit()
        XCTAssertNil(caption.startedAt)
        XCTAssertEqual(caption.text, "")

        caption.update(to: "second", at: 30.0)
        XCTAssertEqual(caption.startedAt, 30.0)
    }

    /// Whitespace-only text is empty text — it must not start a clock, or a card
    /// that draws nothing would still claim a position.
    func testBlankTextStartsNoClock() {
        var caption = AudioRecorder.RemoteCaption()
        caption.update(to: "   \n ", at: 4.0)
        XCTAssertNil(caption.startedAt)
        XCTAssertEqual(caption.text, "")
    }

    /// The office side keeps the same contract, through its single setter.
    func testTheOfficeCaptionTracksItsOwnStartTheSameWay() {
        let r = AudioRecorder()
        XCTAssertNil(r.partialStartedAt)

        r.recordingElapsed = 5.0
        r.setPartialTranscript("we should")
        XCTAssertEqual(r.partialStartedAt, 5.0)

        r.recordingElapsed = 11.0
        r.setPartialTranscript("we should ship it")
        XCTAssertEqual(r.partialStartedAt, 5.0, "same utterance, still its start")

        r.setPartialTranscript("")
        XCTAssertNil(r.partialStartedAt, "cleared text cannot keep a start time")
    }
}
