import XCTest
@testable import MeetingTranscriber

/// Unit tests for live realtime captions on the Remote stream.
///
/// Two decisions are genuinely pure here and both are tested: whether a session
/// wants a second realtime engine at all (`ModelLoader.wantsRemoteRealtime`), and
/// what the caption shows as realtime results arrive and confirmed remote chunks
/// land (`AudioRecorder.RemoteCaption`). Everything else this change touches —
/// the second sidecar process, the tap feeding it, the flushes at the chunk
/// boundary and at Stop — needs the Aggregate Device and a running sidecar, and
/// is listed as an on-device check instead.
@MainActor
final class DualStreamRealtimeTests: XCTestCase {

    // MARK: - 1. The unset path is inert (the regression bar)

    /// With no Remote channel configured, no second realtime engine is ever
    /// wanted — whatever the realtime setting says. This is the predicate the
    /// loader uses both to decide the extra loading step and to tear a leftover
    /// engine down, so a single-stream session cannot start (or keep) one.
    func testNoRemoteChannelNeverWantsARemoteRealtimeEngine() {
        XCTAssertFalse(ModelLoader.wantsRemoteRealtime(remoteChannel: nil,
                                                       realtimeEnabled: true))
        XCTAssertFalse(ModelLoader.wantsRemoteRealtime(remoteChannel: nil,
                                                       realtimeEnabled: false))
    }

    /// With realtime captions switched off there is nothing to draw for either
    /// stream, so a Remote channel alone is not enough.
    func testRealtimeOffNeverWantsARemoteRealtimeEngine() {
        XCTAssertFalse(ModelLoader.wantsRemoteRealtime(remoteChannel: 1,
                                                       realtimeEnabled: false))
    }

    /// Both conditions together — and channel 0 is a valid channel, so the
    /// predicate must test for nil rather than for truthiness.
    func testRemoteChannelPlusRealtimeWantsTheEngine() {
        XCTAssertTrue(ModelLoader.wantsRemoteRealtime(remoteChannel: 1,
                                                      realtimeEnabled: true))
        XCTAssertTrue(ModelLoader.wantsRemoteRealtime(remoteChannel: 0,
                                                      realtimeEnabled: true))
    }

    /// A fresh recorder shows no remote caption, so a single-stream session
    /// renders exactly the cards it rendered before this feature existed.
    func testRecorderStartsWithNoRemoteCaption() {
        XCTAssertTrue(AudioRecorder().remoteCaption.text.isEmpty)
    }

    // MARK: - 2. The caption rule

    func testPartialsReplaceEachOther() {
        var caption = AudioRecorder.RemoteCaption()
        caption.update(to: "so the")
        caption.update(to: "so the budget")
        XCTAssertEqual(caption.text, "so the budget")
    }

    /// The rule that differs from Office: a final is KEPT. Office turns its final
    /// into an unconfirmed segment immediately, so it can clear the partial;
    /// Remote has no unconfirmed segment, so clearing on the final would blank
    /// the caption for the seconds the confirmed remote chunk takes to return.
    func testFinalTextStaysOnScreenUntilTheChunkCommits() {
        var caption = AudioRecorder.RemoteCaption()
        caption.update(to: "so the budget is approved")   // the flush's final
        XCTAssertEqual(caption.text, "so the budget is approved")
        caption.commit()                                   // the confirmed row landed
        XCTAssertTrue(caption.text.isEmpty)
    }

    /// The replacement rule from the other side: once a window is settled the
    /// caption goes, whether that window produced a row (transcribed), produced
    /// nothing (empty transcript, or skipped as silence), or failed. `commit` is
    /// the single call every one of those paths makes.
    func testCommitClearsRegardlessOfOutcome() {
        for text in ["some remote speech", "", "   "] {
            var caption = AudioRecorder.RemoteCaption()
            caption.update(to: text)
            caption.commit()
            XCTAssertTrue(caption.text.isEmpty)
        }
    }

    func testCommitIsIdempotent() {
        var caption = AudioRecorder.RemoteCaption()
        caption.update(to: "hello")
        caption.commit()
        caption.commit()
        XCTAssertTrue(caption.text.isEmpty)
    }

    /// An engine that emits nothing (or whitespace) must not leave an empty amber
    /// card behind — the view draws the card only for non-empty text.
    func testWhitespaceOnlyResultsRenderNothing() {
        var caption = AudioRecorder.RemoteCaption()
        caption.update(to: "hello")
        caption.update(to: "   \n ")
        XCTAssertTrue(caption.text.isEmpty)
    }

    /// Trimming is applied on the way in, so the view never has to.
    func testTextIsTrimmed() {
        var caption = AudioRecorder.RemoteCaption()
        caption.update(to: "  hello from the call\n")
        XCTAssertEqual(caption.text, "hello from the call")
    }

    /// The view keys its scroll-to-bottom off the whole caption value, so it has
    /// to compare by content.
    func testCaptionIsEquatableByContent() {
        var a = AudioRecorder.RemoteCaption()
        var b = AudioRecorder.RemoteCaption()
        XCTAssertEqual(a, b)
        a.update(to: "x")
        XCTAssertNotEqual(a, b)
        b.update(to: "x")
        XCTAssertEqual(a, b)
    }
}
