import XCTest
@testable import MeetingTranscriber

/// Unit tests for `ATNDBeamService.parse` — the wire format of the array's
/// multicast notices.
///
/// These exist because of a real, total failure on the owner's ATND1061: the app
/// received nothing at all while a Python receiver on the same wire printed every
/// packet. The cause was one byte. The array puts a SPACE before the CR (the
/// silence notice measures 44 bytes on the wire for 42 bytes of text), our
/// simulator did not send it, and the parser's exact-token-count check turned
/// that extra empty token into a silent reject of EVERY notice.
///
/// So the bar here is the bytes the device actually sends, not the bytes our
/// simulator found convenient. Section 1 is that bar; the simulator was fixed to
/// match in the same change.
final class ATNDNoticeParseTests: XCTestCase {

    // MARK: - 1. Real device wire format (trailing space before CR)

    func testTalkingNoticeWithTrailingSpaceIsParsed() {
        // Captured shape from the owner's array: beam 0, angle 77, rotate 62.
        let notice = ATNDBeamService.parse("MD camera_control_notice 0000 00 NC 1,0,77,62,0 ")
        XCTAssertEqual(notice, .talking(.init(channel: 1, elevation: 77, rotation: 62)),
                       "A trailing space before CR is what the real array sends — it must parse.")
    }

    func testSilenceNoticeWithTrailingSpaceIsParsed() {
        let notice = ATNDBeamService.parse("MD camera_control_notice 0000 00 NC 0,,,,0 ")
        XCTAssertEqual(notice, .silence)
    }

    /// The exact byte count that exposed the bug: 42 chars of text, then space,
    /// then CR. If this drifts, the arithmetic in the parse comment is wrong too.
    func testSilenceNoticeIsFortyFourBytesOnTheWire() {
        let text = "MD camera_control_notice 0000 00 NC 0,,,,0"
        XCTAssertEqual(text.count, 42)
        XCTAssertEqual((text + " \r").utf8.count, 44)
    }

    // MARK: - 2. Still accepts the form without the trailing space

    func testTalkingNoticeWithoutTrailingSpaceStillParses() {
        let notice = ATNDBeamService.parse("MD camera_control_notice 0000 00 NC 1,5,30,180,0")
        XCTAssertEqual(notice, .talking(.init(channel: 6, elevation: 30, rotation: 180)),
                       "Tolerating both forms costs nothing and avoids betting on one spelling.")
    }

    func testSilenceNoticeWithoutTrailingSpaceStillParses() {
        XCTAssertEqual(ATNDBeamService.parse("MD camera_control_notice 0000 00 NC 0,,,,0"), .silence)
    }

    // MARK: - 3. Channel is converted to the human-facing 1...6

    func testWireChannelZeroBecomesBeamOne() {
        XCTAssertEqual(ATNDBeamService.parse("MD camera_control_notice 0000 00 NC 1,0,10,20,0 "),
                       .talking(.init(channel: 1, elevation: 10, rotation: 20)))
    }

    func testWireChannelFiveBecomesBeamSix() {
        XCTAssertEqual(ATNDBeamService.parse("MD camera_control_notice 0000 00 NC 1,5,10,20,0 "),
                       .talking(.init(channel: 6, elevation: 10, rotation: 20)))
    }

    func testOutOfRangeChannelIsRejected() {
        XCTAssertNil(ATNDBeamService.parse("MD camera_control_notice 0000 00 NC 1,6,10,20,0 "))
    }

    // MARK: - 4. Other traffic on the same socket is ignored, not misread

    func testLevelMeterNoticeIsIgnored() {
        // Arrives on the same socket at 10 Hz; must be a quiet nil, not an error.
        let line = "MD level_meter_notice 0000 00 NC 25,0,0,0,0,0,0,,,,,,0,0,0,,,,,,,,0,,,,,,,,,,15,15,15,15,15,15,,,, "
        XCTAssertNil(ATNDBeamService.parse(line))
    }

    func testGarbageIsRejected() {
        XCTAssertNil(ATNDBeamService.parse(""))
        XCTAssertNil(ATNDBeamService.parse("   "))
        XCTAssertNil(ATNDBeamService.parse("MD camera_control_notice 0000 00 NC"))
        XCTAssertNil(ATNDBeamService.parse("MD camera_control_notice 0000 00 XX 1,0,77,62,0 "))
        XCTAssertNil(ATNDBeamService.parse("MD camera_control_notice 0000 00 NC 1,0,77 "))
    }
}
