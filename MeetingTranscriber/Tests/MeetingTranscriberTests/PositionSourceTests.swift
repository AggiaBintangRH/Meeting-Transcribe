import XCTest
@testable import MeetingTranscriber

/// Unit tests for the diarization source selector (`PositionSource`) — the pure
/// coverage decision `derivedRows` makes once per segment. No app state, no GUI:
/// `plan(pyannoteRanges:)` is the whole switch, so the modes are assertable
/// directly.
///
/// The regression bar for this feature is that `both` stays byte-identical to the
/// pre-selector behaviour (pyannote shows, ATND fills pyannote's own complement);
/// section 1 is that bar.
final class PositionSourceTests: XCTestCase {

    // MARK: - Helpers

    private func range(_ start: Double, _ end: Double, _ id: Int, _ name: String) -> LabeledRange {
        (start: start, end: end, id: id, name: name)
    }

    /// Two pyannote turns with a 2s hole between them — the shape the gap-fill
    /// exists for.
    private var pyannote: [LabeledRange] {
        [range(0, 3, 1, "Speaker 1"), range(5, 8, 2, "Speaker 2")]
    }

    private func assertSame(_ got: [LabeledRange], _ want: [LabeledRange],
                            _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, message, file: file, line: line)
        guard got.count == want.count else { return }
        for (g, w) in zip(got, want) {
            XCTAssertEqual(g.start, w.start, accuracy: 1e-9, message, file: file, line: line)
            XCTAssertEqual(g.end, w.end, accuracy: 1e-9, message, file: file, line: line)
            XCTAssertEqual(g.id, w.id, message, file: file, line: line)
            XCTAssertEqual(g.name, w.name, message, file: file, line: line)
        }
    }

    // MARK: - 1. `both` — today's behaviour, unchanged

    /// `both` displays pyannote AND hands pyannote's own ranges to the gap-fill,
    /// so ATND fills exactly the complement. This is the pre-selector code path
    /// (`positionGapFill(window:covered: ranges)` + `ranges + fills`).
    func testBothSelectsTodaysCoverage() {
        let plan = PositionSource.both.plan(pyannoteRanges: pyannote)
        assertSame(plan.displayRanges, pyannote, "both must still display pyannote")
        XCTAssertNotNil(plan.gapFillCoverage, "both must still gap-fill")
        assertSame(plan.gapFillCoverage ?? [], pyannote,
                   "both must hand pyannote's own ranges in as the covered set")
    }

    /// Empty pyannote (nothing diarized yet — the freshly-committed text case)
    /// still gap-fills, with an empty covered set, so ATND tiles the window.
    func testBothWithNoPyannoteStillFillsEverything() {
        let plan = PositionSource.both.plan(pyannoteRanges: [])
        XCTAssertTrue(plan.displayRanges.isEmpty)
        XCTAssertEqual(plan.gapFillCoverage?.count, 0)
    }

    // MARK: - 2. `pyannote` — no position anywhere

    /// `nil` coverage is the "don't gap-fill at all" signal — `derivedRows` maps
    /// over it, so no position id is even produced, let alone displayed.
    func testPyannoteSelectsNoCoverage() {
        let plan = PositionSource.pyannote.plan(pyannoteRanges: pyannote)
        XCTAssertNil(plan.gapFillCoverage, "voice-only must not run the gap-fill")
        assertSame(plan.displayRanges, pyannote, "voice-only still displays pyannote")
    }

    /// The live-partial label is gated by the same flag, so it can't leak a
    /// position name into a mode that shows none.
    func testUsesPositionFlag() {
        XCTAssertTrue(PositionSource.both.usesPosition)
        XCTAssertTrue(PositionSource.atnd.usesPosition)
        XCTAssertFalse(PositionSource.pyannote.usesPosition)
    }

    // MARK: - 3. `atnd` — position tiles the whole window

    /// Empty coverage means the complement `positionGapFill` walks is the ENTIRE
    /// window, and no pyannote range is displayed. pyannote itself is untouched —
    /// the ranges are simply not consulted for display.
    func testAtndSelectsFullWindowCoverage() {
        let plan = PositionSource.atnd.plan(pyannoteRanges: pyannote)
        XCTAssertTrue(plan.displayRanges.isEmpty, "position-only must not display pyannote rows")
        XCTAssertNotNil(plan.gapFillCoverage)
        XCTAssertTrue(plan.gapFillCoverage?.isEmpty ?? false,
                      "empty covered set ⇒ the complement is the whole window")
    }

    /// The plan never invents, reorders or mutates ranges — every mode's display
    /// output is either the input verbatim or empty. Guards the invariant that a
    /// position id can only ever enter the rows via `fills`.
    func testPlanNeverFabricatesRanges() {
        for source in PositionSource.allCases {
            let plan = source.plan(pyannoteRanges: pyannote)
            XCTAssertTrue(plan.displayRanges.isEmpty || plan.displayRanges.count == pyannote.count,
                          "\(source.rawValue) displayRanges must be pyannote's own or empty")
            for r in plan.displayRanges {
                XCTAssertLessThan(r.id, PositionDiarizer.positionIDBase,
                                  "\(source.rawValue) leaked a position id into displayRanges")
            }
            for r in plan.gapFillCoverage ?? [] {
                XCTAssertLessThan(r.id, PositionDiarizer.positionIDBase,
                                  "\(source.rawValue) leaked a position id into the covered set")
            }
        }
    }

    // MARK: - 4. Defaults reading

    /// A missing key is every existing install — it must land on `both`, and an
    /// unrecognised value must not silently change the policy either.
    func testCurrentDefaultsToBoth() {
        let d = UserDefaults(suiteName: "PositionSourceTests.\(UUID().uuidString)")!
        XCTAssertEqual(PositionSource.current(d), .both, "absent key")
        d.set("nonsense", forKey: PositionSource.defaultsKey)
        XCTAssertEqual(PositionSource.current(d), .both, "unknown value")
        d.set("atnd", forKey: PositionSource.defaultsKey)
        XCTAssertEqual(PositionSource.current(d), .atnd)
        d.set("pyannote", forKey: PositionSource.defaultsKey)
        XCTAssertEqual(PositionSource.current(d), .pyannote)
    }
}
