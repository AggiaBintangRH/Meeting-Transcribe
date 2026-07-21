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
        XCTAssertTrue(PositionSource.atndTimingPyannoteIdentity.usesPosition,
                      "the boundaries come from ATND, so the position layer must run")
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

    // MARK: - 4. `atndTiming` — ATND boundaries, pyannote identity

    /// Coverage is the same as `atnd` (position tiles the whole window); the only
    /// difference is the relabel flag, and only this mode sets it.
    func testAtndTimingSelectsFullWindowCoveragePlusRelabel() {
        let plan = PositionSource.atndTimingPyannoteIdentity.plan(pyannoteRanges: pyannote)
        XCTAssertTrue(plan.displayRanges.isEmpty)
        XCTAssertTrue(plan.gapFillCoverage?.isEmpty ?? false,
                      "empty covered set ⇒ ATND tiles the window, as in `atnd`")
        XCTAssertTrue(plan.relabelFromPyannote)
        for other in [PositionSource.both, .atnd, .pyannote] {
            XCTAssertFalse(other.plan(pyannoteRanges: pyannote).relabelFromPyannote,
                           "\(other.rawValue) must not relabel")
        }
    }

    /// A position span sitting inside one pyannote turn adopts that turn's id and
    /// name — and keeps its OWN start/end, which is the whole point of the mode.
    func testRelabelAdoptsTheOverlappingTurnKeepingBoundaries() {
        let spans = [range(1, 2.5, 100_000, "Position 1")]
        let out = PositionRelabel.fromPyannote(spans, pyannote: pyannote)
        assertSame(out, [range(1, 2.5, 1, "Speaker 1")],
                   "must take pyannote's identity at ATND's boundaries")
    }

    /// Straddling two turns → the one with the GREATER overlap wins (1.0s of
    /// Speaker 1 vs 2.0s of Speaker 2), not the first or the earlier one.
    func testRelabelPicksTheGreaterOverlap() {
        let spans = [range(2, 7, 100_001, "Position 2")]
        let out = PositionRelabel.fromPyannote(spans, pyannote: pyannote)
        XCTAssertEqual(out.first?.id, 2)
        XCTAssertEqual(out.first?.name, "Speaker 2")
        // Flip the balance and the answer flips with it.
        let flipped = PositionRelabel.fromPyannote([range(0.5, 5.5, 100_001, "Position 2")],
                                                   pyannote: pyannote)
        XCTAssertEqual(flipped.first?.id, 1)
    }

    /// The hole between the two pyannote turns: nothing overlaps, so the span
    /// keeps its POSITION id and name. Honest — pyannote has no answer there.
    func testRelabelKeepsThePositionIDWhereNoTurnOverlaps() {
        let spans = [range(3.2, 4.8, 100_002, "Position 3")]
        let out = PositionRelabel.fromPyannote(spans, pyannote: pyannote)
        assertSame(out, spans, "an uncovered span must keep its own label")
        XCTAssertGreaterThanOrEqual(out[0].id, PositionDiarizer.positionIDBase)
    }

    /// No pyannote at all (nothing diarized yet) → every span is untouched, so
    /// the mode degrades exactly to `atnd` rather than blanking the labels.
    func testRelabelWithNoPyannoteIsIdentity() {
        let spans = [range(0, 3, 100_000, "Position 1"), range(3, 6, 100_001, "Position 2")]
        assertSame(PositionRelabel.fromPyannote(spans, pyannote: []), spans,
                   "no pyannote turns ⇒ nothing to adopt")
    }

    /// The id invariant: relabeling can only ever move an id pyannote → display.
    /// Every id in the result is either one of pyannote's own or the span's
    /// original — the reverse direction (a position id reaching pyannote's world)
    /// is structurally impossible because `pyannote` is only ever read here.
    func testRelabelNeverInventsAnID() {
        let spans = [range(0, 2, 100_000, "Position 1"),
                     range(2, 4, 100_001, "Position 2"),
                     range(4, 9, 100_002, "Position 3")]
        let allowed = Set(pyannote.map(\.id) + spans.map(\.id))
        for r in PositionRelabel.fromPyannote(spans, pyannote: pyannote) {
            XCTAssertTrue(allowed.contains(r.id), "invented id \(r.id)")
        }
    }

    // MARK: - 5. Defaults reading

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
        d.set("atndTiming", forKey: PositionSource.defaultsKey)
        XCTAssertEqual(PositionSource.current(d), .atndTimingPyannoteIdentity)
    }
}
