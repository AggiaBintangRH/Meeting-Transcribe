import Foundation

/// Which diarization layer the transcript's speaker labels are DRAWN from.
///
/// The two layers answer different questions and can honestly disagree: pyannote
/// answers "who is this voice", the ATND beam answers "which seat is this coming
/// from". Moving one person between three seats is one speaker to pyannote and
/// three positions to ATND — both correct. The default fusion (`both`) lets
/// pyannote win wherever it has a turn, which makes the position layer invisible
/// exactly when the two disagree. This selector exists so either layer can be
/// isolated for validation and per-room calibration.
///
/// DISPLAY ONLY. Nothing here changes what pyannote computes: `liveTurns`,
/// `overlapRegions`, `repairWindows`, `speakerCount` and the Python-owned
/// `SpeakerProfileStore` stay pure pyannote in every mode. Position ids
/// (>= `PositionDiarizer.positionIDBase`) exist only inside the local `filled`
/// array in `derivedRows` — feeding one into the sidecar's world would desync its
/// embedding indices.
enum PositionSource: String, CaseIterable {
    /// pyannote authoritative, ATND fills only what pyannote left uncovered.
    /// The default, and byte-identical to the behaviour before this setting.
    case both
    /// ATND position spans cover the whole window; pyannote still runs untouched
    /// underneath, its labels just aren't displayed.
    case atnd
    /// Pure pyannote — no gap-fill, no position label on the live partial.
    case pyannote

    static let defaultsKey = "atnd.position.source"

    /// Read the owner's choice. Unknown/absent value → `both`, so a missing key
    /// (every existing install) keeps today's behaviour.
    static func current(_ defaults: UserDefaults = .standard) -> PositionSource {
        PositionSource(rawValue: defaults.string(forKey: defaultsKey) ?? "") ?? .both
    }

    /// True when the ATND layer contributes anything to the display at all —
    /// gates the live-partial position label alongside the gap-fill itself.
    var usesPosition: Bool { self != .pyannote }
}

/// A labeled stretch of recording time: the tuple shape `speakerRanges`,
/// `positionGapFill` and `derivedRows` already pass around. Named so the
/// coverage plan below can be written (and tested) without repeating it.
typealias LabeledRange = (start: Double, end: Double, id: Int, name: String)

/// What one `derivedRows` call should hand to the display path, decided purely
/// from the selected source and the pyannote ranges for that window.
///
/// Deliberately a plan value rather than inline branching: the deferred fourth
/// mode ("ATND holds the boundaries, pyannote only renames them") becomes one
/// more `case` in `plan(for:pyannoteRanges:)` and nothing in `derivedRows` moves.
struct PositionCoveragePlan {
    /// pyannote ranges that appear in the rendered rows.
    let displayRanges: [LabeledRange]
    /// Coverage handed to `positionGapFill` — its complement over the window is
    /// what ATND fills. `nil` means "don't gap-fill at all" (no position labels).
    let gapFillCoverage: [LabeledRange]?
}

extension PositionSource {
    /// The single switch every mode goes through.
    ///
    /// - `both`: pyannote shows, ATND fills pyannote's own complement — today.
    /// - `atnd`: nothing pyannote shows, and the coverage passed to the gap-fill
    ///   is EMPTY, so its complement is the entire window and the position spans
    ///   tile all of it. pyannote is not consulted for display, only skipped.
    /// - `pyannote`: pyannote shows, no gap-fill runs.
    func plan(pyannoteRanges: [LabeledRange]) -> PositionCoveragePlan {
        switch self {
        case .both:
            return PositionCoveragePlan(displayRanges: pyannoteRanges,
                                        gapFillCoverage: pyannoteRanges)
        case .atnd:
            return PositionCoveragePlan(displayRanges: [], gapFillCoverage: [])
        case .pyannote:
            return PositionCoveragePlan(displayRanges: pyannoteRanges,
                                        gapFillCoverage: nil)
        }
    }
}
