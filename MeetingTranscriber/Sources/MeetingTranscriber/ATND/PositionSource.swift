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
    /// ATND decides WHEN (its boundary timeline tiles the window, so no holes),
    /// pyannote decides WHO (each span adopts the id+name of the pyannote turn it
    /// overlaps most). The two layers stop competing over the same question:
    /// position knows the direction to the millisecond, voice knows the identity
    /// that matches a stored profile across meetings.
    ///
    /// Useful side effect: because a relabeled row carries a PYANNOTE id wherever
    /// pyannote had a turn, overlap repair and speaker profiles keep working on
    /// it — unlike `atnd`, where every row carries a position id and `applyRepair`
    /// matches nothing. A span no pyannote turn overlaps keeps its position id.
    case atndTimingPyannoteIdentity = "atndTiming"
    /// Talker direction WHILE RECORDING, then nothing once the recording stops —
    /// the finished transcript is purely the voice engine's (owner, 2026-08-18:
    /// *"pokoknya ATND gak akan digunakan ketika stop recording, full
    /// diarization model"*).
    ///
    /// THE ONLY TIME-DEPENDENT MODE, and that is why it is a case rather than a
    /// checkbox beside the others: the four above are fixed rules, so a flag that
    /// silently rewrote one of them into a schedule would make the rest untrue as
    /// written. Here the schedule IS the mode.
    ///
    /// It exists because the batch engines have no live labels — with CAM++ or
    /// spectral the room has no names at all until Stop — so direction is the
    /// only thing that can fill the meeting as it happens, while the whole-file
    /// pass is the answer anyone keeps.
    ///
    /// ⚠ IT IS NOT THE SAME AS `both`, and the difference is the whole request.
    /// `both` also lets voice win, but leaves direction filling whatever voice did
    /// NOT cover. This drops the position layer entirely once recording ends, so a
    /// stretch the engine never labelled falls back to the nearest voice turn
    /// instead of to a seat — see `derivedRows`'s `filled.isEmpty` branch, which
    /// is what stops it becoming SPEAKER UNKNOWN.
    case atndLiveOnly = "atndLive"

    static let defaultsKey = "atnd.position.source"

    /// Read the owner's choice. Unknown/absent value → `both`, so a missing key
    /// (every existing install) keeps today's behaviour.
    static func current(_ defaults: UserDefaults = .standard) -> PositionSource {
        PositionSource(rawValue: defaults.string(forKey: defaultsKey) ?? "") ?? .both
    }

    /// True when the ATND layer contributes anything to the display at all —
    /// gates the live-partial position label alongside the gap-fill itself.
    /// True when the ATND layer contributes anything to the display at all —
    /// gates the live-partial position label alongside the gap-fill itself.
    ///
    /// `atndLiveOnly` is TRUE here on purpose: while recording it behaves exactly
    /// like `both`, and the live caption is the surface that mode exists to serve.
    /// What changes at Stop is handled by `effective(recording:)`, not here.
    var usesPosition: Bool { self != .pyannote }

    /// The mode that actually applies right now.
    ///
    /// One place resolves the schedule, so `derivedRows`, the live caption and any
    /// future reader cannot disagree about when the position layer stops counting
    /// — the two-readers-of-one-setting shape this project has been bitten by more
    /// than once.
    /// ⚠ IT RESOLVES TO ONE OF THE FOUR FIXED MODES, NEVER BACK TO ITSELF. That
    /// is the property worth having: after this call no time-dependent mode
    /// remains anywhere downstream, so `plan`, `usesPosition` and every future
    /// reader only ever handle rules that mean one thing. Returning `self` while
    /// recording — the first version — type-checked and behaved correctly, and
    /// still let the schedule leak past the one place that is supposed to end it.
    func effective(recording: Bool) -> PositionSource {
        guard self == .atndLiveOnly else { return self }
        return recording ? .both : .pyannote
    }
}

/// A labeled stretch of recording time: the tuple shape `speakerRanges`,
/// `positionGapFill` and `derivedRows` already pass around. Named so the
/// coverage plan below can be written (and tested) without repeating it.
typealias LabeledRange = (start: Double, end: Double, id: Int, name: String)

/// What one `derivedRows` call should hand to the display path, decided purely
/// from the selected source and the pyannote ranges for that window.
///
/// Deliberately a plan value rather than inline branching — which is exactly how
/// the fourth mode ("ATND holds the boundaries, pyannote only renames them")
/// landed: one more `case` in `plan(pyannoteRanges:)` plus one flag, and the only
/// thing `derivedRows` gained is a single `relabel` step.
struct PositionCoveragePlan {
    /// pyannote ranges that appear in the rendered rows.
    let displayRanges: [LabeledRange]
    /// Coverage handed to `positionGapFill` — its complement over the window is
    /// what ATND fills. `nil` means "don't gap-fill at all" (no position labels).
    let gapFillCoverage: [LabeledRange]?
    /// Rename the assembled ranges from pyannote, keeping their ATND boundaries.
    /// Only `atndTimingPyannoteIdentity` sets it; `derivedRows` applies it as one
    /// step right after `filled` is built, so no mode needs a special case there.
    var relabelFromPyannote = false
}

extension PositionSource {
    /// The single switch every mode goes through.
    ///
    /// - `both`: pyannote shows, ATND fills pyannote's own complement — today.
    /// - `atnd`: nothing pyannote shows, and the coverage passed to the gap-fill
    ///   is EMPTY, so its complement is the entire window and the position spans
    ///   tile all of it. pyannote is not consulted for display, only skipped.
    /// - `pyannote`: pyannote shows, no gap-fill runs.
    /// - `atndTimingPyannoteIdentity`: the SAME coverage as `atnd` — position
    ///   tiles the whole window — plus the relabel flag, so the spans keep their
    ///   ATND boundaries but take pyannote's identity.
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
        case .atndTimingPyannoteIdentity:
            return PositionCoveragePlan(displayRanges: [], gapFillCoverage: [],
                                        relabelFromPyannote: true)
        case .atndLiveOnly:
            // WHILE RECORDING it is `both`, verbatim. After Stop this case is
            // never reached, because `effective(recording:)` has already resolved
            // it to `.pyannote` — resolving in ONE place is what keeps the two
            // halves of a time-dependent mode from drifting apart.
            //
            // Falling through to `both`'s plan rather than repeating the literal:
            // if `both` is ever retuned, this mode must move with it, and a copy
            // would silently not.
            return PositionSource.both.plan(pyannoteRanges: pyannoteRanges)
        }
    }
}

/// The relabel step of `atndTimingPyannoteIdentity`, factored out of
/// `derivedRows` so it can be tested without app state.
enum PositionRelabel {
    /// Give each range in `ranges` the id and name of the pyannote turn it
    /// overlaps MOST in time, leaving its start/end alone. Greatest overlap (not
    /// midpoint, not first hit) because an ATND span can straddle a pyannote
    /// boundary, and the speaker who held most of the span is the one the words
    /// in it mostly belong to.
    ///
    /// A range no pyannote turn overlaps keeps its own id and name — which for a
    /// gap-fill span is a POSITION id. That is deliberate: pyannote genuinely has
    /// no answer there (freshly-committed text, the first 1–2 s it never turns),
    /// and inventing one would be worse than showing the seat. Those ids stay
    /// inside `derivedRows`' local `filled` array exactly as in `both`/`atnd`.
    static func fromPyannote(_ ranges: [LabeledRange],
                             pyannote: [LabeledRange]) -> [LabeledRange] {
        guard !pyannote.isEmpty else { return ranges }
        return ranges.map { r in
            var bestID = r.id
            var bestName = r.name
            var bestOverlap = 0.0
            for p in pyannote {
                let o = min(r.end, p.end) - max(r.start, p.start)
                if o > bestOverlap {
                    bestOverlap = o
                    bestID = p.id
                    bestName = p.name
                }
            }
            return (start: r.start, end: r.end, id: bestID, name: bestName)
        }
    }
}
