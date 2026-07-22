import Foundation

// Speaker identity SPACES: the office/remote/position id ranges, the remote
// label composition, the boundary guards that keep the ranges disjoint, and the
// three-way rename that routes on the id range alone. Moved verbatim from the
// core file.
extension AudioRecorder {

    /// Rename a speaker profile and refresh all its rows in the transcript.
    ///
    /// THREE-WAY on the id range alone — the whole point of the disjoint bases
    /// (pyannote < 10_000 ≤ remote < 100_000 ≤ position). A position id must
    /// never reach the Python-owned profile stores at all; a remote id must
    /// never reach profiles.json, and an office id never profiles-remote.json.
    func renameSpeaker(id: Int, to name: String) {
        if id >= PositionDiarizer.positionIDBase {
            positionDiarizer?.rename(clusterID: id - PositionDiarizer.positionIDBase, to: name)
        } else {
            // `SpeakerProfileStore` picks the file and strips `remoteIDBase`
            // itself, so the file choice cannot drift from the id range.
            SpeakerProfileStore.rename(id: id, to: name)
        }
        // Only ONE of the two turn collections can contain this id (their ranges
        // are disjoint), so mapping both is safe and keeps the branch out.
        liveTurns = liveTurns.map {
            $0.id == id
                ? DiarizationService.Turn(start: $0.start, end: $0.end, id: $0.id, name: name)
                : $0
        }
        remoteLiveTurns = remoteLiveTurns.map {
            $0.id == id
                ? DiarizationService.Turn(start: $0.start, end: $0.end, id: $0.id, name: name)
                : $0
        }
        // Repaired/preserved pinned rows carry their own name copy — keep it in sync.
        for i in segments.indices where segments[i].pinnedSpeakerID == id {
            segments[i].pinnedSpeakerName = name
        }
        rebuildDisplayRows()
    }

    /// Prefix every Remote row's speaker label carries. Office and Remote are
    /// two SEPARATE identity spaces on purpose (owner requirement): office
    /// "Speaker 1" and remote "R1" are different people, and the label must make
    /// that impossible to misread.
    nonisolated static let remoteNamePrefix = "Remote Speaker - "

    /// Label a Remote row shows before (or without) remote diarization: the
    /// stream is known to be "not the room" and nothing more.
    nonisolated static let remoteSpeakerLabel = remoteNamePrefix + "Speaker Unknown"

    /// Compose the displayed Remote label from a remote profile name ("R1" →
    /// "Remote Speaker - R1").
    nonisolated static func remoteDisplayName(_ name: String) -> String {
        remoteNamePrefix + name
    }

    /// Inverse of `remoteDisplayName`, for the rename dialog: the user edits the
    /// PROFILE name ("R1"), not the composed row label, so the prefix is not
    /// saved into profiles-remote.json and then re-prefixed on the next rebuild.
    nonisolated static func remoteBaseName(_ display: String) -> String {
        display.hasPrefix(remoteNamePrefix)
            ? String(display.dropFirst(remoteNamePrefix.count))
            : display
    }

    /// Offset the sidecar adds to remote-local speaker ids, mirroring the proven
    /// `PositionDiarizer.positionIDBase = 100_000` split: a downstream consumer
    /// tells the three label spaces apart with one integer comparison
    /// (pyannote < 10_000 ≤ remote < 100_000 ≤ position). It is what keeps a
    /// remote id out of profiles.json and an office id out of profiles-remote.json
    /// — see `renameSpeaker` and `SpeakerProfileStore.Space`.
    nonisolated static let remoteIDBase = 10_000

    // MARK: Id-space guards (phase 5)
    //
    // The disjoint ranges above hold BY CONSTRUCTION today: office turns live in
    // `liveTurns`, remote ones in `remoteLiveTurns`, position ids never leave the
    // `filled` array in `derivedRows`. Correct, but silent if a future edit hands
    // the wrong collection to an Office-only consumer — the damage (a remote voice
    // matched onto an office profile, desyncing the Python-owned store) would not
    // surface until profiles.json was already corrupt.
    //
    // So the ranges are also checked at the boundaries that matter. The predicates
    // are pure and testable; the wrappers `assert` and are compiled out of release
    // builds, both the check and its message — a shipped app never pays for them,
    // and never crashes on them either.

    /// Ids in `turns` that are NOT office (pyannote) ids, i.e. ≥ `remoteIDBase`.
    /// Empty is the invariant every Office-only consumer relies on.
    nonisolated static func nonOfficeIDs(in turns: [DiarizationService.Turn]) -> [Int] {
        turns.map(\.id).filter { $0 >= remoteIDBase }
    }

    /// Ids in `turns` that are NOT remote ids — office ids below `remoteIDBase`
    /// or position ids at/above `PositionDiarizer.positionIDBase`. The mirror of
    /// `nonOfficeIDs`: profiles-remote.json must never see an office id either.
    nonisolated static func nonRemoteIDs(in turns: [DiarizationService.Turn]) -> [Int] {
        turns.map(\.id).filter {
            $0 < remoteIDBase || $0 >= PositionDiarizer.positionIDBase
        }
    }

    /// Pass-through that asserts (debug only) every turn is office-space.
    /// `site` names the consumer so a tripped assertion points straight at the
    /// wrong-collection call rather than at this helper.
    @discardableResult
    nonisolated static func officeTurnsOnly(_ turns: [DiarizationService.Turn],
                                            _ site: @autoclosure () -> String)
        -> [DiarizationService.Turn] {
        assert(nonOfficeIDs(in: turns).isEmpty,
               "\(site()) was given non-office speaker ids \(nonOfficeIDs(in: turns)) — "
               + "Office-only state must never see remote (≥ \(remoteIDBase)) or position "
               + "(≥ \(PositionDiarizer.positionIDBase)) ids.")
        return turns
    }

    /// The remote twin of `officeTurnsOnly`.
    @discardableResult
    nonisolated static func remoteTurnsOnly(_ turns: [DiarizationService.Turn],
                                            _ site: @autoclosure () -> String)
        -> [DiarizationService.Turn] {
        assert(nonRemoteIDs(in: turns).isEmpty,
               "\(site()) was given non-remote speaker ids \(nonRemoteIDs(in: turns)) — "
               + "remote state only holds ids in [\(remoteIDBase), "
               + "\(PositionDiarizer.positionIDBase)).")
        return turns
    }
}
