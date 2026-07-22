import Foundation

/// Swift-side view of the speaker-profile store (models/speaker-profiles/).
/// Python owns the voice embeddings; Swift reads and renames the display
/// names in profiles.json (the sidecar reloads names before every job).
struct SpeakerProfile: Identifiable, Codable, Equatable {
    let id: Int
    var name: String
    var count: Int
}

enum SpeakerProfileStore {

    /// The two identity spaces the sidecar keeps, each in its own pair of files.
    /// They are deliberately NOT one store with a name prefix: `ProfileStore.assign()`
    /// scores an incoming embedding against every centroid it holds, so sharing a
    /// file would let a remote voice match an office profile. Separate files make
    /// that impossible rather than merely filtered.
    ///
    /// Remote ids arrive from the sidecar offset by `AudioRecorder.remoteIDBase`;
    /// `localID(_:)` takes the offset back off before anything is written, so
    /// profiles-remote.json only ever contains the sidecar's own local ids.
    enum Space {
        case office
        case remote

        var fileName: String {
            switch self {
            case .office: return "profiles.json"
            case .remote: return "profiles-remote.json"
            }
        }

        /// Wire id → the id as stored on disk for this space.
        func localID(_ id: Int) -> Int {
            self == .remote ? id - AudioRecorder.remoteIDBase : id
        }

        /// Which space a wire id belongs to (pyannote < 10_000 ≤ remote).
        /// Position ids (≥ 100_000) never reach here — `renameSpeaker` routes
        /// them to the position diarizer before this type is consulted. The
        /// assert is that routing's tripwire: a position id arriving here would
        /// be filed as a remote profile and quietly corrupt profiles-remote.json.
        /// Debug-only (compiled out of release) — a mis-filed rename is a bug to
        /// catch in development, not a reason to crash on a user's meeting.
        static func forWireID(_ id: Int) -> Space {
            assert(isProfileID(id),
                   "position id \(id) reached SpeakerProfileStore — renameSpeaker "
                   + "must branch on PositionDiarizer.positionIDBase first.")
            return id >= AudioRecorder.remoteIDBase ? .remote : .office
        }
    }

    /// True when `id` belongs to one of the two voice-profile stores at all.
    /// Position ids are not voice profiles — they are ATND beam clusters, owned
    /// by `PositionDiarizer` and stored nowhere near profiles.json. Pure, so the
    /// boundary can be tested without tripping the assert that enforces it.
    static func isProfileID(_ id: Int) -> Bool {
        id < PositionDiarizer.positionIDBase
    }

    static var fileURL: URL { fileURL(.office) }

    static func fileURL(_ space: Space) -> URL {
        PythonRuntime.profilesDir.appendingPathComponent(space.fileName)
    }

    static func load(_ space: Space = .office) -> [SpeakerProfile] {
        guard let data = try? Data(contentsOf: fileURL(space)),
              let profiles = try? JSONDecoder().decode([SpeakerProfile].self, from: data)
        else { return [] }
        return profiles.sorted { $0.id < $1.id }
    }

    /// Rename by WIRE id — the space and the on-disk id are both derived from it,
    /// so a caller can never write a remote id into profiles.json by mistake.
    static func rename(id: Int, to name: String) {
        let space = Space.forWireID(id)
        var profiles = load(space)
        guard let index = profiles.firstIndex(where: { $0.id == space.localID(id) }) else { return }
        profiles[index].name = name.trimmingCharacters(in: .whitespaces)
        save(profiles, space)
    }

    static func delete(id: Int) {
        // Removes the name entry; the embedding remains until python resaves,
        // after which the voice will be re-enrolled as a new speaker.
        let space = Space.forWireID(id)
        save(load(space).filter { $0.id != space.localID(id) }, space)
    }

    private static func save(_ profiles: [SpeakerProfile], _ space: Space) {
        guard let data = try? JSONEncoder().encode(profiles) else { return }
        let url = fileURL(space)
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url)
    }
}
