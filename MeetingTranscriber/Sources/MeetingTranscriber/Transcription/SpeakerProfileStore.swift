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

    static var fileURL: URL {
        PythonRuntime.profilesDir
            .appendingPathComponent("profiles.json")
    }

    static func load() -> [SpeakerProfile] {
        guard let data = try? Data(contentsOf: fileURL),
              let profiles = try? JSONDecoder().decode([SpeakerProfile].self, from: data)
        else { return [] }
        return profiles.sorted { $0.id < $1.id }
    }

    static func rename(id: Int, to name: String) {
        var profiles = load()
        guard let index = profiles.firstIndex(where: { $0.id == id }) else { return }
        profiles[index].name = name.trimmingCharacters(in: .whitespaces)
        save(profiles)
    }

    static func delete(id: Int) {
        // Removes the name entry; the embedding remains until python resaves,
        // after which the voice will be re-enrolled as a new speaker.
        save(load().filter { $0.id != id })
    }

    private static func save(_ profiles: [SpeakerProfile]) {
        guard let data = try? JSONEncoder().encode(profiles) else { return }
        try? FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: fileURL)
    }
}
