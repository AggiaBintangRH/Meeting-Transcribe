import Foundation

/// One speaker turn on the recording clock: who spoke, from when to when, and how
/// sure we are that it is them.
///
/// TOP-LEVEL, not nested in a service, since 2026-07-30. It used to be
/// `DiarizationService.Turn`, which tied the app's central identity type to ONE
/// diarizer — the pyannote sidecar. Splitting that sidecar into a pipeline
/// service (turns, no identity) and a WeSpeaker service (identity) made the
/// mismatch structural: the type every consumer reads is the COMPOSITION of the
/// two, so it can no longer belong to either.
///
/// What that buys, stated plainly so it is not overclaimed: identity is now
/// *structurally available* to any diarizer that can produce time spans — the
/// composition step takes local labels and returns these. It is NOT wired to
/// anything new. MOSS still labels per chunk with no cross-chunk stitching (the
/// owner deferred that, "nanti saja"), and the ATND position layer still owns its
/// own cluster ids and has no embeddings. Both remain exactly as they were.
///
/// The `id` alone says which identity space a turn belongs to:
/// pyannote < 10 000 ≤ remote < 100 000 ≤ position < 1 000 000 ≤ MOSS. See
/// `AudioRecorder+SpeakerIDSpaces`, which asserts the range at every boundary.
struct SpeakerTurn: Decodable, Sendable {
    let start: Double
    let end: Double
    let id: Int          // persistent profile id
    let name: String     // profile display name (renameable)
    /// Cosine similarity that MATCHED this voice to its saved profile, when
    /// it matched one. `nil` means the sidecar sent no `conf` key — a
    /// brand-new profile's first appearance was never scored against
    /// anything, so there is no measurement to show. Never treat `nil` as 0:
    /// "no number" and "certainly wrong" are opposite claims.
    ///
    /// Optional + defaulted on purpose: the synthesized `Decodable` then
    /// tolerates the key's absence (every pre-confidence payload still
    /// decodes), and the default keeps every existing memberwise
    /// construction — the offset map, the rename maps, the tests — compiling
    /// unchanged.
    var conf: Double? = nil
}
