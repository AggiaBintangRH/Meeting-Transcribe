import Foundation

/// Recording-time seconds as `mm:ss`, in ONE place.
///
/// There were two of these until the 2026-08-10 cleanup — `TranscriptView.clock`
/// and `TranscriptPDF.clock` — and they had already drifted: `%02d:%02d` on
/// screen against `%d:%02d` in the exported PDF, so the same measurement printed
/// as `03:07` in the app and `3:07` in the document the client keeps. Nobody
/// chose that; it is what two copies do.
///
/// Both copies also rendered past the hour as `72:05`. That is UNCHANGED here
/// rather than quietly fixed: no recording in this project has reached an hour of
/// a single speaker turn, `mm:ss` is what both surfaces have always shown, and a
/// tidy-up is the wrong moment to alter what a document says. When it matters,
/// this is the one function to change.
enum Clock {

    /// `mm:ss`, zero-padded, rounded to the nearest second.
    static func mmss(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}
