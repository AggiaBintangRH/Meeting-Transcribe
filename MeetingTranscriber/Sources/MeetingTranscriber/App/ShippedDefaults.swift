import Foundation

/// The configuration a FRESH INSTALL starts in — one statement of each value,
/// read by every site that needs it.
///
/// ⚠ WHY A TYPE AND NOT A LITERAL AT EACH READER. Before this existed, the
/// shipped default for `overlap.detect.enabled` was written **eight** times:
/// three `@AppStorage(...) = false` declarations in Settings and five
/// `d.object(forKey:) as? Bool ?? false` reads in `ModelLoader` and
/// `AudioRecorder`. `chunked.model` was written **twelve** times. Changing a
/// default therefore meant finding every copy, and missing one produces the
/// failure this project has hit repeatedly and documents under its own heading:
/// two readers of one fact, disagreeing. Here it would have been silent and
/// user-visible in the worst way — the Settings toggle drawn ON while the
/// loader, reading its own literal, never loaded the model.
///
/// ⚠ THESE ARE DEFAULTS, NOT PINS. Each applies only while its key is absent,
/// which is exactly what "a fresh Mac needs no setting up" means. A value the
/// user has chosen is never overwritten from here — the one migration that
/// deliberately overwrites a stored choice is `adoptMeasuredEngineDefaults`,
/// and it now reads these same constants so the two cannot drift apart.
///
/// ⚠ DELIBERATELY ABSENT, and each for its own reason:
///
///   `mic.deviceUID`          machine-specific. This Mac captures through
///                            BlackHole; the client's captures through
///                            Dante/Q-SYS. A shipped default would point a
///                            fresh install at a device that is not there.
///   `diarization.numSpeakers` a PER-MEETING fact, not a setup choice. Owner's
///                            own 43-minute recording had 7 people; a shipped
///                            5 would have merged two of them, silently.
///   `chunked.intervalSec`    120 s is measured better for MOSS and UNMEASURED
///                            for Qwen3, which is the shipped ASR. 30 s stays
///                            until someone measures it.
enum ShippedDefaults {
    /// Correct on 4 of the 5 known-answer recordings — the best score of the six
    /// engines, tied with DiariZen, and the only top scorer that also marks its
    /// own overlap. See `adoptMeasuredEngineDefaults` for the full evidence.
    static let diarizationEngine = ModelLoader.pyannoteEngineID

    /// Clean punctuated text at 27x realtime, and the highest agreement with an
    /// unrelated model (95.9 % vs MOSS) on the 43-minute meeting.
    static let chunkedModel = "qwen3"

    /// Parakeet TDT 0.6b v3. Measured on this M4 at ~128x realtime against
    /// Nemotron's ~14x on the identical call shape — 9x faster — and it is the
    /// one realtime engine measured to return the EMPTY STRING over silence
    /// rather than a hallucination, so it carries no drop gate to over-delete
    /// with. Its 25-language roster has no Indonesian, which costs nothing:
    /// meetings here are English only.
    static let realtimeModel = "parakeet"

    /// Both ON, owner-chosen 2026-08-21. They only add work at Stop, and repair
    /// SKIPS rather than guesses when separation cannot help — which on single-
    /// mic audio is most of the time, and is the safe outcome.
    static let overlapDetect = true
    static let overlapRepair = true
}
