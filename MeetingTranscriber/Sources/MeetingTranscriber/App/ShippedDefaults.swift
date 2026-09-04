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
    /// How long a pause must last before the realtime lane calls it the end of
    /// an utterance. **One speech→silence edge is one FLUSH is one ROW**
    /// (`AudioRecorder.swift:1458` → `:1094` / `:1174`), so this value alone
    /// decides how often the transcript starts a new line while recording.
    ///
    /// ⚠ IT WAS 300 ms, AND 300 ms IS AN ORDINARY PAUSE INSIDE A SENTENCE — so
    /// one sentence became several rows and the live transcript read as choppy
    /// rather than as turn-taking. Measured over Silero at the real call shape
    /// (85 ms tap buffers, the last probability per buffer, the same
    /// hysteresis), on the 43-minute 7-person meeting:
    ///
    ///     300 ms   120 rows   18.0/min   median utterance 2.9 s   max  9.1 s
    ///     400 ms    96 rows   14.4/min                    3.8 s   max 12.1 s
    ///     500 ms    78 rows   11.7/min                    4.0 s   max 18.2 s
    ///   → 600 ms    70 rows   10.5/min                    4.3 s   max 27.3 s
    ///     800 ms    49 rows    7.3/min                    5.6 s   max 37.2 s
    ///    1200 ms    24 rows    3.6/min                    9.5 s   max 74.4 s ✗
    ///
    /// The row RATE is what the user sees and the MAX is what makes a value
    /// unsafe, so 600 is chosen against both. Every realtime sidecar caps its
    /// utterance buffer at `MAX_BUFFER = 60 s` and **trims from the FRONT**, so
    /// audio past that ceiling is discarded silently — the failure direction
    /// this project ranks worst. At 600 ms the longest utterance measured is
    /// 27.3 s (p95 13.6 s), i.e. 2.2× of margin; at 1200 ms it is 74.4 s and
    /// real speech would be dropped with no trace. **Do not raise this past
    /// ~800 ms without re-measuring that column.**
    ///
    /// Two couplings, both checked rather than assumed. The chunk boundary is
    /// `elapsed >= interval && !speaking`, so a longer hold waits for a bigger
    /// pause — which is the direction the 2026-08-13 audit wanted (3.5 % of
    /// words are lost at boundaries). And `positionDiarizer.noteSpeech` no
    /// longer gates anything: `gateOnSpeech` is false since the beam was
    /// ungated, so the ATND layer is untouched by this value.
    static let vadMinSilenceMs = 600.0

    static let overlapDetect = true
    static let overlapRepair = true
}
