import Foundation

// MARK: - One class per chunked ASR model
// Each model knows its HuggingFace repo and how to express the user's
// language setting for its mlx-audio implementation.

protocol ChunkedASRModel: Sendable {
    /// Catalog entry (name, badges, install check).
    var info: ModelInfo { get }
    /// HuggingFace repo passed to mlx-audio's load().
    var repoID: String { get }
    /// This model's OWN sidecar script, relative to `scripts/`.
    ///
    /// ONE SERVICE PER MODEL, finished 2026-07-30: there is no shared
    /// `chunked-asr-service.py` any more, so there is no file left to fall back
    /// to. Making this a protocol REQUIREMENT rather than a lookup with a
    /// `default:` is what makes the mapping exhaustive — a new model cannot
    /// compile without naming its own script, so it can never silently inherit
    /// another model's sidecar (or point at a deleted one). All the scripts speak
    /// the SAME frames, which is why one Swift client drives them all.
    var scriptName: String { get }
    /// stderr log basename for that sidecar — one log per service, never two
    /// writers on one file (a mistake this project made and fixed 2026-07-15,
    /// and one the split would make live again).
    var logName: String { get }
    /// Map the settings language code ("id", "en", …) to what this model
    /// expects. Return nil to omit (auto-detect).
    func languageArgument(for code: String) -> String?
}

/// Qwen3-ASR 1.7B (bf16) — primary. Best WER, Indonesian support.
final class Qwen3ASRModel: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "qwen3")
    var repoID: String { info.hfRepo }
    var scriptName: String { ChunkedASRService.Config.qwen3ScriptName }
    var logName: String { "qwen3" }

    /// Qwen3-ASR does its own language identification; it accepts ISO
    /// codes when forced. "auto" → omit and let LID decide.
    func languageArgument(for code: String) -> String? {
        code == "auto" ? nil : code
    }
}

/// Whisper large-v3 — widest language coverage, safe fallback.
final class WhisperLargeV3Model: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "whisper")
    var repoID: String { info.hfRepo }
    var scriptName: String { ChunkedASRService.Config.whisperScriptName }
    var logName: String { "whisper" }

    /// Whisper uses ISO-639-1 codes directly ("id", "en", …).
    func languageArgument(for code: String) -> String? {
        code == "auto" ? nil : code
    }
}

/// Voxtral Mini 4B Realtime 2602 — best WER on its 13 languages, and by far the
/// slowest (~27 s per 30 s chunk on the owner's M4), which is why a Remote channel
/// and MOSS-as-diarization both REFUSE to start with it selected rather than
/// substitute another model. Those refusals live in `AudioRecorder`; the sidecar
/// knows nothing about them.
final class VoxtralMiniModel: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "voxtral")
    var repoID: String { info.hfRepo }
    var scriptName: String { ChunkedASRService.Config.voxtralScriptName }
    var logName: String { "voxtral" }

    /// Voxtral covers 13 languages; pass ISO code, omit on auto.
    func languageArgument(for code: String) -> String? {
        code == "auto" ? nil : code
    }
}

/// Granite Speech 4.1 2B NAR — FIVE languages (en, fr, de, es, pt). No Japanese:
/// that belongs to ibm-granite's BASE Granite Speech card, not to the NAR
/// checkpoint we ship (its own card says `language: [en, fr, de, es, pt]`).
final class GraniteSpeechModel: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "granite")
    var repoID: String { info.hfRepo }
    var scriptName: String { ChunkedASRService.Config.graniteScriptName }
    var logName: String { "granite" }

    /// The picker is selectable (owner, 2026-07-31) and the code IS forwarded —
    /// but `generate()` has no language parameter and `**kwargs` swallows the
    /// flag without raising, so the transcript never changes. Measured: `"fr"`,
    /// `"de"` and even a nonsense `"xx"` all returned byte-identical output.
    ///
    /// Forwarded rather than dropped ON PURPOSE. Returning nil here would make
    /// the choice vanish silently at the Swift boundary, giving TWO places where
    /// it disappears; this way it travels the whole path and dies where the truth
    /// actually is — inside a model that has no such concept — and the sidecar
    /// log records what was asked for.
    func languageArgument(for code: String) -> String? {
        code == Languages.auto.code ? nil : code
    }
}

/// MOSS-Transcribe-Diarize 0.9B — speaker-attributed ASR. Unlike the four above
/// it also returns WHO spoke and WHEN, which is why it can additionally be
/// selected as the diarization engine (Settings → Models → Diarization).
///
/// Runs PyTorch on MPS rather than MLX, and "Whisper as ASR + MOSS as diarizer"
/// needs it in a process the ASR selection does not own — which is why it was the
/// first model to get its own sidecar, and the template the others followed.
final class MossTranscribeDiarizeModel: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "moss")
    var repoID: String { info.hfRepo }
    /// The ASR-role service and its own log (2026-07-31). The DIARIZATION role
    /// runs `moss-diar/moss-diar-service.py` → `moss-diar` log and names both
    /// LITERALLY at `Config.mossDiarization()` — the two roles can be two live
    /// processes at once, so neither may be derived from the other.
    var scriptName: String { ChunkedASRService.Config.mossASRScriptName }
    var logName: String { "moss-asr" }

    /// Selectable and forwarded, exactly as Granite's is, and equally inert:
    /// MOSS is steered only by its prompt, and a prompt asking for Indonesian or
    /// Chinese was measured to return identical English. The sidecar accepts
    /// `--language` and documents it as ignored.
    func languageArgument(for code: String) -> String? {
        code == Languages.auto.code ? nil : code
    }
}

// MARK: - Factory

enum ChunkedASRModelFactory {
    /// The model chosen in Settings → Models → Chunked.
    static func fromSettings() -> any ChunkedASRModel {
        switch UserDefaults.standard.string(forKey: "chunked.model") ?? "qwen3" {
        case "whisper": return WhisperLargeV3Model()
        case "voxtral": return VoxtralMiniModel()
        case "granite": return GraniteSpeechModel()
        case "moss":    return MossTranscribeDiarizeModel()
        default:        return Qwen3ASRModel()
        }
    }
}

// MARK: - Catalog lookup helper

extension ModelCatalog {
    static func chunkedModel(id: String) -> ModelInfo {
        chunked.first { $0.id == id } ?? chunked[0]
    }
}
