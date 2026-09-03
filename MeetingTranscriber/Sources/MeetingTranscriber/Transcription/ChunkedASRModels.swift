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

    /// The interpreter this model's sidecar runs under, or nil for the main
    /// `.venv` that five of the six use.
    ///
    /// ⚠ NOT A PREFERENCE — a dependency conflict that cannot be resolved. The
    /// main `.venv` needs `transformers` 5.x for the MLX stack; VibeVoice pins
    /// `>=4.51.3,<5.0.0`, so the two can never share an interpreter. Exactly the
    /// situation that gave DiCoW `.venv-dicow` on 2026-07-16, and NeMo and
    /// DiariZen their own after it.
    ///
    /// Defaulted in an extension rather than required, deliberately: the answer
    /// for a NEW model is almost always "the main venv", and a protocol
    /// requirement here would make every existing class restate it. That is the
    /// opposite trade from `scriptName`, which IS required precisely because
    /// there is no safe default — inheriting another model's sidecar is silent,
    /// while inheriting the main venv fails loudly at import.
    var venvName: String? { get }
}

extension ChunkedASRModel {
    var venvName: String? { nil }
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

final class VibeVoiceASRModel: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "vibevoice")
    var repoID: String { info.hfRepo }
    var scriptName: String { "vibevoice-asr/vibevoice-asr-service.py" }
    var logName: String { "vibevoice-asr" }

    /// The ONE model here that does not run in the main `.venv` — see
    /// `ChunkedASRModel.venvName`. `transformers <5.0` against the MLX stack's
    /// 5.x is a hard conflict, not a version preference.
    var venvName: String? { ".venv-vibevoice" }

    /// Selectable and forwarded, and inert — the fifth member of that group
    /// (Granite, Voxtral, MOSS, Parakeet). `streaming_generate` has no language
    /// parameter; checked in the vendored source on 2026-09-02, not inferred
    /// from the model card's "10 languages". The sidecar logs what it was asked
    /// for and ignores it, so the choice dies where the truth is.
    func languageArgument(for code: String) -> String? {
        code == Languages.auto.code ? nil : code
    }
}

// MARK: - Factory

enum ChunkedASRModelFactory {
    /// The model chosen in Settings → Models → Chunked.
    static func fromSettings() -> any ChunkedASRModel {
        switch UserDefaults.standard.string(forKey: "chunked.model") ?? ShippedDefaults.chunkedModel {
        case "whisper": return WhisperLargeV3Model()
        case "voxtral": return VoxtralMiniModel()
        case "granite": return GraniteSpeechModel()
        case "moss":    return MossTranscribeDiarizeModel()
        case "vibevoice": return VibeVoiceASRModel()
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
