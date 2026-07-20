import Foundation

// MARK: - One class per chunked ASR model
// Each model knows its HuggingFace repo and how to express the user's
// language setting for its mlx-audio implementation.

protocol ChunkedASRModel: Sendable {
    /// Catalog entry (name, badges, install check).
    var info: ModelInfo { get }
    /// HuggingFace repo passed to mlx-audio's load().
    var repoID: String { get }
    /// Map the settings language code ("id", "en", …) to what this model
    /// expects. Return nil to omit (auto-detect).
    func languageArgument(for code: String) -> String?
}

/// Qwen3-ASR 1.7B (bf16) — primary. Best WER, Indonesian support.
final class Qwen3ASRModel: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "qwen3")
    var repoID: String { info.hfRepo }

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

    /// Whisper uses ISO-639-1 codes directly ("id", "en", …).
    func languageArgument(for code: String) -> String? {
        code == "auto" ? nil : code
    }
}

/// Voxtral Mini 4B Realtime 2602 — best WER on its 13 languages.
final class VoxtralMiniModel: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "voxtral")
    var repoID: String { info.hfRepo }

    /// Voxtral covers 13 languages; pass ISO code, omit on auto.
    func languageArgument(for code: String) -> String? {
        code == "auto" ? nil : code
    }
}

/// Granite Speech 4.1 2B NAR — English/French/German/Spanish/Portuguese only.
final class GraniteSpeechModel: ChunkedASRModel {
    let info = ModelCatalog.chunkedModel(id: "granite")
    var repoID: String { info.hfRepo }

    /// generate() has no language parameter (swallowed by **kwargs) — never send one.
    func languageArgument(for code: String) -> String? { nil }
}

// MARK: - Factory

enum ChunkedASRModelFactory {
    /// The model chosen in Settings → Models → Chunked.
    static func fromSettings() -> any ChunkedASRModel {
        switch UserDefaults.standard.string(forKey: "chunked.model") ?? "qwen3" {
        case "whisper": return WhisperLargeV3Model()
        case "voxtral": return VoxtralMiniModel()
        case "granite": return GraniteSpeechModel()
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
