import Foundation

// MARK: - Model catalog (matches download-best-models.sh)

struct ModelInfo: Identifiable {
    let id: String        // stored in settings
    let name: String
    let detail: String
    let badges: [String]
    let hfRepo: String    // used to check the models/ folder
}

enum ModelCatalog {
    static let realtime = ModelInfo(
        id: "nemotron",
        name: "Nemotron 3.5 ASR Streaming 0.6B",
        detail: "Cache-aware streaming FastConformer-RNNT",
        badges: ["MLX", "40 locales", "112x RT", "0.6B"],
        hfRepo: "mlx-community/nemotron-3.5-asr-streaming-0.6b"
    )

    static let chunked: [ModelInfo] = [
        ModelInfo(id: "qwen3",
                  name: "Qwen3-ASR 1.7B (bf16)",
                  detail: "Best WER · Indonesian supported · full precision",
                  badges: ["MLX bf16", "52 languages", "30–36x RT", "1.7B"],
                  hfRepo: "mlx-community/Qwen3-ASR-1.7B-bf16"),
        ModelInfo(id: "whisper",
                  name: "Whisper large-v3 (fp16)",
                  detail: "Widest language coverage — safe fallback",
                  badges: ["MLX fp16", "99+ languages", "5–8x RT", "1.5B"],
                  // fp16 MLX weights, runs via mlx-whisper (tokenizer bundled)
                  hfRepo: "mlx-community/whisper-large-v3-mlx"),
        ModelInfo(id: "voxtral",
                  name: "Voxtral Mini 4B Realtime 2602 (fp16)",
                  detail: "5.9% WER FLEURS · 13 languages · full precision",
                  badges: ["MLX fp16", "13 languages", "~10x RT", "4B"],
                  hfRepo: "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16"),
        ModelInfo(id: "granite",
                  name: "Granite Speech 4.1 2B NAR (bf16)",
                  detail: "EN/FR/DE/ES/PT only · non-autoregressive · lowercase, sparse punctuation",
                  badges: ["MLX bf16", "5 languages", "~14x RT", "2B"],
                  hfRepo: "mlx-community/granite-speech-4.1-2b-nar-mlx"),
    ]

    static let diarization = ModelInfo(
        id: "pyannote",
        name: "pyannote community-1",
        detail: "Best open-source DER (19.9% AMI SDM) · runs after recording ends",
        badges: ["PyTorch MPS", "19.9% DER", "pyannote.audio 4.0"],
        hfRepo: "pyannote/speaker-diarization-community-1"
    )

    static let vad = ModelInfo(
        id: "silero",
        name: "Silero VAD v6.2.1",
        detail: "16% fewer errors than v5 · runs on CPU",
        badges: ["1.2 MB", "CPU", "100x+ RT"],
        hfRepo: "silero-vad-v6.2.1" // copied folder, not a HF hub repo
    )

    static let overlapSeparation = ModelInfo(
        id: "mossformer2",
        name: "MossFormer2 (librimix-2spk) standalone",
        detail: "8 kHz · 2-speaker LibriMix checkpoint · experimental — single-mic separation quality varies",
        badges: ["PyTorch CPU", "8 kHz", "2-speaker", "LibriMix"],
        hfRepo: "alibabasglab/mossformer2-librimix-2spk"
    )

    /// DiCoW v3.3 Large — diarization-conditioned Whisper (target-speaker ASR).
    /// Overlap attempt #4 (2026-07-16). Runs in its own `.venv-dicow`.
    static let overlapDicow = ModelInfo(
        id: "dicow",
        name: "DiCoW v3.3 Large",
        detail: "Target-speaker ASR — transcribes one speaker at a time from the diarization mask · experimental",
        badges: ["PyTorch", "Whisper-based", "Target-speaker", "6 GB"],
        hfRepo: "BUT-FIT/DiCoW_v3_3_large"
    )

    /// Engines offered on Settings → Models → Overlap (`overlap.engine`).
    static let overlapEngines: [ModelInfo] = [overlapSeparation, overlapDicow]

    static func overlapEngine(id: String) -> ModelInfo {
        overlapEngines.first { $0.id == id } ?? overlapSeparation
    }

    /// Project models folder (HF_HOME used by download-best-models.sh)
    static let modelsDir = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Documents/AI/Meeting Transcribe/models")

    static func isInstalled(_ model: ModelInfo) -> Bool {
        let fm = FileManager.default
        if model.id == "silero" {
            return fm.fileExists(atPath: modelsDir.appendingPathComponent(model.hfRepo).path)
        }
        if model.id == "mossformer2" {
            // Non-standard layout: downloaded flat into models/mossformer2/<name>/,
            // not the HF-hub cache. masknet.ckpt is the largest of the 3 weights.
            let path = modelsDir
                .appendingPathComponent("mossformer2/mossformer2-librimix-2spk/masknet.ckpt")
            return fm.fileExists(atPath: path.path)
        }
        let hubName = "models--" + model.hfRepo.replacingOccurrences(of: "/", with: "--")
        return fm.fileExists(atPath: modelsDir.appendingPathComponent("hub/\(hubName)").path)
    }
}

// MARK: - Languages offered in pickers

enum Languages {
    static let all: [(code: String, name: String)] = [
        ("auto", "Auto-detect"), ("id", "Indonesian"), ("en", "English"),
        ("ms", "Malay"), ("zh", "Chinese"), ("ja", "Japanese"),
    ]
}
