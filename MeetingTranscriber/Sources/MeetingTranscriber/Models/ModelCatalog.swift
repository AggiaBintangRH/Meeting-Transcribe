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
    /// The realtime engines, NEMOTRON FIRST — first is the default, and the
    /// default must stay what it has always been.
    ///
    /// One entry, one load, however many streams the session has: whichever
    /// engine is selected serves the Office and Remote lanes from a single
    /// process (see `RealtimeASRService`), so there is no second row to show. A
    /// dual-stream session's overlay is identical to a single-stream one here.
    static let realtimeModels: [ModelInfo] = [
        // ⚠ FIRST ENTRY = THE DEFAULT. `realtimeModel(id:)` falls back to
        // `realtimeModels[0]`, and SettingsMatrixTests asserts that entry equals
        // `RealtimeASRService.defaultModelID` — otherwise an unknown stored id
        // resolves to a model the loader would not have started. Parakeet moved
        // to the front on 2026-08-21 when it became the shipped default; the
        // two must always move together.
        //
        // Measured on this M4 (2026-08-11), whole-buffer re-transcribe — the
        // shape the sidecar actually uses — best of 2 with the warm-up
        // discarded: 10 s buffer 0.083 s, 30 s 0.235 s, 60 s 0.462 s. That is
        // 121–130x realtime against Nemotron's steady ~14x, and two ACTIVE lanes
        // sit at ~31 % duty where Nemotron's sit at 285 %. The badge quotes the
        // 30 s figure, which is the one a meeting actually spends its time on.
        // CC BY-4.0, so it is shippable to a paying client — the same licence
        // question that ruled out five of DiariZen's six checkpoints.
        ModelInfo(id: "parakeet",
                  name: "Parakeet TDT 0.6b v3",
                  detail: "FastConformer-TDT · CC BY-4.0 · 25 European languages · no Indonesian/Chinese/Japanese",
                  badges: ["MLX", "25 languages", "128x RT", "0.6B", "2.3 GB"],
                  hfRepo: "mlx-community/parakeet-tdt-0.6b-v3"),
        ModelInfo(id: "nemotron",
                  name: "Nemotron 3.5 ASR Streaming 0.6B",
                  detail: "Cache-aware streaming FastConformer-RNNT",
                  badges: ["MLX", "40 locales", "112x RT", "0.6B"],
                  hfRepo: "mlx-community/nemotron-3.5-asr-streaming-0.6b"),
        // Measured on this M4 (2026-08-11), same whole-buffer shape and same
        // best-of-2 method as the Parakeet row: 10 s buffer 0.151 s, 30 s
        // 0.609 s, 60 s 0.621 s — i.e. 49–97x realtime, between Parakeet's
        // ~128x and Nemotron's ~14x. The badge quotes the 30 s figure, which is
        // the WORST of the three and the one a meeting actually spends its time
        // on; the 60 s row is faster only because that window held less speech,
        // since cost here tracks tokens generated rather than seconds of audio.
        //
        // Apache 2.0, so it is shippable to a paying client.
        //
        // "3 languages" is the roster mainline mlx-audio can EXPRESS, not the 30
        // FunAudioLLM advertises for this checkpoint — `_map_language()` accepts
        // eleven ISO codes (English, Japanese and nine spellings of Chinese) and
        // RAISES on everything else. The badge states what the code does, which
        // is the Granite lesson applied one step further on.
        ModelInfo(id: "funasr",
                  name: "Fun-ASR MLT Nano 2512",
                  detail: "SenseVoice encoder + Qwen3 decoder · Apache 2.0 · English, Chinese, Japanese",
                  badges: ["MLX", "3 languages", "49x RT", "1.0B", "2.0 GB"],
                  hfRepo: "mlx-community/Fun-ASR-MLT-Nano-2512-fp16"),
        // The FOURTH realtime engine, 2026-09-02, and the same checkpoint as the
        // "vibevoice" chunked entry in its OTHER role — two services, two logs,
        // and two live processes if both slots pick it (~22 GB).
        //
        // ⚠ THE BADGE SAYS 5x AND THAT IS NOT THE WHOLE STORY, so the detail
        // line carries the half that matters. Measured 2026-09-02 with model
        // load excluded: 5.2x realtime, CONSTANT at 30 s and 60 s, 38 % duty on
        // two active lanes. Every other engine here re-transcribes its lane's
        // whole buffer per partial, so cost grows with utterance length —
        // Nemotron is ~14x per second and still needs cadence stretching at
        // 285 % duty. This one advances a KV cache one window at a time, so it
        // is the slowest per second of audio and among the lightest on duty.
        // Judge it on duty; that is what decides whether captions keep up.
        //
        // "11 GB RAM" and "own runtime" are on the badge for the same reason
        // they are on the chunked entry: the cost belongs at the point of
        // choosing, not in a log after a meeting.
        ModelInfo(id: "vibevoice",
                  name: "VibeVoice-ASR-Streaming 1.5B",
                  detail: "Speaker-attributed streaming — advances a cache per window, so cost does not grow with utterance length",
                  badges: ["PyTorch MPS", "MIT", "5x RT", "38% duty",
                           "11 GB RAM", "own runtime"],
                  hfRepo: "microsoft/VibeVoice-ASR-Streaming-1.5B"),
    ]

    /// The realtime engine for a stored id, falling back to the FIRST entry —
    /// the `chunkedModel(id:)` pattern, and the same reason: a stored id naming
    /// an engine this build no longer has must resolve to a real model rather
    /// than to a `default:` that quietly points at a deleted script.
    static func realtimeModel(id: String) -> ModelInfo {
        realtimeModels.first { $0.id == id } ?? realtimeModels[0]
    }

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
                  // 6, not 5: the model card lists Japanese alongside
                  // EN/FR/DE/ES/PT (checked 2026-07-29 — the older "5, no
                  // Japanese" note in this repo was wrong).
                  detail: "EN/FR/DE/ES/PT/JA only · non-autoregressive · lowercase, sparse punctuation",
                  badges: ["MLX bf16", "6 languages", "~14x RT", "2B"],
                  hfRepo: "mlx-community/granite-speech-4.1-2b-nar-mlx"),
        // The only chunked model that is not purely a recogniser: it returns the
        // speaker label and the timestamps WITH the words, from one forward pass.
        // Runs on PyTorch/MPS (every other entry is MLX), in its own sidecar.
        ModelInfo(id: "moss",
                  name: "MOSS-Transcribe-Diarize 0.9B",
                  detail: "Speaker-attributed ASR — transcript, speaker labels and timestamps from one model",
                  badges: ["PyTorch MPS", "float32", "~4.7x RT", "0.9B", "3.6 GB"],
                  hfRepo: "OpenMOSS-Team/MOSS-Transcribe-Diarize"),
        // The SECOND speaker-attributed ASR here, added 2026-09-02 at the
        // client's request. MIT, so it may ship; runs on PyTorch/MPS in the ONLY
        // sidecar with its own interpreter on the chunked side (transformers
        // <5.0 against the MLX stack's 5.x — see VibeVoiceASRModel.venvName).
        //
        // ⚠ THE BADGES ARE THE MEASURED NUMBERS, INCLUDING THE UNFLATTERING ONE.
        // Driven on this M4 on 2026-09-02 before it was wired: 10.96 GB RSS and
        // a 9.8 GB MPS pool, flat from 30 s to 121 s of audio. That is above the
        // 11.8 GB ceiling of the client's 16 GB Mac once the other ~4 GB of
        // sidecars are counted, so it is a 64 GB-machine model here. The size
        // badge says so rather than leaving it to be discovered mid-meeting.
        //
        // Speaker counting, auto, no way to pin it (`streaming_generate` has no
        // num_speakers and the package has none anywhere): Overlap123 3/3 ✓,
        // Meeting5People 4 against 5 ✗ — where all five diarization engines get
        // 5 — and the client's own ATND recording 1 against 5. Kept out of the
        // detail line only because it is a diarization property and this is the
        // ASR slot; it is in the sidecar docstring in full.
        ModelInfo(id: "vibevoice",
                  name: "VibeVoice-ASR-Streaming 1.5B",
                  detail: "Speaker-attributed streaming ASR — text and speaker labels from one model",
                  badges: ["PyTorch MPS", "float32", "~4.6x RT", "10 languages",
                           "11 GB RAM", "own runtime"],
                  hfRepo: "microsoft/VibeVoice-ASR-Streaming-1.5B"),
    ]

    static let diarization = ModelInfo(
        id: "pyannote",
        name: "pyannote community-1",
        detail: "Best open-source DER (19.9% AMI SDM) · runs after recording ends",
        badges: ["PyTorch MPS", "19.9% DER", "pyannote.audio 4.0"],
        hfRepo: "pyannote/speaker-diarization-community-1"
    )

    /// WeSpeaker ResNet34-LM — speaker EMBEDDINGS, not diarization. It turns a
    /// span of speech into a vector, which is what lets a voice be matched against
    /// the saved profile store and keep its name across chunks and sessions.
    ///
    /// Its own catalog entry since 2026-07-30, when `diarize-service.py` was split
    /// into a pipeline sidecar and this one (ONE SERVICE PER MODEL, owner
    /// 2026-07-29). The weights were always downloaded — `download-best-models.sh`
    /// has fetched them since dual-stream — they were just loaded inside the
    /// pyannote process and so had no row of their own.
    ///
    /// It gets its OWN loading step, ordered BEFORE pyannote: ~26 MB against the
    /// pipeline's minute, so "the embedder is missing" is reported in seconds
    /// instead of after a long load that was going to fail anyway.
    static let speakerEmbedding = ModelInfo(
        id: "wespeaker",
        name: "WeSpeaker ResNet34-LM",
        detail: "Speaker embeddings — matches each voice to its saved profile so names persist",
        badges: ["PyTorch MPS", "26 MB", "voice profiles"],
        hfRepo: "pyannote/wespeaker-voxceleb-resnet34-LM"
    )

    /// MOSS-Transcribe-Diarize used as the DIARIZER rather than as the chunked
    /// recogniser (`diarization.engine == "moss"`). Its own id so the two roles
    /// can be selected independently, but the SAME `hfRepo` — one download, one
    /// install check, and when it fills both roles it is also one process (see
    /// `ModelLoader.needsSecondMossProcess`).
    static let mossDiarization = ModelInfo(
        id: "moss-diar",
        name: "MOSS-Transcribe-Diarize 0.9B",
        detail: "Speaker labels come from the ASR model itself — per chunk, anonymous, no voice profiles",
        badges: ["PyTorch MPS", "float32", "~6.4 s / 30 s chunk", "0.9B"],
        hfRepo: "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    )

    /// The SPECTRAL engine — the vendored Apache-2.0 `diarize` package
    /// (github.com/FoxNoseTech/diarize @ 4f25d27): Silero VAD → sliding-window
    /// WeSpeaker embeddings → GMM-BIC speaker counting → spectral clustering →
    /// Viterbi smoothing. A WHOLE-FILE pass at Stop and nothing else (v1).
    ///
    /// `hfRepo` is the WeSpeaker checkpoint on purpose, and it is not a fudge: the
    /// engine's ONE modification to upstream (`vendor/diarize/torch_embedder.py`)
    /// swaps upstream's self-downloading ONNX embedder for exactly these weights,
    /// so this really is the model the install check must look for. Silero comes
    /// from the `silero_vad` wheel's own package data, not from `models/`. Two
    /// catalog entries sharing an `hfRepo` is the `moss` / `moss-diar` precedent.
    static let spectralDiarization = ModelInfo(
        id: "spectral",
        name: "Spectral clustering diarization",
        detail: "Silero VAD → WeSpeaker embeddings → GMM-BIC + spectral clustering · CPU · whole recording at stop",
        badges: ["PyTorch CPU", "whole-file only", "Apache 2.0", "vendored"],
        hfRepo: "pyannote/wespeaker-voxceleb-resnet34-LM"
    )

    /// The NEMO engine — NVIDIA NeMo's `ClusteringDiarizer` (nemo_toolkit 3.0.0,
    /// Apache 2.0): MarbleNet VAD → multi-scale TitaNet-Large embeddings → NME-SC
    /// spectral clustering, which estimates the speaker count ITSELF → multi-scale
    /// label fusion. A WHOLE-FILE pass at Stop and nothing else, like spectral.
    ///
    /// `hfRepo` is NOT an HF repo id: both checkpoints are `.nemo` files
    /// downloaded flat into `models/nemo/`, so `isInstalled` special-cases this
    /// entry the way it does silero and mossformer2. The string names the folder
    /// rather than a hub path, and nothing else reads it.
    ///
    /// Measured on this M4 before it was offered at all: 16–23× realtime on MPS,
    /// 5 speakers / 9 segments on `Meeting5People.wav` and 8 speakers / 324 turns
    /// on a real 48.2-minute meeting. Its cold import is ~51 s — the slowest load
    /// in the app — and its peak RSS scales with the audio (1.15 GB for 98 s,
    /// 13.33 GB for 67 min).
    static let diarizenDiarization = ModelInfo(
        id: "diarizen",
        name: "DiariZen meeting (BUT Speech@FIT)",
        detail: "WavLM+Conformer end-to-end segmentation → WeSpeaker clustering · trained on far-field meeting audio · whole recording at stop",
        badges: ["PyTorch MPS", "whole-file only", "MIT", "own runtime"],
        hfRepo: "BUT-FIT/diarizen-meeting-base"
    )

    static let nemoDiarization = ModelInfo(
        id: "nemo",
        name: "NVIDIA NeMo clustering diarization",
        detail: "MarbleNet VAD → TitaNet-Large embeddings → NME-SC clustering · counts speakers itself · whole recording at stop",
        badges: ["PyTorch MPS", "whole-file only", "Apache 2.0", "own runtime"],
        hfRepo: "nemo"
    )

    /// The CAM++ engine — a pipeline built around a speaker EMBEDDING model,
    /// because that is all CAM++ is: Silero VAD → CAM++ vectors over 2 s sliding
    /// sub-windows → spectral clustering with an eigengap speaker count. A
    /// WHOLE-FILE pass at Stop and nothing else, like spectral, NeMo and DiariZen.
    ///
    /// The same SHAPE as `spectralDiarization` and deliberately not the same
    /// stages: that engine's GMM-BIC speaker COUNTING is the part measured
    /// failing (20 speakers on a 67-minute meeting, 13 on a 3-person clip), and
    /// this replaces exactly that stage while keeping the rest.
    ///
    /// Measured on this M4 before it was offered at all: 5 speakers on
    /// `Meeting5People.wav` (truth 5) in 1.7 s, 3 on `Overlap123.wav` (truth 3)
    /// in 1.1 s, and 3 speakers / 99 turns on the 67-minute meeting where
    /// spectral returns 20 / 1869.
    ///
    /// **Apache 2.0, and the licence chose the checkpoint.** The MLX-native CAM++
    /// on the hub declares no licence at all, and this app ships to a paying
    /// client — the same question that ruled out five of DiariZen's six
    /// checkpoints. Unlike NeMo, DiCoW and DiariZen it needs NO interpreter of
    /// its own: torch, scipy and silero-vad are already in the main `.venv`, so
    /// it costs the bundle only its 66 MB of weights.
    static let camPlusDiarization = ModelInfo(
        id: "campplus",
        name: "CAM++ clustering diarization",
        detail: "Silero VAD → CAM++ embeddings → spectral clustering with an eigengap speaker count · CPU · whole recording at stop",
        badges: ["PyTorch CPU", "whole-file only", "Apache 2.0", "vendored"],
        hfRepo: "Wespeaker/wespeaker-voxceleb-campplus-LM"
    )

    /// The SEVENTH engine (2026-09-02), owner-requested after the client asked
    /// for VibeVoice by name. The same checkpoint as the chunked and realtime
    /// entries, in its third role: it transcribes and attributes in ONE pass, so
    /// the `Speaker N:` runs inside its transcript are the diarization. Nothing
    /// here embeds or clusters.
    ///
    /// ⚠ THE BADGES CARRY THE UNFLATTERING MEASUREMENTS ON PURPOSE, because this
    /// is where the choice is made. Auto speaker counts on the known-answer
    /// files: Overlap123 3/3 ✓, Meeting5People 4 against 5 ✗ (every other engine
    /// gets 5), the client's ATND recording 1 against 5. And "no SPK" is the one
    /// that matters most — `streaming_generate` has no `num_speakers`, so the
    /// control that rescues pyannote, spectral and CAM++ on that ATND file
    /// (4 → 5) cannot rescue this one.
    static let vibeVoiceDiarization = ModelInfo(
        id: "vibevoice",
        name: "VibeVoice speaker-attributed diarization",
        detail: "Transcribes and attributes in one pass — one KV cache over the whole recording, so labels never need stitching · MPS · whole recording at stop",
        badges: ["PyTorch MPS", "whole-file only", "MIT", "no SPK control",
                 "11 GB RAM", "own runtime", "~2.9 s boundaries"],
        hfRepo: "microsoft/VibeVoice-ASR-Streaming-1.5B"
    )

    /// Engines offered on Settings → Models → Diarization (`diarization.engine`).
    static let diarizationEngines: [ModelInfo] = [diarization, spectralDiarization,
                                                  nemoDiarization, diarizenDiarization,
                                                  camPlusDiarization, mossDiarization,
                                                  vibeVoiceDiarization]

    /// The short name an engine goes by in prose — "NeMo", not "NVIDIA NeMo
    /// clustering diarization". Card names are too long for a sentence.
    ///
    /// Returns nil for an unknown value ON PURPOSE, so
    /// `testEveryDiarizationEngineHasAShortName` fails the moment a seventh
    /// engine joins `diarizationEngines`. A `default:` here would hand the new
    /// engine somebody else's name — the fall-through this file already warns
    /// about for `scriptName` and that the rail hit for real with DiariZen.
    static func diarizationEngineShortName(_ engineValue: String) -> String? {
        switch engineValue {
        case ModelLoader.pyannoteEngineID: return "pyannote"
        case ModelLoader.spectralEngineID: return "spectral"
        case ModelLoader.nemoEngineID:     return "NeMo"
        case ModelLoader.diarizenEngineID: return "DiariZen"
        case ModelLoader.camPlusEngineID:  return "CAM++"
        case ModelLoader.vibeVoiceEngineID: return "VibeVoice"
        case ModelLoader.mossEngineID:     return "MOSS"
        default:                           return nil
        }
    }

    /// The engines this chunked model offers, written out for a sentence:
    /// "pyannote, spectral, NeMo, DiariZen or CAM++".
    ///
    /// WHY THIS IS DERIVED AND NOT TYPED OUT. The sentence it feeds has now gone
    /// stale TWICE — it missed DiariZen until 2026-08-10 and then missed CAM++
    /// until the 2026-08-13 audit — each time telling the user they had fewer
    /// options than they did. Both times the list beside it, which reads this
    /// same filter, was already correct. A hand-written list next to a derived
    /// one is a copy that only ever drifts one way.
    /// The engines available once the chunked model is NOT MOSS — the list the
    /// MOSS notice offers as the way out.
    ///
    /// A named entry point rather than the caller passing some arbitrary
    /// non-MOSS id: `diarizationEngineIsSelectable` branches only on "is this
    /// MOSS", so any other id answers the same, and a literal like `"qwen3"` at
    /// the call site would read as though that particular model mattered.
    static var diarizationEnginesWithoutMoss: String {
        diarizationEngineNames(forChunkedModel: "")
    }

    /// The engines that CANNOT mark overlap themselves, written for a sentence —
    /// the ones the separate detector exists for.
    ///
    /// Derived for the same reason the list above is: the Detect overlap tab said
    /// "MOSS and spectral" while `ModelLoader.marksItsOwnOverlap` put FOUR engines
    /// in that set, so users of NeMo and CAM++ were told a feature they needed did
    /// not concern them. Third time this shape of sentence has gone stale in a
    /// week; none of them will again.
    static var diarizationEnginesWithoutOwnOverlap: String {
        prose(diarizationEngines
                .map { diarizationEngineValue($0) }
                .filter { !ModelLoader.marksItsOwnOverlap(diarEngine: $0) }
                .compactMap { diarizationEngineShortName($0) })
    }

    /// "a", "a or b", "a, b or c" — one place, so every derived list reads alike.
    private static func prose(_ names: [String]) -> String {
        guard let last = names.last else { return "" }
        guard names.count > 1 else { return last }
        return names.dropLast().joined(separator: ", ") + " or " + last
    }

    static func diarizationEngineNames(forChunkedModel chunkedID: String) -> String {
        prose(diarizationEngines
                .map { diarizationEngineValue($0) }
                .filter { ModelLoader.diarizationEngineIsSelectable($0, chunkedID: chunkedID) }
                .compactMap { diarizationEngineShortName($0) })
    }

    /// The `diarization.engine` VALUE a given engine card selects.
    ///
    /// The catalog id and the setting value are deliberately different strings
    /// for MOSS. Catalog ids are unique across the whole catalog, and the SAME
    /// model already appears as the chunked entry `"moss"` — so the diarization
    /// entry has to be `"moss-diar"`. The setting, meanwhile, is a plain engine
    /// choice ("pyannote" | "spectral" | "moss") that the loader and the recorder
    /// read directly. Both facts are load-bearing, so the conversion lives here,
    /// once, rather than as a string comparison repeated in each tab — where
    /// writing `engine = m.id` would silently store a value nothing ever matches.
    ///
    /// SPECTRAL DELIBERATELY USES ONE STRING FOR BOTH (`"spectral"`), like
    /// pyannote and unlike MOSS. The MOSS divergence exists solely because that
    /// model has a second catalog entry in the CHUNKED list and two entries cannot
    /// share an id; spectral is a diarizer and only a diarizer, so there is no
    /// collision to work around. Inventing a `"spectral-diar"` id purely for
    /// symmetry would add the very indirection that made the MOSS card look
    /// selected while the engine never switched.
    static func diarizationEngineValue(_ model: ModelInfo) -> String {
        switch model.id {
        case mossDiarization.id:     return ModelLoader.mossEngineID
        case spectralDiarization.id: return ModelLoader.spectralEngineID
        case nemoDiarization.id:     return ModelLoader.nemoEngineID
        case diarizenDiarization.id: return ModelLoader.diarizenEngineID
        case camPlusDiarization.id:  return ModelLoader.camPlusEngineID
        case vibeVoiceDiarization.id: return ModelLoader.vibeVoiceEngineID
        default:                     return ModelLoader.pyannoteEngineID
        }
    }

    /// The inverse: the engine card a stored `diarization.engine` value selects.
    static func diarizationEngine(forEngine engine: String) -> ModelInfo {
        switch engine {
        case ModelLoader.mossEngineID:     return mossDiarization
        case ModelLoader.spectralEngineID: return spectralDiarization
        case ModelLoader.nemoEngineID:     return nemoDiarization
        case ModelLoader.diarizenEngineID: return diarizenDiarization
        case ModelLoader.camPlusEngineID:  return camPlusDiarization
        case ModelLoader.vibeVoiceEngineID: return vibeVoiceDiarization
        default:                           return diarization
        }
    }

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

    /// Qwen3-ForcedAligner 0.6B — forced alignment, not recognition. It takes the
    /// chunked model's own text plus the same audio and returns the start/end time
    /// of each word, which is what lets a chunk be split by speaker at the exact
    /// word instead of by estimated character position. Its OWN sidecar since
    /// 2026-07-29 (`AlignerService`), asked AFTER the chunk's text is already on
    /// screen — so alignment refines rows and never delays them.
    /// MOSS needs no aligner, and the tab says so instead of showing a 1.2 GB
    /// model it will not load (`ModelLoader.wantsAligner` has excluded MOSS since
    /// the aligner became its own sidecar; only the UI never mentioned it).
    ///
    /// ⚠ **NOT a claim of word timestamps, and the wording is careful for a
    /// reason.** MOSS returns `{start, end, speaker, text}` per SEGMENT and its
    /// own sidecar docstring says *"No `conf` and no `words`, ever"*. What makes
    /// the aligner redundant is not that MOSS times each word — it does not — but
    /// that the aligner's PURPOSE is already served: word times exist to split one
    /// chunk's text between speakers at the exact word, and MOSS emits a separate
    /// timed segment per speaker in the first place, so there is nothing left to
    /// split. Claiming word-level timing here would be the fabrication direction
    /// this project ranks worst.
    static let wordAlignerMoss = ModelInfo(
        id: "aligner-moss",
        name: "MOSS (built in)",
        detail: "MOSS times and labels its own segments · nothing left for a word aligner to split",
        badges: ["PyTorch MPS", "no extra cost", "part of transcription"],
        hfRepo: "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    )

    /// The aligner OFFERED for a given chunked model, and this is a BICONDITIONAL
    /// like `overlapDetectors(forDiarEngine:)` and the MOSS⟺MOSS rule — one
    /// function, because two half-rules living apart is how they come to disagree.
    ///
    /// It mirrors `ModelLoader.wantsAligner`'s `chunkedID != "moss"` term rather
    /// than restating it: that function decides whether the 1.2 GB process LOADS,
    /// this one decides what the tab SHOWS, and the tab claiming a model the
    /// loader refuses to start is exactly the UI-lies-about-the-app defect the
    /// 2026-08-10 pass was about.
    static func wordAligners(forChunkedModel id: String) -> [ModelInfo] {
        id == "moss" ? [wordAlignerMoss] : [wordAligner]
    }

    static let wordAligner = ModelInfo(
        id: "aligner",
        name: "Qwen3-ForcedAligner 0.6B (bf16)",
        detail: "Word-level timestamps for the chunked transcript · own sidecar, asked after the text is shown",
        badges: ["MLX bf16", "word timestamps", "~0.2 s/chunk", "0.6B"],
        hfRepo: "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"
    )

    /// Engines offered on Settings → Models → Overlap (`overlap.engine`).
    static let overlapEngines: [ModelInfo] = [overlapSeparation, overlapDicow]

    // MARK: - Overlap DETECTION (a different job from repair)
    //
    // Detection marks WHERE two people spoke together; repair tries to recover the
    // words. They need different things: repair works from pyannote turns that
    // intersect, detection reads the audio directly and needs no turns at all —
    // which is why it is the only option under MOSS and spectral, engines that
    // assign exactly one speaker per instant.

    /// pyannote's segmentation network, taken from the community-1 checkpoint this
    /// app already ships. Its output is a POWERSET over speaker combinations, so
    /// "two speakers active" is a class it predicts directly rather than something
    /// inferred from separate per-speaker curves.
    ///
    /// Measured on the owner's M4 before this was offered at all: 32 MB, loads in
    /// ~0 s, scans at ~160x realtime (a 43-minute meeting in 16 s), and — checked
    /// in BOTH directions — 16.2 % of a clip that genuinely contains overlap,
    /// 0.0 % of a clean one, 0.1–1.8 % of the owner's real meetings. It does not
    /// simply fire on room noise.
    static let overlapDetectPyannote = ModelInfo(
        id: "pyannote-segmentation",
        name: "pyannote segmentation",
        detail: "Powerset speaker-activity network from speaker-diarization-community-1 · needs no speaker turns · ~160x realtime",
        badges: ["PyTorch", "32 MB", "already installed", "detect only"],
        hfRepo: "pyannote/speaker-diarization-community-1"
    )

    /// DiariZen's own overlap detection — NOT a separate model, and the card says
    /// so rather than implying a second download.
    ///
    /// Its Conformer head is a POWERSET over speaker combinations: for this
    /// checkpoint, 11 classes = 1 silence + 4 single-speaker + **6 pairs**
    /// (`powerset_max_classes = 2`, read off the loaded model). Two people at once
    /// is predicted directly at 50 fps, so the overlap regions fall out of the
    /// diarization pass that already ran — no second network, no second scan, no
    /// extra seconds. Measured: 11 intersecting turn pairs over 13.30 s on
    /// `recordings/Overlap123.wav`.
    static let overlapDetectDiarizen = ModelInfo(
        id: "diarizen-powerset",
        name: "DiariZen (built in)",
        detail: "Found by the diarization pass itself · no second model, no extra pass",
        badges: ["PyTorch MPS", "no extra cost", "part of diarization"],
        hfRepo: "BUT-FIT/diarizen-meeting-base"
    )

    /// Every detector the app can offer. The list exists rather than a hardcoded
    /// reference so the picker, the stored setting and the "installed" check all
    /// read one array.
    static let overlapDetectors: [ModelInfo] = [overlapDetectPyannote,
                                                overlapDetectDiarizen]

    /// The detectors OFFERED for a given diarization engine, and this is a
    /// BICONDITIONAL like the MOSS⟺MOSS rule — one function, because two half-rules
    /// living apart is how they come to disagree.
    ///
    /// Under DiariZen the detector IS DiariZen: its own head already answers the
    /// question, so running pyannote's segmentation network over the same audio
    /// would pay for a second opinion nobody asked for. Under every other engine
    /// DiariZen is not loaded at all, so its detection cannot be offered.
    ///
    /// Read by the tab's picker AND by `overlapDetector(id:forDiarEngine:)`, which
    /// is what corrects a stored id that is no longer on offer — the trap this
    /// project has hit before with a picker nothing read.
    static func overlapDetectors(forDiarEngine engine: String) -> [ModelInfo] {
        engine == ModelLoader.diarizenEngineID ? [overlapDetectDiarizen]
                                               : [overlapDetectPyannote]
    }

    static func overlapDetector(id: String) -> ModelInfo {
        overlapDetectors.first { $0.id == id } ?? overlapDetectPyannote
    }

    /// The detector actually in force: the stored id when the current engine still
    /// offers it, otherwise that engine's own. A stored id must never be able to
    /// name a detector this session cannot run — the value-outliving-its-control
    /// failure the 2026-08-06 settings pass exists to forbid.
    static func overlapDetector(id: String, forDiarEngine engine: String) -> ModelInfo {
        let offered = overlapDetectors(forDiarEngine: engine)
        return offered.first { $0.id == id } ?? offered[0]
    }

    static func overlapEngine(id: String) -> ModelInfo {
        overlapEngines.first { $0.id == id } ?? overlapSeparation
    }

    /// Project models folder (HF_HOME used by download-best-models.sh). Derived
    /// from PythonRuntime.projectDir so it tracks wherever the repo actually is
    /// — not a hardcoded ~/Documents path that only matched the first machine.
    static let modelsDir = PythonRuntime.projectDir.appendingPathComponent("models")

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
        if model.id == diarizenDiarization.id {
            // BOTH halves are required and both are checked. The checkpoint is a
            // normal HF-hub snapshot, but the ENGINE is a vendored source tree
            // that `.venv-diarizen` was built from — an editable install of it
            // freezes as a `git+` ref, which `build.sh` strips, so the tree being
            // present is what proves the bundle can rebuild the runtime at all.
            let hub = ModelCatalog.modelsDir
                .appendingPathComponent("hub/models--BUT-FIT--diarizen-meeting-base")
            let vendor = PythonRuntime.scriptsDir
                .appendingPathComponent("diarizen/vendor/diarizen")
            return fm.fileExists(atPath: hub.path) && fm.fileExists(atPath: vendor.path)
        }
        if model.id == nemoDiarization.id {
            // Non-standard layout, the mossformer2 precedent: NeMo's checkpoints
            // are two flat `.nemo` archives under models/nemo/, not an HF-hub
            // cache. BOTH are required and both are checked — the sidecar hands
            // `ClusteringDiarizer` two absolute paths and a missing one fails deep
            // inside the pipeline at Stop, which is the worst moment to find out.
            // (Local paths, never NeMo's pretrained NAMES, is also what keeps the
            // NGC downloader out of a 100 %-offline build.)
            let dir = modelsDir.appendingPathComponent("nemo")
            return ["titanet_large.nemo", "vad_multilingual_marblenet.nemo"].allSatisfy {
                fm.fileExists(atPath: dir.appendingPathComponent($0).path)
            }
        }
        let hubName = "models--" + model.hfRepo.replacingOccurrences(of: "/", with: "--")
        return fm.fileExists(atPath: modelsDir.appendingPathComponent("hub/\(hubName)").path)
    }
}

// MARK: - Languages offered in pickers

/// Which languages each chunked model is offered, and what happens to a stored
/// choice the selected model cannot honour.
///
/// THE STRUCTURE: the three tiers are not a taste ranking — each tier boundary
/// is literally a model's published roster, so "how many independently-trained
/// models chose to support this language" is readable straight off the table:
///
/// | Cumulative | is exactly | Count |
/// |---|---|---|
/// | tier 1 | Granite Speech 4.1's languages | 6 |
/// | tier 1 + 2 | Voxtral Mini's languages | 13 |
/// | tier 1 + 2 + 3 | Qwen3-ASR's languages | 30 |
///
/// Whisper large-v3 supports 100. The other 70 are DELIBERATELY NOT OFFERED
/// (owner, 2026-07-29): the request was "only the ones with good WER", and
/// per-language WER for Whisper exists publicly only as a figure in OpenAI's
/// release — not as numbers that could be quoted honestly. Tier membership is
/// the verifiable proxy that was agreed on instead. Every code below was checked
/// against the INSTALLED `mlx_whisper.tokenizer.LANGUAGES` (all 30 present), so
/// nothing here can be rejected by Whisper's own tokenizer.
///
/// The old single `all` list (auto/id/en/ms/zh/ja) was wrong in both directions
/// at once: it hid 94 of Whisper's languages, and it offered Indonesian and
/// Malay for Voxtral and Granite, which have neither — picking Indonesian sent
/// `--language id` to a 13-language model.
enum Languages {
    typealias Entry = (code: String, name: String)

    /// Always first in a picker, for every model that takes a language at all.
    static let auto: Entry = ("auto", "Auto-detect")

    /// Tier 1 — the six every chunked model here supports (= Granite's roster).
    static let tier1: [Entry] = [
        ("en", "English"), ("de", "German"), ("fr", "French"),
        ("es", "Spanish"), ("pt", "Portuguese"), ("ja", "Japanese"),
    ]

    /// Granite Speech 4.1 2B **NAR** — the five its own card publishes.
    /// Deliberately NOT `tier1`: see `supported(forModel:)` for why the two
    /// stopped being the same list.
    static let graniteNAR: [Entry] = [
        ("en", "English"), ("fr", "French"), ("de", "German"),
        ("es", "Spanish"), ("pt", "Portuguese"),
    ]

    /// Tier 2 — tier 1 + these seven is exactly Voxtral Mini's 13.
    static let tier2: [Entry] = [
        ("ar", "Arabic"), ("hi", "Hindi"), ("it", "Italian"), ("nl", "Dutch"),
        ("zh", "Chinese"), ("ko", "Korean"), ("ru", "Russian"),
    ]

    /// Tier 3 — tiers 1+2 + these seventeen is exactly Qwen3-ASR's 30.
    static let tier3: [Entry] = [
        ("id", "Indonesian"), ("ms", "Malay"), ("th", "Thai"),
        ("vi", "Vietnamese"), ("tr", "Turkish"), ("yue", "Cantonese"),
        ("sv", "Swedish"), ("da", "Danish"), ("fi", "Finnish"),
        ("pl", "Polish"), ("cs", "Czech"), ("tl", "Filipino"),
        ("fa", "Persian"), ("el", "Greek"), ("hu", "Hungarian"),
        ("mk", "Macedonian"), ("ro", "Romanian"),
    ]

    /// The full offered set (30). Whisper and Qwen3 get this.
    static let allTiers: [Entry] = tier1 + tier2 + tier3

    /// The realtime (Nemotron) picker's list — UNCHANGED on purpose.
    ///
    /// Nemotron 3.5's card claims 40 locales but does not publish the roster,
    /// and the NVIDIA model card is gated (HTTP 401), so there is no honest
    /// source for 40 codes. Rather than invent them, `realtime.language` keeps
    /// exactly the six entries the single old list offered. Do not "unify" this
    /// with the tiers: the tiers are evidence about the CHUNKED models, and
    /// applying them here would be a claim about Nemotron nobody has checked.
    static let realtime: [Entry] = [
        auto, ("id", "Indonesian"), ("en", "English"),
        ("ms", "Malay"), ("zh", "Chinese"), ("ja", "Japanese"),
    ]

    /// Parakeet TDT 0.6b v3 — the 25 its OWN checkpoint's card publishes.
    ///
    /// Read from the front-matter of
    /// `models/hub/models--mlx-community--parakeet-tdt-0.6b-v3/**/README.md`, not
    /// from NVIDIA's card for the base model and not from the family. That is
    /// the Granite lesson, which cost a correction in BOTH directions: the card
    /// of the checkpoint on disk is the authority for the checkpoint on disk.
    ///
    /// Worth stating because it is the practical difference between the two
    /// realtime engines: there is NO Indonesian, Malay, Chinese or Japanese
    /// here — four of the five concrete options the Nemotron picker offers. A
    /// stored `realtime.language` naming one of them resolves to Auto-detect
    /// while Parakeet is selected (`resolveRealtime`), rather than travelling to
    /// a sidecar that has never heard of it.
    static let parakeet: [Entry] = [
        ("en", "English"), ("es", "Spanish"), ("fr", "French"),
        ("de", "German"), ("bg", "Bulgarian"), ("hr", "Croatian"),
        ("cs", "Czech"), ("da", "Danish"), ("nl", "Dutch"),
        ("et", "Estonian"), ("fi", "Finnish"), ("el", "Greek"),
        ("hu", "Hungarian"), ("it", "Italian"), ("lv", "Latvian"),
        ("lt", "Lithuanian"), ("mt", "Maltese"), ("pl", "Polish"),
        ("pt", "Portuguese"), ("ro", "Romanian"), ("sk", "Slovak"),
        ("sl", "Slovenian"), ("sv", "Swedish"), ("ru", "Russian"),
        ("uk", "Ukrainian"),
    ]

    /// Fun-ASR MLT Nano 2512 — what the CODE accepts, not what the card claims.
    ///
    /// FunAudioLLM's card lists 30 languages for this checkpoint. Mainline
    /// mlx-audio's `_map_language()` maps eleven ISO codes and RAISES a
    /// `ValueError` on anything else, so 30 is a property of the weights and 3
    /// is a property of the thing that runs them. The Granite lesson said the
    /// checkpoint's own card beats the family's; this is the step after that —
    /// the IMPLEMENTATION beats the checkpoint's card when they disagree.
    ///
    /// Nine of those eleven codes (cjy, cmn, gan, hak, hsn, nan, wuu, yue, zh)
    /// all resolve to the same prompt word, 中文, so offering them separately
    /// would be nine ways to pick Chinese. Only `zh` is listed.
    ///
    /// UNLIKE every other picker in this file, a wrong code here is not merely
    /// ignored — it raises inside the model and takes the lane down with it.
    /// `resolveRealtime` is what stops that, and the sidecar checks again.
    static let funasr: [Entry] = [
        ("en", "English"), ("zh", "Chinese"), ("ja", "Japanese"),
    ]

    // MARK: - The REALTIME pickers, per engine
    //
    // Deliberately their own four functions rather than extra cases in
    // `supported(forModel:)` and friends below. Those are documented as
    // reasoning about the CHUNKED models — their `default:` falls back to the
    // full 30 "because ChunkedASRModelFactory falls back to Qwen3", which is a
    // sentence about a different factory entirely. Folding a realtime engine
    // into that default would silently offer Nemotron 30 languages nobody has
    // checked, which is exactly what `Languages.realtime`'s own comment forbids.

    /// What a realtime engine actually supports — WITHOUT `auto`.
    static func realtimeSupported(forModel id: String) -> [Entry] {
        switch id {
        case RealtimeASRService.parakeetModelID: return parakeet
        case RealtimeASRService.funasrModelID: return funasr
        // Nemotron keeps EXACTLY the five it has always offered. Its card claims
        // 40 locales and publishes no roster, and NVIDIA's own card is gated
        // (HTTP 401), so there is nothing honest to widen it to.
        default: return realtime.filter { $0.code != auto.code }
        }
    }

    /// What a realtime picker shows: `auto` first, then the engine's languages.
    static func realtimeEntries(forModel id: String) -> [Entry] {
        [auto] + realtimeSupported(forModel: id)
    }

    /// The code that may actually be sent for this realtime engine. Resolved at
    /// the READ boundary (`RealtimeASRService.Config.fromSettings`), so a stored
    /// value from the other engine is overridden while this one is selected and
    /// is still there when the user switches back.
    static func resolveRealtime(language code: String, forModel id: String) -> String {
        guard code != auto.code else { return auto.code }
        return realtimeSupported(forModel: id).contains { $0.code == code }
            ? code : auto.code
    }

    /// Why this realtime engine's setting has no effect, or nil when it uses one.
    ///
    /// Shown next to a LIVE picker, never a disabled one — the 2026-07-31
    /// reversal, made after the owner was shown the same class of measurement
    /// twice. The control is real, the stored choice survives an engine switch,
    /// the code is even forwarded to the sidecar and logged there, and the MODEL
    /// ignores it. MEASURED for Parakeet on 2026-08-11: `ParakeetTDT.generate()`
    /// has no `language` parameter at all — only `**kwargs`, the same
    /// swallow-without-raising shape as Granite's and Voxtral's — and driven
    /// with None, "fr", "de" and a nonsense "xx" it returned byte-identical
    /// 264-character output every time and raised nothing.
    ///
    /// Do NOT "fix" the sidecar to honour it: there is nothing there to honour.
    static func realtimeNote(forModel id: String) -> String? {
        switch id {
        case RealtimeASRService.parakeetModelID:
            // Short on screen; the evidence stays here. Measured 2026-08-11:
            // `ParakeetTDT.generate()` has no `language` parameter, only
            // `**kwargs` — driven with None, "fr", "de" and a nonsense "xx" it
            // returned byte-identical 264-char output and raised nothing. Same
            // swallow-without-raising shape as Granite and Voxtral.
            return "Detected automatically — this setting will not change the transcript."
        default:
            return nil
        }
    }

    /// One line naming what a realtime engine handles, or nil when there is no
    /// honest roster to state. Nemotron gets nil for the reason its picker's
    /// comment gives: 40 locales claimed, none published, and inventing them
    /// would be worse than saying nothing.
    static func realtimeSupportedSentence(forModel id: String) -> String? {
        switch id {
        case RealtimeASRService.parakeetModelID:
            return "Supported languages: "
                + parakeet.map { $0.1 }.joined(separator: ", ") + "."
        case RealtimeASRService.funasrModelID:
            // Stated as what this build supports, deliberately NOT as what the
            // model card claims (30). See `funasr` above.
            return "Supported languages: "
                + funasr.map { $0.1 }.joined(separator: ", ") + "."
        default:
            return nil
        }
    }

    /// Does this chunked model take a language at all?
    ///
    /// Verified in `ChunkedASRModels.swift`: `GraniteSpeechModel` and
    /// `MossTranscribeDiarizeModel` both return nil from `languageArgument` for
    /// EVERY code — Granite's `generate()` has no language parameter (it is
    /// swallowed by `**kwargs`), and MOSS is steered by its prompt. A picker for
    /// them cannot change one word of output, so it is shown DISABLED with the
    /// reason rather than populated: an active-looking control that does nothing
    /// is worse than none.
    /// Whether this model's picker can be USED. Every model now can — see below.
    ///
    /// DELIBERATELY NO LONGER derived from `noLanguageParameterNote`. Those two
    /// were one thing until 2026-07-31, which meant "we can explain why it does
    /// nothing" and "you may not touch it" could not be separated. The owner
    /// asked three times for Granite and MOSS to be selectable and, after being
    /// shown the measurements twice, confirmed. So the note stays — the UI still
    /// says plainly that the setting has no effect on those two — but the control
    /// is live.
    ///
    /// WHAT THIS MEANS, so nobody later reads it as a bug: for Granite and MOSS
    /// the chosen code is stored, survives a model switch, and is even passed to
    /// the sidecar — and the MODEL ignores it. Measured, not assumed:
    ///   * Granite driven directly with `language="fr"`, `"de"` and a nonsense
    ///     `"xx"` returned byte-identical 492-character output every time and
    ///     raised nothing (`generate()` takes `**kwargs`, so the flag is
    ///     swallowed, not rejected); its implementation contains zero
    ///     occurrences of "language".
    ///   * MOSS asked in its prompt for Indonesian and for Chinese returned
    ///     identical English, 6 segments, format intact.
    /// Both detect the spoken language from the audio and cannot be told
    /// otherwise. Do NOT "fix" the sidecars to honour this flag — there is
    /// nothing there to honour.
    static func acceptsLanguage(model id: String) -> Bool { true }

    /// Why this model's setting has no effect, or nil when it really uses one.
    /// Shown next to a LIVE picker now, not a disabled one — the text is the only
    /// thing left telling the truth, so it must not be dropped.
    static func noLanguageParameterNote(forModel id: String) -> String? {
        switch id {
        case "granite":
            return "Granite Speech has no language parameter — it detects the language itself. You can set this, but it will not change the transcript."
        case "moss", "moss-diar":
            return "MOSS is driven by its prompt and has no language parameter — it detects the language itself. You can set this, but it will not change the transcript."
        // FOUND 2026-07-31, and it had been believed to work: the checkpoint's
        // `model_type` is `voxtral_realtime`, whose `generate()` has NO language
        // parameter — only `**kwargs`, which swallows the flag exactly as
        // Granite's does. MEASURED on real speech: no language, "fr", "de" and a
        // nonsense "xx" all returned the same 226 characters, and none raised.
        // The sidecar comment claiming Voxtral "really is honoured (contrast
        // granite-service.py)" was simply wrong and has been corrected.
        case "voxtral":
            return "Voxtral's realtime checkpoint has no language parameter — it detects the language itself. You can set this, but it will not change the transcript."
        // The FOURTH of this group, added 2026-09-02, and checked the way the
        // Voxtral entry above taught: read the signature, do not trust the card.
        // `VibeVoiceASRForConditionalGeneration.streaming_generate` takes
        // prompt_text, chunk_duration, text_audio_delay, sample_rate,
        // max_new_tokens_per_chunk, temperature, context_info, encode_mode,
        // repetition_penalty and pad_last_chunk — and no language. The model
        // card advertises 10 languages; the API offers no way to ask for one.
        case "vibevoice":
            return "VibeVoice has no language parameter — it detects the language itself. You can set this, but it will not change the transcript."
        default:
            return nil
        }
    }

    /// The languages a model actually supports — WITHOUT `auto` (that is a UI
    /// affordance, not a language). Pure and static so it is unit-testable.
    ///
    /// An unknown id gets the full set because `ChunkedASRModelFactory` falls
    /// back to Qwen3 for one, so the offered list matches the model that would
    /// really run.
    static func supported(forModel id: String) -> [Entry] {
        switch id {
        case "voxtral": return tier1 + tier2       // 13
        // Granite is tier 1 MINUS Japanese — NOT tier1, and this cost a
        // correction twice. An earlier note said five, a later one "corrected"
        // it to six because ibm-granite's *base* Granite Speech card lists
        // Japanese. We do not ship the base model: we ship the **NAR** variant,
        // whose own card (models/hub/…granite-speech-4.1-2b-nar-mlx, README
        // front-matter) says exactly `language: [en, fr, de, es, pt]`. Read the
        // card of the checkpoint that is actually on disk, not the family's.
        //
        // `tier1` keeps Japanese because it is not Granite's list after all —
        // it is the shared base of VOXTRAL's 13, and Voxtral's card really does
        // list `ja` (verified in the same way).
        case "granite": return graniteNAR          // 5
        // MOSS's card says "50+" and never enumerates them, so there is no
        // upstream roster to mirror. Its picker is selectable (owner, 2026-07-31)
        // and therefore needs rows, so it offers the project's shared 30 — the
        // same set Whisper and Qwen3 publish. That is stated as OUR list, not as
        // a claim about MOSS: `noLanguageParameterNote` says the setting changes
        // nothing for this model, and `supportedSentence` says the roster is
        // unpublished. Nothing here invents a per-language claim upstream did not
        // make, which is the same rule that keeps Nemotron's 40 locales unlisted.
        case "moss", "moss-diar": return allTiers
        default: return allTiers                   // whisper, qwen3 — 30
        }
    }

    /// What a picker shows: `auto` first, then the model's languages. Models
    /// that take no language get `auto` alone (their picker is disabled).
    static func pickerEntries(forModel id: String) -> [Entry] {
        [auto] + supported(forModel: id)
    }

    /// The language code that may actually be sent for this model.
    ///
    /// An unsupported stored value resolves to "auto" rather than being passed
    /// through — the failure this prevents is a stale UserDefaults value from an
    /// earlier model reaching a sidecar (`--language id` to Voxtral). Resolving
    /// at READ time means the user's stored setting is never rewritten behind
    /// their back; it is only overridden while an incompatible model is picked.
    static func resolve(language code: String, forModel id: String) -> String {
        guard code != auto.code else { return auto.code }
        guard acceptsLanguage(model: id) else { return auto.code }
        return supported(forModel: id).contains { $0.code == code } ? code : auto.code
    }

    /// Display name for a code, for the "…was dropped" note. Falls back to the
    /// raw code so an unknown one is still named rather than silently vanishing.
    /// One line naming what a model handles, for the models whose picker is
    /// disabled. A STATEMENT of coverage, never an offer of choice — both such
    /// models detect the language from the audio themselves.
    ///
    /// An empty roster is reported as unpublished rather than as "none": MOSS's
    /// card claims 50+ languages without ever enumerating them, and inventing a
    /// list would be worse than admitting there isn't one. Same wording rule the
    /// Nemotron tab already follows for its unpublished 40 locales.
    static func supportedSentence(forModel id: String) -> String {
        // MOSS's rows are the project's shared 30, chosen only so its picker has
        // something to show — upstream publishes no roster at all. Naming them
        // here would turn a UI convenience into a factual claim about the model,
        // so it says what is actually known instead.
        if id == "moss" || id == "moss-diar" {
            return "Upstream states MOSS handles 50+ languages but does not publish the list; the codes offered here are this app's shared list."
        }
        let names: [String] = supported(forModel: id).map { $0.1 }
        guard !names.isEmpty else {
            return "Upstream states this model handles many languages but does not publish the list."
        }
        return "Supported languages: " + names.joined(separator: ", ") + "."
    }

    static func name(for code: String) -> String {
        if code == auto.code { return auto.name }
        // `parakeet` is in the list because nine of its codes (bg, hr, et, lv,
        // lt, mt, sk, sl, uk) appear in no tier and in no other roster — without
        // it the "…was dropped" note would name a raw code instead of a language.
        return (allTiers + realtime + parakeet).first { $0.code == code }?.name ?? code
    }
}
