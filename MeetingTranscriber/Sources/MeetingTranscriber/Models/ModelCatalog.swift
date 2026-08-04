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
    /// One entry, one load, however many streams the session has: the realtime
    /// sidecar serves the Office and Remote lanes from a single process (see
    /// `NemotronASRService`), so there is no second row to show. A dual-stream
    /// session's overlay is identical to a single-stream one here.
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

    /// Engines offered on Settings → Models → Diarization (`diarization.engine`).
    static let diarizationEngines: [ModelInfo] = [diarization, spectralDiarization,
                                                  mossDiarization]

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
        default:                     return ModelLoader.pyannoteEngineID
        }
    }

    /// The inverse: the engine card a stored `diarization.engine` value selects.
    static func diarizationEngine(forEngine engine: String) -> ModelInfo {
        switch engine {
        case ModelLoader.mossEngineID:     return mossDiarization
        case ModelLoader.spectralEngineID: return spectralDiarization
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
    static let wordAligner = ModelInfo(
        id: "aligner",
        name: "Qwen3-ForcedAligner 0.6B (bf16)",
        detail: "Word-level timestamps for the chunked transcript · own sidecar, asked after the text is shown",
        badges: ["MLX bf16", "word timestamps", "~0.2 s/chunk", "0.6B"],
        hfRepo: "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"
    )

    /// Engines offered on Settings → Models → Overlap (`overlap.engine`).
    static let overlapEngines: [ModelInfo] = [overlapSeparation, overlapDicow]

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
        return (allTiers + realtime).first { $0.code == code }?.name ?? code
    }
}
