import SwiftUI

/// Models → Chunked: select the batch ASR model for the final transcript.
struct ChunkedModelTab: View {
    /// THE MASTER SWITCH — distinct from `chunked.finalPass` below, which reads
    /// like one and is not: that key governs only the extra pass after Stop, and
    /// with it off chunked ASR still runs all meeting. This one stops the model
    /// loading at all. See `ModelLoader.wantsChunked`.
    @AppStorage("chunked.enabled")     private var enabled = true
    @AppStorage("chunked.model")       private var model = ShippedDefaults.chunkedModel
    /// Written ONLY by `selectChunkedModel` below — this tab does not otherwise
    /// own the diarization engine.
    @AppStorage("diarization.engine")  private var diarizationEngine = ShippedDefaults.diarizationEngine
    @AppStorage("chunked.language")    private var language = "auto"
    @AppStorage("chunked.intervalSec") private var intervalSec = 30
    // Pipeline-level, NOT per-model, so both are shown for every model.
    @AppStorage("chunked.finalPass")       private var finalPass = true
    /// The SCOPE of that pass. Restored 2026-09-04 after the owner reported the
    /// tail-only pin as a defect — "it not remove the chunk it use the chunked
    /// text instead re transcribe at start to stop". A predecessor key,
    /// `chunked.continueOnStop`, sat here until 2026-08-06 with the INVERTED
    /// sense ("Continue from live text (tail only)"); the new key is deliberately
    /// NOT that one, so a value stored under the old name cannot be read with the
    /// opposite meaning.
    @AppStorage("chunked.fullPassAtStop")  private var fullPassAtStop = ShippedDefaults.chunkedFullPassAtStop

    /// Set when switching models drops the picked language, so the user is told
    /// which one went and why instead of finding the picker quietly on
    /// Auto-detect. Cleared as soon as they pick a language themselves.
    @State private var droppedLanguage: String?

    var body: some View {
        Group {
            // The switch comes FIRST and everything it controls follows, the same
            // shape as Overlap and Detect overlap. The warning is deliberately
            // ABOVE the toggle, not below it: what is lost here is the transcript
            // itself, and a consequence printed underneath is read after the
            // decision instead of before it.

            SettingToggle(label: "Enable Chunked ASR", isOn: $enabled)

            chunkedBody
            
        }
    }

    /// Model choice and every chunked setting. Only reachable while `enabled` is
    /// on — with it off no chunked sidecar is loaded at all, so a visible picker
    /// would be choosing a model for something that does not run.
    @ViewBuilder
    private var chunkedBody: some View {
        Group {
            // PIPELINE-LEVEL SETTINGS FIRST, directly under the master switch
            // (owner, 2026-08-06). These four apply to every model identically —
            // what happens at Stop, which language is asked for, and how long a
            // chunk is — so they belong with the switch that turns the pass on,
            // not scattered below a model choice they do not depend on.
            //
            // Everything after them is about WHICH model and ITS own options.
            // The one coupling worth knowing: the language picker's roster is
            // per-model (Whisper 100, Qwen3 30, Voxtral 13, Granite 5), so
            // choosing a model below can narrow the list above — `onChange` at
            // the end of this view resets a code that stops being supported and
            // says so, rather than leaving the picker quietly on Auto-detect.
            SettingBlock(title: "Chunked interval — faster updates vs higher accuracy") {
                Picker("", selection: $intervalSec) {
                    Text("15 s").tag(15)
                    Text("30 s").tag(30)
                    Text("60 s").tag(60)
                    Text("120 s").tag(120)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text(intervalHint)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            languageBlock

            stopPassBlock

            // THE SELECTED MODEL'S OWN OPTIONS, above the card list rather than
            // below it (owner, 2026-08-06) and inside a bordered box. Position
            // and border do the same job: these belong to ONE card, unlike
            // everything above them, which applies to every model. Loose at the
            // bottom of a long tab they read as more pipeline settings.
            //
            // Shown ONLY while that model is selected — the rule OverlapTab
            // follows for its two engines. The branches are mutually exclusive,
            // so two option boxes can never be on screen at once.
            if model == "whisper" {
                ModelOptionsBox(
                    title: "WHISPER OPTIONS",
                    note: "Every control starts at Whisper's own default — untouched, this "
                        + "transcribes exactly as before.") {
                    WhisperOptionsBlock()
                }
            }

            if model == "qwen3" {
                ModelOptionsBox(
                    title: "QWEN3 OPTIONS",
                    note: "Every control starts at Qwen3's own default — untouched, this "
                        + "transcribes exactly as before.") {
                    Qwen3OptionsBlock()
                }
            }

            // Shows only when a Remote channel is configured and Voxtral is picked —
            // the model choice and the thing that makes it unworkable live on
            // different tabs, so the warning has to appear on both.
            DualStreamVoxtralWarning()

            ForEach(ModelCatalog.chunked) { m in
                Button(action: {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                        selectChunkedModel(m.id)
                    }
                }) {
                    ModelCardView(model: m, selected: model == m.id)
                }
                .buttonStyle(.plain)
            }

            ModelInstallStatus(model: ModelCatalog.chunkedModel(id: model))

            // The transcript shows a per-row `asr` number, and its ABSENCE is the
            // thing that needs explaining — it looks like a missing feature
            // otherwise. It is a property of the model, not of the audio.
            Text("Transcript confidence is available with Whisper only — the other models expose no confidence signal.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)

            // NO note for MOSS (owner, 2026-08-06). There were two, then one, now
            // none. What it said is still true and still happens — picking MOSS
            // here also sets `diarization.engine` (see `selectChunkedModel`) —
            // but the Diarization tab shows that selection plainly, and the rail
            // moves Aligner and Overlap into "NOT USED BY YOUR MODELS" the moment
            // it takes effect. The consequence is visible; the paragraph was not
            // carrying it alone.

        }
        // The picked language can become unsupported the moment the model
        // changes, so the reset happens here — not at the next recording, where
        // it would be invisible. Only the note is new to the user: the value
        // could never have been sent anyway (`Languages.resolve` is applied
        // again when the sidecar config is built), so this is the same decision
        // made where it can be explained.
        .onChange(of: model) { _, newModel in
            // A model that has no language parameter is NOT a reason to erase
            // the choice: it ignores the setting either way (the disabled picker
            // says so, and the sidecar config resolves to auto), so leaving the
            // stored code alone means it is still there when a model that CAN
            // use it is selected again.
            guard Languages.acceptsLanguage(model: newModel) else {
                droppedLanguage = nil
                return
            }
            let resolved = Languages.resolve(language: language, forModel: newModel)
            if resolved != language {
                droppedLanguage = language
                language = resolved
            } else {
                droppedLanguage = nil
            }
        }
    }

    /// The stop-time pass — SHARED by every model, deliberately. What happens at
    /// Stop is a property of the pipeline, like the language and the chunk
    /// interval it now sits beside at the top of the tab; only the decoding
    /// options are per-model, and those stay below the card list. The measured
    /// cost of the full pass does depend on the selected model, so that one line
    /// (`fullPassCostNote`) is the only part that varies.
    @ViewBuilder
    private var stopPassBlock: some View {
        SettingBlock(title: "") {
            SettingToggle(label: "Re-transcribe at stop", isOn: $finalPass)
            Text("After you stop, transcribe with this model once more so the audio no chunk "
                 + "covered yet is as accurate as the rest.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)

            if finalPass {
                SettingToggle(label: "…\(AudioRecorder.fullPassToggleLabel), not just the last seconds",
                              isOn: $fullPassAtStop)
                if fullPassAtStop {
                    // The cost is the whole decision, so it is stated where the
                    // decision is made and it names the SELECTED model — the one
                    // line on this tab that varies per model, for that reason.
                    Text("On: the recording is transcribed again from the beginning and the live "
                         + "chunk text is replaced. Windows are cut at silence, so nothing is "
                         + "split mid-word. " + AudioRecorder.fullPassCostNote(chunkedModelID: model))
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                    if let refusal = AudioRecorder.chunkedFullPassRefusalMessage(chunkedModelID: model) {
                        // Shown where the choice is made AND enforced as a hard
                        // startup refusal, so it cannot be discovered at Stop.
                        Text(refusal)
                            .font(.system(size: 11))
                            .foregroundColor(Theme.amber)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    Text("Off: only the seconds after the last chunk are transcribed. Everything "
                         + "before that keeps the text the live chunks produced.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if !finalPass {
                // Deliberately NOT naming the engine. It said "(Nemotron)" until
                // 2026-08-11, when a second realtime engine landed and the
                // sentence became capable of naming a model this session is not
                // running — the same defect as the rail reading "pyannote" for a
                // DiariZen session. Which engine it is lives on the Realtime tab,
                // where it can be right by construction.
                Text("Off: nothing is transcribed at stop. The last seconds keep the realtime "
                     + "engine's text that is already on screen instead of being deleted — so "
                     + "no speech is lost, but that tail has no transcript confidence and no word "
                     + "alignment, and it is the less accurate engine. With MOSS selected as both "
                     + "the chunked model and the diarization engine, the tail also keeps no "
                     + "speaker labels, since MOSS produces them in the same pass as the words.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

        }
    }

    /// Language picker for the SELECTED model only, plus the reason when it is
    /// inert.
    ///
    /// Two of the five models (Granite, MOSS) take no language argument at all.
    /// Their picker used to be DISABLED for that reason; since 2026-07-31 it is
    /// selectable on owner request, so the only thing left telling the user the
    /// setting does nothing is `noLanguageParameterNote` below. Do not drop that
    /// text — `LanguageSupportTests.testEveryModelsPickerIsSelectableButTwoStillWarn`
    /// fails if it goes, and without it the control is a silent lie.
    ///
    /// `accepts` is retained rather than deleted: it is the one place that would
    /// disable a picker again if a future model genuinely cannot take a code, and
    /// `Languages.acceptsLanguage` is where that decision belongs.
    @ViewBuilder
    private var languageBlock: some View {
        let accepts = Languages.acceptsLanguage(model: model)
        let entries = Languages.pickerEntries(forModel: model)

        SettingBlock(title: "Language") {
            // Reading through `resolve` means the picker can never display a
            // blank row for a stored code this model does not have, without
            // rewriting what is on disk.
            LanguagePicker(
                selection: Binding(
                    get: { Languages.resolve(language: language, forModel: model) },
                    set: { language = $0; droppedLanguage = nil }),
                entries: entries)
                .disabled(!accepts)
                .opacity(accepts ? 1 : 0.45)

            if let why = Languages.noLanguageParameterNote(forModel: model) {
                Text(why)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
                // The roster, stated as well as offered. The picker above IS
                // selectable for these two now, but selecting changes nothing —
                // measured, not assumed: Granite returns byte-identical output
                // for "fr", "de" and even a nonsense "xx" (its `**kwargs` swallows
                // the flag without raising), and MOSS returns identical English
                // when its prompt asks for Indonesian or Chinese. Both detect the
                // spoken language from the audio. This line names what each one
                // can recognise, which the rows alone cannot convey.
                Text(Languages.supportedSentence(forModel: model))
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("\(entries.count - 1) languages — the list this model publishes. Other models offer different lists, so switching models can drop a language that only some of them support.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let dropped = droppedLanguage {
                Text("\(Languages.name(for: dropped)) is not supported by \(ModelCatalog.chunkedModel(id: model).name) — the language was set back to Auto-detect.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.amber)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Facts: Whisper large-v3 is trained on 30 s windows (its optimum);
    /// chunks under ~15 s measurably increase WER from lost context;
    /// longer chunks improve accuracy further at the cost of update delay.
    private var intervalHint: String {
        switch intervalSec {
        case 15:  return "Fastest updates — below models' optimal context; accuracy drops noticeably."
        case 30:  return "Recommended — Whisper's training window (30 s) and a strong balance for Qwen3/Voxtral."
        case 60:  return "Higher accuracy — more context per chunk, updates once a minute."
        default:  return "Best accuracy — maximum context; transcript updates every 2 minutes."
        }
    }

    /// Picking MOSS as the chunked model also selects MOSS as the diarization
    /// engine, because that pairing is the ONE configuration where MOSS is used
    /// as its authors built it: a single model returning the words, the speakers
    /// and the times from ONE forward pass. Any other engine alongside MOSS-as-ASR
    /// throws that away — MOSS's own attribution is discarded and the text is
    /// re-split by another diarizer's spans, which is an estimate.
    ///
    /// It is a SELECTION, not a lock. The engine card in Models → Diarization
    /// visibly moves to MOSS and the owner can move it straight back; nothing here
    /// prevents MOSS-as-ASR with pyannote if that is deliberately wanted. The
    /// house rule this respects is "never substitute SILENTLY" — the substitution
    /// is on screen, one tab away, and reversible.
    ///
    /// Deliberately NOT symmetric: choosing MOSS as the ENGINE must not force the
    /// chunked model, because "another model does the ASR, a second MOSS process
    /// labels" is a real supported mode (`needsSecondMossProcess`).
    private func selectChunkedModel(_ id: String) {
        model = id
        // ONE rule, both directions (2026-08-06): MOSS chunked ⟺ MOSS diarization.
        // Correcting it here as well as in `DiarizationTab.onAppear` is what stops
        // an engine staying stored with no card on screen selecting it. Asking
        // `diarizationEngineIsSelectable` rather than testing for MOSS by hand
        // means this site cannot drift from the card filter.
        if !ModelLoader.diarizationEngineIsSelectable(diarizationEngine, chunkedID: id) {
            diarizationEngine = ModelLoader.fallbackDiarizationEngine(chunkedID: id)
        }
    }
}

/// The BEST-ACCURACY preset for Whisper, and the measurement behind it.
///
/// Owner asked for a preset and asked for it to be MEASURED, not assumed. It was,
/// on `recordings/meeting-2026-07-30T04-53-29Z.wav` (67 min of real meeting
/// audio), driving `whisper-service.py` directly: 8 consecutive 30 s chunks, then
/// the 10 LOWEST-CONFIDENCE chunks out of 40 sampled across the whole meeting
/// (conf 0.766–0.867) so the hard cases were covered, not just the easy ones.
///
/// THE RESULT IS THAT THE SHIPPED DEFAULTS ARE THE MOST ACCURATE SETTING. Every
/// knob measured either did nothing at all or removed real speech:
///
/// | knob | measured |
/// |---|---|
/// | `best_of` 2 and 5 | **0 of 10 hard chunks changed**, byte-identical, same time. Not broken — the sidecar logs `decoding: best_of=5` and mlx_whisper receives it; it only applies on temperature fallback, and no chunk, however poor, ever fell back |
/// | `compression_threshold` 2.0 / 3.5 | no change |
/// | `no_speech_threshold` 0.3 / 0.9 | no change (the silence rule needs a LOW logprob too, and the default never binds) |
/// | `logprob_threshold` −2.0 | no change |
/// | `logprob_threshold` −0.5 | changed 3 of 10 — and **deleted real sentences**, at 10× the time (204 s vs 20 s) |
/// | `hallucination_silence_sec` 2 s | changed 4 of 8 — also **deleted real sentences** |
///
/// THE TRAP THIS MEASUREMENT WALKED INTO, recorded because it nearly produced the
/// opposite preset: both knobs that "worked" RAISED the confidence number.
/// `logprob −0.5` took conf 0.9056 → 0.9110 and `hallucination_silence` 0.9056 →
/// 0.9126. Confidence is `exp(word-weighted mean avg_logprob)`, so **deleting the
/// least-certain words raises it mechanically** — it can always be improved by
/// removing text. What the diffs actually showed being removed:
///
///   "Unfortunately, I never got to listen to them."
///   "Now one of two things can happen."
///   "…than be scared by a jobless future I started to realize that I was a little"
///
/// Coherent English on audio measured at RMS 0.062–0.092, against this project's
/// 0.004 silence floor — speech, not silence. That is the over-deletion direction
/// CLAUDE.md names as the dangerous one, since deleted text leaves no trace in the
/// transcript. **Confidence is therefore NOT a usable score for comparing decode
/// settings, and any future preset work here must diff the TEXT.**
enum WhisperPreset {
    /// The measured-best values. Identical to the shipped defaults on purpose —
    /// that is the finding, not a placeholder. Kept as an explicit list anyway so
    /// the button restores it from any customisation, and so
    /// `WhisperPresetTests` fails if a default ever drifts away from what was
    /// measured instead of the two silently disagreeing.
    static let bestAccuracy: [(key: String, value: Any)] = [
        ("whisper.initialPrompt", ""),
        ("whisper.bestOf", 0),
        ("whisper.noSpeechThreshold", 0.6),
        ("whisper.logprobThreshold", -1.0),
        ("whisper.compressionThreshold", 2.4),
        ("whisper.hallucinationSilenceSec", 0.0),
    ]

    static func applyBestAccuracy() {
        let d = UserDefaults.standard
        for (key, value) in bestAccuracy { d.set(value, forKey: key) }
    }

    /// Whether the stored options already ARE the preset. Compared with a
    /// tolerance on the doubles because a slider writes 0.6000000000000001.
    static var bestAccuracyIsActive: Bool {
        let d = UserDefaults.standard
        for (key, value) in bestAccuracy {
            switch value {
            case let v as String:
                if (d.string(forKey: key) ?? "") != v { return false }
            case let v as Int:
                if (d.object(forKey: key) as? NSNumber)?.intValue ?? 0 != v { return false }
            case let v as Double:
                let stored = (d.object(forKey: key) as? NSNumber)?.doubleValue
                if abs((stored ?? v) - v) > 0.0001 { return false }
            default:
                return false
            }
        }
        return true
    }
}

/// Whisper's own decoding options, shown only while Whisper is the selected
/// chunked model (Models → Chunked). Every control's first position is what the
/// app has always done — leaving this whole block alone changes nothing.
///
/// Two of these can make the transcript WORSE in ways that are invisible in the
/// output, so the copy names the failure rather than only the feature: an
/// initial prompt can put its own words into the transcript, and the
/// hallucination-silence skip switches Whisper to its own word timing, which
/// this project measured against the Qwen3 aligner and did not adopt.
struct WhisperOptionsBlock: View {
    @AppStorage("whisper.initialPrompt")          private var initialPrompt = ""
    @AppStorage("whisper.bestOf")                 private var bestOf = 0
    @AppStorage("whisper.noSpeechThreshold")      private var noSpeechThreshold = 0.6
    @AppStorage("whisper.logprobThreshold")       private var logprobThreshold = -1.0
    @AppStorage("whisper.compressionThreshold")   private var compressionThreshold = 2.4
    // `whisper.task` and `whisper.autoDetectLanguage` had controls here and were
    // removed on 2026-08-06 (owner) — both are now fixed at their defaults, in
    // `ChunkedASRService.WhisperOptions.fromSettings`, not merely hidden.
    @AppStorage("whisper.hallucinationSilenceSec") private var hallucinationSilenceSec = 0.0

    var body: some View {
        Group {
            SettingBlock(title: "Preset") {
                HStack(spacing: 10) {
                    Button(action: {
                        WhisperPreset.applyBestAccuracy()
                        // Re-read into this view's @AppStorage mirrors: the
                        // preset writes UserDefaults directly, and a slider bound
                        // to a stale @AppStorage would write its old value back.
                        initialPrompt = ""; bestOf = 0
                        noSpeechThreshold = 0.6; logprobThreshold = -1.0
                        compressionThreshold = 2.4; hallucinationSilenceSec = 0.0
                    }) {
                        Text("Apply best accuracy")
                            .font(.system(size: 12, weight: .bold))
                            .padding(.horizontal, 12).padding(.vertical, 7)
                            .background(RoundedRectangle(cornerRadius: 7).fill(Theme.chip))
                    }
                    .buttonStyle(.plain)

                    if WhisperPreset.bestAccuracyIsActive {
                        Label("in use", systemImage: "checkmark.circle.fill")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(Theme.teal)
                    }
                }

                // The honest version. Measured on 67 minutes of the owner's own
                // meeting audio, including the 10 worst chunks of 40 sampled —
                // see `WhisperPreset` for the table and the diffs.
                Text("Measured, not assumed: on real meeting audio the shipped defaults ARE the "
                     + "most accurate setting. Best-of changed nothing even on the 10 worst "
                     + "chunks, and the two knobs that did change something deleted real "
                     + "sentences while making the confidence number look better. This button "
                     + "restores those values.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Initial prompt — vocabulary help, with a real risk") {
                TextField("e.g. ATND1061, pyannote, Qwen3", text: $initialPrompt,
                          axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                    .font(.system(size: 12))

                Text("Names and jargon Whisper keeps getting wrong. Risk: it is context, not an instruction — on unclear audio Whisper can write words FROM the prompt that nobody said. Keep it a short word list. Logged per chunk in logs/whisper.log.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Sampling candidates (best-of)") {
                Picker("", selection: $bestOf) {
                    Text("Default").tag(0)
                    Text("2").tag(2)
                    Text("5").tag(5)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text("Candidates sampled only when a chunk fails the checks below. No effect on a clean chunk. (No beam search: this runtime has no beam decoder.)")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Silence threshold — \(String(format: "%.2f", noSpeechThreshold))") {
                Slider(value: $noSpeechThreshold, in: 0.2...0.9, step: 0.05)

                Text("How sure Whisper must be a window has no speech before skipping it. Lower risks losing speech; higher risks invented text on silence. The app's own hallucination filter is separate and unaffected.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Confidence floor — \(String(format: "%.1f", logprobThreshold))") {
                Slider(value: $logprobThreshold, in: -3.0...(-0.2), step: 0.1)

                Text("Below this average token log-probability Whisper retries the chunk hotter. Raising it retries more often — slower, not always better.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Repetition limit — \(String(format: "%.1f", compressionThreshold))") {
                Slider(value: $compressionThreshold, in: 1.5...4.0, step: 0.1)

                Text("Text compressing better than this is looping, and the chunk is retried. Lower catches loops sooner but can reject genuinely repetitive speech.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Skip long silences inside a chunk") {
                Picker("", selection: $hallucinationSilenceSec) {
                    Text("Off").tag(0.0)
                    Text("2 s").tag(2.0)
                    Text("5 s").tag(5.0)
                    Text("10 s").tag(10.0)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text("Jumps over silent stretches longer than this — where Whisper is most likely to invent a caption. Turning it on also switches Whisper to its own word timing internally; word times for speaker rows still come from the aligner.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("Changing any of these restarts the Whisper sidecar at the next recording; the options in force are written to logs/whisper.log.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Qwen3-ASR's own decoding options, shown only while Qwen3 is the selected
/// chunked model (Models → Chunked). Every control's first position is what the
/// app has always done — leaving this whole block alone changes nothing.
///
/// TWO CONTROLS, NOT FIFTEEN, and the copy says why rather than leaving it
/// looking unfinished. Qwen3's `generate()` advertises fifteen knobs; each was
/// measured on a real recording before it was given a control, because a picker
/// that reaches the model and changes nothing is worse than no picker (this
/// project shipped exactly that for Granite's language on 2026-08-01). Only
/// these two moved the text. The sampling group is additionally inert as
/// shipped: temperature is 0, so decoding is greedy and top-p/top-k cannot bite.
///
/// Both controls here can make the transcript WORSE in ways that are invisible
/// in the output, so — as in `WhisperOptionsBlock` — the copy names the failure
/// and not only the feature.
struct Qwen3OptionsBlock: View {
    @AppStorage("qwen3.systemPrompt")         private var systemPrompt = ""
    @AppStorage("qwen3.repetitionPenalty")    private var repetitionPenalty = 0.0
    @AppStorage("qwen3.repetitionContextSize") private var repetitionContextSize = 100

    var body: some View {
        Group {
            SettingBlock(title: "Vocabulary prompt — spelling help, with a real risk") {
                TextField("e.g. ATND1061, pyannote, PREP framework",
                          text: $systemPrompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                    .font(.system(size: 12))

                Text("A short list of names and spellings, as context before the audio. Risk: it does not stay where you point it — a prompt also shifts wording and punctuation elsewhere in the chunk. Logged per chunk in logs/qwen3.log.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Repetition penalty") {
                Picker("", selection: $repetitionPenalty) {
                    Text("Off").tag(0.0)
                    Text("1.1").tag(1.1)
                    Text("1.2").tag(1.2)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text("Helps when a chunk loops on one phrase. Risk: it penalises ordinary repetition too — measured here, 1.2 already dropped a real word and a full stop. Off is the default and the safe choice.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Shown only with a penalty active, deliberately: Qwen3 reads this
            // number only where it builds the penalty, so on its own it is a
            // control that cannot affect anything. Hiding it is the honest
            // version of the disabled-picker rule the Language block follows.
            if repetitionPenalty > 0 {
                SettingBlock(title: "Repetition window — \(repetitionContextSize) tokens") {
                    Picker("", selection: $repetitionContextSize) {
                        Text("20").tag(20)
                        Text("100").tag(100)
                        Text("250").tag(250)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    Text("How far back the penalty looks. Smaller hits only immediate loops; larger also penalises a word said earlier in the chunk. 100 is the default.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Text("Changing any of these restarts the Qwen3 sidecar at the next recording, and the options in force are written to logs/qwen3.log at startup — a run at defaults logs \"decoding: defaults\".")
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

