import SwiftUI

/// Models → Chunked: select the batch ASR model for the final transcript.
struct ChunkedModelTab: View {
    @AppStorage("chunked.model")       private var model = "qwen3"
    @AppStorage("chunked.language")    private var language = "auto"
    @AppStorage("chunked.intervalSec") private var intervalSec = 30
    // Pipeline-level, like the interval and the language above — NOT per-model,
    // so this pair is shown for every model. Defaults are today's behaviour, and
    // note that `continueOnStop` defaults to TRUE here while the Diarization
    // tab's identically-named toggle defaults to FALSE; the copy below says so.
    @AppStorage("chunked.finalPass")       private var finalPass = true
    @AppStorage("chunked.continueOnStop")  private var continueOnStop = true

    /// Set when switching models drops the picked language, so the user is told
    /// which one went and why instead of finding the picker quietly on
    /// Auto-detect. Cleared as soon as they pick a language themselves.
    @State private var droppedLanguage: String?

    var body: some View {
        Group {
            Text("Accurate rolling transcript during the meeting: every interval, the chunk is re-transcribed with this model (cut at silence, never mid-word).")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

            // Shows only when a Remote channel is configured and Voxtral is picked —
            // the model choice and the thing that makes it unworkable live on
            // different tabs, so the warning has to appear on both.
            DualStreamVoxtralWarning()

            ForEach(ModelCatalog.chunked) { m in
                Button(action: {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                        model = m.id
                    }
                }) {
                    ModelCardView(model: m, selected: model == m.id)
                }
                .buttonStyle(.plain)
            }

            // The transcript shows a per-row `asr` number, and its ABSENCE is the
            // thing that needs explaining — it looks like a missing feature
            // otherwise. It is a property of the model, not of the audio.
            Text("Transcript confidence is available with Whisper only — the other models expose no confidence signal.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)

            // MOSS is the one entry above that is not purely a recogniser, so the
            // consequences of picking it are not the same as picking another
            // model — and they land on tabs the user is not looking at.
            if model == "moss" {
                Text("MOSS also returns speaker labels and timestamps with the words. Those labels are only used if you select MOSS as the diarization engine too (Models → Diarization) — otherwise pyannote splits its text exactly as it does for any other model. MOSS reports no transcript confidence, and the word aligner is not used with it (it already returns its own times).")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Per-model options, shown ONLY while that model is selected —
            // the same rule OverlapTab follows for its two engines, after the
            // owner found one engine's controls visible while the other was
            // picked. Two models have them now; the branches are mutually
            // exclusive, so no two option blocks can ever be on screen at once.
            if model == "whisper" {
                WhisperOptionsBlock()
            }

            if model == "qwen3" {
                Qwen3OptionsBlock()
            }

            SettingBlock(title: "Chunk interval — faster updates vs higher accuracy") {
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

            stopPassBlock

            languageBlock
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
    /// Stop is a property of the pipeline, exactly like the chunk interval and
    /// the language above; only the decoding options are per-model. The measured
    /// cost of the full pass does depend on the selected model, so that one line
    /// (`fullPassCostNote`) is the only part that varies.
    @ViewBuilder
    private var stopPassBlock: some View {
        let refusal = AudioRecorder.chunkedFullPassRefusalMessage(chunkedModelID: model)

        SettingBlock(title: "Run a transcription pass at stop") {
            SettingToggle(label: "Run a transcription pass at stop", isOn: $finalPass)
            Text("After you stop, transcribe the recording with this model once more so the last "
                 + "seconds — the part no chunk covered yet — are as accurate as the rest.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)

            SettingToggle(label: "Continue from live text (tail only)", isOn: $continueOnStop)
                .disabled(!finalPass)
            Text(continueOnStop
                 ? "On: only the audio since the last chunk is transcribed and appended. This is "
                   + "what the app has always done, and it is the default — note that the "
                   + "same-named toggle in Models → Diarization defaults to OFF, so the pair look "
                   + "alike but start differently."
                 : "Off: the WHOLE recording is transcribed again from start to end, window by "
                   + "window, replacing the text collected during the meeting. More consistent — "
                   + "every chunk is decoded by the same model under the same settings — but it "
                   + "runs after you press Stop and its windows are cut on a fixed clock rather "
                   + "than at silence, so a boundary can land mid-word. "
                   + AudioRecorder.fullPassCostNote(chunkedModelID: model))
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)

            if !finalPass {
                Text("Off: nothing is transcribed at stop. The last seconds keep the realtime "
                     + "(Nemotron) text that is already on screen instead of being deleted — so "
                     + "no speech is lost, but that tail has no transcript confidence and no word "
                     + "alignment, and it is the less accurate engine. With MOSS selected as both "
                     + "the chunked model and the diarization engine, the tail also keeps no "
                     + "speaker labels, since MOSS produces them in the same pass as the words.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Shown where the choice is made, not only at Stop. The same refusal
            // is enforced as a hard startup failure, so this is a warning the
            // user can act on rather than the only thing standing in the way.
            if finalPass, !continueOnStop, let refusal {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11, weight: .bold))
                    Text(refusal + " Recording will not start while this is selected.")
                        .fixedSize(horizontal: false, vertical: true)
                }
                .font(.system(size: 11))
                .foregroundColor(Theme.red)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Theme.red.opacity(0.10))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Theme.red.opacity(0.35), lineWidth: 1)
                        )
                )
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
    @AppStorage("whisper.task")                   private var task = "transcribe"
    @AppStorage("whisper.autoDetectLanguage")     private var autoDetectLanguage = false
    @AppStorage("whisper.hallucinationSilenceSec") private var hallucinationSilenceSec = 0.0

    var body: some View {
        Group {
            Text("Whisper options — these change how Whisper decodes. Every setting below starts at Whisper's own default, so an untouched block transcribes exactly as before.")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

            SettingBlock(title: "Initial prompt — vocabulary help, with a real risk") {
                TextField("e.g. Aggia, ATND1061, pyannote, Qwen3", text: $initialPrompt,
                          axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                    .font(.system(size: 12))

                Text("Text given to Whisper as if it had just been said, before the chunk. Useful for names, jargon and spellings it keeps getting wrong.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Risk: this is context, not an instruction. Whisper continues text, so when the audio is unclear it can write words FROM THE PROMPT that nobody said — not just \"no improvement\". Keep it short and factual (a word list, not a sentence about the meeting). While a prompt is set, every chunk logs \"prompt active\" to logs/whisper.log so suspicious text can be traced back to it.")
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

                Text("How many candidates Whisper samples when it falls back to sampling (it only does that after a chunk fails its quality checks below). No effect on a clean chunk; costs time on a difficult one.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("There is no beam-search setting: the MLX Whisper runtime has no beam decoder — asking for one stops the model loading at all. Best-of is the only search width available here.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Silence threshold — \(String(format: "%.2f", noSpeechThreshold))") {
                Slider(value: $noSpeechThreshold, in: 0.2...0.9, step: 0.05)

                Text("How sure Whisper must be that a window contains no speech before it skips it. Lower drops more audio as silence (risking lost speech); higher transcribes more of it (risking invented text on silence).")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("This is Whisper's own check, inside decoding. The app keeps its separate hallucination filter on the text Whisper does return (short canned captions, repetition loops, implausibly sparse text) — that one is not affected by this slider, and every line it removes is logged with its reason in logs/whisper.log.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Confidence floor — \(String(format: "%.1f", logprobThreshold))") {
                Slider(value: $logprobThreshold, in: -3.0...(-0.2), step: 0.1)

                Text("Below this average token log-probability Whisper treats its own output as failed and retries the chunk at a higher temperature. Raising it (towards 0) makes it retry more often — slower, and not always better.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Repetition limit — \(String(format: "%.1f", compressionThreshold))") {
                Slider(value: $compressionThreshold, in: 1.5...4.0, step: 0.1)

                Text("Text that compresses better than this is looping (\"yeah yeah yeah…\") and the chunk is retried. Lower catches loops sooner but can reject legitimately repetitive speech.")
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

                Text("Makes Whisper jump over silent stretches longer than this instead of decoding them — the stretches where it is most likely to invent a closing caption.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Turning this on ALSO switches Whisper to its own word timestamps — Whisper only consults this threshold when it is timing words, so the setting would otherwise do nothing at all. Those timings stay internal to this skip and are never used for speaker rows: the word aligner (Models → Aligner) remains the source of word times. This project measured Whisper's native word timing against that aligner and chose the aligner, so this is worth turning on for the silence skip, not for the timings.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Task") {
                Picker("", selection: $task) {
                    Text("Transcribe").tag("transcribe")
                    Text("Translate to English").tag("translate")
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text("Translate outputs English no matter what was spoken. Meetings here are English, so this normally does nothing — and it makes the transcript no longer a record of the words actually said.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Language detection") {
                SettingToggle(label: "Let Whisper detect the language per chunk",
                              isOn: $autoDetectLanguage)

                Text("On, the Language setting below is ignored and Whisper decides from the audio of each chunk — which can change mid-meeting on a noisy or short chunk. Off (default) sends the language you picked, which is steadier.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("Changing any of these restarts the Whisper sidecar at the next recording, and the options in force are written to logs/whisper.log at startup — a run at defaults logs \"decoding: defaults\".")
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
            Text("Qwen3 options — these change how Qwen3 decodes. Every setting below starts at Qwen3's own default, so an untouched block transcribes exactly as before.")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

            SettingBlock(title: "Vocabulary prompt — spelling help, with a real risk") {
                TextField("e.g. Aggia, ATND1061, pyannote, PREP framework",
                          text: $systemPrompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                    .font(.system(size: 12))

                Text("A short list of names, jargon and spellings, given to Qwen3 as context before the audio. Tested here: listing \"PREP framework\" changed a transcribed \"Prep\" into \"PREP\", which is what this is for.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Risk: it does not stay where you point it. Testing showed a non-empty prompt ALSO changes wording and punctuation in parts of the audio it says nothing about — in one case merging two sentences into one. Keep it to a short word list, not a sentence describing the meeting. While a prompt is set, every chunk logs \"prompt active\" to logs/qwen3.log so suspicious text can be traced back to it.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Unlike Whisper's initial prompt, this one is not continued as text: instructions in it are ignored, and made-up words in it did not appear in the transcript. The risk here is the shifted wording above, not invented vocabulary.")
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

                Text("Pushes Qwen3 away from repeating tokens it has just produced, which can help when a chunk loops on the same phrase. Off is the model's own default and the safe choice.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Risk: this penalises ordinary repetition too — English repeats words legitimately. Measured on a clean recording, 1.2 already dropped a real word and a sentence-ending period, and higher values degrade badly (at 2.0 the text came back with broken capitalisation and punctuation scattered mid-word). Nothing above 1.2 is offered for that reason. Turn it on only for a chunk that is actually looping, and compare the result.")
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

                    Text("How far back the penalty looks. Smaller punishes only immediate loops and leaves normal speech alone; larger also penalises a word said earlier in the same chunk, which is usually legitimate. 100 is the model's default.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("This appears only while a penalty is set, because Qwen3 reads it nowhere else — on its own it would be a control that changes nothing.")
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
