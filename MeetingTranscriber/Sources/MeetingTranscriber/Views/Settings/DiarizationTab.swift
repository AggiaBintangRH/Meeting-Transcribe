import SwiftUI

/// Models → Diarization: pick the engine, then its settings.
///
/// Three engines with genuinely different shapes, so the tab shows only the
/// selected one's blocks. Everything under `pyannoteSettings` is pyannote-specific
/// — a live pass, a cluster threshold and an overlap mode are all things the other
/// two simply do not have — and showing a control that does nothing is the exact
/// complaint the Overlap tab already had to fix once.
///
/// **THE SPECTRAL BLOCK SHOWS ITS INERT CONTROLS RATHER THAN HIDING THEM**, which
/// is the opposite of what the MOSS block does, and the difference is not style.
/// MOSS owns its own `moss.*` stop keys, so pyannote's `diarization.live` /
/// `diarization.continueOnStop` are genuinely not part of a MOSS session and the
/// tab has nothing to say about them. Spectral reads the SAME `diarization.*` keys
/// pyannote does — a user who set them under pyannote still has them set here, and
/// two of them will now be ignored. Hiding those is the failure the language-picker
/// work fixed on 2026-07-31: a control the engine cannot act on stays visible and
/// says so, rather than vanishing or quietly lying.
struct DiarizationTab: View {
    @AppStorage("diarization.engine")      private var engine = "pyannote"
    @AppStorage("chunked.model")           private var chunkedModel = "qwen3"
    @AppStorage("diarization.enabled")     private var enabled = true
    @AppStorage("diarization.finalPass")   private var finalPass = true
    @AppStorage("diarization.continueOnStop") private var continueOnStop = false
    @AppStorage("diarization.intervalSec") private var intervalSec = 30
    // REMOVED 2026-08-06 (owner), controls and reads both: `diarization.live`
    // (always on), `.detectOverlap` (the Detect overlap tab owns this now),
    // `.resetOnStart` (always fresh) and `.numSpeakers` (always auto). Each is
    // pinned at its old default in the Audio layer — see
    // `lockDiarizationSettings` and `AudioRecorder.diarNumSpeakers` — so a value
    // stored before the control disappeared cannot still decide behaviour.
    @AppStorage("diarization.clusterThreshold") private var clusterThreshold = 0.6
    // The MOSS engine's own stop pair. Named `moss.*` rather than reusing
    // `diarization.*` because the defaults must differ: MOSS's tail-only default
    // is TRUE (today's behaviour), while `diarization.continueOnStop` is FALSE.
    @AppStorage("moss.finalPass")      private var mossFinalPass = true
    @AppStorage("moss.continueOnStop") private var mossTailOnly = true

    /// Set when the stored engine was MOSS and the chunked model is not, so the
    /// reset announces itself instead of the user finding pyannote selected with
    /// no explanation — the `droppedLanguage` precedent in `ChunkedModelTab`.
    @State private var mossEngineDropped = false

    private var isMoss: Bool { engine == ModelLoader.mossEngineID }

    private var engineBoxTitle: String {
        isMoss ? "MOSS" : (isSpectral ? "SPECTRAL" : "PYANNOTE")
    }

    /// MOSS keeps its OWN stop pair (`moss.finalPass` / `moss.continueOnStop`) —
    /// their defaults differ from the pyannote pair, so these are two settings
    /// with one control, not one setting with two names.
    private var stopPassBinding: Binding<Bool> { isMoss ? $mossFinalPass : $finalPass }
    private var tailBinding: Binding<Bool> { isMoss ? $mossTailOnly : $continueOnStop }
    private var isSpectral: Bool { engine == ModelLoader.spectralEngineID }
    var body: some View {
        Group {
            
            SettingToggle(label: "Enable speaker diarization (who spoke when)", isOn: $enabled)

            // PIPELINE-LEVEL SETTINGS ABOVE THE CARDS (owner, 2026-08-06), the
            // same shape as the Chunked tab. The bindings follow the engine
            // because MOSS keeps its own pair (`moss.*`) — their defaults differ,
            // so they are not one key with two readers.
            // NOTHING SHARED APPLIES UNDER SPECTRAL (owner, 2026-08-06). It has no
            // live pass, so there is no interval to time and no live labels for a
            // tail to continue from; and the stop pass is not optional there, it
            // IS the labels. `runsSpectralOfficePass` reads none of the three, so
            // hiding them here is not hiding something still in force — it is the
            // UI agreeing with a rule that already ignored them.
            if !isSpectral {
            SettingBlock(title: "Run at stop") {
                SettingToggle(label: "Run a diarization pass at stop", isOn: stopPassBinding)
                Text("One more pass after you stop, to finalise speaker labels.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // VISIBILITY IS THE OWNER'S RULE, chosen with the consequence stated:
            // these appear while the stop pass is OFF. Note that `continueOnStop`
            // is read ONLY inside `if willRunStopPass` (AudioRecorder.stop), so in
            // this branch the tail toggle is stored but does nothing. Kept as
            // asked — flipping the condition is the whole change if that is ever
            // revisited.
            if !stopPassBinding.wrappedValue {
                if !isMoss {
                    SettingBlock(title: "Chunked interval") {
                        Picker("", selection: $intervalSec) {
                            Text("15s").tag(15)
                            Text("30s").tag(30)
                            Text("45s").tag(45)
                            Text("60s").tag(60)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()

                        Text("How often live labelling runs. Separate from the ASR interval, and used only while live labels are on.")
                            .font(.system(size: 11))
                            .foregroundColor(Theme.textFaint)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                SettingBlock(title: "Tail") {
                    SettingToggle(label: "Continue from live labels (tail only)", isOn: tailBinding)
                    Text("On: only the audio after the last live chunk is diarized, so numbering stays stable. Off: the whole recording is re-diarized.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            }

            // THE SELECTED ENGINE'S OWN SETTINGS, directly under the shared ones
            // and inside a border (owner, 2026-08-06) — the shape the Chunked tab
            // uses for its per-model options. Position and border do one job:
            // these belong to ONE card, unlike the three settings above them,
            // which apply to every engine.
            // A BOX ONLY WHERE THERE IS A CONTROL (owner, 2026-08-06). pyannote is
            // the one engine with a setting of its own; spectral and MOSS had
            // nine and fourteen blocks of pure prose between them, which is a
            // manual, not a settings page. Each keeps ONE line — the one that
            // prevents a real failure — and nothing else.
            if !isMoss && !isSpectral {
                ModelOptionsBox(title: "PYANNOTE SETTINGS",
                                note: "Only for the engine selected below.") {
                    pyannoteSettings
                }
            } else if isSpectral {
                Text("No settings. This engine has no live pass — it diarizes the whole "
                     + "recording once, after you press Stop, every time.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("No settings of its own. Chunk length is the quality knob and lives in "
                     + "Models → Chunked: 60 s or 120 s separates speakers noticeably better "
                     + "than 30 s.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // RESTORED here: this notice went missing during the restructuring
            // above, and it is the half of the MOSS⟺MOSS rule the user can see.
            // Without it the engine silently changes under them.
            if mossEngineDropped {
                Text(chunkedModel == "moss"
                     ? "MOSS transcribes and labels in one pass, so with MOSS as the chunked "
                       + "model it is also the diarizer. Pick a different chunked model to use "
                       + "pyannote or spectral."
                     : "MOSS diarization is offered only when MOSS is also the chunked model. "
                       + "The engine was set back to pyannote.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.amber)
                    .fixedSize(horizontal: false, vertical: true)
            }

            ForEach(ModelCatalog.diarizationEngines.filter {
                ModelLoader.diarizationEngineIsSelectable(
                    ModelCatalog.diarizationEngineValue($0), chunkedID: chunkedModel)
            }) { m in
                // `diarizationEngineValue`, not `m.id`: the MOSS card's catalog id
                // is "moss-diar" (the chunked entry already owns "moss") while the
                // setting's value is "moss". Writing the id here would store a
                // string neither the loader nor the recorder ever matches, and the
                // engine would silently never switch.
                let value = ModelCatalog.diarizationEngineValue(m)
                Button(action: {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                        engine = value
                    }
                }) {
                    ModelCardView(model: m, selected: engine == value)
                }
                .buttonStyle(.plain)
            }

            // The selected ENGINE's card, found by its setting value — the MOSS
            // card's catalog id is "moss-diar" while the stored value is "moss",
            // so `first { $0.id == engine }` would silently report the wrong model.
            if let selected = ModelCatalog.diarizationEngines.first(where: {
                ModelCatalog.diarizationEngineValue($0) == engine
            }) {
                ModelInstallStatus(model: selected)
            }

        }
        // A stored engine must never outlive the card that selects it. This runs
        // on appear AND whenever the chunked model changes, so the pairing cannot
        // be left inconsistent by either route.
        .onAppear { dropMossEngineIfUnavailable() }
        .onChange(of: chunkedModel) { _, _ in dropMossEngineIfUnavailable() }
    }

    /// A stored engine must never outlive the card that selects it — in EITHER
    /// direction now: MOSS left selected beside another ASR, or pyannote/spectral
    /// left selected beside MOSS.
    private func dropMossEngineIfUnavailable() {
        guard !ModelLoader.diarizationEngineIsSelectable(engine, chunkedID: chunkedModel) else {
            mossEngineDropped = false
            return
        }
        engine = ModelLoader.fallbackDiarizationEngine(chunkedID: chunkedModel)
        mossEngineDropped = true
    }

    @ViewBuilder
    private var pyannoteSettings: some View {
        Group {
            SettingBlock(title: "Speaker profiles") {
                Text("Every recording starts with an empty speaker store, so speakers are numbered from 1 each time. spk appears once a voice matches its profile.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Voice-separation threshold — \(String(format: "%.2f", clusterThreshold))") {
                Slider(value: $clusterThreshold, in: 0.4...0.8, step: 0.05)
                Text("How readily two turns are judged the same voice. Lower merges more, higher splits more. Per-room calibration — leave at 0.60 unless voices keep getting merged.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
