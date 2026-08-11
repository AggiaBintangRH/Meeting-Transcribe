import SwiftUI

/// Models → Diarization: pick the engine, then its settings.
///
/// Four engines in three genuinely different shapes (pyannote live+stop; spectral
/// and NeMo whole-file batch; MOSS per chunk), so the tab shows only the selected
/// one's blocks. Everything under `pyannoteSettings` is pyannote-specific
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

    // `engineBoxTitle` USED TO LIVE HERE and was DEAD — nothing read it. The only
    // options box left is pyannote's, whose title is a literal and which is shown
    // only under pyannote, so a computed title had no caller. It was still being
    // maintained (a DiariZen case was added to it on 2026-08-10 before anyone
    // noticed), which is the cost of dead code in a file where comments are
    // load-bearing: it reads like the thing that names the box, and it is not.

    /// MOSS keeps its OWN stop pair (`moss.finalPass` / `moss.continueOnStop`) —
    /// their defaults differ from the pyannote pair, so these are two settings
    /// with one control, not one setting with two names.
    private var stopPassBinding: Binding<Bool> { isMoss ? $mossFinalPass : $finalPass }
    private var tailBinding: Binding<Bool> { isMoss ? $mossTailOnly : $continueOnStop }
    private var isSpectral: Bool { engine == ModelLoader.spectralEngineID }
    /// NeMo, the fourth engine (2026-08-07). It has the SPECTRAL shape exactly —
    /// no live pass, one whole-file pass at Stop, no tail — so everywhere below it
    /// is grouped with spectral rather than given a branch of its own.
    private var isNemo: Bool { engine == ModelLoader.nemoEngineID }
    /// DiariZen, the fifth engine (2026-08-10). Same SPECTRAL shape as NeMo —
    /// no live pass, one whole-file pass at Stop, no tail — so it joins the
    /// batch group rather than getting a branch of its own.
    private var isDiarizen: Bool { engine == ModelLoader.diarizenEngineID }
    /// CAM++, the sixth engine (2026-08-11). Same SPECTRAL shape again — no live
    /// pass, one whole-file pass at Stop, no tail — so it joins the batch group.
    private var isCamPlus: Bool { engine == ModelLoader.camPlusEngineID }
    /// The two WHOLE-FILE BATCH engines. Shared pyannote controls apply to
    /// neither: there is no live pass to time, no live labels for a tail to
    /// continue from, and the stop pass is not optional — it IS the labels.
    /// `runsBatchOfficePass` reads none of the three, so hiding them here is not
    /// hiding something still in force; it is the UI agreeing with a rule that
    /// already ignores them.
    private var isBatchEngine: Bool { isSpectral || isNemo || isDiarizen || isCamPlus }
    var body: some View {
        Group {
            
            SettingToggle(label: "Enable speaker diarization (who spoke when)", isOn: $enabled)

            // PIPELINE-LEVEL SETTINGS ABOVE THE CARDS (owner, 2026-08-06), the
            // same shape as the Chunked tab. The bindings follow the engine
            // because MOSS keeps its own pair (`moss.*`) — their defaults differ,
            // so they are not one key with two readers.
            // NOTHING SHARED APPLIES UNDER EITHER BATCH ENGINE (owner, 2026-08-06;
            // NeMo joined 2026-08-07) — see `isBatchEngine` for the whole reason.
            if !isBatchEngine {
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
            if !isMoss && !isBatchEngine {
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
            } else if isNemo {
                // TWO lines, not nine blocks of prose (the 2026-08-06 rule): what
                // this engine does differently, and what it still gives you. The
                // second half is the part a user would otherwise have to discover —
                // it is the same identity path pyannote uses, so nothing about
                // profiles or renaming changes here, unlike under MOSS.
                //
                // `spk` IS CALLED OUT SEPARATELY, and the wording is the 2026-08-10
                // audit's. This block used to say all three "work normally", which
                // was true of the wiring and wrong in practice: `spk` is the cosine
                // `assign()` returns on a MATCH, and a first appearance has none. A
                // batch engine calls `identify` ONCE over a store the session just
                // reset, so every voice is a first appearance and no row gets a
                // number. pyannote differs only because it identifies per 30 s
                // chunk, and chunk 2 onward matches chunk 1's fresh profiles.
                // A promise the architecture cannot keep is worse than a caveat.
                Text("No settings — the whole recording is diarized once at Stop, "
                     + "speaker count automatic.\n"
                     + "Profiles and renaming work. spk confidence usually will not "
                     + "appear.\n"
                     + "It cannot mark two people at once, so overlap repair needs "
                     + "Detect overlap switched on.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            } else if isDiarizen {
                // ITS OWN BRANCH, added 2026-08-10. Without one it fell through to
                // the `else` below and showed MOSS's copy — "chunk length is the
                // quality knob… Models → Chunked", which is a CHUNKED-ASR setting
                // and has no effect on this engine at all. A settings page that
                // hands out another model's advice is worse than a blank one.
                //
                // The last paragraph is the part that differs from NeMo's, and it
                // is a measurement rather than a family resemblance: this engine's
                // powerset head predicts two-speaker frames directly, and its turns
                // were measured intersecting (11 pairs / 13.30 s on Overlap123).
                // So overlap really is marked here, and repair does NOT need the
                // separate detector — see `usesDetectedRegionsForRepair`.
                // THREE SHORT SENTENCES (owner, 2026-08-10, restating the
                // 2026-08-06 rule): this was three paragraphs and the owner cut it
                // — "nobody reads it, the client just picks". Every clause that
                // survived prevents a wrong expectation; the REASONS moved into
                // this comment, where the next maintainer needs them and the client
                // does not. Do not grow it back.
                Text("No settings — the whole recording is diarized once at Stop, "
                     + "speaker count automatic.\n"
                     + "Profiles and renaming work. spk confidence usually will not "
                     + "appear.\n"
                     + "It marks overlapping speech itself, so Detect overlap is not "
                     + "used here.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            } else if isCamPlus {
                // ITS OWN BRANCH, for the reason DiariZen's exists: without one it
                // would fall through to the `else` below and hand the user MOSS's
                // advice about chunk length, a CHUNKED-ASR setting with no effect
                // on this engine.
                //
                // The third line is where it differs from DiariZen's: this engine
                // clusters one label per window, so its turns never intersect and
                // it cannot mark overlap — the same sentence NeMo's branch carries,
                // and it is a property of the clustering, not a family resemblance.
                // Same three-sentence limit the owner set on 2026-08-10.
                Text("No settings — the whole recording is diarized once at Stop, "
                     + "speaker count automatic.\n"
                     + "Profiles and renaming work. spk confidence usually will not "
                     + "appear.\n"
                     + "It cannot mark two people at once, so overlap repair needs "
                     + "Detect overlap switched on.")
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
                       + "pyannote, spectral, NeMo or DiariZen."
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
