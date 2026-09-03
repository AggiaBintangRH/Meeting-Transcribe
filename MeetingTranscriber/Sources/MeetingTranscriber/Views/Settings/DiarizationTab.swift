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
    @AppStorage("diarization.engine")      private var engine = ShippedDefaults.diarizationEngine
    @AppStorage("chunked.model")           private var chunkedModel = ShippedDefaults.chunkedModel
    @AppStorage("diarization.enabled")     private var enabled = true
    @AppStorage("diarization.finalPass")   private var finalPass = true
    @AppStorage("diarization.continueOnStop") private var continueOnStop = false
    @AppStorage("diarization.intervalSec") private var intervalSec = 30
    /// Read-only here, and it is the SAME key `SpeakerCountView` writes and
    /// `AudioRecorder.diarNumSpeakers` reads — one value, not a third reader that
    /// could disagree. Used only to say whether a count is pinned, because
    /// "speaker count automatic" was hand-written into four engine notes and was
    /// wrong in all four (2026-08-14 audit).
    @AppStorage("diarization.numSpeakers") private var office = 0
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
    /// VibeVoice, the seventh engine (2026-09-02). SPECTRAL SHAPE in every way
    /// the shared controls care about — no live pass, one whole-file pass at
    /// Stop, no tail — so it joins the batch group. It is speaker-attributed ASR
    /// like MOSS, but it does NOT take MOSS's branch: MOSS's advice is about
    /// chunk length, and this engine reads no chunk setting at all.
    private var isVibeVoice: Bool { engine == ModelLoader.vibeVoiceEngineID }
    /// The two WHOLE-FILE BATCH engines. Shared pyannote controls apply to
    /// neither: there is no live pass to time, no live labels for a tail to
    /// continue from, and the stop pass is not optional — it IS the labels.
    /// `runsBatchOfficePass` reads none of the three, so hiding them here is not
    /// hiding something still in force; it is the UI agreeing with a rule that
    /// already ignores them.
    private var isBatchEngine: Bool {
        isSpectral || isNemo || isDiarizen || isCamPlus || isVibeVoice
    }

    /// The first two lines of every whole-file engine's note, DERIVED from the
    /// settings actually in force.
    ///
    /// ⚠ ALL FOUR WERE HAND-WRITTEN AND ALL FOUR WERE STALE — found by the
    /// 2026-08-14 audit, and it is the third time this exact class has appeared
    /// (three engine lists, all stale, in one sweep on 2026-08-13). Each said:
    ///
    ///   * "**No settings**" — while three controls sat directly above the text;
    ///   * "the whole recording is diarized **once at Stop**" — false since
    ///     2026-08-13, when switching the stop pass off gave these engines a live
    ///     per-interval path instead;
    ///   * "**speaker count automatic**" — flatly contradicting
    ///     `ModelLoader.honoursSpeakerCount`, which is `true` for all four;
    ///   * and spectral's added "This engine has **no live pass**".
    ///
    /// A sentence beside a control it contradicts is worse than no sentence, so
    /// this one reads the same values the controls bind to, and `overlapLine`
    /// reads the loader's rule. Nothing about these four notes is hand-written any
    /// more — which is the only form of the fix that a seventh engine cannot make
    /// stale again.
    private var batchEngineNote: String {
        let pass = stopPassBinding.wrappedValue
            ? "The whole recording is diarized once at Stop."
            : "Labels are made live, one window every \(intervalSec)s"
              + (tailBinding.wrappedValue ? ", plus the tail at Stop." : ".")
        // NOT "automatic" — these four read the number, which is exactly what the
        // old text denied. Auto is only what happens when nobody sets one, and in
        // the live mode above it is measured to be the failing case.
        let count = office > 0
            ? "Speaker count pinned to \(office)."
            : "Speaker count automatic — set one beside the record button if the "
              + "transcript shows people who were not there."
        return pass + " " + count + "\nProfiles and renaming work. "
             + "spk confidence usually will not appear."
             + "\n" + overlapLine
    }

    /// The third line, DERIVED from `ModelLoader.marksItsOwnOverlap`.
    ///
    /// It was per-engine prose, and spectral's block simply did not have it — so
    /// the one engine of the four whose note said nothing about overlap was one
    /// that cannot mark it, i.e. the case where the sentence matters. Deriving it
    /// gives spectral the line back and makes a seventh engine impossible to miss,
    /// which is the same reason `diarizationEnginesWithoutOwnOverlap` exists.
    private var overlapLine: String {
        ModelLoader.marksItsOwnOverlap(diarEngine: engine)
            ? "It marks overlapping speech itself, so Detect overlap is not used here."
            : "It cannot mark two people at once, so overlap repair needs Detect "
            + "overlap switched on."
    }
    var body: some View {
        Group {
            
            SettingToggle(label: "Enable speaker diarization (who spoke when)", isOn: $enabled)

            // PIPELINE-LEVEL SETTINGS ABOVE THE CARDS (owner, 2026-08-06), the
            // same shape as the Chunked tab. The bindings follow the engine
            // because MOSS keeps its own pair (`moss.*`) — their defaults differ,
            // so they are not one key with two readers.
            // NOTHING SHARED APPLIES UNDER EITHER BATCH ENGINE (owner, 2026-08-06;
            // NeMo joined 2026-08-07) — see `isBatchEngine` for the whole reason.
            // EVERY ENGINE NOW HAS THIS BLOCK (owner, 2026-08-13). The four
            // whole-file engines used to be excluded because they had no live
            // path, so switching the stop pass off left a meeting with no labels
            // at all. They have one now — a window is just a short recording, sent
            // as an ordinary whole-file job and stitched across windows by
            // `identify`. See `AudioRecorder+BatchLiveDiarization`.
            SettingBlock(title: "Run at stop") {
                SettingToggle(label: "Run a diarization pass at stop", isOn: stopPassBinding)
                Text("On: the whole recording is re-diarized from start to end after you stop — the best labels, and the only mode a set speaker count reaches. Off: labels come from the live passes instead.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
                // NO AUTO-COUNT WARNING HERE (owner, 2026-08-13). One was added
                // with this block and removed on sight. The measurement behind it
                // stands and is kept where it belongs — in
                // `AudioRecorder+BatchLiveDiarization`, which explains why the SPK
                // picker matters for this mode — but the owner has seen those
                // numbers twice and does not want the tab repeating them.
            }

            // TWO CONTROLS AT THE TOP, FOR EVERY ENGINE, AND NOTHING ELSE UNTIL
            // THE STOP PASS IS OFF (owner, 2026-08-14): *"pertama hanya ada Enable
            // Diarization, Run Diarization at Stop … ketika false baru ada Interval
            // Chunked, Continue Tail."*
            //
            // ⚠ THE BEHAVIOUR MOVED WITH THE LAYOUT, which is the whole point.
            // Both settings below describe the LIVE path, and the live path is
            // what runs in place of a stop pass. Until today `continueOnStop` was
            // read only when the stop pass was ON — visible where inert, active
            // where invisible — and the owner paid for that: a stale `true` turned
            // the office pass into a tail, a tail is a `chunk` job carrying no
            // `num_speakers`, and the SPK picker went dim with nothing in the UI
            // able to explain or undo it. `runsTailPassAtStop`, `mossStopMode` and
            // `remoteStopMode` now all read it in this branch only, so a stop pass
            // is unconditionally a full re-diarization.
            //
            // NO `isBatchEngine` GUARD ON THE TAIL, deliberately. It was written
            // that way for an hour and it was the stale-list mistake again: those
            // four engines HAVE had a live path since 2026-08-13, so they have a
            // leftover window at Stop exactly like pyannote does, and `stop()`
            // flushes it under this same setting.
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
                    Text("On: at stop, the audio after the last live window is diarized too, so the final seconds are not left unlabelled. Off: whatever the live passes produced is the transcript.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
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
                Text(batchEngineNote)
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
                Text(batchEngineNote)
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
                Text(batchEngineNote)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            } else if isVibeVoice {
                // ITS OWN BRANCH, for the reason CAM++'s and DiariZen's exist:
                // without one it falls through to the `else` and hands the user
                // MOSS's advice about chunk length, which this engine does not
                // read. What it needs said instead is the thing no other engine
                // here has to say — the SPK control does nothing under it.
                Text(batchEngineNote)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
                Text("The speaker count beside the record button has no effect "
                     + "here: this model has no such parameter, so the count is "
                     + "always automatic. It also transcribes as it labels, so "
                     + "its turn boundaries fall on a ~2.9 s grid rather than on "
                     + "the word.")
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
                Text(batchEngineNote)
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
                // The engine list is DERIVED, never typed out — it had gone stale
                // twice (missing DiariZen, then CAM++), each time telling the user
                // they had fewer options than they did, while the card list two
                // blocks below read the same filter and was right both times.
                Text(chunkedModel == "moss"
                     ? "MOSS transcribes and labels in one pass, so with MOSS as the chunked "
                       + "model it is also the diarizer. Pick a different chunked model to use "
                       + ModelCatalog.diarizationEnginesWithoutMoss + "."
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
