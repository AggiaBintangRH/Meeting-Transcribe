import SwiftUI

/// Models → Realtime: which engine writes the live captions, and its settings.
///
/// TWO engines since 2026-08-11 (Nemotron, Parakeet).
///
/// LAYOUT — the `ChunkedModelTab` order, adopted here on 2026-08-11 (owner:
/// *"move the model realtime config to top, the choose model is bottom, like
/// other"*): switch → the selected engine's options in a bordered box → the
/// card list → install status.
///
/// EVERY setting here is per-engine, so there is no pipeline-level block at all
/// and only ever ONE options box on screen. Language was briefly placed above
/// the cards because both engines read one key (`realtime.language`); the owner
/// corrected it the same day and was right — the ROSTERS barely overlap
/// (Nemotron offers Indonesian, Malay, Chinese and Japanese; Parakeet has none
/// of the four), so a picker that looked global had its options changing
/// underneath it.
///
/// COPY IS DELIBERATELY SHORT (owner, 2026-08-11: *"di settingnya banyak sekali
/// kata kata, buat simple saja"*). Where a control needs a caveat it gets ONE
/// short line and the evidence stays in these comments — the reasoning is for
/// whoever edits this next, not for someone choosing a setting.
///
/// BOTH ENGINES NOW SHOW THE SAME TWO CONTROLS: caption interval, then language.
/// Nemotron's "Chunk size" picker was REMOVED the same day (owner: *"oke hapus
/// aja deh"*) after measurement showed every option except the shipped one was
/// simply slower — 2.08 s per partial at 1120 ms against 3.42 / 5.49 / 18.04 at
/// 560 / 160 / 80 — with no evidence any of them was more accurate. The value is
/// pinned at `RealtimeASRService.pinnedChunkMs` and its key is no longer read.
/// The capability is untouched: the sidecar still declares `ATT_CONTEXT` and
/// still takes `--chunk-ms`, so restoring the picker is one edit if evidence
/// ever justifies it.
struct RealtimeModelTab: View {
    @AppStorage("realtime.enabled")  private var enabled = true
    @AppStorage("realtime.model")    private var model = RealtimeASRService.defaultModelID
    @AppStorage("realtime.language") private var language = "auto"
    /// The caption cadence — ONE key shared by both engines, so the choice
    /// survives an engine switch. Default is the value both sidecars used before
    /// the control existed, so an untouched install is unchanged.
    @AppStorage("realtime.partialMs")
    private var partialMs = RealtimeASRService.defaultPartialMs

    var body: some View {
        Group {
            SettingToggle(label: "Enable realtime transcription", isOn: $enabled)

            // THE SELECTED ENGINE'S OWN OPTIONS, above the card list rather than
            // below it, and only ever ONE box on screen — the `ChunkedModelTab`
            // shape.
            //
            // LANGUAGE LIVES IN HERE, NOT ABOVE (owner, 2026-08-11: *"karena
            // bahasa tiap model berbeda, masukan setting bahasanya per model"*).
            // It sat at pipeline level for a few hours on the reasoning that both
            // engines read one key — which is true and is not the point. The
            // ROSTERS barely overlap: Nemotron offers Indonesian, Malay, Chinese
            // and Japanese, and Parakeet has NONE of the four, so the list of
            // codes changes underneath a picker that looked like a global. Shown
            // per engine, the roster is visibly a property of the card selected
            // below it, which is what it has always been.
            // Every engine spelled out, and the fall-through goes to the DEFAULT
            // engine's own box rather than to whichever branch happened to be
            // last. `modelTabStatus` shipped a `default:` that answered
            // "pyannote" for a DiariZen session — a new engine inheriting
            // another engine's name, with nothing broken and nothing to notice.
            ModelOptionsBox(title: optionsTitle, note: optionsNote) {
                partialIntervalBlock
                languageBlock
            }

            ForEach(ModelCatalog.realtimeModels) { m in
                Button(action: {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                        model = m.id
                    }
                }) {
                    ModelCardView(model: m, selected: model == m.id)
                }
                .buttonStyle(.plain)
            }

            // Install status on the SELECTED model only (2026-08-06): only the
            // selected one can stop a session starting, and a badge on every card
            // has no room to say what to do about it.
            ModelInstallStatus(model: ModelCatalog.realtimeModel(id: model))
        }
    }

    /// The selected engine's box heading. A stored id naming an engine this
    /// build no longer has resolves through `ModelCatalog.realtimeModel(id:)`,
    /// so the heading can never name a model that is not there.
    private var optionsTitle: String {
        ModelCatalog.realtimeModel(id: model).name.uppercased() + " OPTIONS"
    }

    /// One short line per engine — the measured headline, nothing more.
    private var optionsNote: String {
        switch model {
        case RealtimeASRService.parakeetModelID: return "25 languages · ~130x realtime."
        case RealtimeASRService.funasrModelID:   return "3 languages · ~49x realtime."
        default:                                 return "40 locales."
        }
    }

    /// The caption cadence — how much new speech a lane collects before the live
    /// caption is redrawn. **Shown for ALL engines**, same control, same values,
    /// one shared key (owner, 2026-08-11: *"jangan dibedakan, soalnya sama
    /// tentang waktu interval"*).
    ///
    /// It was Parakeet-only for a few hours because Nemotron is ~9x slower per
    /// partial, so its cadence is not a free choice. That is a reason for the
    /// two SIDECARS to honour the number differently, not a reason for one
    /// engine to lack the control — and lacking it left Nemotron's caption speed
    /// pinned to a constant nothing in the UI could reach.
    ///
    /// How each honours it, which is what `partialHint` has to convey:
    /// - **Parakeet** — the exact cadence. 0.235 s per partial on a 30 s buffer,
    ///   so duty is simply 2 lanes × 0.235 / interval.
    /// - **Nemotron** — a FLOOR. `PARTIAL_DUTY` still stretches it to
    ///   `max(interval, buffer × 0.15)`, so a short interval is honoured early
    ///   in an utterance (where responsiveness is felt) and overridden later
    ///   (where it would put the lane past 100 % duty).
    @ViewBuilder
    private var partialIntervalBlock: some View {
        SettingBlock(title: "Caption interval") {
            Picker("", selection: $partialMs) {
                Text("0.75 s").tag(750)
                Text("1 s").tag(1000)
                Text("1.5 s").tag(1500)
                Text("2 s").tag(2000)
                Text("3 s").tag(3000)
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            Text(partialHint)
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Language — PER MODEL, inside the selected engine's box. The key is shared
    /// (`realtime.language`) but the ROSTER is not: Nemotron publishes no list
    /// and offers the project's five, Parakeet has 25 from its own card and none
    /// of Nemotron's four non-European ones. A code the selected engine lacks
    /// resolves to Auto-detect at the read boundary and the picker says so.
    @ViewBuilder
    private var languageBlock: some View {
        SettingBlock(title: "Language") {
            LanguagePicker(selection: $language,
                           entries: Languages.realtimeEntries(forModel: model))

            // Selectable everywhere, with the truth beside it — the 2026-07-31
            // reversal. For Parakeet the choice is stored, survives an engine
            // switch, is forwarded to the sidecar and logged there, and is then
            // ignored by the model.
            if let why = Languages.realtimeNote(forModel: model) {
                Text(why)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let roster = Languages.realtimeSupportedSentence(forModel: model) {
                Text(roster)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // The engines' rosters barely overlap: four of Nemotron's five
            // concrete options (Indonesian, Malay, Chinese, Japanese) are absent
            // from Parakeet. A stored code the selected engine lacks resolves to
            // Auto-detect at the read boundary rather than travelling to a
            // sidecar that has never heard of it, so say so here instead of
            // letting the picker look like it was ignored.
            if language != "auto",
               Languages.resolveRealtime(language: language,
                                         forModel: model) == "auto" {
                Text("\(Languages.name(for: language)) not supported here — using Auto-detect.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.amber)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Duty numbers are measured, not estimated: 0.235 s per partial on a 30 s
    /// buffer, doubled for two simultaneously ACTIVE lanes, over the cadence.
    private var partialHint: String {
        switch partialMs {
        case 750:  return "Fastest updates · ~63% load"
        case 1000: return "~47% load"
        case 1500: return "Recommended · ~31% load"
        case 2000: return "~24% load"
        default:   return "Lightest · ~16% load"
        }
    }

    /// Duty is measured, one lane, against this engine's own stretched cadence
    /// (1.5 s floor, then 15 % of buffer). Double it when Office and Remote are
    /// both talking. Over 100 % the lane cannot keep up and falls further behind
    /// as the meeting runs.
}
