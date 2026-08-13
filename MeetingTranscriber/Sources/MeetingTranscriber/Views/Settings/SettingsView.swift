import SwiftUI

/// Settings window: left rail with sections, "Models" section holds 3 sub-tabs.
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss

    // Main sections (left rail) — room to grow (General, Export, …)
    enum Section: String, CaseIterable {
        case models = "Models"
        case microphone = "Microphone"
        case atnd = "ATND"
        // Things that happen AROUND a meeting rather than to its audio (owner,
        // 2026-08-10). Last in the rail because it is the only section that acts
        // after a recording is over, and the rail otherwise reads as pipeline
        // order.
        case utils = "Utils"

        var icon: String {
            switch self {
            case .models: return "cpu"
            case .microphone: return "mic.fill"
            case .atnd: return "antenna.radiowaves.left.and.right"
            case .utils: return "wrench.and.screwdriver.fill"
            }
        }

        /// Header line under the section title (sections with sub-tabs use theirs).
        var subtitle: String {
            switch self {
            case .models, .atnd: return ""
            case .microphone: return "Input device and channel selection"
            case .utils: return "Export and other post-meeting options"
            }
        }
    }

    // Sub-tabs inside Models.
    //
    // DECLARATION ORDER IS THE RAIL ORDER — `modelRail` reads `allCases` — and it
    // follows the pipeline: what you hear live, what produces the final text, who
    // said it, then the two overlap stages, then the two refinements.
    //
    // Detect overlap sits immediately ABOVE Overlap because that is the direction
    // the data flows: under MOSS and spectral the detector is what gives repair
    // its regions, so a user reading downwards meets the prerequisite first.
    enum ModelTab: String, CaseIterable, SettingsSubTab {
        case realtime = "Realtime"
        case chunked = "Chunked"
        case diarization = "Diarization"
        /// Detection is a DIFFERENT job from repair, done by a different model
        /// that needs no speaker turns — so it gets its own tab rather than a
        /// block inside Overlap. See `OverlapDetectTab`.
        case overlapDetect = "Detect overlap"
        case overlap = "Overlap"
        case aligner = "Aligner"
        case vad = "VAD"

        var title: String { rawValue }

        var icon: String {
            switch self {
            case .realtime: return "bolt.fill"
            case .chunked: return "doc.text.fill"
            case .aligner: return "timeline.selection"
            case .vad: return "waveform.and.mic"
            case .diarization: return "person.2.fill"
            case .overlap: return "person.wave.2.fill"
            case .overlapDetect: return "waveform.badge.magnifyingglass"
            }
        }

        var subtitle: String {
            switch self {
            case .realtime: return "Live captions while recording"
            case .chunked: return "Accurate final transcript"
            case .aligner: return "Word timestamps — word-exact speaker attribution"
            case .vad: return "Voice activity detection"
            case .diarization: return "Speaker identification — who spoke when"
            case .overlap: return "Overlap recovery — separate simultaneous speech"
            case .overlapDetect: return "Mark where two people spoke at once"
            }
        }
    }

    // Sub-tabs inside ATND
    enum ATNDSubTab: String, CaseIterable, SettingsSubTab {
        case connection = "Connection"
        case command = "Command"
        case position = "Position ID"

        var title: String { rawValue }

        var icon: String {
            switch self {
            case .connection: return "network"
            case .command: return "terminal.fill"
            case .position: return "person.line.dotted.person.fill"
            }
        }

        var subtitle: String {
            switch self {
            case .connection: return "ATND1061 beamforming array — beam and talker data over IP"
            case .command: return "Send IP control commands to the array"
            case .position: return "Speaker identity from talker direction — replaces voice-based labels while the array is connected"
            }
        }
    }

    @State private var section: Section = .models
    @State private var modelTab: ModelTab = .realtime
    @State private var atndTab: ATNDSubTab = .connection
    @Namespace private var tabIndicator

    var body: some View {
        HStack(spacing: 0) {
            leftRail
            Rectangle().fill(Theme.divider).frame(width: 1)
            content
        }
        .frame(minWidth: 860, idealWidth: 920, maxWidth: .infinity,
               minHeight: 540, idealHeight: 620, maxHeight: .infinity)
        .background(Theme.sidebar)
    }

    // MARK: Left rail

    /// One short word or two for the rail's right-hand column. Deliberately terse:
    /// the rail is 232 pt wide and the value sits beside the tab name, so anything
    /// longer would wrap or clip. The full story lives in the tab itself.
    private func modelTabStatus(_ t: ModelTab) -> (String, Bool) {
        switch t {
        case .chunked:
            return (ModelCatalog.chunkedModel(id: chunkedModel).name, false)
        case .diarization:
            guard diarEnabled else { return ("off", false) }
            switch ModelLoader.wantedDiarizationStack(diarizationEnabled: true,
                                                      engine: diarEngine,
                                                      chunkedID: chunkedModel,
                                                      chunkedEnabled: chunkedEnabled) {
            // EVERY CASE NAMED, no `default:` — and that is the point. DiariZen
            // landed with this switch ending in `default: return "pyannote"`, so
            // the rail showed **pyannote** for a DiariZen session: the exact
            // fall-through this project already wrote down for
            // `ChunkedASRModel.scriptName` ("a switch over model ids needs a
            // default, and a default is how a future model falls through to a
            // deleted file"). Spelled out here so the COMPILER reports the next
            // engine instead of the rail quietly naming the wrong one.
            case .pyannote:          return ("pyannote", false)
            case .spectral:          return ("spectral", false)
            case .nemo:              return ("NeMo", false)
            case .diarizen:          return ("DiariZen", false)
            case .camPlus:           return ("CAM++", false)
            case .mossOwnASR:        return ("included", true)
            case .mossSecondProcess: return ("MOSS", false)
            case .none:              return ("off", false)
            }
        case .aligner:
            // MOSS attributes its own text, so the job is done and the rail says
            // so — the `built in` the Detect overlap row uses for pyannote, and
            // the twin of diarization's `included` two cases above. Reporting
            // "on" here would promise a 1.2 GB model that never loads; "off"
            // would deny work that is happening.
            if ModelLoader.alignmentIsBuiltIn(chunkedID: chunkedModel,
                                              chunkedEnabled: chunkedEnabled) {
                return ("built in", true)
            }
            return (alignEnabled ? "on" : "off", false)
        case .overlap:  return (repairEnabled ? "on" : "off", false)
        case .overlapDetect:
            // pyannote marks overlap itself, so the detector is redundant there —
            // said in the rail so the user sees it without opening the tab.
            if diarEnabled, diarEngine == ModelLoader.pyannoteEngineID {
                return ("built in", true)
            }
            return (detectEnabled ? "on" : "off", false)
        case .realtime: return (realtimeEnabled ? "on" : "off", false)
        // VAD returned "" until 2026-08-10 — the only switchable tab in the rail
        // with no status, so it read as though it had nothing to switch. It always
        // did: `vad.enabled` is the same key `switchedOffModelTabs` consults two
        // functions below, so the tab could sit dimmed under SWITCHED OFF while
        // showing no reason for it. Same shape as its four neighbours now.
        case .vad:      return (vadEnabled ? "on" : "off", false)
        }
    }

    /// The rail's model list in THREE groups, each in pipeline order (see
    /// `ModelTab`): what is running, what the owner switched off, and what cannot
    /// apply at all. The last two are dimmed and moved down, never hidden and
    /// never disabled — the switch that brings a tab back is INSIDE it.
    ///
    /// The two dimmed groups are kept APART rather than merged into one "inactive"
    /// list, because the difference is the only thing the user can act on:
    /// SWITCHED OFF is undone by the tab's own toggle, NOT USED BY YOUR MODELS is
    /// not undone by anything on that page. One heading over both would tell a
    /// user that turning Aligner on might help under MOSS, which is false.
    private var modelRail: some View {
        VStack(alignment: .leading, spacing: 2) {
            let inapplicable = inapplicableModelTabs
            let off = switchedOffModelTabs
            let running = ModelTab.allCases.filter {
                !inapplicable.contains($0) && !off.contains($0)
            }
            ForEach(running, id: \.self) { railRow($0, dim: false) }
            railGroup("SWITCHED OFF", ModelTab.allCases.filter { off.contains($0) })
            railGroup("NOT USED BY YOUR MODELS",
                      ModelTab.allCases.filter { inapplicable.contains($0) })
        }
    }

    @ViewBuilder
    private func railGroup(_ heading: String, _ tabs: [ModelTab]) -> some View {
        if !tabs.isEmpty {
            Text(heading)
                .font(.system(size: 9, weight: .bold)).kerning(0.4)
                .foregroundColor(Theme.textFaint)
                .padding(.horizontal, 12).padding(.top, 14).padding(.bottom, 4)
            ForEach(tabs, id: \.self) { railRow($0, dim: true) }
        }
    }

    private func railRow(_ t: ModelTab, dim: Bool) -> some View {
        // A dimmed tab shows NO status, deliberately, and for a different reason
        // in each group. Under SWITCHED OFF the value would just be "off" — the
        // heading already says that, in one place instead of once per row. Under
        // NOT USED BY YOUR MODELS the raw value actively contradicts the heading:
        // "Aligner — on" makes it look like it is running, "Overlap — off" like
        // switching it on would help, and neither is true. The reason is a
        // sentence either way, and it waits in the tab where there is room for it.
        let status = dim ? ("", false) : modelTabStatus(t)
        let active = modelTab == t
        return Button(action: {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.85)) { modelTab = t }
        }) {
            HStack(spacing: 8) {
                Image(systemName: t.icon)
                    .font(.system(size: 11, weight: .bold)).frame(width: 15)
                Text(t.title)
                    .font(.system(size: 12, weight: .bold))
                    .lineLimit(1)
                Spacer(minLength: 6)
                if !status.0.isEmpty {
                    Text(status.0)
                        .font(.system(size: 10))
                        .lineLimit(1)
                        .minimumScaleFactor(0.85)
                        .foregroundColor(active ? Theme.selectedTabText
                                         : (status.1 ? Theme.teal : Theme.textFaint))
                }
            }
            .foregroundColor(active ? Theme.selectedTabText
                             : (dim ? Theme.textFaint : Theme.textSecondary))
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(RoundedRectangle(cornerRadius: 8)
                .fill(active ? Theme.selectedTabBackground : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }


    // MARK: - Which model tabs can apply to the current choice
    //
    // Two of the six sub-tabs can be made irrelevant by the other two: the ALIGNER
    // has no work when the chunked model attributes its own text, and OVERLAP has
    // nothing to find when the diarization engine never marks two speakers at one
    // instant. Until now both sat in the bar looking exactly as usable as the
    // rest, and a user could spend time on a setting that would never run.
    //
    // They are dimmed and moved to the END of the bar, never hidden and never
    // disabled — the reason is a sentence, and a sentence needs the content area,
    // which the user only reaches by clicking. See `inapplicabilityNote`.
    //
    // Both answers come from the loader's own pure rules, asked with the feature
    // switched ON, so the question is "could this ever apply?" rather than "is it
    // on right now".

    @AppStorage("chunked.model")          private var chunkedModel = "qwen3"
    @AppStorage("diarization.engine")     private var diarEngine = "pyannote"
    @AppStorage("diarization.enabled")    private var diarEnabled = true
    @AppStorage("overlap.engine")         private var overlapEngineID = ModelCatalog.overlapSeparation.id
    @AppStorage("align.enabled")          private var alignEnabled = true
    @AppStorage("overlap.repair.enabled") private var repairEnabled = false
    @AppStorage("realtime.enabled")       private var realtimeEnabled = true
    @AppStorage("overlap.detect.enabled") private var detectEnabled = false
    @AppStorage("vad.enabled")            private var vadEnabled = true
    @AppStorage("chunked.enabled")        private var chunkedEnabled = true

    private var inapplicableModelTabs: Set<ModelTab> {
        var out: Set<ModelTab> = []
        // Asked with alignment switched ON — "could this ever apply?" — but with
        // `chunkedEnabled` as the user really has it: with no chunked pass there
        // are no segments to split into words, whatever the aligner's own toggle
        // says. That is an inapplicability, not a preference, so it belongs here
        // rather than in `switchedOffModelTabs`.
        //
        // …but NOT when MOSS is doing the same job itself. `wantsAligner` is
        // false for both reasons and they are not alike: with no chunked pass
        // nothing happens, while under MOSS the attribution HAPPENS — MOSS emits
        // a timed segment per speaker rather than words for the app to sort. The
        // DiariZen precedent below is the same call, made the same way.
        if !ModelLoader.wantsAligner(alignEnabled: true, chunkedID: chunkedModel,
                                     chunkedEnabled: chunkedEnabled),
           !ModelLoader.alignmentIsBuiltIn(chunkedID: chunkedModel,
                                           chunkedEnabled: chunkedEnabled) {
            out.insert(.aligner)
        }
        // Asked with repair switched ON, so the question is "could this ever
        // apply?" rather than "is it on right now" — but `detectEnabled` is passed
        // as the user really has it, because under MOSS and spectral that switch
        // is what decides whether repair has any regions at all. Turning Detect
        // overlap on therefore lifts this tab out of the dimmed group, which is
        // the truth: repair really does start working at that moment.
        if ModelLoader.wantedOverlapEngine(repairEnabled: true, engineID: overlapEngineID,
                                           diarEngine: diarEngine,
                                           detectEnabled: detectEnabled) == nil {
            out.insert(.overlap)
        }
        // Detection needs rows to mark, nothing more — so the tab APPLIES under
        // every engine, and is dimmed only when there is no diarization at all.
        //
        // Deliberately NOT asked through `wantsOverlapDetect`, which is a different
        // question. That rule answers "does this session load the pyannote
        // segmentation SIDECAR?", and it is false under DiariZen — but only because
        // DiariZen IS the detector there, not because the page has nothing to
        // offer. The tab shows DiariZen as the detection model and its switch turns
        // the marking on and off, so filing it under NOT USED would hide a control
        // that works (tried on 2026-08-10 and reverted the same day).
        if !diarEnabled { out.insert(.overlapDetect) }
        return out
    }

    /// Tabs whose feature is simply SWITCHED OFF: they would work, the owner has
    /// turned them off. Dimmed and moved down like the inapplicable ones, but
    /// under their own heading — see `modelRail` for why the two never merge.
    ///

    /// Subtracting `inapplicableModelTabs` is what keeps a tab out of BOTH lists.
    /// The stronger statement wins: under MOSS the Aligner cannot run whatever its
    /// toggle says, so filing it under "switched off" would invite the user to
    /// turn on something that would still not run.
    private var switchedOffModelTabs: Set<ModelTab> {
        var out: Set<ModelTab> = []
        if !chunkedEnabled  { out.insert(.chunked) }
        if !realtimeEnabled { out.insert(.realtime) }
        if !diarEnabled     { out.insert(.diarization) }
        if !detectEnabled   { out.insert(.overlapDetect) }
        if !repairEnabled   { out.insert(.overlap) }
        if !alignEnabled    { out.insert(.aligner) }
        if !vadEnabled      { out.insert(.vad) }
        return out.subtracting(inapplicableModelTabs)
    }

    /// Why the open tab cannot apply — the real reason from the rule that decided
    /// it, never a generic "not available". nil when the tab applies normally.
    private var inapplicabilityNote: String? {
        guard section == .models, inapplicableModelTabs.contains(modelTab) else { return nil }
        switch modelTab {
        // Aligner deliberately has NO banner (owner, 2026-08-06). The tab still
        // sits under "NOT USED BY YOUR MODELS", which already says it, and the
        // Aligner tab's own copy states the MOSS exclusion — a third place saying
        // the same thing is noise. It was also the only banner that could go
        // stale: it named the chunked MODEL as the reason, and since
        // `chunked.enabled` the aligner is equally inapplicable with the pass
        // switched off, where that sentence would simply have been wrong.
        case .overlapDetect:
            return "Speaker diarization is switched off, so there are no rows to mark."
        case .overlap:
            if !diarEnabled {
                return "Speaker diarization is switched off, so there are no speaker turns "
                     + "to look for overlap in. Nothing on this page will run."
            }
            if diarEngine == ModelLoader.mossEngineID {
                return "MOSS never marks two speakers at the same instant — its segments "
                     + "tile exactly, one speaker each — so it cannot tell repair where to "
                     + "work. Switch on Models → Detect overlap and this page becomes "
                     + "active: that detector reads the audio directly and hands its "
                     + "regions to the engine you pick here."
            }
            // Named per engine rather than defaulted to "the spectral engine": a
            // banner that names the wrong engine is worse than none, and with four
            // engines the `default` arm is no longer a safe place to hide one.
            let engineName = diarEngine == ModelLoader.nemoEngineID
                ? "The NeMo engine" : "The spectral engine"
            return "\(engineName) assigns exactly one speaker per instant, so its "
                 + "turns never intersect and it cannot tell repair where to work. Switch "
                 + "on Models → Detect overlap and this page becomes active: that detector "
                 + "reads the audio directly and hands its regions to the engine you pick "
                 + "here."
        default:
            return nil
        }
    }

    private var leftRail: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("SETTINGS")
                .font(.system(size: 13, weight: .heavy))
                .kerning(1.2)
                .foregroundColor(Theme.textPrimary)
                .padding(.bottom, 14)

            ForEach(Section.allCases, id: \.self) { s in
                Button(action: {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.85)) { section = s }
                }) {
                    HStack(spacing: 10) {
                        Image(systemName: s.icon)
                            .font(.system(size: 12, weight: .bold))
                            .frame(width: 16)
                        Text(s.rawValue)
                            .font(.system(size: 13, weight: .bold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.85)
                        Spacer(minLength: 0)
                    }
                    .foregroundColor(section == s ? Theme.selectedTabText : Theme.textSecondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(section == s ? Theme.selectedTabBackground : .clear)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                // The model sub-tabs live HERE now, nested under their section,
                // rather than in a horizontal bar above the content: a vertical
                // list has room for a status beside each name and for a heading
                // over the ones that cannot apply.
                if s == .models, section == .models {
                    modelRail.padding(.leading, 8).padding(.top, 2).padding(.bottom, 6)
                }
            }

            Spacer()

            Button(action: { dismiss() }) {
                HStack(spacing: 8) {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .bold))
                    Text("Close")
                        .font(.system(size: 12, weight: .bold))
                }
                .foregroundColor(Theme.textSubtle)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(RoundedRectangle(cornerRadius: 8).fill(Theme.chip))
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .frame(width: 232)
    }

    // MARK: Content

    private var content: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 3) {
                Text(section.rawValue)
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                Text(headerSubtitle)
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textDim)
                    .animation(.none, value: modelTab)
                    .animation(.none, value: atndTab)
            }
            .padding(.horizontal, 24)
            .padding(.top, 20)
            .padding(.bottom, 14)

            if section == .atnd {
                SubTabBar(selection: $atndTab, geometryID: "activeATNDTab", namespace: tabIndicator)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
            }

            Rectangle().fill(Theme.divider).frame(height: 1)

            ScrollView {
                VStack(alignment: .leading, spacing: SettingsLayout.row) {
                    if let note = inapplicabilityNote {
                        HStack(alignment: .top, spacing: 9) {
                            Image(systemName: "info.circle")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(Theme.amber)
                            Text(note)
                                .font(.system(size: 12))
                                .foregroundColor(Theme.amber)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(RoundedRectangle(cornerRadius: 8)
                            .fill(Theme.amber.opacity(0.10)))
                    }
                    switch section {
                    case .models:
                        switch modelTab {
                        case .realtime:    RealtimeModelTab()
                        case .chunked:     ChunkedModelTab()
                        case .aligner:     AlignerTab()
                        case .vad:         VADTab()
                        case .diarization: DiarizationTab()
                        case .overlap:     OverlapTab()
                        case .overlapDetect: OverlapDetectTab()
                        }
                    case .microphone:
                        MicrophoneTab()
                    case .atnd:
                        switch atndTab {
                        case .connection: ATNDConnectionTab()
                        case .command:    ATNDCommandTab()
                        case .position:   ATNDPositionTab()
                        }
                    case .utils:
                        UtilsTab()
                    }
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
                .id("\(section)-\(modelTab)-\(atndTab)") // re-render for transition
                .transition(.opacity.combined(with: .move(edge: .trailing)))
            }
            .animation(.spring(response: 0.32, dampingFraction: 0.86), value: modelTab)
            .animation(.spring(response: 0.32, dampingFraction: 0.86), value: atndTab)
            .animation(.spring(response: 0.32, dampingFraction: 0.86), value: section)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
    }

    /// Sections with sub-tabs show the active sub-tab's subtitle.
    private var headerSubtitle: String {
        switch section {
        case .models:
            // The Aligner's own subtitle promises WORD-exact attribution, which
            // is what the Qwen3 aligner delivers and what MOSS does not: MOSS
            // reports `{start, end, speaker, text}` per SEGMENT and no word times
            // at all. Left alone, the header contradicted the card two lines
            // below it, which says there is nothing for a word aligner to split.
            //
            // Overridden HERE rather than by making `subtitle` take arguments:
            // it is a plain property on the enum, read by three sections, and
            // this is the one case in the app where the header depends on
            // another tab's model.
            if modelTab == .aligner,
               ModelLoader.alignmentIsBuiltIn(chunkedID: chunkedModel,
                                              chunkedEnabled: chunkedEnabled) {
                return "Speaker attribution — done by the transcription model itself"
            }
            return modelTab.subtitle
        case .atnd:   return atndTab.subtitle
        default:      return section.subtitle
        }
    }
}
