import SwiftUI

/// Models → Overlap: overlap-repair settings (experimental).
///
/// Two selectable engines, each with its own settings — only the selected
/// engine's blocks are shown, so e.g. MossFormer2's debug-row toggle stays
/// hidden while DiCoW is picked:
///  • MossFormer2 (attempt #3, 2026-07-14) — separates the waveform into two
///    per-speaker tracks at stop and re-transcribes each.
///  • DiCoW v3.3 (attempt #4, 2026-07-16) — diarization-conditioned Whisper,
///    transcribes one speaker at a time from its mask.
/// `overlap.repair.enabled` (the shared on/off) and `overlap.engine` apply to
/// both; window size and debug rows are per-engine.
struct OverlapTab: View {
    @AppStorage("overlap.repair.enabled")           private var enabled = false
    @AppStorage("overlap.engine")                   private var engine = "mossformer2"
    @AppStorage("overlap.mossformer.windowSec")     private var mossWindowSec = 10
    @AppStorage("overlap.mossformer.showDebugRows") private var mossShowDebugRows = true
    @AppStorage("overlap.dicow.windowSec")          private var dicowWindowSec = 10
    @AppStorage("overlap.dicow.showDebugRows")      private var dicowShowDebugRows = true

    @AppStorage("diarization.engine") private var diarEngine = "pyannote"

    private var dicowSelected: Bool { engine == ModelCatalog.overlapDicow.id }

    var body: some View {
        Group {
            Text("Engine used to recover speech when two people talk at once.")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

            // Both engines locate their windows from pyannote turns, so neither
            // has anything to work on under the MOSS diarization engine. Said
            // here because the setting that disables this feature lives on a
            // different tab, and the engine is not even loaded (ModelLoader
            // skips the step) — the toggle below would otherwise look active.
            if diarEngine == ModelLoader.mossEngineID {
                Text("Not active: overlap repair finds its windows in pyannote's speaker turns, and the diarization engine is currently set to MOSS. Switch it back in Models → Diarization to use this.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            ForEach(ModelCatalog.overlapEngines) { m in
                Button(action: {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                        engine = m.id
                    }
                }) {
                    ModelCardView(model: m, selected: engine == m.id)
                }
                .buttonStyle(.plain)
            }

            SettingToggle(label: "Repair overlap regions at stop (experimental)", isOn: $enabled)

            if !dicowSelected {
                SettingBlock(title: "Show raw separated tracks") {
                    SettingToggle(label: "Show \"MossFormer2 Index 1/2\" rows", isOn: $mossShowDebugRows)
                        .disabled(!enabled)

                    Text("Adds two extra rows per repaired overlap showing each separated track's raw transcription verbatim (before combine/replace) — useful for checking what the separator produced. Turn off for a clean transcript.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }

                SettingBlock(title: "Window around each overlap") {
                    Picker("", selection: $mossWindowSec) {
                        Text("5s").tag(5)
                        Text("8s").tag(8)
                        Text("10s").tag(10)
                        Text("15s").tag(15)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .disabled(!enabled)

                    Text("How much audio around each detected overlap is fed to the separator — this many seconds on each side of the overlap's midpoint (default 10s → a 20-second window). Longer windows give the model more context but take longer to process.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }

                SettingBlock(title: "What to expect") {
                    Text("On a single microphone, separating two people talking at once is unreliable: the two output tracks are often blended or near-duplicates rather than two clean voices. When that happens the region keeps its original text unchanged — an internal quality check (near-duplicate guard + speaker attribution) skips it. Expect many, sometimes most, regions to be skipped; that is correct, safe behaviour, not a failure.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("Every decision (overwrite or skip, with before/after text) is logged to logs/overlap-repair-decisions.log. Only two-speaker overlaps are handled — windows touching three or more speakers are skipped.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if dicowSelected {
                SettingBlock(title: "Show raw per-speaker output") {
                    SettingToggle(label: "Show \"DiCoW · <speaker>\" rows", isOn: $dicowShowDebugRows)
                        .disabled(!enabled)

                    Text("Adds one extra row per speaker per repaired overlap showing what DiCoW transcribed for that speaker, verbatim and ungated — including text the quality checks rejected. Turn off for a clean transcript.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }

                SettingBlock(title: "Window around each overlap") {
                    Picker("", selection: $dicowWindowSec) {
                        Text("5s").tag(5)
                        Text("8s").tag(8)
                        Text("10s").tag(10)
                        Text("14s").tag(14)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .disabled(!enabled)

                    Text("How much audio around each detected overlap is fed to DiCoW — this many seconds on each side of the overlap's midpoint (default 10s → a 20-second window). Capped at 14s because DiCoW reads at most 30 seconds at once; longer windows are skipped rather than processed unreliably.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }

                SettingBlock(title: "What to expect") {
                    Text("DiCoW transcribes one speaker at a time using that speaker's diarization mask, so it handles overlaps in the ASR instead of splitting the waveform. It does recover speech that is otherwise lost — but this engine was tried once before (July 2026) and removed, because on real recordings it sometimes put one speaker's words under another speaker, repeated the same words for both, or ran away and attributed far more words to a speaker than they had time to say.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("Those three failures are now each checked for: text that matches the other speaker's existing words better than its own, text duplicated across both speakers, and text too long for the speaker's own speaking time are all rejected, and the region keeps its original text. Expect regions to be skipped; that is correct, safe behaviour, not a failure.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("Every decision (overwrite or skip, with before/after text and the reason) is logged to logs/overlap-repair-decisions.log — read it before trusting a repaired transcript. Only two-speaker overlaps are handled — windows touching three or more speakers are skipped.")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Text("The selected engine loads up front when this is on (visible in the loading overlay) and runs only after you stop recording. Requires diarization to be enabled so overlaps can be located.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
