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

    @AppStorage("diarization.engine")     private var diarEngine = "pyannote"
    @AppStorage("overlap.detect.enabled") private var detectOn = false

    private var dicowSelected: Bool { engine == ModelCatalog.overlapDicow.id }

    private var windowBinding: Binding<Int> { dicowSelected ? $dicowWindowSec : $mossWindowSec }
    private var debugRowsBinding: Binding<Bool> {
        dicowSelected ? $dicowShowDebugRows : $mossShowDebugRows
    }

    /// Engines that assign exactly one speaker per instant, and so depend on the
    /// standalone detector to locate overlap. Mirrors
    /// `ModelLoader.wantedOverlapEngine`'s own test — THREE engines since NeMo
    /// (2026-08-07), and it must keep mirroring it: this copy decides only what the
    /// tab SAYS, while the loader decides what runs, so a divergence would print a
    /// reassurance about a session that behaves the other way.
    private var needsDetector: Bool {
        diarEngine == ModelLoader.mossEngineID || diarEngine == ModelLoader.spectralEngineID
            || diarEngine == ModelLoader.nemoEngineID
    }

    private var diarEngineName: String {
        switch diarEngine {
        case ModelLoader.mossEngineID: return "MOSS"
        case ModelLoader.nemoEngineID: return "the NeMo engine"
        default:                       return "the spectral engine"
        }
    }

    var body: some View {
        Group {
            // The switch comes FIRST, and everything it controls appears only once
            // it is on — the same shape as Detect overlap. Previously the engine
            // cards sat ABOVE their own switch and the settings below it were
            // merely `.disabled`, so the page showed a model choice, then a row of
            // greyed controls, and only in the middle the one control that decides
            // whether any of it runs. Nothing is hidden that is still in effect:
            // while this is off, no engine is loaded at all.
            SettingToggle(label: "Repair overlap regions at stop (experimental)", isOn: $enabled)

            if enabled {
                overlapEngineSection
            }

            Text("The selected engine loads up front when this is on (visible in the loading overlay) and runs only after you stop recording. Requires diarization to be enabled so overlaps can be located.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Engine choice and the selected engine's own settings. Only reachable while
    /// `enabled` is on, which is why nothing in here carries `.disabled(!enabled)`
    /// any more — a control that cannot be seen cannot be misread as inert.
    /// The two SHARED settings, then the engine cards. Both settings exist for
    /// each engine under its own key with its own default, so the bindings follow
    /// the selection rather than one key being read by two engines — the same
    /// reason `DiarizationTab` splits MOSS's stop pair from pyannote's.
    ///
    /// They sit ABOVE the cards (owner, 2026-08-06) because they are the two
    /// questions worth asking whichever engine is picked; the cards answer a third.
    @ViewBuilder
    private var overlapEngineSection: some View {
        Group {
            SettingBlock(title: "Window around each overlap") {
                // Per-engine tags: DiCoW caps at 14 s because its sidecar rejects
                // windows over 30 s (±14 = 28), MossFormer2 has no such limit.
                // Same control, different ceilings — not a cosmetic difference.
                Picker("", selection: windowBinding) {
                    Text("5s").tag(5)
                    Text("8s").tag(8)
                    Text("10s").tag(10)
                    Text(dicowSelected ? "14s" : "15s").tag(dicowSelected ? 14 : 15)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text("Seconds of audio on each side of the overlap's midpoint. Longer gives the model more context and takes longer.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Show model speaker rows") {
                SettingToggle(label: "Show model speaker rows", isOn: debugRowsBinding)
                Text("Adds the engine's raw per-speaker output as extra rows, verbatim and ungated — useful for checking what it produced. Turn off for a clean transcript.")
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

            ModelInstallStatus(model: ModelCatalog.overlapEngine(id: engine))
        }
    }
}
