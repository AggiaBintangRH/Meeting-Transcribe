import SwiftUI

/// Settings → Models → Detect overlap.
///
/// A tab of its own, and deliberately NOT a block inside the existing Overlap
/// tab, because it is a different job done by a different model:
///
///   * **Overlap REPAIR** (the other tab) tries to recover the words lost when
///     two people talk at once. It needs overlap REGIONS, which it takes from
///     pyannote turns that intersect.
///   * **Overlap DETECTION** (here) only marks where that happened. It needs no
///     turns at all — it reads the audio directly with pyannote's segmentation
///     model, so it works under engines that cannot mark overlap themselves.
///
/// WHY IT EXISTS. MOSS and spectral both assign exactly one speaker per instant —
/// MOSS's segments tile exactly, spectral's Viterbi smoothing admits no
/// intersection — so neither can report that two people spoke together. Under
/// those engines this is the ONLY way to know a row's words are approximate.
/// Under pyannote it is redundant: that engine already reports overlap itself.
///
/// The numbers below are measured on this machine, not estimated:
///   * the model is 32 MB and already ships inside the app — no download;
///   * it loads in ~0 s and scans at ~160x realtime (43-minute meeting: 16 s);
///   * verified in BOTH directions — 16.2 % of a clip that really contains
///     overlap, 0.0 % of a clean one, and 0.1–1.8 % on the owner's real
///     meetings, so it does not simply fire on room noise.
struct OverlapDetectTab: View {
    @AppStorage("overlap.detect.enabled") private var enabled = false
    @AppStorage("overlap.detect.model")   private var detector = ModelCatalog.overlapDetectPyannote.id
    @AppStorage("diarization.engine")     private var engine = "pyannote"
    @AppStorage("diarization.enabled")    private var diarOn = true

    /// pyannote already marks overlap on its own, so switching this on there would
    /// pay for a second pass to learn what the transcript already knows.
    private var redundantHere: Bool {
        diarOn && engine == ModelLoader.pyannoteEngineID
    }

    var body: some View {
        VStack(alignment: .leading, spacing: SettingsLayout.row) {
            if redundantHere {
                Note(color: Theme.textDim,
                     icon: "checkmark.circle",
                     text: "The pyannote engine already reports overlapping speech "
                         + "itself, so this detector has nothing to add while it is "
                         + "selected. It matters for MOSS and spectral, which assign "
                         + "exactly one speaker per instant and cannot mark overlap "
                         + "at all.")
            } else if !diarOn {
                Note(color: Theme.amber, icon: "info.circle",
                     text: "Speaker diarization is switched off, so there are no rows "
                         + "to mark. Turn it on to use this.")
            }
            // NO banner under MOSS or spectral (owner, 2026-08-06). It explained
            // why the detector matters there, which is the tab's whole subject —
            // the subtitle and the toggle already say it, and a banner that fires
            // on the ordinary case is a banner nobody reads.
            //
            // The two above are kept because each reports something the tab cannot
            // do rather than describing what it is for: pyannote makes it
            // redundant, and diarization off leaves it nothing to mark.

            // The switch comes FIRST and the model picker only appears once it is
            // on. A picker above its own switch asks the user to choose a model
            // for something that is not running — and because nothing is loaded
            // while this is off, that choice would have no effect until they
            // scrolled past it and found the switch.
            SettingBlock(title: "Detection") {
                SettingToggle(label: "Mark rows where two people spoke at once",
                              isOn: $enabled)
                Text("Runs once at Stop, over the whole recording. Measured on this "
                     + "Mac: about 16 seconds for a 43-minute meeting.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if enabled {
                SettingBlock(title: "Detection model") {
                    // No VStack of its own: this list is inside a SettingBlock, so
                    // it already gets `SettingsLayout.inBlock`. The hard-coded 8
                    // that used to be here happened to agree — a second copy of a
                    // number is how the two drift apart later.
                    ForEach(ModelCatalog.overlapDetectors) { m in
                        Button(action: {
                            withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                                detector = m.id
                            }
                        }) {
                            ModelCardView(model: m, selected: detector == m.id)
                        }
                        .buttonStyle(.plain)
                    }

                    ModelInstallStatus(model: ModelCatalog.overlapDetector(id: detector))
                }
            }
        }
    }

    private struct Note: View {
        let color: Color
        let icon: String
        let text: String
        var body: some View {
            HStack(alignment: .top, spacing: 9) {
                Image(systemName: icon).font(.system(size: 12, weight: .bold))
                    .foregroundColor(color)
                Text(text).font(.system(size: 12)).foregroundColor(color)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 8).fill(color.opacity(0.10)))
        }
    }
}
