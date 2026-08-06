import SwiftUI

/// Models → Aligner: the forced aligner that gives every word in a chunk its
/// own start/end time (`align.enabled`).
///
/// Its own tab rather than a toggle buried at the bottom of Chunked: this is a
/// 1.2 GB model that loads with the session, so it belongs on the card-based
/// model surface every other model uses.
///
/// The card doubles as the on/off control, but it is no longer the ONLY one: an
/// explicit `SettingToggle` was added on owner request (2026-07-31). The old
/// reasoning — "one aligner, so a toggle next to a single-choice card would be
/// redundant" — missed that every other Settings tab states its on/off as a
/// labelled switch, so this tab read as having no setting at all. A card that is
/// secretly a switch is discoverable only by clicking it.
///
/// Both drive the same `align.enabled`, so they can never disagree.
struct AlignerTab: View {
    @AppStorage("align.enabled") private var enabled = false
    @AppStorage("chunked.model") private var chunkedModel = "qwen3"

    private var model: ModelInfo { ModelCatalog.wordAligner }

    var body: some View {
        Group {
            
            SettingBlock(title:"") {
                SettingToggle(label: "Enable word alignment", isOn: $enabled)
            }

            Button(action: {
                withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                    enabled.toggle()
                }
            }) {
                ModelCardView(model: model, selected: enabled)
            }
            .buttonStyle(.plain)

            ModelInstallStatus(model: model)
        }
    }
}
