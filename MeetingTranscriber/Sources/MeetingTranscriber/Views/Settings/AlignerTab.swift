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
///
/// UNDER MOSS THE TAB SHOWS MOSS (owner, 2026-08-12), the `DiariZen (built in)`
/// pattern from Detect overlap. `ModelLoader.wantsAligner` has excluded MOSS
/// since the aligner became its own sidecar — the process is torn down so a
/// session cannot inherit a live aligner and start aligning MOSS text by
/// accident — and the rail has correctly filed this tab under NOT USED BY YOUR
/// MODELS all along. What was missing was the tab's own half: it went on showing
/// a 1.2 GB Qwen3 card and an inviting switch, with **no sentence anywhere
/// saying why neither would act.** That is the rule the 2026-08-06 settings pass
/// set out ("a control that cannot act says so") broken in the quietest way.
struct AlignerTab: View {
    @AppStorage("align.enabled") private var enabled = false
    @AppStorage("chunked.model") private var chunkedModel = "qwen3"
    @AppStorage("chunked.enabled") private var chunkedEnabled = true

    /// The model in force for the CURRENT chunked choice — one function, shared
    /// with nothing else that could disagree with it.
    private var model: ModelInfo {
        ModelCatalog.wordAligners(forChunkedModel: chunkedModel)[0]
    }

    /// MOSS does this itself, so there is no separate aligner to switch on.
    ///
    /// Read from `ModelLoader` rather than restated as `chunkedModel == "moss"`,
    /// which is what this was for an hour: the rail groups and labels the tab
    /// from the same rule, and a local copy is exactly how the page comes to say
    /// one thing while the row beside it says another. It also carries the
    /// `chunkedEnabled` term the local copy lacked — with the chunked pass off,
    /// MOSS is not attributing anything either, and the note below must give the
    /// reason the user can actually act on.
    private var alignerIsTheTranscriber: Bool {
        ModelLoader.alignmentIsBuiltIn(chunkedID: chunkedModel,
                                       chunkedEnabled: chunkedEnabled)
    }

    /// Asked with alignment ON — "could this ever apply?" — so the note below
    /// reports an inapplicability rather than the user's own preference. Matches
    /// `SettingsView.inapplicableModelTabs` exactly, which is what stops the rail
    /// and the tab telling two different stories about one session.
    private var applies: Bool {
        ModelLoader.wantsAligner(alignEnabled: true, chunkedID: chunkedModel,
                                 chunkedEnabled: chunkedEnabled)
    }

    var body: some View {
        Group {

            SettingBlock(title: "") {
                SettingToggle(label: "Enable word alignment", isOn: $enabled)
                if !applies {
                    // The switch stays REACHABLE and its stored value survives —
                    // the 2026-07-31 language-picker reversal: a control that
                    // disappears reads as a bug, a control that explains itself
                    // reads as an answer, and switching the chunked model back
                    // must find the choice where it was left.
                    Text(note)
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            // Under MOSS the card is NOT a button. Elsewhere it doubles as the
            // switch, but here there is nothing to switch: tapping "MOSS (built
            // in)" would move `align.enabled`, which this session does not read.
            if alignerIsTheTranscriber {
                ModelCardView(model: model, selected: true)
            } else {
                Button(action: {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                        enabled.toggle()
                    }
                }) {
                    ModelCardView(model: model, selected: enabled)
                }
                .buttonStyle(.plain)
            }

            ModelInstallStatus(model: model)
        }
    }

    /// Why the switch will not act. Two distinct reasons, and they must not be
    /// collapsed: one is answered by choosing a different chunked model, the
    /// other by switching the chunked pass back on.
    private var note: String {
        if alignerIsTheTranscriber {
            // NOT "timing with the words" — that was the first wording and it
            // claims word-level times MOSS does not produce. What MOSS returns is
            // one timed, speaker-labelled SEGMENT at a time, which is why there
            // is nothing left to attribute rather than a second way of doing it.
            return "MOSS already labels each segment with its speaker, so there is nothing "
                 + "for a word aligner to attribute. The Qwen3 aligner is not loaded this "
                 + "session and this switch will not change the transcript."
        }
        return "The chunked pass is off, so there are no segments to split into words. "
             + "This switch will not change anything until you switch it back on in "
             + "Models → Chunked."
    }
}
