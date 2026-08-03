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
    private var installed: Bool { ModelCatalog.isInstalled(model) }

    var body: some View {
        Group {
            Text("Gives every word in a chunk its own start and end time, so a speaker change can split the text at the exact word instead of at an estimated position.")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

            Text("The text appears as soon as the chunked model returns it, split between speakers by the estimate; the rows are rebuilt word-exactly when the alignment arrives, usually under a second later. The transcript never waits for it.")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)

            // MOSS already returns its own per-segment times, and its speaker
            // labels are anonymous per chunk, so aligning its text would refine
            // boundaries the model itself drew. Excluded in `ModelLoader
            // .wantsAligner` rather than left to this toggle, so the card below
            // is genuinely inert — say so.
            if chunkedModel == "moss" {
                Text("Not used with MOSS: it returns its own timestamps. Pick another chunked model in Models → Chunked to use the aligner.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Word-level alignment") {
                SettingToggle(label: "Enable word alignment", isOn: $enabled)

                // Stated here because the aligner is a whole extra process and
                // this tab is where someone decides to pay for it. MEASURED
                // 2026-07-31 on the owner's M4: 2.20 GB resident, the second
                // largest sidecar after the chunked model itself (4.29 GB), in a
                // default session totalling 9.49 GB.
                Text(enabled
                     ? "On — the aligner runs as its own process and holds about 2.2 GB while a session is active."
                     : "Off — text is split between speakers by estimated character position. Turning this on costs about 2.2 GB of memory for a second process.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Button(action: {
                withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                    enabled.toggle()
                }
            }) {
                ModelCardView(model: model, selected: enabled)
            }
            .buttonStyle(.plain)

            HStack(spacing: 6) {
                Image(systemName: installed ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .font(.system(size: 11))
                    .foregroundColor(installed ? Theme.teal : Theme.red)
                Text(installed
                     ? "Downloaded — use the switch above, or click the card, to turn word alignment on or off."
                     : "Not downloaded. Run download-best-models.sh before turning this on, or the session will fail to start.")
                    .font(.system(size: 11))
                    .foregroundColor(installed ? Theme.textFaint : Theme.red)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
            }

            SettingBlock(title: "What it changes") {
                Text("Without it, a chunk is divided between speakers by estimated character position: the text is cut proportionally, so a speaker switch in the middle of a sentence puts a few words on the wrong side. With it, each word goes to the speaker whose turn actually covers that word's timestamp.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("This matters most with the ATND array, where the beam direction can change mid-sentence.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "What to expect") {
                Text("Works with every chunked model — the aligner only ever sees the audio and the text, never the recogniser. It runs in its own process, alongside transcription rather than after it, so it adds nothing to the time before text appears.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Live captions are not aligned — they are replaced by the chunked pass within one interval anyway, so only the final transcript is word-exact.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Forced alignment fits whatever text the recogniser produced onto the audio, so a misheard word also carries a misplaced time. When alignment is unreliable — the timings disagree with the chunk, or words run past the end of the audio — that chunk silently keeps the estimate rather than showing wrong times. The aligner's own decisions are logged to logs/aligner.log; how the words were then used is logged to logs/position-diarization.log as WORDS OK or WORDS SKIP.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
