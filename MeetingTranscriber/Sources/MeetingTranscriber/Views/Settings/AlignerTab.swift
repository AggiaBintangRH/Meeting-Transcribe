import SwiftUI

/// Models → Aligner: the forced aligner that gives every word in a chunk its
/// own start/end time (`align.enabled`).
///
/// Its own tab rather than a toggle buried at the bottom of Chunked: this is a
/// 1.2 GB model that loads with the session, so it belongs on the card-based
/// model surface every other model uses.
///
/// The card itself is the on/off control — there is only one aligner, so a
/// separate toggle next to a single-choice card would be redundant.
struct AlignerTab: View {
    @AppStorage("align.enabled") private var enabled = false

    private var model: ModelInfo { ModelCatalog.wordAligner }
    private var installed: Bool { ModelCatalog.isInstalled(model) }

    var body: some View {
        Group {
            Text("Gives every word in a chunk its own start and end time, so a speaker change can split the text at the exact word instead of at an estimated position.")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

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
                     ? "Downloaded — click the card to turn word alignment on or off."
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
                Text("Works with every chunked model — the aligner only ever sees the audio and the text, never the recogniser. Alignment adds roughly 0.3 s per chunk on top of transcription.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Live captions are not aligned — they are replaced by the chunked pass within one interval anyway, so only the final transcript is word-exact.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Forced alignment fits whatever text the recogniser produced onto the audio, so a misheard word also carries a misplaced time. When alignment is unreliable — the timings disagree with the chunk, or words run past the end of the audio — that chunk silently falls back to the estimate rather than showing wrong times. Every decision is logged to logs/position-diarization.log as WORDS OK or WORDS SKIP.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
