import SwiftUI

/// Models → Chunked: select the batch ASR model for the final transcript.
struct ChunkedModelTab: View {
    @AppStorage("chunked.model")       private var model = "qwen3"
    @AppStorage("chunked.language")    private var language = "auto"
    @AppStorage("chunked.intervalSec") private var intervalSec = 30

    var body: some View {
        Group {
            Text("Accurate rolling transcript during the meeting: every interval, the chunk is re-transcribed with this model (cut at silence, never mid-word).")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

            ForEach(ModelCatalog.chunked) { m in
                Button(action: {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                        model = m.id
                    }
                }) {
                    ModelCardView(model: m, selected: model == m.id)
                }
                .buttonStyle(.plain)
            }

            SettingBlock(title: "Chunk interval — faster updates vs higher accuracy") {
                Picker("", selection: $intervalSec) {
                    Text("15 s").tag(15)
                    Text("30 s").tag(30)
                    Text("60 s").tag(60)
                    Text("120 s").tag(120)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text(intervalHint)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Language") {
                LanguagePicker(selection: $language)
            }
        }
    }

    /// Facts: Whisper large-v3 is trained on 30 s windows (its optimum);
    /// chunks under ~15 s measurably increase WER from lost context;
    /// longer chunks improve accuracy further at the cost of update delay.
    private var intervalHint: String {
        switch intervalSec {
        case 15:  return "Fastest updates — below models' optimal context; accuracy drops noticeably."
        case 30:  return "Recommended — Whisper's training window (30 s) and a strong balance for Qwen3/Voxtral."
        case 60:  return "Higher accuracy — more context per chunk, updates once a minute."
        default:  return "Best accuracy — maximum context; transcript updates every 2 minutes."
        }
    }
}
