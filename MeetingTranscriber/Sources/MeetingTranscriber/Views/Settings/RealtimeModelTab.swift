import SwiftUI

/// Models → Realtime: Nemotron streaming settings.
struct RealtimeModelTab: View {
    @AppStorage("realtime.enabled")  private var enabled = true
    @AppStorage("realtime.chunkMs")  private var chunkMs = 160
    @AppStorage("realtime.language") private var language = "auto"

    var body: some View {
        Group {
            ModelCardView(model: ModelCatalog.realtime, selected: true)

            SettingToggle(label: "Enable realtime transcription", isOn: $enabled)

            SettingBlock(title: "Chunk size — latency vs accuracy") {
                Picker("", selection: $chunkMs) {
                    Text("80 ms").tag(80)
                    Text("160 ms").tag(160)
                    Text("560 ms").tag(560)
                    Text("1120 ms").tag(1120)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text(chunkHint)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }

            SettingBlock(title: "Language") {
                LanguagePicker(selection: $language, entries: Languages.realtime)

                // The chunked tab now offers a per-model list built from each
                // model's published roster. Nemotron has none, so this list is
                // deliberately the old short one — said out loud so the
                // difference between the two tabs does not read as an oversight.
                Text("Nemotron's card states 40 locales but does not publish which ones, so this list is not the full set — unlike the chunked models, whose languages come from their published rosters.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var chunkHint: String {
        switch chunkMs {
        case 80:   return "Lowest latency — captions appear almost instantly."
        case 160:  return "Balanced — recommended for meetings."
        case 560:  return "Higher accuracy, ~0.5s delay."
        default:   return "Best accuracy, ~1s delay."
        }
    }
}
