import SwiftUI

/// Models → Diarization: pyannote community-1 settings.
struct DiarizationTab: View {
    @AppStorage("diarization.enabled")     private var enabled = true
    @AppStorage("diarization.live")        private var live = true
    @AppStorage("diarization.finalPass")   private var finalPass = true
    @AppStorage("diarization.continueOnStop") private var continueOnStop = false
    @AppStorage("diarization.intervalSec") private var intervalSec = 30
    @AppStorage("diarization.resetOnStart") private var resetOnStart = true
    @AppStorage("diarization.detectOverlap") private var detectOverlap = true
    @AppStorage("diarization.numSpeakers") private var numSpeakers = 0

    var body: some View {
        Group {
            ModelCardView(model: ModelCatalog.diarization, selected: true)

            SettingToggle(label: "Enable speaker diarization (who spoke when)", isOn: $enabled)

            SettingBlock(title: "How it runs") {
                SettingToggle(label: "Live labels during recording (chunked)", isOn: $live)
                Text("Diarizes each chunk as it's recorded and matches voices against your saved speaker profiles, so names appear during the meeting instead of only at the end.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                SettingToggle(label: "Run a diarization pass at stop", isOn: $finalPass)
                Text("After you stop, run one more diarization pass to finalize speaker labels. Turn off to keep only the live labels.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                SettingToggle(label: "Continue from live labels (tail only)", isOn: $continueOnStop)
                    .disabled(!finalPass)
                Text(continueOnStop
                     ? "At stop, only the audio since the last live chunk is diarized and appended — speaker names and numbering stay stable. If live labels are off, the whole recording is diarized as one tail."
                     : "At stop, the entire recording is re-diarized and replaces the live labels — best global clustering, but speaker numbering can change.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Live diarization interval") {
                Picker("", selection: $intervalSec) {
                    Text("15s").tag(15)
                    Text("30s").tag(30)
                    Text("45s").tag(45)
                    Text("60s").tag(60)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(!live)

                Text("How often live diarization runs — separate from the chunked-ASR interval. Longer chunks give steadier voice matching and less CPU load; shorter chunks label speakers sooner. Only used when live labels are on.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Overlapping speech") {
                SettingToggle(label: "Detect overlapping speech (show both speakers)", isOn: $detectOverlap)
                Text(detectOverlap
                     ? "When two people talk at once, both are kept and their rows are flagged as overlapping. Note: on a single mic the words in an overlap are approximate."
                     : "Each moment is assigned to one speaker only (cleaner rows, no overlaps). Best when people rarely talk over each other.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Speaker profiles") {
                SettingToggle(label: "Start each recording fresh (reset profiles)", isOn: $resetOnStart)
                Text(resetOnStart
                     ? "Every recording begins with an empty speaker store, so speakers are numbered from 1 each time. Turn off to remember voices across meetings."
                     : "Voices are remembered across meetings — a speaker matched in a past recording keeps the same name.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Number of speakers") {
                Picker("", selection: $numSpeakers) {
                    Text("Auto").tag(0)
                    Text("2").tag(2)
                    Text("3").tag(3)
                    Text("4").tag(4)
                    Text("5").tag(5)
                    Text("6").tag(6)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text(numSpeakers == 0
                     ? "Auto-detect — flexible, but may split or merge voices on hard audio."
                     : "Fixed at \(numSpeakers) — more accurate when you know the participant count.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }

            Text("The pipeline and voice-embedding model load up front (visible in the loading overlay). Speaker profiles are saved locally under models/speaker-profiles and reused across meetings — rename a speaker in the transcript and it's remembered next time.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
