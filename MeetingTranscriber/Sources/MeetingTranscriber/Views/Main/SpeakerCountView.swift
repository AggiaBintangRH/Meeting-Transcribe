import SwiftUI

/// How many speakers to ask each diarizer for, under the record card.
///
/// Moved out of the status chips (owner, 2026-08-13). It sat beside the RMS
/// meter, where a row of read-only readings made a real, writable setting look
/// like one more number the app was reporting. Under the mic it reads as what it
/// is: something you set before you press record.
///
/// TWO COUNTS, because there are two recordings. Office is the room; Remote is
/// the far end of the call, and they are separate identity spaces holding
/// different people. Until today the remote passes were pinned to Auto and the
/// room's number was, for five days, wrongly forced onto them — a 5-person room
/// with one caller split that caller into five profiles.
///
/// ⚠ **The far end is not visible from your chair.** That was the argument for
/// not offering this at all, and it is still the risk: a pinned count is an EXACT
/// constraint on pyannote, spectral and NeMo, so a guess here does not degrade
/// gently — it merges people who spoke or invents people who did not. Both rows
/// therefore default to Auto and the remote row says so.
struct SpeakerCountView: View {
    @ObservedObject var recorder: AudioRecorder

    /// The two counts. `AudioRecorder.diarNumSpeakers` and `.remoteNumSpeakers`
    /// read these same keys, so control and passes are ONE value rather than two
    /// readers agreeing.
    @AppStorage("diarization.numSpeakers")       private var office = 0
    @AppStorage("diarization.remoteNumSpeakers") private var remote = 0

    @AppStorage("diarization.engine")         private var engine = "pyannote"
    @AppStorage("diarization.enabled")        private var diarOn = true
    @AppStorage("diarization.finalPass")      private var finalPass = true
    @AppStorage("diarization.continueOnStop") private var continueOnStop = false
    /// -1 or absent = no Remote channel selected.
    ///
    /// ⚠ **The row STAYS VISIBLE then, dimmed, with the reason** — it was hidden
    /// outright for about an hour on 2026-08-13 and the owner's first words on
    /// seeing it were *"kok cuma ada office"*, which is precisely the outcome this
    /// project's own rule predicts: **a control that disappears reads as a bug, a
    /// dimmed one with a reason reads as an answer** (the language-picker
    /// reversal, 2026-07-31, and the SPK chip after it). Hiding it was argued as
    /// "a greyed picker would invent a stream the session does not have"; that is
    /// answered by SAYING there is no stream, which the note below does, and which
    /// also tells the user where to go and turn one on.
    @AppStorage("mic.remoteChannel")          private var remoteChannel = -1

    /// Is a Remote (conferencing) channel configured at all?
    private var hasRemote: Bool { remoteChannel >= 0 }

    /// Whether the number reaches the engine in THIS session — the loader's own
    /// rule, never restated here. A second copy is how a control comes to promise
    /// something the pass does not do.
    private var honoured: Bool {
        ModelLoader.speakerCountReachesEngine(diarEngine: engine,
                                              diarizationEnabled: diarOn,
                                              finalPass: finalPass,
                                              continueOnStop: continueOnStop)
    }

    /// Locked for the whole session, like every control behind the gear
    /// (2026-08-12). The stop passes read both values, so a change made
    /// mid-meeting would decide the speaker count of a recording already half
    /// over.
    private var locked: Bool { recorder.state != .idle }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("SPEAKERS")
                .font(.system(size: 9, weight: .bold)).kerning(0.6)
                .foregroundColor(Theme.textFaint)

            row(label: "Office", value: $office, tint: Theme.teal)
            row(label: "Remote", value: $remote, tint: Theme.remoteRole,
                available: hasRemote)

            // Two different sentences, because they need two different actions.
            // "No Remote channel" is fixed in Settings and the note says where;
            // the Auto advice only makes sense once there IS a far end.
            Text(hasRemote
                 ? "Auto is safest for the far end — you can see the room, not the call."
                 : "No Remote channel — the far end is not being recorded. "
                   + "Pick one in Settings → Microphone.")
                .font(.system(size: 10))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(help)
    }

    /// `available` = there is a stream for this row to describe. False only for
    /// Remote with no channel selected: the row still shows, dimmed and inert,
    /// with the reason underneath.
    private func row(label: String, value: Binding<Int>, tint: Color,
                     available: Bool = true) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "person.2")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(available ? tint : Theme.textDim)
            Text(label)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(Theme.textDim)
                .frame(width: 52, alignment: .leading)

            // A `Menu`, NOT a `Picker(.menu)` — the reason is carried over from
            // the chip this replaced: on macOS the picker draws its own bevelled
            // pop-up button, a second background and border with its own padding,
            // so it sits heavier than everything around it. A borderless Menu is
            // the same native dropdown drawing nothing but text.
            Menu {
                Button("Auto") { value.wrappedValue = 0 }
                // 2 is the floor because one speaker is not diarization, and 20 is
                // NeMo's own `max_num_speakers` — the tightest real bound any
                // engine here declares, so no entry promises more than some engine
                // could deliver.
                ForEach(2...20, id: \.self) { n in
                    Button("\(n)") { value.wrappedValue = n }
                }
            } label: {
                HStack(spacing: 3) {
                    Text(text(value.wrappedValue))
                        .font(.system(size: 11, weight: .semibold).monospaced())
                        .foregroundColor(honoured && available
                                         ? Theme.textBody : Theme.textDim)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 7, weight: .bold))
                        .foregroundColor(Theme.textDim)
                }
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .disabled(locked || !available)

            Spacer(minLength: 0)
        }
        .opacity(locked || !available ? 0.5 : 1)
    }

    /// Padded on BOTH sides to the width of the longest entry so a row never
    /// changes width as the value changes. Done in TEXT, not layout: macOS `Menu`
    /// measures its own label and honours neither a `.frame` nor an overlaid
    /// template — two attempts proved that on the chip this replaced — while
    /// `.monospaced()` gives a space exactly a digit's advance.
    private func text(_ value: Int) -> String {
        let v = value == 0 ? "Auto" : "\(value)"
        guard v.count < 4 else { return v }
        let pad = 4 - v.count
        return String(repeating: " ", count: (pad + 1) / 2) + v
             + String(repeating: " ", count: pad / 2)
    }

    private var help: String {
        if locked {
            return "Fixed while a meeting is running — the stop passes read these."
        }
        if !honoured {
            return "This engine counts speakers on its own and ignores a set number."
        }
        return "How many people to find. Auto lets the engine decide — worth "
             + "setting a number for the spectral engine, which has been measured "
             + "counting 13 speakers on a 3-person recording."
    }
}
