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
/// gently — it merges people who spoke or invents people who did not.
///
/// Both rows therefore default to Auto. The line that SAID so under the Remote
/// row was removed on the owner's instruction (2026-08-13); the caution now lives
/// in the tooltip and here, not on screen. Recorded rather than quietly dropped,
/// because the risk did not go away with the sentence.
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
                                              finalPass: finalPass)
    }

    /// Locked for the whole session, like every control behind the gear
    /// (2026-08-12). The stop passes read both values, so a change made
    /// mid-meeting would decide the speaker count of a recording already half
    /// over.
    private var locked: Bool { recorder.state != .idle }

    var body: some View {
        // CENTRED under the record button, which is itself centred (owner,
        // 2026-08-13). The BLOCK centres; the rows inside stay a fixed width and
        // left-align within it, so both pickers sit on one vertical line. Centring
        // each row independently would let "Office" and "Remote" — different
        // string widths — push their pickers to different places, which reads as
        // sloppier than not centring at all.
        VStack(alignment: .leading, spacing: 14) {
            Text("SPEAKERS")
                .font(.system(size: 15, weight: .bold)).kerning(1.1)
                .foregroundColor(Theme.textFaint)
                .frame(maxWidth: .infinity, alignment: .center)

            row(label: "Office", value: $office, tint: Theme.teal)
            row(label: "Remote", value: $remote, tint: Theme.remoteRole,
                available: hasRemote)

            // THE NUMBER WILL NOT REACH THE ENGINE — say so on screen, not only in
            // a tooltip (owner, 2026-08-14). Both pickers stay usable and the
            // stored value survives, which is the 2026-07-31 language-picker rule:
            // a control that vanishes reads as a bug, one that says why it is inert
            // reads as an answer. `inertReason` is the ONE source; the tooltip
            // prints the same string.
            //
            // NOT AMBER. The owner removed an amber block from the Diarization tab
            // on sight the day before, and it was a different thing wearing the
            // same colour — a measurement they had already been shown twice. This
            // is a STATE the user asked to be told about, so it is drawn like the
            // "No Remote channel" line beside it: the same faint text, in the same
            // place, saying the same kind of thing.
            //
            // Deliberately NOT hidden while `locked`: a running meeting shows the
            // lock tooltip, but hiding the reason mid-session would make the
            // control's inertness look like the lock, which is a different and
            // temporary cause.
            if let reason = inertReason {
                Text(reason)
                    .font(.system(size: 13))
                    .foregroundColor(Theme.textFaint)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .center)
            }

            // The "Auto is safest for the far end" line was removed on the
            // owner's instruction. What is left is not advice but a STATE — there
            // is no second recording — and it names the one place that changes it,
            // so it stays.
            if !hasRemote {
                Text("No Remote channel — pick one in Settings → Microphone.")
                    .font(.system(size: 13))
                    .foregroundColor(Theme.textFaint)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
        // A CARD, like the record card above it (owner, 2026-08-13). Same fill and
        // corner radius, plus a hairline border — the record card has no stroke
        // because its own size carries it; this block is short enough that without
        // one it read as text floating on the sidebar rather than as a control.
        .padding(.vertical, 16)
        .padding(.horizontal, 18)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Theme.card)
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Theme.overlayBorder, lineWidth: 1)
                )
        )
        .padding(.horizontal, 16)
        .help(help)
    }

    /// `available` = there is a stream for this row to describe. False only for
    /// Remote with no channel selected: the row still shows, dimmed and inert,
    /// with the reason underneath.
    private func row(label: String, value: Binding<Int>, tint: Color,
                     available: Bool = true) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "person.2")
                .font(.system(size: 17, weight: .bold))
                .foregroundColor(available ? tint : Theme.textDim)
            Text(label)
                .font(.system(size: 17, weight: .semibold))
                .foregroundColor(Theme.textBody)
                // Wide enough for "Office" and "Remote" at 17pt, so the two
                // pickers line up under each other.
                .frame(width: 80, alignment: .leading)

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
                HStack(spacing: 5) {
                    Text(text(value.wrappedValue))
                        .font(.system(size: 17, weight: .bold).monospaced())
                        .foregroundColor(honoured && available
                                         ? Theme.textBright : Theme.textDim)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 11, weight: .bold))
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
        if let reason = inertReason { return reason }
        // NAMES THE MEASUREMENT, NOT ONE ENGINE. It said "worth setting a number
        // for the spectral engine, which has been measured counting 13 speakers on
        // a 3-person recording" — true, and by 2026-08-14 measurably incomplete:
        // on a 30 s window of Meeting5People.wav (5 real people) CAM++ returned 11
        // speakers and spectral 12, and NeMo has been measured at 12 on a 2-person
        // clip. Naming one engine reads as "the others are fine".
        return "How many people to find. Auto lets each engine decide, and on short "
             + "windows three of them have been measured inventing speakers — set a "
             + "number whenever you know it."
    }

    /// WHY the number will not reach the engine, or nil when it will.
    ///
    /// ONE SOURCE for the tooltip AND the on-screen warning (owner, 2026-08-14:
    /// *"buat peringatan kalau model itu tidak bisa set speaker"*). Writing the
    /// warning as its own string was the obvious move and is the mistake this
    /// project keeps paying for — two expressions of one fact drift, and the
    /// drift is always toward under-reporting (the three hand-written engine
    /// lists, all three stale, found in one sweep on 2026-08-13).
    ///
    /// THREE REASONS, and until 2026-08-14 all three printed the first one. Under
    /// pyannote with the tail on, "this engine counts on its own" is simply FALSE
    /// — pyannote reads the number perfectly well, on a full stop pass — and it is
    /// also unactionable, which is worse: the owner hit exactly this and had no way
    /// to learn that one toggle two screens away was the cause. The rule is
    /// `ModelLoader.speakerCountReachesEngine`, so the split here stays keyed on
    /// the same facts it is, never re-derived.
    private var inertReason: String? {
        guard !honoured else { return nil }
        guard diarOn else {
            return "Speaker diarization is off, so nobody is being counted. "
                 + "Settings → Diarization."
        }
        // The engine CAN read the number; this session's own pass throws it away.
        // Both legs name the control, because a reason the user cannot act on is
        // the failure this whole split exists to fix.
        if ModelLoader.honoursSpeakerCount(diarEngine: engine) {
            // ONE leg now, not two. The tail case disappeared on 2026-08-14 when a
            // stop pass became unconditionally a full pass; keeping a branch for a
            // state that can no longer occur is how a stale reason survives.
            return "Not sent while the stop pass is off — live windows and the "
                 + "tail need not contain everyone. "
                 + "Settings → Diarization → Run a diarization pass at stop."
        }
        // MOSS, and only MOSS. Deliberately says what it CANNOT do rather than
        // naming a setting: there is no control anywhere that would change this,
        // so pointing at one would be worse than pointing at nothing. See
        // `ModelLoader.honoursSpeakerCount` for why it is permanent — the labels
        // come out of a language model as text tags, with no clustering stage to
        // bound and no count parameter in the checkpoint.
        let name = ModelCatalog.diarizationEngineShortName(engine) ?? "This engine"
        return "\(name) writes speaker labels as it transcribes and has no speaker "
             + "count to set. Pick another engine if you need to fix the number."
    }
}
