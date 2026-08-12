import SwiftUI

/// Top-right status chips: speaker count, RMS meter, ATND, recording state.
struct StatusChipsView: View {
    @ObservedObject var recorder: AudioRecorder
    @ObservedObject private var beam = ATNDBeamService.shared

    /// Speakers to ask the diarizer for; 0 = auto. Shares the key
    /// `AudioRecorder.diarNumSpeakers` reads, so this control and the passes are
    /// the same value by construction rather than by two readers agreeing.
    @AppStorage("diarization.numSpeakers")   private var numSpeakers = 0
    @AppStorage("diarization.engine")        private var engine = "pyannote"
    @AppStorage("diarization.enabled")       private var diarOn = true
    @AppStorage("diarization.finalPass")     private var finalPass = true
    @AppStorage("diarization.continueOnStop") private var continueOnStop = false

    private var isRecording: Bool { recorder.state == .recording }

    /// Whether the number reaches the engine in THIS session. Asked of the
    /// loader's own rule, never restated here — a second copy is how the chip
    /// would come to promise something the pass does not do.
    ///
    /// It takes the two pyannote pass settings as well as the engine, because
    /// under pyannote a TAIL stop pass is a `chunk` job and carries no count. The
    /// chip lit for exactly that session until the 2026-08-10 audit.
    private var honoured: Bool {
        ModelLoader.speakerCountReachesEngine(diarEngine: engine,
                                              diarizationEnabled: diarOn,
                                              finalPass: finalPass,
                                              continueOnStop: continueOnStop)
    }

    var body: some View {
        HStack(spacing: 8) {
            speakerCountChip
            chip {
                Image(systemName: "waveform")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Theme.teal)
                label("RMS", dim: true)
                label(String(format: "%.4f", recorder.rms), color: Theme.textBody)
            }
            vadChip
            atndChip
            remoteChip
            chip {
                Circle()
                    .fill(isRecording ? Theme.red : Theme.teal)
                    .frame(width: 7, height: 7)
                label(isRecording ? "REC" : "IDLE", weight: .heavy, color: Theme.textBright)
            }
        }
    }

    /// How many speakers to tell the diarizer to find — Auto, or a fixed 2…20.
    ///
    /// IN THE MAIN WINDOW, not Settings (owner, 2026-08-10), because it is the one
    /// diarization value that changes per MEETING rather than per setup: you know
    /// before you press record how many people are in the room.
    ///
    /// WHY IT EARNS A CONTROL AT ALL, since three of the five engines count
    /// correctly on their own — measured auto vs pinned on the same files:
    /// **spectral returns 13 speakers on a 3-person clip and 20 on a 67-minute
    /// meeting**, and pinning fixes both. That one engine is the whole case.
    ///
    /// It stays VISIBLE under engines that ignore it rather than vanishing, and
    /// says so instead — the language-picker rule (2026-07-31): the stored choice
    /// survives an engine switch, and a control that disappears reads as a bug
    /// while a dimmed one with a reason reads as an answer. `honoured` drives only
    /// the colour and the tooltip; the value is written and forwarded either way,
    /// and dies in the sidecar that does not read it.
    private var speakerCountChip: some View {
        chip {
            // ALWAYS teal, like the RMS waveform beside it (owner, 2026-08-10).
            // It briefly tracked `honoured`, which left the icon grey next to its
            // neighbour and read as "this chip is broken" rather than "this engine
            // ignores the number". The icon is the chip's IDENTITY — RMS's is teal
            // whatever the level — so the inert state is carried by the VALUE text
            // and the tooltip instead, where it describes the thing that is
            // actually inert.
            Image(systemName: "person.2")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(Theme.teal)
            label("SPK", dim: true)
            // A `Menu`, NOT a `Picker(.menu)`. The picker was tried first and is
            // the obvious choice, but on macOS it draws its own bevelled pop-up
            // button INSIDE the chip: a second background, a second border and its
            // own internal padding, so this one chip sat wider and visually
            // heavier than its five neighbours (owner: "padding nya sama jadi
            // tertata rapih"). A borderless Menu with the chip's own `label` gives
            // the same dropdown — still native, still scrolling past ~20 items —
            // while drawing nothing but text.
            Menu {
                Button("Auto") { numSpeakers = 0 }
                // 2 is the floor because "one speaker" is not diarization, and 20
                // is NeMo's own `max_num_speakers` ceiling — the tightest real
                // bound any engine here declares, so no entry can promise more
                // than some engine could deliver.
                ForEach(2...20, id: \.self) { n in
                    Button("\(n)") { numSpeakers = n }
                }
            } label: {
                HStack(spacing: 3) {
                    // CONSTANT WIDTH BY CHARACTER COUNT, not by frames.
                    //
                    // The chip must not shrink from "Auto" to "2" (owner,
                    // 2026-08-10). Two layout attempts failed first — a `.frame`
                    // and then a hidden "Auto" template with the value overlaid —
                    // because macOS's `Menu` measures its own label and does not
                    // honour either. Rather than keep guessing at a control whose
                    // sizing I cannot inspect, this pads the STRING: `label` renders
                    // `.monospaced()`, where a space has exactly a digit's advance,
                    // so four characters are four characters wide whatever the font
                    // metrics are. "Auto" is the longest value this can show.
                    label(speakerCountText,
                          color: honoured ? Theme.textBody : Theme.textDim)
                    // Our own indicator: the native one sits after a gap sized for
                    // the bezel we just removed, which reopens the same misalignment.
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 7, weight: .bold))
                        .foregroundColor(Theme.textDim)
                }
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            // LOCKED FOR THE SESSION, like every control behind the gear (owner,
            // 2026-08-12). This one lives in the MAIN window, so shutting the
            // settings sheet left it reachable — and it is a real setting: the
            // stop passes read it, so a change made mid-meeting would decide the
            // speaker count of a recording already half over.
            //
            // Its own doc still stands: it is read LATE and has no second reader
            // during recording, so it was never INCOHERENT. What changed is the
            // rule — a meeting's settings are fixed once it starts.
            .disabled(recorder.state != .idle)
        }
        .help(speakerCountHelp)
    }

    /// The value as a FIXED-WIDTH, CENTRED string: `Auto`, ` 2  `, ` 20 `.
    ///
    /// Padded to the width of the longest entry so the chip never changes size —
    /// see the call site for why this is done in text rather than in layout — and
    /// padded on BOTH sides so a short value sits in the middle of the slot rather
    /// than hugging the label (owner, 2026-08-10).
    private var speakerCountText: String {
        let value = numSpeakers == 0 ? "Auto" : "\(numSpeakers)"
        let slack = max(0, 4 - value.count)
        // NON-BREAKING spaces, not plain ones: an ordinary space at either end is a
        // legitimate line-break opportunity, and text layout is free to discard it
        // when measuring — which would silently undo the whole point of padding
        // here. U+00A0 carries the same advance in a monospaced font and is never
        // collapsed.
        let pad = String(repeating: "\u{00A0}", count: slack / 2)
        // The odd character goes on the RIGHT, which is where a reader's eye is
        // least likely to notice it: the value stays aligned with the chevron.
        let tail = String(repeating: "\u{00A0}", count: slack - slack / 2)
        return pad + value + tail
    }

    private var speakerCountHelp: String {
        // The lock comes FIRST: while a meeting is running it is the only reason
        // that matters, and the engine/pass explanations below would send the
        // user to a Settings page they cannot open yet.
        guard recorder.state == .idle else {
            return "Locked for this meeting — set the speaker count before recording starts."
        }
        guard diarOn else {
            return "Speaker diarization is switched off, so nothing is counted."
        }
        // TWO REASONS THE NUMBER CAN BE INERT, and they are fixed in different
        // places — one by switching engine, one by changing a Diarization setting.
        // A tooltip that named the wrong one would send the user to the wrong page,
        // which is the same rule the rail's inapplicability banners follow.
        guard ModelLoader.honoursSpeakerCount(diarEngine: engine) else {
            return "The \(ModelCatalog.diarizationEngine(forEngine: engine).name) engine "
                 + "always counts the speakers itself and ignores this. Your choice is "
                 + "kept for when you switch engines."
        }
        guard honoured else {
            return finalPass
                ? "Not used with \"Continue from live labels (tail only)\" switched on: "
                  + "at Stop only the last few seconds are diarized, and a short tail "
                  + "need not contain everyone, so no count is sent. Switch that off in "
                  + "Settings → Models → Diarization."
                : "Not used while the diarization pass at Stop is switched off — the "
                  + "count is only sent on that pass. Switch it on in "
                  + "Settings → Models → Diarization."
        }
        return numSpeakers == 0
            ? "Auto — the engine decides. Worth setting a number for the spectral "
              + "engine, which has been measured counting 13 speakers on a 3-person "
              + "recording; pyannote and NeMo count correctly on their own."
            // NAMES THE ROOM, not "the recording". With a Remote channel there are
            // two recordings, and this number describes only the one you can see —
            // `AudioRecorder.remoteNumSpeakers` keeps the far end automatic, so
            // saying "the recording" here would claim a reach the passes do not
            // have. True of a single-stream session too: the room IS the recording.
            : "The diarizer is told to find exactly \(numSpeakers) speakers in the "
              + "room; remote participants are always counted automatically. Set it "
              + "back to Auto if you are unsure — too low a number merges two people "
              + "into one."
    }

    /// Live beam data from the ATND1061's multicast stream: which beam channel
    /// is speaking, and where it points. Independent of recording.
    private var atndChip: some View {
        chip {
            Image(systemName: "dot.radiowaves.left.and.right")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(beam.talker != nil ? Theme.teal : Theme.textDim)

            // Channel reads bright, its bearing sits back — the beam is the
            // fact, the angles are the detail.
            if let t = beam.talker, case .listening = beam.state {
                label("CH\(t.channel)", weight: .heavy, color: Theme.teal)
                label(String(format: "%2d° %3d°", t.elevation, t.rotation),
                      color: Theme.textSubtle)
            } else {
                label("ATND", dim: true)
                label("–", dim: true)
            }
        }
        .frame(minWidth: stableWidth(104), alignment: .leading)
        .animation(.easeOut(duration: 0.15), value: beam.talker)
        .help(atndHelp)
    }

    /// The chip is too small to show a failure inline — surface it on hover.
    private var atndHelp: String {
        switch beam.state {
        case .failed(let message):  return message
        case .listening:            return "Listening for ATND1061 beam notices."
        case .waitingForControl:    return "Waiting for the TCP connection — the array only sends beam data once connected. Connect in Settings → ATND."
        case .off:                  return "ATND1061 is off — enable it in Settings → ATND."
        }
    }

    /// How many speakers the REMOTE (conferencing) stream has produced — its own
    /// identity space, so the number is unrelated to the room's speakers.
    ///
    /// Present only for a dual-stream session: `remoteStreamActive` is false for
    /// the entire life of a single-stream app, so this `if` renders nothing and
    /// the chip row is byte-for-byte the pre-dual-stream one. Amber, because
    /// amber is the Remote capture role everywhere in the app.
    @ViewBuilder
    private var remoteChip: some View {
        if recorder.remoteStreamActive {
            chip {
                Image(systemName: "phone.and.waveform")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Theme.remoteRole)
                label("REMOTE", dim: true)
                // "–" until the first remote diarization result lands, matching
                // how the ATND chip shows its own not-yet-known state.
                if let count = recorder.remoteSpeakerCount {
                    label("\(count)", weight: .heavy, color: Theme.remoteRole)
                } else {
                    label("–", dim: true)
                }
            }
            .frame(minWidth: stableWidth(86), alignment: .leading)
            .help("Speakers identified on the Remote (conferencing) stream. "
                  + "Remote speakers are a separate identity space from the room's.")
        }
    }

    /// VAD status: TALKING (voice detected) / SILENT / – when idle or VAD off.
    private var vadChip: some View {
        chip {
            if !isRecording {
                Image(systemName: "person.wave.2")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Theme.textDim)
                label("VAD", dim: true)
                label("–", dim: true)
            } else if !recorder.vadEnabled {
                Image(systemName: "person.wave.2")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Theme.textDim)
                label("VAD", dim: true)
                label("OFF", dim: true)
            } else {
                VoiceBars(level: recorder.rms, active: recorder.isSpeaking)
                label(recorder.isSpeaking ? "TALKING" : "SILENT",
                      weight: .heavy,
                      color: recorder.isSpeaking ? Theme.teal : Theme.textSubtle)
            }
        }
        .frame(minWidth: stableWidth(96), alignment: .leading)
        .animation(.easeOut(duration: 0.15), value: recorder.isSpeaking)
    }

    /// Chip text. Every label refuses to wrap or shrink — without this the
    /// row silently breaks "IDLE" into "IDL/E" once the window gets narrow.
    private func label(_ text: String,
                       weight: Font.Weight = .bold,
                       dim: Bool = false,
                       color: Color? = nil) -> some View {
        Text(text)
            .font(.system(size: 11, weight: weight).monospaced())
            .foregroundColor(color ?? (dim ? Theme.textDim : Theme.textBody))
            .lineLimit(1)
            .fixedSize(horizontal: true, vertical: false)
    }

    private func chip<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 6) { content() }
            .padding(.horizontal, 12)
            .frame(height: 30)
            .background(RoundedRectangle(cornerRadius: 8).fill(Theme.chip))
    }

    /// Reserve a minimum width, but ONLY while recording.
    ///
    /// Three chips (VAD, ATND, REMOTE) pinned a `minWidth` unconditionally so the
    /// row would not jitter as their text changes — `SILENT`→`TALKING`, a bearing
    /// appearing, a remote count arriving. That is a real need and the widths are
    /// sized for the longest string each can show.
    ///
    /// The cost was visible at rest, which is most of the time: the frame is
    /// applied AFTER `chip()`, so the rounded background stretches while the
    /// content stays `.leading` — dead space inside the right of three chips and
    /// not the other three, which is exactly the uneven padding the owner reported
    /// (2026-08-10). Idle, every chip now hugs its content and the row is uniform;
    /// recording, the reservations come back and nothing moves. Jitter can only
    /// happen while the values are changing, so this buys the stability where it
    /// exists and pays nothing where it does not.
    private func stableWidth(_ width: CGFloat) -> CGFloat? {
        isRecording ? width : nil
    }
}

/// Mini equalizer: 3 bars that rise and fall with the live voice level.
/// Teal + moving while talking; dim + flat when silent.
private struct VoiceBars: View {
    let level: Float   // live RMS
    let active: Bool   // VAD says speech

    /// Per-bar sensitivity so the bars move independently.
    private let gains: [Float] = [22, 40, 30]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<3, id: \.self) { i in
                Capsule()
                    .fill(active ? Theme.teal : Theme.textDim)
                    .frame(width: 2.5, height: height(i))
                    .animation(.easeOut(duration: 0.12), value: level)
            }
        }
        .frame(height: 12, alignment: .center)
    }

    private func height(_ bar: Int) -> CGFloat {
        guard active else { return 3 }
        let h = CGFloat(min(level * gains[bar], 1.0)) * 12
        return max(3, h)
    }
}
