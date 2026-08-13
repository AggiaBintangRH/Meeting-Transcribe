import SwiftUI

/// Top-right status chips: RMS meter, VAD, ATND, remote speakers, recording
/// state. All READ-ONLY — the one writable control that used to live here (the
/// speaker count) moved under the mic on 2026-08-13, which is what makes this a
/// row of readings rather than a mix the eye cannot tell apart.
struct StatusChipsView: View {
    @ObservedObject var recorder: AudioRecorder
    @ObservedObject private var beam = ATNDBeamService.shared

    // NO diarization settings are read here any more. The speaker-count control
    // moved under the mic on 2026-08-13 (`SpeakerCountView`), and its five
    // `@AppStorage` keys went WITH it rather than being left behind unused —
    // a second reader of a settings key is how two parts of the UI come to
    // disagree about one meeting, which is the failure this codebase has paid
    // for more than once.
    private var isRecording: Bool { recorder.state == .recording }

    var body: some View {
        HStack(spacing: 8) {
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
