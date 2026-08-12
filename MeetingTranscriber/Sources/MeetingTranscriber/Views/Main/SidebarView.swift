import SwiftUI

/// Left panel: app title, settings button, record card.
struct SidebarView: View {
    @ObservedObject var recorder: AudioRecorder
    let openSettings: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Spacer()
            RecordCardView(recorder: recorder)
            Spacer()
            Spacer()
            // Under the record card, at the very bottom (owner, 2026-08-10).
            // AFTER the second Spacer rather than between them, so it is pinned to
            // the floor instead of floating in whatever space is left — and so the
            // mic card keeps the exact vertical position it has always had when no
            // meeting is finished, which is every launch.
            MeetingActionsView(recorder: recorder)
        }
        .frame(width: 350)
        .background(Theme.sidebar)
    }

    /// Settings are LOCKED for the whole session — not just while the tape is
    /// rolling (owner, 2026-08-12).
    ///
    /// `.processing` is included deliberately: the stop passes are still reading
    /// settings then (`diarization.finalPass`, `diarization.numSpeakers`, the
    /// chunked stop plan), so a change made during the overlay would land
    /// mid-meeting on a transcript already half-decided. `.preparing` is the
    /// model load itself.
    ///
    /// This is the STRUCTURAL half of a rule the code previously had to enforce
    /// by hand: `lockDiarizationSettings` exists because three keys were read
    /// per chunk AND again at Stop, and Settings being reachable mid-recording
    /// let the two reads disagree about one meeting. That lock stays — it is
    /// still what guarantees coherence — but the door is now shut as well.
    ///
    /// A finished meeting (`.idle` with `meetingFinished`) is deliberately NOT
    /// locked: nothing is reading settings any more, and a change there applies
    /// to the next session, which is when it can be honoured coherently.
    private var settingsLocked: Bool { recorder.state != .idle }

    private var lockedReason: String {
        switch recorder.state {
        case .preparing:  return "Models are loading — settings are locked until the session starts."
        case .recording:  return "Settings are locked while recording. Stop the meeting to change them."
        case .processing: return "The stop passes are still running and are reading these settings. "
                               + "They unlock when processing finishes."
        case .idle:       return ""
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Button(action: openSettings) {
                Image(systemName: settingsLocked ? "lock.fill" : "gearshape.fill")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(settingsLocked ? Theme.textFaint : Theme.textPrimary)
                    .frame(width: 36, height: 36)
                    .background(Circle().fill(Theme.logoBackground))
            }
            .buttonStyle(.plain)
            .disabled(settingsLocked)
            // A control that cannot act must SAY why — the rule this project
            // applies to every inert control. A greyed gear with no explanation
            // reads as a bug; one that names the reason reads as an answer.
            .help(settingsLocked ? lockedReason : "Settings")

            Text("MEETING TRANSCRIBER")
                .font(.system(size: 17, weight: .heavy))
                .kerning(0.5)
                .foregroundColor(Theme.textPrimary)

            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.top, 28)
    }
}
