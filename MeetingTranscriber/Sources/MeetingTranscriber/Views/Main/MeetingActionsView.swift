import SwiftUI

/// The two actions that only exist once a meeting is FINISHED: export it, or
/// clear it and record another. Bottom of the sidebar, under the record card
/// (owner, 2026-08-10).
///
/// The flow this belongs to, which is the owner's and is deliberately one-way:
///
///     record → stop → passes finish → mic LOCKS, these two appear
///                                   → Start Over clears the panel
///                                   → these hide, mic unlocks
///
/// So the app cannot be recording over a transcript nobody has dealt with. The
/// mic's lock is `AudioRecorder.toggle`'s guard, not just the button's
/// `.disabled` — a view is not a rule.
///
/// BOTH ARE WIRED. Export builds the PDF, chooses its destination from
/// `utils.exportPdfToThumbdrive`, and reports the result in a modal alert — all
/// of that in `ExportCoordinator`, none of it here.
///
/// This header claimed export was "not wired yet" for a few hours after export
/// landed. Nothing caught it; the owner did, by asking. Comments describing
/// whether something works go stale the moment it starts working, and a stale one
/// misleads exactly the reader who trusted it.
struct MeetingActionsView: View {
    @ObservedObject var recorder: AudioRecorder

    var body: some View {
        // Absent, not dimmed, when no meeting is finished. The Settings rail dims
        // tabs because they still name a real choice; these name an object that
        // does not exist yet, and a permanently grey pair at every launch reads as
        // a broken app.
        if recorder.meetingFinished {
            VStack(spacing: 10) {
                exportButton
                startOverButton
                caption
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 20)
        }
    }

    // MARK: - Export

    /// Disabled with nothing to export (owner, 2026-08-10). The button still
    /// APPEARS, because a finished meeting that transcribed nothing is exactly
    /// when the user needs to see that the app noticed — vanishing would leave
    /// them guessing whether the meeting registered at all.
    private var exportButton: some View {
        actionButton(title: "EXPORT TO PDF",
                     icon: "arrow.down.doc.fill",
                     tint: Theme.teal,
                     filled: true,
                     enabled: recorder.canExport,
                     action: runExport)
    }

    /// WHERE the file goes, and WHAT the user is told afterwards, are both
    /// `ExportCoordinator`'s — the destination setting has exactly one reader and
    /// this is not it, and the success/failure popups live beside the flow's other
    /// dialogs so they cannot disagree about whether an export finished.
    ///
    /// The outcome is therefore DISCARDED here on purpose. This view showed a small
    /// caption until 2026-08-10; the owner asked for a popup so a client can see
    /// plainly that the PDF landed, and a caption repeating an alert the user has
    /// just dismissed is noise in the corner of the window.
    ///
    /// Two model reads and a call. The meeting's date is `recorder`'s to know
    /// (`meetingRecordedAt`), not something a button action derives from the
    /// filesystem.
    private func runExport() {
        ExportCoordinator.export(rows: recorder.displayRows,
                                 recordedAt: recorder.meetingRecordedAt)
    }

    // MARK: - Start Over

    /// Always available on a finished meeting — including an empty one, which is
    /// the case that most needs it: it is the ONLY way back to a usable mic.
    private var startOverButton: some View {
        actionButton(title: "START OVER",
                     icon: "arrow.counterclockwise",
                     tint: Theme.textSecondary,
                     filled: false,
                     enabled: true,
                     action: { recorder.startOver() })
    }

    // MARK: - Caption

    /// The ONE thing still worth saying inline. It is not an OUTCOME — those are
    /// popups now — it is a STATE: why the button above cannot be pressed. An
    /// alert would be wrong for it, because the user has not done anything yet and
    /// an unprompted dialog on a fresh screen is the app interrupting itself.
    @ViewBuilder
    private var caption: some View {
        if !recorder.canExport {
            Text("No text was transcribed, so there is nothing to export. "
                 + "Start Over to record again.")
                .font(.system(size: 10))
                .foregroundColor(Theme.textFaint)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Shared button shape

    /// The filled/outline pair `EnrollmentPromptView` already uses, with ITS
    /// constants rather than new ones.
    ///
    /// There is no shared styled-button type in this app — four views each roll
    /// their own — so the honest thing for a fifth is to join the existing
    /// convention instead of starting another. Written first with its own radius
    /// (9, a fourth value beside 7/8/12), its own disabled opacity (0.4, a third
    /// beside 0.35/0.5) and a hardcoded `Color.black.opacity(0.8)` for the
    /// label-on-teal that `Theme.selectedTabText` already names. Every one of those
    /// was a private divergence with nothing relating it to the others.
    private static let cornerRadius: CGFloat = 7
    private static let disabledOpacity: Double = 0.5

    private func actionButton(title: String, icon: String, tint: Color,
                              filled: Bool, enabled: Bool,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .semibold))
                Text(title)
                    .font(.system(size: 11, weight: .heavy))
                    .kerning(1.0)
            }
            .foregroundColor(filled ? Theme.selectedTabText : tint)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 11)
            .background(
                RoundedRectangle(cornerRadius: Self.cornerRadius)
                    .fill(filled ? tint : Theme.chip)
            )
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : Self.disabledOpacity)
    }
}
