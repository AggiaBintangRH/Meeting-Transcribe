import SwiftUI

/// Settings → Utils: things that happen AROUND a meeting rather than to its
/// audio. First occupant is where the PDF export goes (owner, 2026-08-10).
///
/// THE AGREED BEHAVIOUR (owner, 2026-08-10):
///
///   OFF — `Export to PDF` opens a save dialog; the user picks the location.
///   ON  — it writes straight to a connected thumbdrive, no dialog. Then:
///         • exactly one thumbdrive  → straight there
///         • more than one           → a picker listing ONLY removable drives.
///           Internal and network volumes are excluded, not listed-and-ignored.
///         • none                    → a blocking alert, in English, asking the
///           user to insert one, with `Try Again` and `Cancel`. Try Again can be
///           pressed indefinitely. Until one of the two is chosen the app is not
///           interactive — no Settings, no Start Over, no second Export.
///
/// So the switch decides WHERE, never WHETHER. Both positions export.
///
/// WIRED. `utils.exportPdfToThumbdrive` is read by `ExportCoordinator` at the
/// moment of export, and by nothing else — a second reader is how
/// `diarization.finalPass` came to delete remote labels for five days
/// (2026-08-10 audit). If a second one is ever added, this switch stops being a
/// destination and becomes a bug.
///
/// This tab shipped one screen ahead of its reader, for a few hours, carrying an
/// amber "not connected yet" note. That note was removed the moment the reader
/// landed — and it was NOT removed automatically, it was caught by the owner
/// asking whether the two were connected. A control claiming to be dead while it
/// works is the same defect as one claiming to work while dead: both make the UI
/// an unreliable description of the app.
struct UtilsTab: View {
    @AppStorage("utils.exportPdfToThumbdrive") private var exportToThumbdrive = false

    var body: some View {
        Group {
            SettingBlock(title: "PDF export") {
                SettingToggle(label: "Export PDF pass to thumbdrive",
                              isOn: $exportToThumbdrive)
                // The switch decides WHERE, not WHETHER (owner, 2026-08-10). Both
                // positions export; they differ in whether the user is asked. Said
                // as two explicit halves rather than "save to USB", because a
                // one-sided description leaves the off position undefined and the
                // off position is the one that opens a dialog.
                Text("On: the PDF goes straight to the connected USB drive.\n"
                     + "Off: a save dialog asks where to put it.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("With no thumbdrive connected, exporting asks you to insert "
                     + "one and waits.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 2)
            }
        }
    }
}
