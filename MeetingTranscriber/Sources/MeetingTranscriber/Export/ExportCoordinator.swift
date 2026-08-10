import AppKit
import Foundation

/// Runs the `Export to PDF` flow and decides WHERE the file lands (owner,
/// 2026-08-10). `utils.exportPdfToThumbdrive` is read HERE and nowhere else.
///
/// ONE READER, on purpose. `diarization.finalPass` had two that disagreed, and
/// that is what silently deleted every remote label for five days until the
/// 2026-08-10 audit. A destination setting read in a second place would fail the
/// same way: the button would report success while the file went elsewhere.
///
/// The flow:
///
///     OFF → NSSavePanel, the user picks.
///     ON  → 1 drive   : straight there.
///           >1 drives : a picker of REMOVABLE volumes only.
///           0 drives  : "Please insert a thumbdrive" with Try Again / Cancel,
///                       looping until one is chosen.
///
/// Everything here is `runModal()`, which is APPLICATION-modal. That is the
/// owner's requirement stated precisely: while the insert-a-drive alert is up the
/// app is not interactive — no Settings, no Start Over, no second Export — and
/// `Try Again` may be pressed indefinitely.
@MainActor
enum ExportCoordinator {

    enum Outcome {
        case written(URL)
        case cancelled
        case failed(String)
    }

    /// ONE reporter, at the ONE exit. Every outcome is announced exactly once
    /// because there is exactly one place that can announce it — not because each
    /// `guard` remembered to. An earlier shape had three returns and two reporting
    /// rules, and one of them (a cancelled thumbdrive) skipped reporting entirely,
    /// correct only because `report` happens to ignore `.cancelled`.
    @discardableResult
    static func export(rows: [AudioRecorder.SpeakerUtterance],
                       recordedAt: Date) -> Outcome {
        let outcome = chooseDestinationAndWrite(rows: rows, recordedAt: recordedAt)
        report(outcome)
        return outcome
    }

    /// DESTINATION FIRST, RENDER SECOND — and that order is the fix, not a style
    /// choice. Rendering used to happen before anything was asked, so every
    /// cancelled export threw the whole document away: measured 30 ms at 100 rows,
    /// **122 ms at 800**, all of it a synchronous MainActor stall between the click
    /// and the first dialog appearing. Asking first puts the dialog up immediately
    /// and pays the render only once a destination is committed.
    ///
    /// The render still sits OUTSIDE the retry loop's cost: `resolveThumbdrive`
    /// returns a URL before any bytes exist, so N presses of Try Again render zero
    /// times, not N.
    private static func chooseDestinationAndWrite(rows: [AudioRecorder.SpeakerUtterance],
                                                  recordedAt: Date) -> Outcome {
        guard !rows.isEmpty else {
            // Unreachable through the button, which is disabled with no rows.
            // Kept so the rule survives a future caller that forgets.
            return .failed("There is no transcript to export.")
        }
        let filename = TranscriptPDF.suggestedFilename(recordedAt: recordedAt)

        let destination: URL
        if UserDefaults.standard.bool(forKey: "utils.exportPdfToThumbdrive") {
            guard let drive = resolveThumbdrive() else { return .cancelled }
            destination = drive.url.appendingPathComponent(filename)
        } else {
            guard let chosen = askWhereToSave(filename: filename) else { return .cancelled }
            destination = chosen
        }

        guard let data = TranscriptPDF.render(rows: rows,
                                              title: "Meeting Transcript",
                                              recordedAt: recordedAt) else {
            return .failed("The PDF could not be created.")
        }
        return write(data, to: destination)
    }

    // MARK: - Telling the user

    /// A modal alert on success AND on failure (owner, 2026-08-10): the client
    /// must know the PDF really landed, not infer it from a line of small text.
    ///
    /// CANCEL IS SILENT, and that is the one case with no popup. The user closed
    /// the dialog themselves a moment ago; an alert confirming their own decision
    /// is the app talking back rather than informing.
    ///
    /// Lives here rather than in the view because this type already owns every
    /// other dialog in the flow. Split across two places, the insert-a-drive alert
    /// and the result alert could disagree about whether an export was "done".
    private static func report(_ outcome: Outcome) {
        let alert = NSAlert()
        switch outcome {
        case .written(let url):
            alert.alertStyle = .informational
            alert.messageText = "PDF saved"
            // The FULL path, because with the thumbdrive switch on the user never
            // chose a location — "saved successfully" would leave them hunting for
            // a file they were never told the destination of.
            alert.informativeText = url.path
        case .failed(let message):
            alert.alertStyle = .warning
            alert.messageText = "Could not save the PDF"
            alert.informativeText = message
        case .cancelled:
            return
        }
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    // MARK: - Thumbdrive

    /// Loops until the user has a drive or gives up. Returns nil ONLY on Cancel.
    ///
    /// Only `.none` goes round again, and the flat shape is what makes that
    /// visible: the two cases that can answer the question return, and the one
    /// that cannot falls off the end of the switch into the next pass.
    private static func resolveThumbdrive() -> ThumbdriveLocator.Drive? {
        while true {
            // Re-scanned every pass — the whole point of Try Again is that the user
            // plugs something in between presses, so a list captured before the
            // loop would never see it.
            switch ThumbdriveLocator.destination(for: ThumbdriveLocator.connected()) {
            case .single(let drive): return drive
            case .choose(let drives): return askWhichDrive(drives)
            case .none: guard askToInsertDrive() else { return nil }
            }
        }
    }

    /// True = Try Again, false = Cancel.
    private static func askToInsertDrive() -> Bool {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Please insert a thumbdrive"
        alert.informativeText = "Export to PDF is set to save to a thumbdrive, but "
            + "none is connected. Insert one and choose Try Again, or Cancel to stop."
        alert.addButton(withTitle: "Try Again")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
    }

    /// A popup of removable volumes. Nil = Cancel.
    private static func askWhichDrive(_ drives: [ThumbdriveLocator.Drive])
        -> ThumbdriveLocator.Drive? {
        let alert = NSAlert()
        alert.messageText = "Choose a thumbdrive"
        alert.informativeText = "More than one is connected."
        let popup = NSPopUpButton(frame: NSRect(x: 0, y: 0, width: 260, height: 25))
        popup.addItems(withTitles: drives.map(\.name))
        alert.accessoryView = popup
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return nil }
        // Index, not name: two drives can legitimately share a name, and matching
        // by title would then write to whichever happened to be first.
        let index = popup.indexOfSelectedItem
        guard drives.indices.contains(index) else { return nil }
        return drives[index]
    }

    // MARK: - Dialog

    /// Asks WHERE, and returns nothing but the URL — no bytes are written or even
    /// built at this point. Nil = Cancel.
    ///
    /// `directoryURL` is set so the dialog opens somewhere predictable rather than
    /// wherever the app last saved anything. It also makes `.gitignore`'s
    /// `PDFExport/` entry describe a real default instead of a folder name nobody
    /// is steered towards — these documents are real client speech, and the repo
    /// is the one place they must never reach.
    private static func askWhereToSave(filename: String) -> URL? {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = filename
        panel.allowedContentTypes = [.pdf]
        panel.canCreateDirectories = true
        panel.directoryURL = defaultExportDirectory()
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }

    /// `~/Documents/Meeting Transcripts`, created on demand. Falls back to letting
    /// the panel choose if it cannot be made — a missing folder must not stop an
    /// export.
    private static func defaultExportDirectory() -> URL? {
        guard let docs = FileManager.default.urls(for: .documentDirectory,
                                                  in: .userDomainMask).first else { return nil }
        let dir = docs.appendingPathComponent("Meeting Transcripts")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return FileManager.default.fileExists(atPath: dir.path) ? dir : nil
    }

    // MARK: - Write

    /// Reports the REAL error rather than a generic failure. On a thumbdrive the
    /// likely causes are specific and actionable — the drive is full, or it is
    /// mounted read-only, or it was pulled out between the picker and the write.
    private static func write(_ data: Data, to url: URL) -> Outcome {
        do {
            try data.write(to: url, options: .atomic)
            return .written(url)
        } catch {
            return .failed(error.localizedDescription)
        }
    }
}
