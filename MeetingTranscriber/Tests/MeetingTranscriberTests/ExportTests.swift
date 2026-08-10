import XCTest
import PDFKit
@testable import MeetingTranscriber

/// The PDF export and its destination rules (owner, 2026-08-10).
@MainActor
final class ExportTests: XCTestCase {

    private func row(_ speaker: String, _ start: Double?, _ end: Double?,
                     _ text: String) -> AudioRecorder.SpeakerUtterance {
        AudioRecorder.SpeakerUtterance(id: "\(speaker)-\(start ?? -1)", speaker: speaker,
                                       start: start, end: end, text: text, confirmed: true)
    }

    // MARK: - Destination

    /// The three cases the flow branches on, as a pure function over the drive
    /// list — so "what happens with none plugged in" is provable without mounting
    /// anything. That case is the one that blocks the whole UI.
    func testDestinationDependsOnHowManyDrivesAreConnected() {
        let a = ThumbdriveLocator.Drive(url: URL(fileURLWithPath: "/Volumes/A"), name: "A")
        let b = ThumbdriveLocator.Drive(url: URL(fileURLWithPath: "/Volumes/B"), name: "B")

        XCTAssertEqual(ThumbdriveLocator.destination(for: []), .none,
                       "nothing plugged in → the blocking insert-a-drive alert")
        XCTAssertEqual(ThumbdriveLocator.destination(for: [a]), .single(a),
                       "exactly one → straight there, no question asked")
        XCTAssertEqual(ThumbdriveLocator.destination(for: [a, b]), .choose([a, b]),
                       "more than one → the picker")
    }

    /// The real scan must never offer an internal disk. Asserted as a PROPERTY of
    /// whatever is mounted on the test machine rather than against a fixed list,
    /// since a build machine's volumes are not knowable here: the boot volume is
    /// always mounted and must never appear.
    func testTheScanNeverOffersTheBootVolume() {
        let drives = ThumbdriveLocator.connected()
        XCTAssertFalse(drives.contains { $0.url.path == "/" },
                       "the boot volume is not removable and must never be an "
                       + "export destination")
    }

    // MARK: - The document

    /// A real PDF, parsed back with PDFKit rather than trusting the byte count.
    func testRenderProducesAReadablePdfContainingTheTranscript() throws {
        let rows = [row("Speaker 1", 0, 4.2, "If we have a one minute speech to deliver."),
                    row("Speaker 2", 4.5, 9.0, "Then the challenge is consolidating it.")]
        let data = try XCTUnwrap(TranscriptPDF.render(rows: rows, title: "Meeting Transcript",
                                                      recordedAt: Date()))
        let doc = try XCTUnwrap(PDFDocument(data: data), "the bytes must parse as a PDF")
        XCTAssertGreaterThan(doc.pageCount, 0)

        let text = doc.string ?? ""
        XCTAssertTrue(text.contains("Speaker 1"), "speaker names are in the document")
        XCTAssertTrue(text.contains("consolidating it"), "and so is what they said")
        XCTAssertTrue(text.contains("0:00"), "and the time range")
    }

    /// A long transcript must FLOW, not be truncated at one page. The pagination
    /// loop is the part of the renderer that can silently drop content, and a
    /// reader has no way to notice the tail is missing.
    func testALongTranscriptRunsToMultiplePages() throws {
        let rows = (0..<120).map {
            row("Speaker \($0 % 3 + 1)", Double($0) * 5, Double($0) * 5 + 4,
                "This is line number \($0), long enough to occupy a good part of a "
                + "line so that the document is forced past a single page.")
        }
        let data = try XCTUnwrap(TranscriptPDF.render(rows: rows, title: "Long meeting",
                                                      recordedAt: Date()))
        let doc = try XCTUnwrap(PDFDocument(data: data))
        XCTAssertGreaterThan(doc.pageCount, 1, "the transcript must paginate")
        XCTAssertTrue((doc.string ?? "").contains("line number 119"),
                      "and the LAST row must survive — a pagination loop that stops "
                      + "early loses the tail with nothing to show for it")
    }

    /// THE PAGE MUST ACTUALLY BE LEGIBLE — white paper, dark ink.
    ///
    /// This is the test the 2026-08-10 export bug needed and did not have. Every
    /// other check here reads `PDFDocument.string`, which extracts text no matter
    /// what colour it was drawn in or whether the page had a background. The
    /// shipped document had BOTH faults: a transparent page, and body text in
    /// `NSColor.textColor`, which is near-white in dark mode. The title was the
    /// only visible element — by accident, because it alone carried no colour and
    /// defaulted to black.
    ///
    /// So this rasterises and counts pixels. The two assertions are the two bugs:
    /// a mostly-white page (background) and a real minority of dark pixels (ink).
    func testThePageIsWhitePaperWithDarkInkNotInvisibleText() throws {
        let rows = (0..<8).map {
            row("Speaker \($0 % 2 + 1)", Double($0) * 3, Double($0) * 3 + 2.5,
                "Line \($0) of a transcript that must be readable on paper.")
        }
        // WHAT THIS TEST CAN AND CANNOT SEE, stated because I got it wrong once.
        //
        // It catches the TRANSPARENT PAGE half of the 2026-08-10 export bug, proven
        // by negative control: deleting the white fill fails it loudly.
        //
        // It does NOT catch the other half — body text in a dynamic colour like
        // `NSColor.textColor`, which is near-white under the dark appearance the
        // app runs in. Restoring that colour leaves this test GREEN, including when
        // the render is wrapped in `performAsCurrentDrawingAppearance(.darkAqua)`,
        // because a headless XCTest process has no application appearance for the
        // colour to resolve against. Do not add that wrapper back believing it
        // helps; it was tried and measured.
        //
        // That half is pinned at the SOURCE instead, by
        // `layout/pdf-export-uses-fixed-ink` in `sidecar-tests.py`.
        let data = try XCTUnwrap(TranscriptPDF.render(rows: rows, title: "Meeting Transcript",
                                                      recordedAt: Date()))
        let page = try XCTUnwrap(PDFDocument(data: data)?.page(at: 0))
        let bounds = page.bounds(for: .mediaBox)

        // Drawn into a bitmap WE allocate, rather than `page.thumbnail(of:for:)`.
        // That returns an NSImage whose representation is not an NSBitmapImageRep,
        // so there are no pixels to read — and the whole point here is the pixels.
        // The rep is created without a backing colour, so anything not painted by
        // the page stays transparent and would read as neither light nor dark:
        // exactly the transparency this test exists to catch.
        let bitmap = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(bounds.width), pixelsHigh: Int(bounds.height),
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        let saved = NSGraphicsContext.current
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        page.draw(with: .mediaBox, to: try XCTUnwrap(NSGraphicsContext.current).cgContext)
        NSGraphicsContext.current = saved

        // Read the raw bytes rather than `colorAt(x:y:)`. That call allocates an
        // NSColor and runs a colour-space conversion per sample; over the 121k
        // samples below it was measured at 92.7 ms against 0.23 ms for this loop —
        // 400x, for a brightness comparison that needs neither.
        let pixels = try XCTUnwrap(bitmap.bitmapData)
        let rowBytes = bitmap.bytesPerRow
        let pixelBytes = bitmap.bitsPerPixel / 8
        var light = 0, dark = 0
        for y in stride(from: 0, to: bitmap.pixelsHigh, by: 2) {
            for x in stride(from: 0, to: bitmap.pixelsWide, by: 2) {
                let p = pixels + y * rowBytes + x * pixelBytes
                // Unweighted mean of R, G and B. The page is greyscale ink on white,
                // so a luminance weighting would move every sample the same way and
                // change no verdict.
                let brightness = (Int(p[0]) + Int(p[1]) + Int(p[2])) / 3
                if brightness > 217 { light += 1 }         // > 0.85
                else if brightness < 128 { dark += 1 }     // < 0.50
            }
        }
        let total = Double(light + dark)
        XCTAssertGreaterThan(Double(light) / total, 0.80,
                             "the page must be mostly WHITE — a transparent PDF page "
                             + "rasterises dark and is what shipped on 2026-08-10")
        XCTAssertGreaterThan(dark, 200,
                             "and there must be real dark INK on it — white-on-white "
                             + "text extracts perfectly and reads as a blank page")
    }

    /// Rows with no timing print no clock. Realtime rows carry none until the
    /// chunked pass replaces them, and `0:00 – 0:00` would be a measurement nobody
    /// made.
    func testARowWithoutTimesPrintsNoClock() throws {
        let data = try XCTUnwrap(TranscriptPDF.render(
            rows: [row("Speaker 1", nil, nil, "No timing on this one.")],
            title: "T", recordedAt: Date()))
        let text = try XCTUnwrap(PDFDocument(data: data)).string ?? ""
        XCTAssertTrue(text.contains("No timing on this one."))
        XCTAssertFalse(text.contains("0:00"),
                       "an absent time must stay absent, not become zero")
    }
}
