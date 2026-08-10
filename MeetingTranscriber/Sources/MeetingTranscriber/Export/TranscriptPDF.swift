import AppKit
import CoreText
import Foundation

/// Renders a finished meeting's transcript to PDF data (owner, 2026-08-10).
///
/// CORETEXT, NOT a hand-rolled line loop. A transcript is arbitrarily long and
/// must flow across pages; `CTFramesetter` is the only path that breaks lines and
/// pages the way the system does everywhere else, including for scripts this
/// project does not otherwise think about. Measuring strings by hand would put a
/// subtly different line-breaker in the one artefact the client keeps.
///
/// WHAT GOES IN, chosen rather than specified (2026-08-10): speaker name, the
/// time range, and the text — plus a title and the recording's date at the top.
/// `spk` / `asr` confidences are deliberately LEFT OUT: they are decision aids
/// while reading the app, and a number a reader cannot interpret is worse than no
/// number in a document that outlives the session.
enum TranscriptPDF {

    /// FIXED INK, never a dynamic system colour.
    ///
    /// `NSColor.textColor` and `.secondaryLabelColor` resolve against the CURRENT
    /// APPEARANCE. In dark mode they are near-white, so on 2026-08-10 this exporter
    /// produced a document whose only visible element was the title — which alone
    /// had no colour attribute and therefore defaulted to black. The owner found it
    /// by opening the file; the test suite did not, because `PDFDocument.string`
    /// extracts text regardless of what colour it was drawn in.
    ///
    /// A printed page has one appearance. These are it.
    private enum Ink {
        static let body  = NSColor(white: 0.05, alpha: 1)
        static let muted = NSColor(white: 0.40, alpha: 1)
    }

    /// US Letter at 72 dpi, which is what CoreGraphics means by "points".
    private static let pageSize = CGSize(width: 612, height: 792)
    private static let margin: CGFloat = 54

    /// Build the document. Returns nil only if CoreGraphics refuses a context,
    /// which in practice means the machine is out of memory.
    static func render(rows: [AudioRecorder.SpeakerUtterance],
                       title: String,
                       recordedAt: Date) -> Data? {
        let body = attributedBody(rows: rows, title: title, recordedAt: recordedAt)

        let data = NSMutableData()
        guard let consumer = CGDataConsumer(data: data) else { return nil }
        var mediaBox = CGRect(origin: .zero, size: pageSize)
        guard let ctx = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else { return nil }

        let framesetter = CTFramesetterCreateWithAttributedString(body)
        let textRect = CGRect(x: margin, y: margin,
                              width: pageSize.width - margin * 2,
                              height: pageSize.height - margin * 2)
        let path = CGPath(rect: textRect, transform: nil)

        var start = 0
        let total = body.length
        // `total > 0` guards the empty case: CTFramesetter on an empty string
        // returns a zero-length frame forever, so a `while start < total` loop with
        // no rows would still emit one page — and `repeat` would emit it twice.
        while start < total {
            ctx.beginPDFPage(nil)
            // AN OPAQUE WHITE PAGE, drawn before anything else.
            //
            // A CGPDFContext page has NO background — it is transparent. Viewers
            // that composite onto white hide that; anything else does not, and the
            // document is a client artefact that will be opened in readers nobody
            // here chose. Found 2026-08-10 by rendering a page to PNG and counting
            // pixels: 96.7 % came back dark, which was the transparency, not ink.
            ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            ctx.fill(CGRect(origin: .zero, size: pageSize))
            let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: start, length: 0),
                                                 path, nil)
            CTFrameDraw(frame, ctx)
            let consumed = CTFrameGetVisibleStringRange(frame)
            ctx.endPDFPage()
            // A page that consumed NOTHING would loop forever — it happens when a
            // single unbreakable element is taller than the frame. Bail rather than
            // hang the app on a transcript nobody could have predicted.
            guard consumed.length > 0 else { break }
            start += consumed.length
        }
        ctx.closePDF()
        return data as Data
    }

    /// A default filename, built entirely from a fixed date format — so there is
    /// no user-supplied text in it and nothing to sanitise. An earlier version took
    /// the meeting title and this comment still described stripping the slashes and
    /// colons that could arrive with it; the parameter went, the comment did not.
    static func suggestedFilename(recordedAt: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH-mm"
        return "Meeting Transcript \(f.string(from: recordedAt)).pdf"
    }

    // MARK: - Content

    private static func attributedBody(rows: [AudioRecorder.SpeakerUtterance],
                                       title: String,
                                       recordedAt: Date) -> NSAttributedString {
        let out = NSMutableAttributedString()

        let titleStyle = NSMutableParagraphStyle()
        titleStyle.paragraphSpacing = 4
        out.append(NSAttributedString(string: title + "\n", attributes: [
            .font: NSFont.systemFont(ofSize: 18, weight: .semibold),
            .foregroundColor: Ink.body,
            .paragraphStyle: titleStyle,
        ]))

        let stamp = DateFormatter()
        stamp.dateStyle = .long
        stamp.timeStyle = .short
        let headerStyle = NSMutableParagraphStyle()
        headerStyle.paragraphSpacing = 18
        out.append(NSAttributedString(string: stamp.string(from: recordedAt) + "\n", attributes: [
            .font: NSFont.systemFont(ofSize: 10),
            .foregroundColor: Ink.muted,
            .paragraphStyle: headerStyle,
        ]))

        // Hoisted beside the paragraph styles they belong to, rather than rebuilt
        // per row. Measured at ~1 ms over 800 rows, so this is for symmetry and
        // readability — the two styles three lines up were already out here, and
        // the attributes that use them were not.
        let speakerStyle = NSMutableParagraphStyle()
        speakerStyle.paragraphSpacing = 2
        let textStyle = NSMutableParagraphStyle()
        textStyle.paragraphSpacing = 12
        textStyle.lineSpacing = 1.5

        let speakerAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 10, weight: .semibold),
            .foregroundColor: Ink.muted,
            .paragraphStyle: speakerStyle,
        ]
        let textAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 11),
            .foregroundColor: Ink.body,
            .paragraphStyle: textStyle,
        ]

        for row in rows {
            let who = row.speaker ?? "Speaker"
            let when = timeRange(row)
            let heading = when.isEmpty ? who : "\(who)   \(when)"
            out.append(NSAttributedString(string: heading + "\n",
                                          attributes: speakerAttributes))
            out.append(NSAttributedString(string: row.text + "\n",
                                          attributes: textAttributes))
        }
        return out
    }

    /// `mm:ss – mm:ss`, or empty when the row has no times. Realtime rows carry
    /// none until the chunked pass replaces them, and printing `0:00 – 0:00`
    /// would be a measurement nobody made.
    /// The FORMATTER is shared with the on-screen transcript (`Clock.mmss`); the
    /// RULE is not, deliberately. `TranscriptView` shows a range or nothing, while
    /// a printed row with only a start is still worth stamping — the reader cannot
    /// hover it to find out more.
    private static func timeRange(_ row: AudioRecorder.SpeakerUtterance) -> String {
        guard let start = row.start else { return "" }
        guard let end = row.end else { return Clock.mmss(start) }
        return "\(Clock.mmss(start)) – \(Clock.mmss(end))"
    }
}
