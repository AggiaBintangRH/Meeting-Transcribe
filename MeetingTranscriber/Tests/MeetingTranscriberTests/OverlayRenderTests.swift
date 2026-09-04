import XCTest
import SwiftUI
@testable import MeetingTranscriber

/// Renders the real stop-overlay rows to a PNG so the layout can be LOOKED AT.
///
/// ⚠ NOT AN ASSERTION ABOUT APPEARANCE, and it does not pretend to be. It is a
/// developer tool: "it compiles" and "the numbers are right" say nothing about
/// whether a bar is cramped against its caption or runs off its row, and this
/// project has already shipped a PDF export that extracted perfectly and was
/// blank on the page. The one thing it DOES assert is that rendering produces a
/// non-trivial image at all — a view that fails to lay out comes back empty.
final class OverlayRenderTests: XCTestCase {

    @MainActor
    func testStopStepRowsRenderForInspection() throws {
        let dir = ProcessInfo.processInfo.environment["MT_RENDER_DIR"]
        let rows = VStack(alignment: .leading, spacing: 14) {
            OverlayStepRow(name: "Re-transcribing the recording", state: .loading,
                           progress: .init(done: 0, total: 120, elapsed: 0.4))
            OverlayStepRow(name: "Re-transcribing the recording", state: .loading,
                           progress: .init(done: 7, total: 120, elapsed: 33))
            OverlayStepRow(name: "Re-transcribing the recording", state: .loading,
                           progress: .init(done: 96, total: 120, elapsed: 402))
            OverlayStepRow(name: "Labelling speakers", state: .loading, progress: nil)
            OverlayStepRow(name: "Transcribing final audio", state: .done)
            OverlayStepRow(name: "Detecting overlap", state: .skipped("no 2-speaker windows"))
            OverlayStepRow(name: "Repairing overlap", state: .failed("sidecar exited: code 1"))
        }
        .padding(28)   // OverlayCard's own padding
        .frame(width: 420, alignment: .leading)   // OverlayCard's own width
        .background(Theme.card)

        let renderer = ImageRenderer(content: rows)
        renderer.scale = 2
        let image = try XCTUnwrap(renderer.nsImage, "the overlay rows failed to lay out at all")
        XCTAssertGreaterThan(image.size.width, 100)
        XCTAssertGreaterThan(image.size.height, 100)

        guard let dir else { return }   // only writes when explicitly asked
        let tiff = try XCTUnwrap(image.tiffRepresentation)
        let png = try XCTUnwrap(NSBitmapImageRep(data: tiff)?
            .representation(using: .png, properties: [:]))
        try png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("stop-overlay.png"))
    }
}
