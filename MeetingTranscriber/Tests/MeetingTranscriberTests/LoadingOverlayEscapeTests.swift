import XCTest
@testable import MeetingTranscriber

/// The startup overlay must never trap the user.
///
/// A client Mac showed it with CAM++ marked failed, the models below it never
/// attempted, the header still reading "Loading models" and no Close button —
/// the app unusable and Settings unreachable (2026-08-18). `failureMessage` and
/// the row's `.failed` state are set one line apart in `loadAll`, so how they
/// disagreed there is not established; what these pin is that the way OUT no
/// longer depends on them agreeing.
///
/// `items` is deliberately `private(set)` and stays that way — widening it so a
/// test could pose a state the app builds for itself would be testing a
/// different program. So the enum rule is tested directly and the one path that
/// can be driven from outside (`failStartup`) is driven.
@MainActor
final class LoadingOverlayEscapeTests: XCTestCase {

    /// THE RULE THE OVERLAY READS. A red row is a failure; nothing else is.
    func testOnlyAFailedRowCountsAsAFailure() {
        XCTAssertTrue(ModelLoader.ItemState.failed("model not found").isFailed)
        for state: ModelLoader.ItemState in [.pending, .loading, .done,
                                             .skipped("no speech found")] {
            XCTAssertFalse(state.isFailed, "\(state) must not turn the overlay red")
        }
    }

    /// ⚠ SKIPPED IS THE ONE THAT MATTERS. It is a correct outcome, and the
    /// 2026-08-12 work exists because painting it red reported a broken app to
    /// someone whose app was working. A correct outcome must not demand a click.
    func testASkippedStepIsNotAFailure() {
        XCTAssertFalse(ModelLoader.ItemState.skipped("no speech found").isFailed)
    }

    /// A fresh loader offers no way out — otherwise every ordinary session would
    /// end with an extra click.
    func testAFreshLoaderHasNothingToDismiss() {
        XCTAssertFalse(ModelLoader().hasFailure)
    }

    /// A startup refusal does, and it is the path every refusal takes
    /// (Voxtral + dual-stream, chunked off + Remote, and the rest).
    func testAStartupRefusalOffersAWayOut() {
        let loader = ModelLoader()
        loader.failStartup(step: "Remote stream + chunked model",
                           message: "Voxtral cannot keep up with two streams.")
        XCTAssertTrue(loader.hasFailure)
        XCTAssertTrue(loader.items.contains { $0.state.isFailed },
                      "and it is visible as a row, not only as a message")
    }

    /// CLOSE MUST ACTUALLY CLOSE — the reported symptom's other half.
    ///
    /// ⚠ This test was first written asserting `hasFailure` stayed TRUE after
    /// dismissing, rationalised as "the row is the record of what happened". That
    /// was wrong and it was my own bug: once `hasFailure` reads the rows, leaving
    /// them makes Close a dead button and the overlay permanent. Writing the
    /// assertion is what exposed it.
    func testDismissingReallyCloses() {
        let loader = ModelLoader()
        loader.failStartup(step: "Remote stream + chunked model", message: "nope")
        XCTAssertTrue(loader.hasFailure, "precondition: the overlay is up")
        loader.dismissFailure()
        XCTAssertFalse(loader.hasFailure,
                       "Close left something asserting a failure — the overlay would "
                       + "stay and the button would be dead")
    }
}
