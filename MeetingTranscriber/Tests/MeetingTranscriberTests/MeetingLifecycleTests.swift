import XCTest
@testable import MeetingTranscriber

/// The finished-meeting lifecycle (owner, 2026-08-10):
///
///     record → stop → passes finish → mic LOCKS, two actions appear
///                                   → Start Over clears the panel
///                                   → actions hide, mic unlocks
///
/// Everything here is about the ONE-WAY door: once a meeting is finished the app
/// must not be able to start recording over it, and `Start Over` must be the only
/// way past. Both halves are pinned, because a lock nobody can release and a lock
/// that releases itself are equally broken.
@MainActor
final class MeetingLifecycleTests: XCTestCase {

    /// The whole loop, in the order the user walks it.
    func testAFinishedMeetingLocksTheMicUntilStartOverClearsIt() {
        let r = AudioRecorder()
        XCTAssertFalse(r.meetingFinished, "a fresh recorder has no finished meeting")

        r.markMeetingFinished()
        XCTAssertTrue(r.meetingFinished)

        r.startOver()
        XCTAssertFalse(r.meetingFinished,
                       "Start Over is the only way out of the finished state, so it "
                       + "must actually lower the flag — otherwise the mic stays "
                       + "locked forever and the app is unusable")
    }

    /// The LOCK, tested through the model rather than the view. `RecordCardView`
    /// also disables the button, but a view is not a rule: this is what stops a
    /// keyboard path, a menu item or a second window from starting a recording
    /// over a transcript the user has not dealt with.
    /// OBSERVED THROUGH `errorMessage`, and that is not arbitrary. `state` is the
    /// obvious thing to assert and it CANNOT discriminate: `start()` hands off to
    /// `AVCaptureDevice.requestAccess` asynchronously, so `state` is still `.idle`
    /// a line later whether the guard ran or not — a negative control proved that
    /// version of this test passes with the guard deleted. `start()`'s FIRST
    /// statement is `errorMessage = nil`, executed synchronously, so a surviving
    /// message is proof `start()` was never entered.
    func testToggleIsInertWhileAMeetingIsFinished() {
        let r = AudioRecorder()
        r.errorMessage = "left over from the meeting"
        r.markMeetingFinished()

        r.toggle()

        XCTAssertEqual(r.errorMessage, "left over from the meeting",
                       "toggle() must not reach start() while a finished meeting is "
                       + "on screen — start() clears errorMessage on entry, so a "
                       + "cleared message means the mic lock was bypassed")
        XCTAssertEqual(r.state, .idle)
        XCTAssertTrue(r.meetingFinished,
                      "and it must not clear the meeting either — only startOver() "
                      + "does that, deliberately, so the exit is a single door")
    }

    /// `startOver` on a recorder with no finished meeting must do NOTHING. Without
    /// the guard it would be a transcript-eraser callable at any moment, including
    /// mid-recording.
    func testStartOverDoesNothingWithoutAFinishedMeeting() {
        let r = AudioRecorder()
        r.segments = [AudioRecorder.TranscriptSegment(text: "still being recorded",
                                                     confirmed: true)]

        r.startOver()

        XCTAssertEqual(r.segments.count, 1,
                       "startOver() outside the finished state must not touch the "
                       + "transcript — a live recording's segments are not its to clear")
    }

    /// What `startOver` clears is derived from what the views READ, so this asserts
    /// the visible set specifically. If a view starts reading a new property, the
    /// property belongs here too.
    func testStartOverClearsEverythingThePanelShows() {
        let r = AudioRecorder()
        r.segments = [AudioRecorder.TranscriptSegment(text: "hello", confirmed: true)]
        r.displayRows = [AudioRecorder.SpeakerUtterance(id: "row-1", speaker: "Speaker 1",
                                                       text: "hello", confirmed: true)]
        r.partialTranscript = "half a sen"
        r.partialSpeakerName = "Speaker 2"
        r.speakerCount = 3
        r.remoteSpeakerCount = 1
        r.errorMessage = "something went wrong"
        r.chunkedError = "chunked blew up"
        r.diarizationError = "diarization blew up"
        r.overlapRepairError = "repair blew up"
        r.overlapRepairProgress = "2 of 5"
        r.stopSteps = [AudioRecorder.StopStep(id: "diarize", name: "Identifying speakers",
                                              state: .done)]
        r.lastRecordingURL = URL(fileURLWithPath: "/tmp/meeting.wav")
        r.markMeetingFinished()

        r.startOver()

        XCTAssertTrue(r.segments.isEmpty)
        XCTAssertTrue(r.displayRows.isEmpty)
        XCTAssertTrue(r.partialTranscript.isEmpty)
        XCTAssertNil(r.partialSpeakerName)
        XCTAssertNil(r.speakerCount)
        XCTAssertNil(r.remoteSpeakerCount)
        XCTAssertNil(r.errorMessage)
        XCTAssertNil(r.chunkedError)
        XCTAssertNil(r.diarizationError)
        XCTAssertNil(r.overlapRepairError)
        XCTAssertNil(r.overlapRepairProgress)
        XCTAssertTrue(r.stopSteps.isEmpty)
        XCTAssertNil(r.lastRecordingURL,
                     "the recording the panel was showing goes too, so nothing "
                     + "downstream can act on a file the user just dismissed")
    }

    /// Export is offered on a finished meeting even when it is EMPTY — that is the
    /// case where the user most needs to see the app noticed — but it cannot be
    /// pressed, because pressing it would write a blank document.
    func testExportIsRefusedWithNoTranscriptAndAllowedWithOne() {
        let r = AudioRecorder()
        r.markMeetingFinished()
        XCTAssertFalse(r.canExport,
                       "a finished meeting that transcribed nothing has nothing to export")

        r.displayRows = [AudioRecorder.SpeakerUtterance(id: "row-1", speaker: "Speaker 1",
                                                       text: "hello", confirmed: true)]
        XCTAssertTrue(r.canExport)
    }

    /// The reason `meetingFinished` is a FLAG and not `state == .idle &&
    /// !segments.isEmpty`, which is what it started as. A meeting that transcribed
    /// nothing is still finished, and deriving it from the transcript would have
    /// left exactly that user with a dead panel and no way back.
    func testAMeetingThatTranscribedNothingIsStillFinished() {
        let r = AudioRecorder()
        r.markMeetingFinished()

        XCTAssertTrue(r.segments.isEmpty)
        XCTAssertTrue(r.meetingFinished,
                      "an empty transcript must not read as 'no meeting happened' — "
                      + "Start Over is the only way back to a usable mic")
    }
}
