import XCTest
import Combine
@testable import MeetingTranscriber

/// WHY THE ATND POSITION LAYER GOES SILENT WHILE ANGLE DATA IS ARRIVING.
///
/// The owner's report: rows render SPEAKER UNKNOWN even though the array is
/// streaming angle/rotation. Their own log agreed — 186 `no-spans` lines, every
/// one of them `samples=0`, with blackouts of 1 s to 30 s at the START of each
/// session.
///
/// `samples=0` means the CLUSTERER never received anything, so the loss is
/// upstream of it. This file drives `PositionDiarizer` with notice streams shaped
/// like a real array and MEASURES which stage drops them, rather than reasoning
/// about it from the source — the mistake this project keeps correcting.
@MainActor
final class PositionBlackoutTests: XCTestCase {

    /// Feed a notice stream and report what survived to the clusterer.
    ///
    /// `speechAt` decides, per notice, whether our VAD had reported speech at
    /// that instant — the `noteSpeech` the capture tap makes.
    private func drive(seconds: Double,
                       hz: Double = 10,
                       gateOnSpeech: Bool,
                       silenceEvery: Int? = nil,
                       speechAt: (Double) -> Bool) -> (samples: Int, spans: Int) {
        let d = PositionDiarizer()
        var now = 0.0
        d.start(tauDeg: 15, smoothingSec: 0.4, mode: .firstCome,
                gateOnSpeech: gateOnSpeech, now: { now })
        let step = 1.0 / hz
        var i = 0
        while now < seconds {
            if speechAt(now) { d.noteSpeech(true, at: now) }
            // A real array interleaves silence notices whenever its own beam is
            // not locked; `silenceEvery: n` sends one every n notices.
            let isSilence = silenceEvery.map { i % $0 == 0 } ?? false
            ATNDBeamService.shared.rawNotices.send(
                (.now, isSilence ? .silence
                                 : .talking(.init(channel: 1, elevation: 30, rotation: 120))))
            now += step
            i += 1
        }
        let range = 0.0...seconds
        return (d.sampleCount(in: range), d.labeledSpans(in: range).count)
    }

    /// BASELINE — a perfect stream: speech from t=0, never a silence notice.
    /// If this does not produce samples, the harness is wrong and every other
    /// number here is meaningless.
    func testPerfectStreamProducesSamples() {
        let r = drive(seconds: 5, gateOnSpeech: true, speechAt: { _ in true })
        XCTAssertGreaterThan(r.samples, 0, "harness is broken — nothing survives even a clean stream")
        XCTAssertGreaterThan(r.spans, 0)
    }

    /// THE OWNER'S CASE #1 — the array interleaves silence notices.
    /// `.silence` calls `smoother.reset()`, and the smoother needs 0.4 s of
    /// UNINTERRUPTED notices to emit, so a silence every 0.4 s starves it forever.
    func testInterleavedSilenceStarvesTheSmoother() {
        for every in [2, 3, 4, 5, 10] {
            let r = drive(seconds: 20, gateOnSpeech: true,
                          silenceEvery: every, speechAt: { _ in true })
            print("SILENCE every \(every) notice(s) (\(Double(every) * 100)ms): "
                  + "samples=\(r.samples) spans=\(r.spans)")
        }
    }

    /// THE OWNER'S CASE #2 — VAD opens late, then flickers. Every closure calls
    /// `smoother.reset()` too.
    func testVadFlickerStarvesTheSmoother() {
        // Opens at 3 s and stays open.
        let late = drive(seconds: 20, gateOnSpeech: true, speechAt: { $0 >= 3.0 })
        print("VAD opens at 3s, then steady: samples=\(late.samples) spans=\(late.spans)")
        // Open 300 ms in every 700 ms — a normal speaking rhythm with pauses.
        let flick = drive(seconds: 20, gateOnSpeech: true,
                          speechAt: { $0.truncatingRemainder(dividingBy: 0.7) < 0.3 })
        print("VAD flickers 300ms/700ms: samples=\(flick.samples) spans=\(flick.spans)")
    }

    /// THE INVARIANT THE OWNER ASKED FOR, stated as a test:
    /// while the array is sending angle/rotation, a row must never be UNKNOWN.
    ///
    /// Swept across every silence cadence the array could plausibly produce, so
    /// this cannot pass by happening to pick a friendly one.
    func testAngleDataAlwaysYieldsASpeaker() {
        for every in [2, 3, 4, 5, 10] {
            let r = drive(seconds: 20, gateOnSpeech: true, silenceEvery: every,
                          speechAt: { $0.truncatingRemainder(dividingBy: 0.7) < 0.3 })
            XCTAssertGreaterThan(r.spans, 0,
                                 "silence every \(every) notice(s): angle data arrived for "
                                 + "20 s and produced NO span — every row in that stretch "
                                 + "renders SPEAKER UNKNOWN")
        }
    }

    // MARK: - What the removed resets were FOR must still hold

    /// REGRESSION #1: two talkers must still separate. The reset existed so one
    /// speaker's direction never averaged into the next one's, and dropping it
    /// could have smeared a beam move into a single mid-way cluster — which would
    /// trade UNKNOWN for a WRONG name, the worse direction.
    func testABeamMoveStillProducesTwoSpeakers() {
        let d = PositionDiarizer()
        var now = 0.0
        d.start(tauDeg: 15, smoothingSec: 0.4, mode: .firstCome,
                gateOnSpeech: false, now: { now })
        // 6 s at rotation 30, a 1 s silence, then 6 s at rotation 200 — well
        // beyond the 15 degrees two clusters must differ by.
        for (rot, span) in [(30, 0.0..<6.0), (200, 7.0..<13.0)] {
            now = span.lowerBound
            while now < span.upperBound {
                ATNDBeamService.shared.rawNotices.send(
                    (.now, .talking(.init(channel: 1, elevation: 30, rotation: rot))))
                now += 0.1
            }
            // The silence between them, at the array's own rate.
            if rot == 30 {
                while now < 7.0 {
                    ATNDBeamService.shared.rawNotices.send((.now, .silence))
                    now += 0.1
                }
            }
        }
        let spans = d.labeledSpans(in: 0...13)
        let names = Set(spans.map(\.name))
        XCTAssertEqual(names.count, 2,
                       "a 170-degree beam move must still be two speakers, not one "
                       + "smeared cluster — got \(names.sorted())")
    }

    /// REGRESSION #2: a silence LONGER than the smoothing window must still clear
    /// the buffer. That is what now carries the old reset's job, so if the
    /// smoother's own time rule ever stops doing it, this fails rather than the
    /// behaviour drifting silently.
    func testALongSilenceStillBreaksContinuity() {
        var sm = DirectionSmoother(windowSec: 0.4)
        let v = PositionMath.unitVector(rotateDeg: 30, angleDeg: 30)
        // Warm up to the point of emitting.
        var t = 0.0
        var emitted = 0
        while t < 1.0 { if sm.push(t: t, vector: v) != nil { emitted += 1 }; t += 0.1 }
        XCTAssertGreaterThan(emitted, 0, "precondition: the smoother was emitting")
        // A 2 s hole — longer than one window — then resume.
        t += 2.0
        XCTAssertNil(sm.push(t: t, vector: v),
                     "after a gap longer than the window the smoother must warm up "
                     + "again, which is what replaces the removed reset")
    }

    // MARK: - End to end: the ROW, which is what the owner actually sees

    /// THE OWNER'S REQUIREMENT, VERBATIM: *"sampai ATND angle rotate ada tuh
    /// pasti pembuatan row akan ada Speaker nya gak jadi speaker unknown."*
    ///
    /// Every test above measures the diarizer. This one measures the TRANSCRIPT,
    /// because that is the only place the promise is kept or broken — a span the
    /// display never consults is worth nothing, and `derivedRows` is where a nil
    /// speaker becomes the SPEAKER UNKNOWN header in `TranscriptView`.
    ///
    /// It drives BOTH row kinds, because they take different branches and the
    /// owner's two screenshots were one of each: an unconfirmed realtime row
    /// (`best?.name`) and a confirmed chunked row.
    func testNoRowIsUnknownWhileTheArrayIsStreaming() {
        let recorder = AudioRecorder()
        let d = PositionDiarizer()
        var now = 0.0
        d.start(tauDeg: 15, smoothingSec: 0.4, mode: .firstCome,
                gateOnSpeech: true, now: { now })
        // The owner's hardest measured case: silence interleaved every 400 ms AND
        // the VAD flickering — the combination that measured samples=0 before.
        while now < 30.0 {
            if now.truncatingRemainder(dividingBy: 0.7) < 0.3 { d.noteSpeech(true, at: now) }
            let silence = Int(now * 10) % 4 == 0
            ATNDBeamService.shared.rawNotices.send(
                (.now, silence ? .silence
                               : .talking(.init(channel: 1, elevation: 30, rotation: 120))))
            now += 0.1
        }
        recorder.positionDiarizer = d
        recorder.positionSource = .both      // no voice engine: ATND fills everything

        // A confirmed chunk and a short realtime utterance, both inside the span
        // the array was streaming for.
        let chunk = AudioRecorder.TranscriptSegment(
            text: "Hello everyone. Thank you for coming.",
            confirmed: true, window: 0.0...30.0)
        let live = AudioRecorder.TranscriptSegment(
            text: "Fridays to", confirmed: false, window: 20.0...20.4)

        for seg in [chunk, live] {
            let rows = recorder.derivedRows(for: seg, regions: [])
            XCTAssertFalse(rows.isEmpty, "no rows at all for \(seg.text)")
            for row in rows {
                XCTAssertNotNil(row.speaker,
                                "SPEAKER UNKNOWN on \(seg.confirmed ? "a confirmed" : "a realtime") "
                                + "row (\"\(row.text)\") while the array was streaming "
                                + "angle/rotation for the whole window")
            }
        }
    }

    // MARK: - The live caption went blank while the rows stayed labelled (2026-08-18)

    /// ⚠ THE OWNER'S OBSERVATION, FROM A SCREENSHOT: rows read `SPEAKER 1` while the
    /// caption above them read `SPEAKER UNKNOWN`, with the VAD chip on SILENT —
    /// *"itunya silent mungkin karena itu gak jadi speaker unknown"*. Correct.
    ///
    /// The caption asked `dominantCluster(minSamples: 3)` over the last second; the
    /// rows ask `timeline.spans`, whose final boundary is open-ended. Direction is
    /// collected only while VAD hears speech, so a silent second gave the caption
    /// nothing and the two surfaces disagreed about the same instant.
    ///
    /// Drives a real talker, then goes silent, then asks about the silent moment.
    func testTheCaptionKeepsTheLastSeatThroughASilentSecond() {
        let d = PositionDiarizer()
        var now = 0.0
        d.start(tauDeg: 15, smoothingSec: 0.4, mode: .firstCome,
                gateOnSpeech: true, now: { now })
        // 3 s of a settled talker.
        while now < 3.0 {
            d.noteSpeech(true, at: now)
            ATNDBeamService.shared.rawNotices.send(
                (.now, .talking(.init(channel: 1, elevation: 30, rotation: 120))))
            now += 0.1
        }
        XCTAssertNotNil(d.captionLabel(at: now), "precondition: a seat is known by now")
        let seated = d.captionLabel(at: now)!

        // Now 2 s where VAD reports nothing — no samples reach the smoother.
        while now < 5.0 {
            ATNDBeamService.shared.rawNotices.send(
                (.now, .talking(.init(channel: 1, elevation: 30, rotation: 120))))
            now += 0.1
        }

        // The OLD lookup finds nothing here — that is the bug, asserted so this
        // test cannot pass for the wrong reason (e.g. the VAD gate quietly opening).
        XCTAssertNil(d.label(for: max(0, now - 1.0)...now, minSamples: 3),
                     "precondition: the recent-samples lookup really is starved")

        let caption = d.captionLabel(at: now)
        XCTAssertNotNil(caption,
                        "a silent second must not blank the caption — the timeline still "
                        + "says who the beam last settled on, and the rows are printing it")
        XCTAssertEqual(caption?.id, seated.id, "and it must be the SAME seat, not a new one")
        // The rows' own answer, for the same instant — they must agree.
        XCTAssertEqual(d.labeledSpans(in: max(0, now - 1.0)...now).last?.id, caption?.id,
                       "caption and rows must resolve one instant identically")
    }

    /// Before the FIRST boundary there is genuinely no talker, and the caption must
    /// still say so. Without this the fallback could be "always name someone".
    func testTheCaptionIsStillNilBeforeTheFirstBoundary() {
        let d = PositionDiarizer()
        var now = 0.0
        d.start(tauDeg: 15, smoothingSec: 0.4, mode: .firstCome,
                gateOnSpeech: true, now: { now })
        now = 2.0
        XCTAssertNil(d.captionLabel(at: now),
                     "no notice has ever arrived — inventing a seat here would be fabrication")
    }

}
