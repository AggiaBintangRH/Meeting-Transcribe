import Foundation

// Position (ATND beam) diarization: session setup of the diarizer, the
// display-time gap-fill against pyannote's complement, and this domain's own log
// writer. Moved verbatim from the core file.
extension AudioRecorder {

    // MARK: - Position diarization (ATND beam) gap-fill
    //
    // Policy (owner, 2026-07-20): pyannote is AUTHORITATIVE. Wherever pyannote has
    // a turn, its own label wins. ATND position only fills the DISPLAY-time gaps
    // pyannote has not (yet) covered — freshly-committed text, the first ~1-2s
    // pyannote never turns, and the live partial. A pyannote chunk landing later
    // OVERRIDES the fill automatically (it shrinks the gap on the next rebuild).
    // Silence gaps where ATND also heard nothing STAY unknown — never force-filled.

    /// Create + start the position diarizer for this session, but ONLY when the
    /// feature is explicitly enabled AND the beam service is actually listening.
    /// Otherwise leave it nil — `positionGapFill` then returns [] and the display
    /// path is pure pyannote, byte-identical to before this feature existed.
    func configurePositionDiarization() {
        positionDiarizer = nil
        lastLoggedSeparation = nil
        let d = UserDefaults.standard
        // Read once, here — the display policy is fixed for the whole session.
        // Reset to `both` first so a session that bails out below (feature off /
        // ATND down) is on the default policy, matching its nil diarizer.
        positionSource = .both
        // SAY WHY, rather than returning in silence.
        //
        // Both conditions were a bare `guard … else { return }`, and a nil
        // diarizer makes `positionGapFill` return [] before it logs anything — so
        // a session where the beam layer never started produced NO line in any
        // file, and every row that had no diarization turn yet rendered as
        // SPEAKER UNKNOWN with nothing anywhere to explain it. Owner, 2026-08-13:
        // *"i enable the ATND but the speaker is unknown"*.
        //
        // The two conditions are named separately because the fix differs: the
        // first is a SECOND switch most people miss (connecting to the array in
        // ATND → Connection does not turn this on), the second is the array not
        // actually streaming.
        guard d.bool(forKey: "atnd.position.enabled") else {
            positionLog("POSITION LAYER OFF — `atnd.position.enabled` is false. "
                        + "Connecting to the array (ATND → Connection) does not "
                        + "enable this; the beam-direction speaker labels are a "
                        + "separate switch in ATND → Position. Until it is on, "
                        + "rows with no diarization turn stay SPEAKER UNKNOWN.")
            return
        }
        guard ATNDBeamService.shared.state == .listening else {
            positionLog("POSITION LAYER OFF — the feature is enabled but the beam "
                        + "service is \(ATNDBeamService.shared.state), not "
                        + "listening. Check the Device IP and the multicast "
                        + "Interface address in ATND → Connection; the Interface "
                        + "is THIS Mac's address, never the array's.")
            return
        }

        let tauDeg = d.object(forKey: "atnd.position.tauDeg") as? Double ?? 15
        let smoothingMs = d.object(forKey: "atnd.position.smoothingMs") as? Double ?? 400
        let mode: PositionDiarizer.Mode =
            (d.string(forKey: "atnd.position.mode") == "enrollment") ? .enrollment : .firstCome
        positionSource = PositionSource.current(d)

        let diarizer = PositionDiarizer()
        // ⚠ THE BEAM IS NO LONGER GATED ON OUR VAD (owner, 2026-09-02:
        // "speaker nya jangan ada blocking VAD kita aja / jadi full dari ATND").
        // Direction is taken from the array for every notice it sends.
        //
        // WHY THE GATE EXISTED, because it was a good reason and it is not
        // forgotten: measured 2026-07-27, the array's beam follows ANY sound —
        // piano and loudspeaker audio still produced angle/rotation notices with
        // the ATND's own SVAD on — so an ungated collector turns a noise source
        // into a speaker position. The gate was the only defence available then.
        //
        // WHY IT CAN GO NOW, and this is the part that changed rather than the
        // preference: the room has EXCLUSION ZONES. The client's array was
        // reconfigured on 2026-09-02 with EX regions, which forbid the beam from
        // fixed positions in hardware, at the source, regardless of what is
        // playing there. CLAUDE.md already named that "the promising fix for the
        // room-echo problem". The defence moved from a software gate that only
        // guesses to a device rule that cannot be argued with.
        //
        // 🔴 AND THE GATE WAS COSTING MORE THAN IT BOUGHT. It only ever opens
        // while `VoiceActivityDetector` says speech, and on the client's 5.8 dB
        // SNR capture that verdict DIES PARTWAY THROUGH A MEETING. Measured on
        // their own recording (meeting-2026-09-02T05-08-56Z.wav, 91.9 s), the
        // heuristic engine's adaptive floor runs away:
        //
        //     0-10 s  49 % speaking   gate 0.0120
        //    30-40 s  66 % speaking   gate 0.0190
        //    50-60 s   3 % speaking   gate 0.0699
        //    60-90 s   0 % speaking   gate 0.1897   <- shut, permanently
        //
        // `noiseFloor` is an EMA over every buffer that FAILS the gate, and the
        // gate is 6x that floor. At low SNR the pauses are nearly as loud as the
        // speech, so the floor climbs, the gate climbs with it, more buffers fail,
        // and once the gate passes the speech level the SPEECH ITSELF feeds the
        // floor. It is positive feedback with no ceiling. Their log shows the
        // consequence exactly: every `SKIP gap=... samples=0` from 55.7 s to the
        // end of the meeting, and SPEAKER UNKNOWN on every row.
        //
        // ⚠ THAT RUNAWAY IS A SEPARATE, STILL-OPEN DEFECT. It also drives the
        // realtime flush (`asr?.flush()` on the speech→silence edge) and the
        // chunk boundary, so it is not fixed by this change — it is only no
        // longer able to blank the speaker labels. Do not read this line as
        // closing it.
        //
        // The parameter STAYS, with its tests: `PositionSpeechGateTests` and
        // `PositionBlackoutTests` drive `gateOnSpeech: true` directly, so the
        // mechanism is still pinned and can be turned back on in one edit if a
        // room without exclusion zones ever needs it.
        let gateOnSpeech = false
        diarizer.start(tauDeg: tauDeg,
                       smoothingSec: smoothingMs / 1000,
                       mode: mode,
                       gateOnSpeech: gateOnSpeech,
                       now: { [weak self] in self?.recordingElapsed ?? 0 })
        positionDiarizer = diarizer

        // SAY THE LAYER STARTED, not only that it did not. Until now this
        // function logged its two OFF cases and nothing else, so a healthy
        // session and a session whose gate never opened looked identical in the
        // file — an absence readable two ways, which is what cost a day on
        // 2026-08-18 (`configureMoss` had the same gap and was given the same
        // fix). The gate state is named because it is the one thing here that a
        // reader will want to check first.
        positionLog("POSITION LAYER ON — source '\(positionSource.rawValue)', "
                    + "tau \(Int(tauDeg))°, smoothing \(Int(smoothingMs)) ms, "
                    + "beam UNGATED (direction is taken from every ATND notice; "
                    + "our VAD no longer has to agree). Noise the array points at "
                    + "can now become a position — exclusion zones on the device "
                    + "are what keeps that out.")
    }

    /// Position-labeled ranges covering the sub-ranges of `window` that pyannote
    /// has NOT (yet) covered. Empty when the feature is off or ATND was silent.
    func positionGapFill(window: ClosedRange<Double>,
                                 covered: [(start: Double, end: Double, id: Int, name: String)])
        -> [(start: Double, end: Double, id: Int, name: String)] {
        guard let pos = positionDiarizer else { return [] }
        let minGapSec = 0.75   // below this is pyannote boundary slop → filling flickers
        // `covered` is already sorted by start (from speakerRanges). Walk it and
        // emit the complement gaps of `window`.
        var fills: [(start: Double, end: Double, id: Int, name: String)] = []
        var cursor = window.lowerBound
        // Build the ordered list of gap ranges (before/between/after covered ranges).
        var gaps: [(Double, Double)] = []
        for r in covered {
            if r.start > cursor { gaps.append((cursor, r.start)) }
            cursor = max(cursor, r.end)
        }
        if window.upperBound > cursor { gaps.append((cursor, window.upperBound)) }

        // ⚠ THE FLOOR APPLIES ONLY WHERE IT HAS A REASON, and that condition is
        // `covered` being non-empty. The 0.75 s exists to ignore SLOP AROUND A
        // VOICE-TURN BOUNDARY — a sliver between two turns is the diarizer's edge
        // being a little off, not a person. With `covered` empty there is no
        // boundary to be sloppy about: the whole window is one gap because no
        // voice engine labelled it at all, which is the ATND-only case.
        //
        // Applying it there deleted exactly the row the owner photographed. A
        // short realtime utterance ("Fridays to", 0.4 s) is its own segment, its
        // window is the utterance, the single gap is 0.4 s < 0.75 s, so it was
        // skipped, `filled` came back empty and the row rendered SPEAKER UNKNOWN
        // while the array had been streaming angle and rotation throughout.
        // `testNoRowIsUnknownWhileTheArrayIsStreaming` is that row, and it still
        // failed after the smoother fix — which is why this is a second change
        // and not a tidy-up of the first.
        let floorApplies = !covered.isEmpty
        for (a, b) in gaps {
            let dur = b - a
            if floorApplies && dur < minGapSec {
                positionLog("SKIP gap<0.75s [\(fmt3(a))..\(fmt3(b))]")
                continue
            }
            // One fill PER SPAN — a beam change mid-gap splits into multiple rows.
            // The boundary timeline TILES the gap from its first boundary onward:
            // every stretch of elapsed time belongs to whoever the beam had last
            // settled on, so a fill can no longer leave a hole in the middle of a
            // gap. That is the whole point of the switch away from reconstructed
            // turns — the old density gate (minDurationSec/minSamples) DISCARDED a
            // short run, and discarding a run discarded a stretch of TIME, whose
            // words then had no range to land in and rendered as SPEAKER UNKNOWN.
            // Debounce still decides WHETHER a boundary exists, upstream in
            // `ClusterChangeDetector`; it can no longer delete elapsed time.
            //
            // `labeledSpans` already clips to the queried range, so no snapping,
            // hole-closing or dominant-label fallback is needed here any more.
            let spans = pos.labeledSpans(in: a...b)
            if spans.isEmpty {
                // The only way a gap yields nothing now: it lies entirely BEFORE
                // the first beam boundary of the session (pre-speech silence, or
                // ATND not streaming yet). Leaving it UNKNOWN is correct — there is
                // no talker to attribute it to.
                positionLog("SKIP gap=[\(fmt3(a))..\(fmt3(b))] samples=\(pos.sampleCount(in: a...b)) no-spans")
                continue
            }
            for s in spans {
                positionLog("FILL gap=[\(fmt3(a))..\(fmt3(b))] -> \(s.id):\(s.name) [\(fmt3(s.start))..\(fmt3(s.end))]")
                fills.append((s.start, s.end, s.id, s.name))
            }
        }
        // Calibration instrument, not noise: the owner tunes `atnd.position.tauDeg`
        // per room ("the tables differ in every room"), and this is the only line
        // that shows how far apart the seats actually landed — e.g. a phantom
        // cluster 16.2° from Speaker 1 against a 15° threshold while the real seats
        // sit 33–41° apart. Logged only when it CHANGES: `positionGapFill` runs once
        // per segment per rebuild, so logging it unconditionally repeated the same
        // line after every FILL.
        if !fills.isEmpty {
            let tau = UserDefaults.standard.object(forKey: "atnd.position.tauDeg") as? Double ?? 15
            let line = "SEPARATION \(pos.separationDescription()) (threshold \(Int(tau))°)"
            if line != lastLoggedSeparation {
                lastLoggedSeparation = line
                positionLog(line)
            }
        }
        return fills
    }

    /// Append a line to logs/position-diarization.log (position gap-fill decisions).
    /// Mirrors overlapLog: one FILL/SKIP line per display-time gap.
    /// This is the file that proved app-owned logs were unbounded: it reached
    /// 13.2 MB and was still growing on 2026-08-05, because it is written per
    /// display-time gap on every rebuild. `appendAppLog` now rolls it.
    func positionLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "position-diarization", message: message)
    }
}
