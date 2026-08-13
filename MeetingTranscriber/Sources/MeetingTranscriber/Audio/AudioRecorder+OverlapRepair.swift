import Foundation

// Stop-time overlap repair, both engines (MossFormer2 and DiCoW): window
// selection, the two sequential drivers, the splice back into `segments`, and
// this domain's own log writer. The tuned gate constants stay declared in the
// core file (stored properties cannot live in an extension). Moved verbatim —
// no constant, threshold or gate was touched.
extension AudioRecorder {

    // MARK: - Overlap repair (runs at stop; engine per `overlap.engine`)
    //
    // Two engines, same shape: find 2-speaker overlap windows, get one text per
    // speaker for each window, gate it, splice it in. MossFormer2 (attempt #3)
    // separates the waveform then re-ASRs each track; DiCoW (attempt #4) asks a
    // diarization-conditioned Whisper for one speaker at a time. Only one engine
    // loads per session.

    /// A merged overlap window plus the two speakers it touches.
    /// The window is centered on the overlap-region midpoint, ±windowSec.
    private struct RepairWindow {
        var start: Double
        var end: Double
        var speakerIDs: [Int]   // exactly 2
    }

    /// Whether repair takes its overlap spans from the DETECTOR rather than from
    /// intersecting turns.
    ///
    /// Keyed on the ENGINE, not on whether the detector happens to be loaded: the
    /// question is "can this session's diarizer mark overlap itself?", and the
    /// answer is a property of MOSS, spectral and NeMo (one speaker per instant),
    /// not of what else is running. pyannote marks overlap itself, so it keeps
    /// using its own turns and its behaviour is byte-for-byte unchanged — the
    /// detector is a second opinion there, never a substitution.
    ///
    /// **DIARIZEN IS ON PYANNOTE'S SIDE OF THIS LINE, AND IT IS A MEASUREMENT.**
    /// It shipped on the other side (2026-08-10) by analogy with the batch engines
    /// it resembles in every other respect — one whole-file pass, no live labels.
    /// That analogy was wrong on the one question this property actually asks. Its
    /// Conformer head is a POWERSET over speaker combinations, and for this
    /// checkpoint that is **11 classes = 1 silence + 4 single-speaker + 6 PAIRS**
    /// (`powerset_max_classes = 2`, verified on the loaded model). Two people
    /// talking at once is a class the network predicts directly, per 20 ms frame —
    /// not something inferred afterwards from turns that happen to collide. So its
    /// turns really do intersect: measured on `recordings/Overlap123.wav`,
    /// **11 intersecting pairs, 13.30 s**, reproduced with the BUNDLED interpreter
    /// after `./build.sh`.
    ///
    /// The engine's own config agrees: `apply_median_filtering` is **false**, and
    /// measurement shows why it must stay false — the 220 ms kernel halves this
    /// (11 pairs → 6, 12 turns → 7). The checkpoint is tuned to PRESERVE overlap.
    ///
    /// Spectral (Viterbi) and NeMo (NME-SC) remain here for the reason that is
    /// genuinely structural: both assign exactly one label per instant, so
    /// `overlapRegions()` is empty under them however the audio sounds.
    var usesDetectedRegionsForRepair: Bool {
        mossDiarizationActive || spectralDiarizationActive || nemoDiarizationActive
            || camPlusDiarizationActive
    }

    /// Which engine to NAME in the log when repair is skipped for want of the
    /// detector. Derived from the flags rather than a `?:` at the call site: with
    /// three engines a two-way conditional silently mislabels the third, and a log
    /// line that names the wrong engine sends the next debugging session to the
    /// wrong page of Settings.
    ///
    /// DiariZen deliberately has NO case here — it never reaches this log, because
    /// it supplies its own regions (see `usesDetectedRegionsForRepair`). A branch
    /// for it would be unreachable, and an unreachable branch that names an engine
    /// is exactly how the next reader concludes the opposite.
    var batchEngineNameForLog: String {
        if mossDiarizationActive { return "MOSS" }
        if nemoDiarizationActive { return "NEMO" }
        if camPlusDiarizationActive { return "CAM++" }
        return "SPECTRAL"
    }

    /// Start overlap repair only once both the last chunk and the diarization
    /// final pass are done, the feature is on, and the selected engine loaded.
    func maybeStartOverlapRepair() {
        let d = UserDefaults.standard
        guard d.object(forKey: "overlap.repair.enabled") as? Bool ?? false else { return }
        // MOSS and spectral cannot locate overlap in their OWN turns — MOSS's
        // segments tile exactly, spectral assigns one label per frame — so under
        // those engines repair's regions come from the standalone DETECTOR, which
        // reads the audio and needs no turns at all. Without it there is genuinely
        // nothing to repair, and `ModelLoader.wantedOverlapEngine` does not even
        // load an engine, so this must skip rather than fall through to
        // `.failed("engine unavailable")` and blame a missing model for a choice.
        if usesDetectedRegionsForRepair {
            guard modelLoader.overlapDetect != nil else {
                overlapLog("skipped — the \(batchEngineNameForLog) "
                           + "diarization engine assigns exactly one speaker per instant, so "
                           + "overlap repair needs the overlap DETECTOR to locate regions, "
                           + "and it is switched off (Settings → Models → Detect overlap)")
                finishRepairStep(.done)
                return
            }
            // Still scanning. Its completion handler calls back in here — both on
            // success and on failure — so this is a wait, not a bail.
            guard overlapDetectDone else { return }
        }
        guard stopped, finalDiarDone, lastChunkDone else { return }
        guard repairTask == nil, !overlapRepairing else { return }
        guard let recording = lastRecordingURL else { finishRepairStep(.done); return }

        let engine = d.string(forKey: "overlap.engine") ?? ModelCatalog.overlapSeparation.id
        if engine == ModelCatalog.overlapDicow.id {
            guard let service = modelLoader.dicowRepair else {
                finishRepairStep(.failed("engine unavailable")); return
            }
            let windowSec = Double(d.object(forKey: "overlap.dicow.windowSec") as? Int ?? 10)
            // DiCoW's sidecar rejects >30s windows, so drop them here (with a log)
            // rather than burning a request on a guaranteed error.
            let windows = repairWindows(windowSec: windowSec, maxDurationSec: dicowMaxWindowSec)
            guard !windows.isEmpty else {
                overlapLog("DiCoW: no 2-speaker overlap windows to repair")
                finishRepairStep(.done); return
            }
            overlapRepairing = true
            overlapRepairError = nil
            setStopStep("repair", .loading)
            overlapLog("starting overlap repair (DiCoW): \(windows.count) window(s), windowSec=\(Int(windowSec))")
            repairTask = Task { @MainActor [weak self] in
                await self?.runDicowRepair(windows: windows, service: service, recording: recording)
            }
            return
        }

        guard let service = modelLoader.overlapRepair else {
            finishRepairStep(.failed("engine unavailable")); return
        }
        let windowSec = Double(d.object(forKey: "overlap.mossformer.windowSec") as? Int ?? 10)
        let windows = repairWindows(windowSec: windowSec)

        // REMOTE, added 2026-08-13 on measurement. Office was the only stream
        // repaired because separation rescued 0 of 16 real office windows — and
        // that number describes ACOUSTIC mixing in a room, not the remote case.
        // Re-measured on the owner's own remote recordings: 3 of 3 windows
        // separated cleanly (each track matching one real voice at 0.55–0.74 and
        // the other at ≈0), because remote overlap is two independent streams
        // mixed DIGITALLY — no shared room, no reverb, no crosstalk, which is the
        // condition separation models are actually good at.
        //
        // Guarded on the remote RECORDING rather than on `remoteStreamActive`:
        // the file is what the separator is given, and a session can have had a
        // remote channel without a usable file behind it.
        var remoteWindows: [RepairWindow] = []
        let remoteFile = remoteStreamActive ? remoteRecordingURL : nil
        if remoteFile != nil {
            remoteWindows = repairWindows(windowSec: windowSec, remote: true)
        }

        guard !windows.isEmpty || !remoteWindows.isEmpty else {
            overlapLog("no 2-speaker overlap windows to repair")
            finishRepairStep(.done); return
        }
        overlapRepairing = true
        overlapRepairError = nil
        setStopStep("repair", .loading)
        overlapLog("starting overlap repair: \(windows.count) office + "
                   + "\(remoteWindows.count) remote window(s), windowSec=\(Int(windowSec))")
        var jobs: [(windows: [RepairWindow], recording: URL, remote: Bool)] =
            [(windows, recording, false)]
        if let remoteFile { jobs.append((remoteWindows, remoteFile, true)) }
        repairTask = Task { @MainActor [weak self] in
            await self?.runOverlapRepairJobs(jobs, service: service)
        }
    }

    /// Repair bailed out before a driver ever started, so nothing else will settle
    /// the overlay's repair row — do it here and re-check the gate.
    private func finishRepairStep(_ state: ModelLoader.ItemState) {
        setStopStep("repair", state)
        checkStopProcessingDone()
    }

    /// Merged 2-speaker overlap windows to repair, in chronological order.
    /// Each raw window is centered on the overlap-region midpoint, ±windowSec.
    /// `maxDurationSec` (DiCoW only; nil = no limit, i.e. MossFormer2's original
    /// behaviour) drops merged windows longer than the engine can accept.
    private func repairWindows(windowSec: Double,
                               maxDurationSec: Double? = nil,
                               remote: Bool = false) -> [RepairWindow] {
        // Overlap repair rewrites transcript text under a speaker id — the last
        // place an id from the WRONG SPACE should ever reach. So each stream is
        // asked of the collection that actually holds its turns, and the assert
        // names the space it expects: office ids are < 10 000, remote ids are
        // >= 10 000, and a mix-up would splice one stream's words under the other
        // stream's speaker. MOSS keeps its turns in their own collection and never
        // in `liveTurns`, and has no remote path at all.
        let turns: [SpeakerTurn]
        if remote {
            turns = Self.remoteTurnsOnly(remoteLiveTurns, "repairWindows(remote)")
        } else {
            turns = mossDiarizationActive
                ? mossTurns
                : Self.officeTurnsOnly(liveTurns, "repairWindows")
        }
        guard turns.count > 1 else { return [] }

        // Where the overlap SPANS come from. Under pyannote and DiariZen: pairs of
        // turns that intersect. Under the engines that assign one label per instant:
        // the standalone detector, which read the audio itself and is run PER STREAM.
        // `maybeStartOverlapRepair` has already proved the detector ran, so an empty
        // list here means it found no overlap — not that it is missing.
        var spans: [(start: Double, end: Double)] = []
        if usesDetectedRegionsForRepair {
            // The 0.4 s genuine-overlap bar is applied to both sources, so the two
            // paths admit the same thing. The detector's own floor is 0.20 s.
            let detected = remote ? remoteDetectedOverlapRegions : detectedOverlapRegions
            spans = detected.filter { $0.end - $0.start >= 0.4 }
            overlapLog("\(remote ? "REMOTE " : "")regions from the overlap DETECTOR: "
                       + "\(spans.count) of \(detected.count) are >= 0.4s")
        } else {
            for i in 0..<turns.count {
                for j in (i + 1)..<turns.count where turns[i].id != turns[j].id {
                    let a = turns[i].start <= turns[j].start ? turns[i] : turns[j]
                    let b = turns[i].start <= turns[j].start ? turns[j] : turns[i]
                    let os = max(a.start, b.start)
                    let oe = min(a.end, b.end)
                    guard oe - os >= 0.4 else { continue }   // genuine overlap only
                    spans.append((os, oe))
                }
            }
        }

        // Raw windows: each overlap span centered in ±windowSec of context.
        var raw: [(start: Double, end: Double)] = []
        for (os, oe) in spans {
            let mid = (os + oe) / 2
            var ws = max(0, mid - windowSec)
            var we = min(recordingElapsed, mid + windowSec)
            ws = min(ws, os)          // keep the full overlap span inside (long overlaps)
            we = max(we, oe)
            we = min(we, recordingElapsed)
            if we - ws >= 2.0 {
                raw.append((ws, we))
                overlapLog("window [\(fmt(ws))-\(fmt(we))] centered on overlap midpoint \(fmt(mid)) (overlap [\(fmt(os))-\(fmt(oe))])")
            }
        }
        guard !raw.isEmpty else { return [] }

        // Merge intersecting ranges so a later repair never clobbers an earlier one.
        raw.sort { $0.start < $1.start }
        var merged: [(start: Double, end: Double)] = []
        for r in raw {
            if var last = merged.last, r.start <= last.end {
                last.end = max(last.end, r.end)
                merged[merged.count - 1] = last
            } else {
                merged.append(r)
            }
        }

        // Attach the distinct speakers each merged window touches; skip 3+, and
        // skip anything past the engine's window limit.
        var result: [RepairWindow] = []
        for m in merged {
            if let cap = maxDurationSec, m.end - m.start > cap {
                overlapLog("SKIP window [\(fmt(m.start))-\(fmt(m.end))] is "
                    + "\(fmt(m.end - m.start))s, over this engine's \(fmt(cap))s limit")
                continue
            }
            var ids: [Int] = []
            for t in turns where min(t.end, m.end) - max(t.start, m.start) > 0 {
                if !ids.contains(t.id) { ids.append(t.id) }
            }
            if ids.count == 2 {
                result.append(RepairWindow(start: m.start, end: m.end, speakerIDs: ids))
            } else {
                overlapLog("SKIP window [\(fmt(m.start))-\(fmt(m.end))] touches \(ids.count) speakers (only 2 supported)")
            }
        }
        return result
    }

    /// Sequential driver: one window fully completes (through the UI update)
    /// before the next begins. A per-window failure logs and continues.
    private func runOverlapRepair(windows: [RepairWindow],
                                  service: OverlapRepairService,
                                  recording: URL,
                                  remote: Bool = false) async {
        let n = windows.count
        for (i, w) in windows.enumerated() {
            if Task.isCancelled { break }   // a new session owns the transcript now
            overlapRepairProgress = "\(i + 1)/\(n)"
            let tmpDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("moss-\(UUID().uuidString)")
            do {
                let tracks = try await service.separate(audio: recording,
                                                        start: w.start, end: w.end,
                                                        outDir: tmpDir)
                guard tracks.count == 2 else {
                    overlapLog("SKIP [\(fmt(w.start))-\(fmt(w.end))] separator returned \(tracks.count) tracks")
                    cleanup(tmpDir); continue
                }
                guard let chunked = modelLoader.chunkedASR else {
                    overlapLog("SKIP — chunked ASR not available for re-transcription")
                    cleanup(tmpDir); break
                }
                // The confidence is deliberately DISCARDED here: it describes the
                // re-ASR of a SEPARATED track, not the original mixed audio the
                // row's words came from, so showing it against a repaired row
                // would attach a number to the wrong thing.
                let text1 = (try? await chunked.transcribeFile(path: tracks[0].path).text) ?? ""
                let text2 = (try? await chunked.transcribeFile(path: tracks[1].path).text) ?? ""
                processRepair(window: w, tracks: tracks, text1: text1, text2: text2,
                              remote: remote)
            } catch {
                overlapLog("SKIP [\(fmt(w.start))-\(fmt(w.end))] separation failed: \(error.localizedDescription)")
            }
            cleanup(tmpDir)
        }
    }

    /// Run every stream's windows, then settle the overlay's repair row ONCE.
    ///
    /// The finishing used to live at the end of `runOverlapRepair`, which was
    /// correct while there was exactly one stream. With two it would settle the
    /// gate after Office and leave Remote's edits landing on a transcript the user
    /// had already been handed — so the loop and the settling are now separate
    /// jobs, and only this function may settle.
    private func runOverlapRepairJobs(_ jobs: [(windows: [RepairWindow], recording: URL,
                                                remote: Bool)],
                                      service: OverlapRepairService) async {
        for job in jobs where !job.windows.isEmpty {
            if Task.isCancelled { break }
            overlapLog("\(job.remote ? "REMOTE" : "OFFICE"): \(job.windows.count) window(s)")
            await runOverlapRepair(windows: job.windows, service: service,
                                   recording: job.recording, remote: job.remote)
        }
        overlapRepairing = false
        overlapRepairProgress = nil
        repairTask = nil
        setStopStep("repair", .done)
        checkStopProcessingDone()
        overlapLog("overlap repair complete")
    }

    /// Fold each separated track's re-ASR text into its attributed speaker with
    /// COMBINE-OR-REPLACE semantics (via `TranscriptMerge`), then ALSO append the
    /// two raw "MossFormer2 Index N" debug rows. A track that adds no new words is
    /// a NO-OP (its speaker's rows are left untouched); near-duplicate tracks skip
    /// the speaker-row edit entirely. Every decision is logged.
    private func processRepair(window w: RepairWindow,
                               tracks: [OverlapRepairService.SeparatedTrack],
                               text1: String, text2: String,
                               remote: Bool = false) {
        let ws = w.start, we = w.end
        let idA = w.speakerIDs[0], idB = w.speakerIDs[1]
        let anchorA = anchorText(for: idA, ws: ws, we: we)
        let anchorB = anchorText(for: idB, ws: ws, we: we)

        // Attribution 2×2: send each track to whichever speaker it best matches.
        let j11 = jaccard(text1, anchorA), j12 = jaccard(text1, anchorB)
        let j21 = jaccard(text2, anchorA), j22 = jaccard(text2, anchorB)
        let straight = (j11 + j22) >= (j12 + j21)
        let t1Speaker = straight ? idA : idB
        let t1Anchor  = straight ? anchorA : anchorB
        let t2Speaker = straight ? idB : idA
        let t2Anchor  = straight ? anchorB : anchorA

        overlapLog("REPAIR [\(fmt(ws))-\(fmt(we))] jaccard "
            + "t1·A=\(fmt3(j11)) t1·B=\(fmt3(j12)) t2·A=\(fmt3(j21)) t2·B=\(fmt3(j22)) → "
            + "\(straight ? "straight" : "swapped"): track1→speaker \(t1Speaker), track2→speaker \(t2Speaker)")

        // Decide per-speaker merges (skipped wholesale if the tracks look duplicated).
        var decisions: [Int: String] = [:]
        let dup = jaccard(text1, text2)
        if dup > nearDuplicateJaccard {
            overlapLog("  SKIP near-duplicate tracks (jaccard=\(fmt3(dup)) > \(nearDuplicateJaccard)) — speaker rows unchanged")
        } else {
            for (trackText, speakerID, anchor) in [(text1, t1Speaker, t1Anchor),
                                                   (text2, t2Speaker, t2Anchor)] {
                if trackText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    overlapLog("  speaker \(speakerID): NO-OP (empty track)")
                    continue
                }
                let r = TranscriptMerge.merge(existing: anchor, track: trackText)
                switch r.kind {
                case .noop:
                    overlapLog("  speaker \(speakerID): NO-OP (track ⊆ existing, run=\(r.longestRun))"
                        + "\n    existing: \(anchor)\n    track: \(trackText)")
                case .combine:
                    decisions[speakerID] = r.text
                    overlapLog("  speaker \(speakerID): COMBINE (run=\(r.longestRun), "
                        + "inserted=\(r.inserted)/\(trackTokenCount(trackText)) track tokens)"
                        + "\n    before: \(anchor)\n    track: \(trackText)\n    after: \(r.text)")
                case .replace:
                    decisions[speakerID] = r.text
                    overlapLog("  speaker \(speakerID): REPLACE (run=\(r.longestRun))"
                        + "\n    before: \(anchor)\n    track: \(trackText)\n    after: \(r.text)")
                }
            }
            if !decisions.isEmpty {
                if remote { applyRemoteRepair(ws: ws, we: we, decisions: decisions) }
                else      { applyRepair(ws: ws, we: we, decisions: decisions) }
            }
        }

        // Debug rows: raw MossFormer2 separated-track ASR, verbatim, kept at the end
        // of the transcript so they render after it. Toggled by the
        // "Show MossFormer2 Index 1/2 rows" setting (Settings → Models → Overlap).
        let showDebug = UserDefaults.standard.object(forKey: "overlap.mossformer.showDebugRows") as? Bool ?? true
        if showDebug {
            let t1 = text1.isEmpty ? "(empty)" : text1
            let t2 = text2.isEmpty ? "(empty)" : text2
            // The label names the STREAM as well as the track. With both streams
            // repairing, four such rows can share one window, and "Index1" twice
            // over would make them impossible to tell apart in the one place they
            // exist to be read.
            let tag = remote ? "MossFormer2 Remote Index" : "MossFormer2 Index"
            if remote {
                remoteSegments.append(RemoteSegment(text: t1, window: ws...we,
                                                    pinnedSpeakerName: "\(tag)1",
                                                    isSeparationDebug: true))
                remoteSegments.append(RemoteSegment(text: t2, window: ws...we,
                                                    pinnedSpeakerName: "\(tag)2",
                                                    isSeparationDebug: true))
            } else {
                segments.append(TranscriptSegment(text: t1, confirmed: true, window: ws...we,
                                                  pinnedSpeakerName: "\(tag)1",
                                                  isSeparationDebug: true))
                segments.append(TranscriptSegment(text: t2, confirmed: true, window: ws...we,
                                                  pinnedSpeakerName: "\(tag)2",
                                                  isSeparationDebug: true))
            }
        }
        rebuildDisplayRows()
    }

    /// The Remote twin of `applyRepair`, over `remoteSegments`.
    ///
    /// SIMPLER than the office one, and the reason is structural rather than a
    /// shortcut: a remote row is one chunk's text split by turns at display time
    /// and nothing else writes into `remoteSegments`, so there are no ATND
    /// position fills, no MOSS pinned rows and no realtime segments to preserve.
    /// What both versions guarantee is the same invariant — **each affected row's
    /// text survives exactly once**, either folded into a repaired speaker's text
    /// or preserved verbatim.
    /// Internal rather than private ON PURPOSE: this is the function that rewrites
    /// remote transcript text, and its "each row's text survives exactly once"
    /// invariant is the one worth a test rather than a comment.
    func applyRemoteRepair(ws: Double, we: Double, decisions: [Int: String]) {
        let regions = remoteOverlapRegions() + remoteDetectedOverlapRegions
        let affected = remoteSegments.indices.filter { i in
            let s = remoteSegments[i]
            // Repair runs at Stop, over CONFIRMED text only. An unconfirmed
            // realtime row is about to be replaced by the chunked pass anyway, and
            // folding recovered words into it would splice them into text with no
            // future.
            guard s.confirmed, !s.isSeparationDebug else { return false }
            return min(s.window.upperBound, we) - max(s.window.lowerBound, ws) > 0
        }

        var preserved: [RemoteSegment] = []
        var consumedSpan: [Int: (lo: Double, hi: Double)] = [:]

        for i in affected {
            for row in Self.remoteRows([remoteSegments[i]], turns: remoteLiveTurns,
                                       regions: regions) {
                let rs = row.start ?? remoteSegments[i].window.lowerBound
                let re = row.end ?? remoteSegments[i].window.upperBound
                let inWindow = min(re, we) - max(rs, ws) > 0
                if inWindow, let sid = row.speakerID, decisions[sid] != nil {
                    let cur = consumedSpan[sid]
                    consumedSpan[sid] = (min(cur?.lo ?? rs, rs), max(cur?.hi ?? re, re))
                } else {
                    // PRESERVE verbatim. The name is stored WITHOUT the
                    // "Remote Speaker - " prefix `remoteRows` adds at render time,
                    // or the next render would prefix it twice.
                    preserved.append(RemoteSegment(text: row.text,
                                                   window: rs...max(re, rs),
                                                   pinnedSpeakerID: row.speakerID,
                                                   pinnedSpeakerName:
                                                    row.speaker.map(Self.remoteBaseName)))
                }
            }
        }

        for i in affected.sorted(by: >) { remoteSegments.remove(at: i) }

        let repaired: [RemoteSegment] = decisions.map { id, text in
            let span = consumedSpan[id] ?? (ws, we)
            return RemoteSegment(text: text, window: span.lo...max(span.hi, span.lo),
                                 pinnedSpeakerID: id,
                                 pinnedSpeakerName: remoteSpeakerName(for: id))
        }
        remoteSegments.append(contentsOf: preserved + repaired)
        // `remoteRows` sorts by window, so no explicit re-sort is needed here —
        // unlike `insertPinnedSorted`, which orders `segments` itself.
    }

    /// Display name for a REMOTE speaker id, from this session's remote turns.
    /// Its own function rather than a parameter on `speakerName(for:)`: the two
    /// read different collections, and one function reading either would be one
    /// wrong argument away from naming a room speaker on a remote row.
    private func remoteSpeakerName(for id: Int) -> String? {
        remoteLiveTurns.first(where: { $0.id == id })?.name
    }

    // MARK: - Overlap repair — DiCoW (attempt #4)

    /// Sequential driver: one window fully completes (through the UI update)
    /// before the next begins. A per-window failure logs and continues.
    /// Mirrors `runOverlapRepair`, but there is no separation step — DiCoW is
    /// asked directly for one transcription per speaker.
    private func runDicowRepair(windows: [RepairWindow],
                                service: DicowService,
                                recording: URL) async {
        // Same language the chunked pass uses; "auto" → let DiCoW detect.
        // Resolved through the SELECTED chunked model exactly as
        // `ChunkedASRService.Config.fromSettings` does — otherwise "same as the
        // chunked pass" quietly stops being true whenever the stored code is one
        // that model cannot take (it runs on auto, DiCoW would not).
        let code = Languages.resolve(
            language: UserDefaults.standard.string(forKey: "chunked.language") ?? "auto",
            forModel: ChunkedASRModelFactory.fromSettings().info.id)
        let language: String? = code == "auto" ? nil : code

        let n = windows.count
        for (i, w) in windows.enumerated() {
            if Task.isCancelled { break }   // a new session owns the transcript now
            overlapRepairProgress = "\(i + 1)/\(n)"
            let ranges = speakerRanges(in: w.start...w.end)
            let targets = w.speakerIDs.map { id in
                DicowService.Target(
                    sid: id,
                    turns: ranges.filter { $0.id == id }.map { (start: $0.start, end: $0.end) })
            }
            guard targets.allSatisfy({ !$0.turns.isEmpty }) else {
                overlapLog("DiCoW SKIP [\(fmt(w.start))-\(fmt(w.end))] a speaker has no turns in-window")
                continue
            }
            do {
                let results = try await service.transcribeTargets(
                    audio: recording, start: w.start, end: w.end,
                    targets: targets, language: language)
                processDicowRepair(window: w, targets: targets, results: results)
            } catch {
                overlapLog("DiCoW SKIP [\(fmt(w.start))-\(fmt(w.end))] failed: \(error.localizedDescription)")
            }
        }
        overlapRepairing = false
        overlapRepairProgress = nil
        repairTask = nil
        setStopStep("repair", .done)
        checkStopProcessingDone()
        overlapLog("DiCoW overlap repair complete")
    }

    /// Gate DiCoW's per-speaker texts, then fold the survivors into their speaker
    /// with COMBINE-OR-REPLACE semantics (via `TranscriptMerge`), exactly like the
    /// MossFormer2 path. No 2×2 attribution is needed — DiCoW is already
    /// conditioned on each speaker's mask, so a text's speaker is known up front.
    ///
    /// The gates exist because of how attempt #2 (2026-07-14) failed; each one
    /// targets a specific observed failure:
    ///   • near-duplicate texts   → both masks picked up the same voice
    ///   • anchor cross-check     → cross-speaker leakage
    ///   • word-density ceiling   → runaway spans (the "40+ word" bug)
    /// A gated speaker keeps its original, honest text.
    private func processDicowRepair(window w: RepairWindow,
                                    targets: [DicowService.Target],
                                    results: [DicowService.TargetResult]) {
        let ws = w.start, we = w.end
        let idA = w.speakerIDs[0], idB = w.speakerIDs[1]
        func text(_ id: Int) -> String { results.first { $0.sid == id }?.text ?? "" }
        let textA = text(idA), textB = text(idB)
        let anchorA = anchorText(for: idA, ws: ws, we: we)
        let anchorB = anchorText(for: idB, ws: ws, we: we)

        overlapLog("DiCoW REPAIR [\(fmt(ws))-\(fmt(we))] speakers \(idA) & \(idB)"
            + "\n    speaker \(idA): \(textA.isEmpty ? "(empty)" : textA)"
            + "\n    speaker \(idB): \(textB.isEmpty ? "(empty)" : textB)")

        var decisions: [Int: String] = [:]
        // Gate: near-duplicate → the two masks resolved to the same voice.
        let dup = jaccard(textA, textB)
        if !textA.isEmpty, !textB.isEmpty, dup > nearDuplicateJaccard {
            overlapLog("  SKIP near-duplicate texts (jaccard=\(fmt3(dup)) > \(nearDuplicateJaccard)) — speaker rows unchanged")
        } else {
            for (id, txt, anchor, otherAnchor) in [(idA, textA, anchorA, anchorB),
                                                   (idB, textB, anchorB, anchorA)] {
                if txt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    overlapLog("  speaker \(id): NO-OP (empty text)")
                    continue
                }
                // Gate: cross-speaker leakage — the text matches the OTHER
                // speaker's existing rows clearly better than its own.
                let jOwn = jaccard(txt, anchor), jOther = jaccard(txt, otherAnchor)
                if jOther > jOwn + anchorLeakMargin {
                    overlapLog("  speaker \(id): SKIP cross-speaker leakage "
                        + "(own=\(fmt3(jOwn)) vs other=\(fmt3(jOther)), margin \(anchorLeakMargin))")
                    continue
                }
                // Gate: word density — more words than this speaker had time to say.
                let spoken = targets.first { $0.sid == id }?.turns
                    .reduce(0.0) { $0 + ($1.end - $1.start) } ?? 0
                let words = txt.split { !$0.isLetter && !$0.isNumber }.count
                let density = spoken > 0 ? Double(words) / spoken : .infinity
                if density > maxWordsPerSecond {
                    overlapLog("  speaker \(id): SKIP word density \(fmt3(density)) w/s "
                        + "(\(words) words over \(fmt(spoken))s of turns) > \(fmt(maxWordsPerSecond))")
                    continue
                }
                let r = TranscriptMerge.merge(existing: anchor, track: txt)
                switch r.kind {
                case .noop:
                    overlapLog("  speaker \(id): NO-OP (text ⊆ existing, run=\(r.longestRun))"
                        + "\n    existing: \(anchor)\n    dicow: \(txt)")
                case .combine:
                    decisions[id] = r.text
                    overlapLog("  speaker \(id): COMBINE (run=\(r.longestRun), "
                        + "inserted=\(r.inserted)/\(trackTokenCount(txt)) track tokens)"
                        + "\n    before: \(anchor)\n    dicow: \(txt)\n    after: \(r.text)")
                case .replace:
                    decisions[id] = r.text
                    overlapLog("  speaker \(id): REPLACE (run=\(r.longestRun))"
                        + "\n    before: \(anchor)\n    dicow: \(txt)\n    after: \(r.text)")
                }
            }
            if !decisions.isEmpty { applyRepair(ws: ws, we: we, decisions: decisions) }
        }

        // Debug rows: DiCoW's raw per-speaker output, verbatim and ungated, kept at
        // the end of `segments` so they render after the real transcript. Toggled by
        // "Show DiCoW per-speaker rows" (Settings → Models → Overlap). Shown even
        // when a gate skipped the text — seeing what was rejected is the point.
        let showDebug = UserDefaults.standard.object(forKey: "overlap.dicow.showDebugRows") as? Bool ?? true
        if showDebug {
            for (id, txt) in [(idA, textA), (idB, textB)] {
                segments.append(TranscriptSegment(
                    text: txt.isEmpty ? "(empty)" : txt, confirmed: true, window: ws...we,
                    pinnedSpeakerName: "DiCoW · \(speakerName(for: id) ?? "Speaker \(id)")",
                    isSeparationDebug: true))
            }
        }
        rebuildDisplayRows()
    }

    /// Write the repaired speakers' merged text back into `segments` WITHOUT
    /// duplicating any text. For every source segment intersecting [ws,we], its
    /// derived rows are either CONSUMED (a repaired speaker's rows in-window — their
    /// text is already folded into the merged `decisions` text) or PRESERVED verbatim
    /// as a pinned segment (every other row). The consumed rows collapse into one
    /// pinned segment per repaired speaker. Invariant: each row's text survives
    /// exactly once — folded or preserved, never split, never duplicated.
    private func applyRepair(ws: Double, we: Double, decisions: [Int: String]) {
        let regions = overlapRegions()
        let affectedIdx = segments.indices.filter { i in
            let s = segments[i]
            guard s.confirmed, !s.isSeparationDebug, let w = s.window else { return false }
            return min(w.upperBound, we) - max(w.lowerBound, ws) > 0
        }

        // No existing rows to fold into — emit repaired speakers as fresh pinned rows.
        guard !affectedIdx.isEmpty else {
            overlapLog("  applyRepair: no source segments intersect window — inserting repaired rows fresh")
            insertPinnedSorted(decisions.map { id, text in
                TranscriptSegment(text: text, confirmed: true, window: ws...we,
                                  pinnedSpeakerID: id, pinnedSpeakerName: speakerName(for: id))
            })
            return
        }

        var preserved: [TranscriptSegment] = []
        var consumedSpan: [Int: (lo: Double, hi: Double)] = [:]   // per repaired speaker

        for i in affectedIdx {
            for row in derivedRows(for: segments[i], regions: regions) {
                let rs = row.start ?? segments[i].window?.lowerBound ?? ws
                let re = row.end ?? segments[i].window?.upperBound ?? we
                let inWindow = min(re, we) - max(rs, ws) > 0
                if inWindow, let sid = row.speakerID, decisions[sid] != nil {
                    // CONSUMED — folded into this speaker's merged text.
                    let cur = consumedSpan[sid]
                    consumedSpan[sid] = (min(cur?.lo ?? rs, rs), max(cur?.hi ?? re, re))
                } else {
                    // PRESERVE verbatim (other speakers, or this speaker outside window).
                    preserved.append(TranscriptSegment(text: row.text, confirmed: true,
                                                       window: rs...max(re, rs),
                                                       pinnedSpeakerID: row.speakerID,
                                                       pinnedSpeakerName: row.speaker))
                }
            }
        }

        // Drop the affected source segments (descending, so indices stay valid).
        for i in affectedIdx.sorted(by: >) { segments.remove(at: i) }

        // One pinned segment per repaired speaker, spanning its consumed rows.
        let repaired: [TranscriptSegment] = decisions.map { id, text in
            let span = consumedSpan[id] ?? (ws, we)
            return TranscriptSegment(text: text, confirmed: true,
                                     window: span.lo...max(span.hi, span.lo),
                                     pinnedSpeakerID: id,
                                     pinnedSpeakerName: speakerName(for: id))
        }
        insertPinnedSorted(preserved + repaired)
    }

    /// Insert new (non-debug) segments and re-sort the real transcript by start time,
    /// keeping any separation-debug segments pinned at the end.
    private func insertPinnedSorted(_ newSegs: [TranscriptSegment]) {
        var real = segments.filter { !$0.isSeparationDebug }
        let debug = segments.filter { $0.isSeparationDebug }
        real.append(contentsOf: newSegs)
        real.sort { ($0.window?.lowerBound ?? .greatestFiniteMagnitude)
                  < ($1.window?.lowerBound ?? .greatestFiniteMagnitude) }
        segments = real + debug
    }

    /// Display name for a speaker id, from the diarization turns collected so far.
    private func speakerName(for id: Int) -> String? {
        liveTurns.first(where: { $0.id == id })?.name
    }

    /// Existing display-row text for one speaker intersecting [ws, we],
    /// the anchor for text-based track attribution.
    private func anchorText(for id: Int, ws: Double, we: Double) -> String {
        displayRows.filter { row in
            guard row.speakerID == id, let s = row.start, let e = row.end else { return false }
            return min(e, we) - max(s, ws) > 0
        }.map(\.text).joined(separator: " ")
    }

    /// Jaccard similarity of the two texts' lowercased alnum word sets.
    private func jaccard(_ a: String, _ b: String) -> Double {
        func tokens(_ s: String) -> Set<String> {
            Set(s.lowercased().split { !$0.isLetter && !$0.isNumber }
                .map(String.init).filter { !$0.isEmpty })
        }
        let ta = tokens(a), tb = tokens(b)
        if ta.isEmpty && tb.isEmpty { return 1.0 }
        let uni = ta.union(tb).count
        return uni == 0 ? 0.0 : Double(ta.intersection(tb).count) / Double(uni)
    }

    /// Whitespace-split token count — the denominator of a COMBINE's
    /// `inserted=N/M`. N/M says how much of the track actually landed in the
    /// transcript, which is the number that tells a genuine one-word recovery
    /// apart from a wholesale re-wording.
    private func trackTokenCount(_ s: String) -> Int {
        s.split(whereSeparator: { $0.isWhitespace }).count
    }

    func fmt(_ s: Double) -> String { String(format: "%.1f", s) }
    func fmt3(_ x: Double) -> String { String(format: "%.3f", x) }

    private func cleanup(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Append a line to logs/overlap-repair-decisions.log (mandatory decision logging).
    /// Kept separate from each engine's own sidecar stderr log (logs/mossformer2.log,
    /// logs/dicow.log) — they used to share one file with no coordination between
    /// writers. This one is SWIFT-owned and deliberately SHARED by both engines, so
    /// it is NOT a `scripts/<name>/ → logs/<name>.log` service log and is exempt from
    /// that rule (which `layout/*` in sidecar-tests.py pins for the 13 that are).
    private func overlapLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "overlap-repair-decisions", message: message)
    }
}
