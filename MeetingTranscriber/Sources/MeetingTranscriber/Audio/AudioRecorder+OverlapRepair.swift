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

    /// Start overlap repair only once both the last chunk and the diarization
    /// final pass are done, the feature is on, and the selected engine loaded.
    func maybeStartOverlapRepair() {
        let d = UserDefaults.standard
        guard d.object(forKey: "overlap.repair.enabled") as? Bool ?? false else { return }
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
        guard !windows.isEmpty else {
            overlapLog("no 2-speaker overlap windows to repair")
            finishRepairStep(.done); return
        }
        overlapRepairing = true
        overlapRepairError = nil
        setStopStep("repair", .loading)
        overlapLog("starting overlap repair: \(windows.count) window(s), windowSec=\(Int(windowSec))")
        repairTask = Task { @MainActor [weak self] in
            await self?.runOverlapRepair(windows: windows, service: service, recording: recording)
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
                               maxDurationSec: Double? = nil) -> [RepairWindow] {
        // Overlap repair rewrites transcript text under a speaker id — the last
        // place a stray remote id should ever reach.
        let turns = Self.officeTurnsOnly(liveTurns, "repairWindows")
        guard turns.count > 1 else { return [] }

        // Raw windows from pairwise different-speaker overlaps.
        var raw: [(start: Double, end: Double)] = []
        for i in 0..<turns.count {
            for j in (i + 1)..<turns.count where turns[i].id != turns[j].id {
                let a = turns[i].start <= turns[j].start ? turns[i] : turns[j]
                let b = turns[i].start <= turns[j].start ? turns[j] : turns[i]
                let os = max(a.start, b.start)
                let oe = min(a.end, b.end)
                guard oe - os >= 0.4 else { continue }   // genuine overlap only
                // Center the window on the overlap-region midpoint.
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
                                  recording: URL) async {
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
                let text1 = (try? await chunked.transcribeFile(path: tracks[0].path)) ?? ""
                let text2 = (try? await chunked.transcribeFile(path: tracks[1].path)) ?? ""
                processRepair(window: w, tracks: tracks, text1: text1, text2: text2)
            } catch {
                overlapLog("SKIP [\(fmt(w.start))-\(fmt(w.end))] separation failed: \(error.localizedDescription)")
            }
            cleanup(tmpDir)
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
                               text1: String, text2: String) {
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
                    overlapLog("  speaker \(speakerID): COMBINE (run=\(r.longestRun))"
                        + "\n    before: \(anchor)\n    track: \(trackText)\n    after: \(r.text)")
                case .replace:
                    decisions[speakerID] = r.text
                    overlapLog("  speaker \(speakerID): REPLACE (run=\(r.longestRun))"
                        + "\n    before: \(anchor)\n    track: \(trackText)\n    after: \(r.text)")
                }
            }
            if !decisions.isEmpty { applyRepair(ws: ws, we: we, decisions: decisions) }
        }

        // Debug rows: raw MossFormer2 separated-track ASR, verbatim, kept at the end
        // of `segments` so they render after the real transcript. Toggled by the
        // "Show MossFormer2 Index 1/2 rows" setting (Settings → Models → Overlap).
        let showDebug = UserDefaults.standard.object(forKey: "overlap.mossformer.showDebugRows") as? Bool ?? true
        if showDebug {
            let t1 = text1.isEmpty ? "(empty)" : text1
            let t2 = text2.isEmpty ? "(empty)" : text2
            segments.append(TranscriptSegment(text: t1, confirmed: true, window: ws...we,
                                              pinnedSpeakerName: "MossFormer2 Index1",
                                              isSeparationDebug: true))
            segments.append(TranscriptSegment(text: t2, confirmed: true, window: ws...we,
                                              pinnedSpeakerName: "MossFormer2 Index2",
                                              isSeparationDebug: true))
        }
        rebuildDisplayRows()
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
        let code = UserDefaults.standard.string(forKey: "chunked.language") ?? "auto"
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
                    overlapLog("  speaker \(id): COMBINE (run=\(r.longestRun))"
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

    func fmt(_ s: Double) -> String { String(format: "%.1f", s) }
    func fmt3(_ x: Double) -> String { String(format: "%.3f", x) }

    private func cleanup(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Append a line to logs/overlap-repair-decisions.log (mandatory decision logging).
    /// Kept separate from the sidecar's own stderr log (overlap-repair-sidecar.log) —
    /// they used to share one file with no coordination between writers.
    private func overlapLog(_ message: String) {
        let dir = PythonRuntime.dataDir.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("overlap-repair-decisions.log")
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        let stamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let line = "[\(stamp)] \(message)\n"
        guard let handle = try? FileHandle(forWritingTo: file) else { return }
        handle.seekToEndOfFile()
        if let data = line.data(using: .utf8) { handle.write(data) }
        try? handle.close()
    }
}
