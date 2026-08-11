import Foundation

// Derivation of the rendered transcript rows from the raw ASR segments and the
// diarization turns — the office pipeline, its pure sentence/range helpers, and
// the remote-row + merge steps that join the two identity spaces at display
// time. Moved verbatim from the core file.
extension AudioRecorder {

    // MARK: - Display rows (speaker · time · text)

    /// Rebuild the rendered transcript from the raw ASR segments and the
    /// diarization turns collected so far. Each confirmed chunk is split into
    /// one row per speaker turn; undiarized/realtime text stays a single row.
    func rebuildDisplayRows() {
        var rows: [SpeakerUtterance] = []
        // Pyannote-derived regions PLUS anything the standalone detector found.
        // The union is used for DISPLAY only; overlap REPAIR still reads
        // `overlapRegions()` alone, because its engines were built against
        // pyannote's intersecting turns and feeding them detector output would
        // change what they operate on without a single measurement behind it.
        let regions = overlapRegions() + detectedOverlapRegions
        for seg in segments {
            rows.append(contentsOf: derivedRows(for: seg, regions: regions))
        }
        // Office and Remote are two independent label spaces sharing one clock;
        // they only ever meet here, at the final sort. With no remote segments
        // the merge returns `rows` untouched — the single-stream path is exactly
        // what it was.
        let merged = Self.mergeRowsByStartTime(
            office: rows,
            remote: Self.remoteRows(remoteSegments, turns: remoteLiveTurns))
        // Relabel sub-second speaker "islands" first (an ATND beam flicker that
        // caught one word), THEN coalesce — because relabeling restores the
        // adjacency that lets the surrounding rows merge.
        let settled = Self.coalesceAdjacentSameSpeaker(
            Self.absorbShortSpeakerIslands(merged, mossSession: mossDiarizationActive))
        // Speaker confidence is annotated LAST, once the spans are final: the
        // number is a minimum over the turns a row spans, and coalescing changes
        // exactly that span. Computing it earlier would describe a row that no
        // longer exists.
        displayRows = Self.annotateSpeakerConfidence(rows: settled,
                                                     officeTurns: liveTurns,
                                                     remoteTurns: remoteLiveTurns)
    }

    /// Attach each row's speaker confidence: the MINIMUM matched cosine over the
    /// turns of that same speaker overlapping the row's final span.
    ///
    /// Minimum, not mean: the row is claiming one identity for its whole span, so
    /// it is worth no more than the weakest evidence inside it. Averaging would
    /// let one strong turn mask a marginal one, which is the direction that
    /// misleads (overstated confidence is invisible; understated is merely
    /// cautious — the same asymmetry the hallucination gates are tuned around).
    ///
    /// `nil` is left in place for every legitimate "not measured" case and is
    /// NEVER replaced with 0: a turn the sidecar sent without a `conf` (a
    /// brand-new profile's first appearance) is skipped rather than counted as
    /// zero, an ATND position id (≥ 100 000) simply matches no pyannote turn, and
    /// unconfirmed realtime rows are not annotated at all.
    ///
    /// Reads EXACTLY ONE turn collection per row, chosen by `isRemote` — the same
    /// one-comparison id-space routing used everywhere else. Office and remote ids
    /// are disjoint by construction, so this is belt-and-braces; it is spelled out
    /// because a row taking its confidence from the other space's turns would be a
    /// silent, plausible-looking wrong number.
    nonisolated static func annotateSpeakerConfidence(
        rows: [SpeakerUtterance],
        officeTurns: [SpeakerTurn],
        remoteTurns: [SpeakerTurn]) -> [SpeakerUtterance] {
        rows.map { row in
            var row = row
            guard row.confirmed, let id = row.speakerID,
                  let start = row.start, let end = row.end
            else { return row }
            let turns = row.isRemote ? remoteTurns : officeTurns
            var lowest: Double?
            for turn in turns where turn.id == id {
                guard let conf = turn.conf else { continue }  // not measured
                // Genuine overlap only. A zero-width row touches nothing and so
                // shows no number — the safe direction.
                guard min(turn.end, end) > max(turn.start, start) else { continue }
                lowest = lowest.map { Swift.min($0, conf) } ?? conf
            }
            row.speakerConf = lowest
            return row
        }
    }

    /// The lower of two confidences, treating `nil` as "no measurement" rather
    /// than as a low value: it neither wins nor erases the other side.
    nonisolated static func minConfidence(_ a: Double?, _ b: Double?) -> Double? {
        guard let a else { return b }
        guard let b else { return a }
        return Swift.min(a, b)
    }

    /// Longest a mislabelled "island" row may be to get absorbed. An ATND beam
    /// flicker onto a neighbour's seat catches a word or two (~0.2 s here); a
    /// genuine mid-sentence interjection is rarely this brief, and never labelled
    /// by a zero-width position span the way a flicker is.
    nonisolated static let islandMaxDurationSec = 0.8

    /// Reassign a very short row to its neighbours when the rows on BOTH sides
    /// are the same *other* speaker.
    ///
    /// The word-attribution log showed `Speaker 2@92.6 Speaker 1@94.120
    /// Speaker 2@94.120` — a Speaker-1 span starting at the same instant the next
    /// Speaker-2 span does, i.e. essentially zero width, that captured the single
    /// word "to" in the middle of one Speaker-2 sentence. The word is real and
    /// correctly placed in time; only its speaker is wrong. Handing it to the
    /// speaker surrounding it on both sides fixes the label and lets the sentence
    /// rejoin. Guarded tightly so a genuine short interjection is not swallowed:
    /// the island must be shorter than `islandMaxDurationSec`, confirmed, same
    /// stream, and flanked by the SAME speaker with a non-nil id on each side.
    ///
    /// `mossSession` is what exempts MOSS since 2026-08-05, replacing an id-range
    /// test. Once a MOSS session's turns go through `identify`, their ids are
    /// PROFILE ids (< 10 000) and no longer sit above `mossIDBase` — so the old
    /// `(island.speakerID ?? 0) < mossIDBase` test stopped exempting anything and
    /// would have started absorbing exactly the rows it was written to protect.
    /// The session is the honest signal anyway: in a MOSS session `liveTurns` is
    /// empty and EVERY row comes from `mossTurns`, so there is no ATND-flicker
    /// case here for the absorber to fix.
    nonisolated static func absorbShortSpeakerIslands(
        _ rows: [SpeakerUtterance], mossSession: Bool = false) -> [SpeakerUtterance] {
        guard rows.count >= 3, !mossSession else { return rows }
        var out = rows
        for i in 1..<(out.count - 1) {
            let island = out[i], before = out[i - 1], after = out[i + 1]
            guard let flank = before.speakerID, after.speakerID == flank,
                  island.speakerID != flank,
                  // Never override MOSS. This absorber exists for ONE failure:
                  // an ATND beam flicker onto a neighbour's seat, which produces
                  // a near-zero-width span whose timing is right and whose
                  // identity is wrong. A MOSS row is the opposite case — the
                  // model attributed those words from the audio itself, and a
                  // short reply between two turns of one speaker ("Hi.", "Yeah.")
                  // is the single most common shape a real conversation has.
                  // Rewriting it from its neighbours would silently delete the
                  // one thing this engine is selected for. Measured: a genuine
                  // 0.8 s "Hi." between two turns of S1 was being absorbed and
                  // then coalesced away entirely.
                  // (the MOSS exemption moved to `mossSession` above — see there)
                  island.confirmed, before.confirmed, after.confirmed,
                  island.isRemote == before.isRemote, island.isRemote == after.isRemote,
                  let s = island.start, let e = island.end,
                  e - s < islandMaxDurationSec
            else { continue }
            out[i].speaker = before.speaker
            out[i].speakerID = flank
            // The identity here came from a HEURISTIC (flanked on both sides),
            // not from a voice embedding scored against a profile, so any
            // confidence carried on this row described the speaker it no longer
            // claims to be. Clear it rather than let it follow the new label.
            out[i].speakerConf = nil
        }
        return out
    }

    /// Join consecutive rows that belong to the same speaker into one.
    ///
    /// A chunk boundary can fall mid-sentence, and each chunk becomes its own
    /// segment, so "…just one minute? You" and "know," end up as two rows even
    /// though one person said them without pausing. assignSentences already
    /// merges within a chunk; this closes the seam BETWEEN chunks.
    ///
    /// Two rows merge only when they are adjacent in the final (time-sorted)
    /// list, share a non-nil `speakerID`, agree on `isRemote`, and are both
    /// confirmed. Adjacent in the sorted list with the same speaker means nobody
    /// spoke between them — so it is genuinely one turn. The guards matter:
    /// - non-nil id: two UNKNOWN rows are not known to be the same person;
    /// - same isRemote: office and remote are separate label spaces;
    /// - a different speaker (or a remote row) sitting between two same-speaker
    ///   rows keeps them apart, because it breaks adjacency;
    /// - unconfirmed rows are provisional and left alone.
    /// Text is joined with a space, the span widens to cover both, and the
    /// overlap flag is OR-ed. This is display-only — `derivedRows` (what overlap
    /// repair reads) is untouched, and `anchorText` joins per-speaker text with
    /// a space anyway, so its result is byte-identical before and after.
    /// Seconds between a turn and a window: 0 if they overlap, else the gap to
    /// the nearer edge. Used to attribute an uncovered tail to the nearest talker.
    nonisolated static func timeDistance(from turn: SpeakerTurn,
                                         to window: ClosedRange<Double>) -> Double {
        if turn.end < window.lowerBound { return window.lowerBound - turn.end }
        if turn.start > window.upperBound { return turn.start - window.upperBound }
        return 0
    }

    nonisolated static func coalesceAdjacentSameSpeaker(
        _ rows: [SpeakerUtterance]) -> [SpeakerUtterance] {
        var out: [SpeakerUtterance] = []
        out.reserveCapacity(rows.count)
        for row in rows {
            if var last = out.last,
               let id = last.speakerID, row.speakerID == id,
               last.isRemote == row.isRemote,
               last.confirmed, row.confirmed {
                last.text = [last.text, row.text]
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .joined(separator: " ")
                if let e = row.end { last.end = max(last.end ?? e, e) }
                last.overlapped = last.overlapped || row.overlapped
                // The merged row now covers both chunks, so it can be no more
                // trustworthy than the WORSE of them. `nil` (a model that reports
                // nothing) is ignored rather than treated as zero.
                last.asrConf = minConfidence(last.asrConf, row.asrConf)
                out[out.count - 1] = last
            } else {
                out.append(row)
            }
        }
        return out
    }

    /// The display rows one raw segment expands into — the single source of truth
    /// shared by `rebuildDisplayRows` and `applyRepair` (so repair sees exactly the
    /// rows the transcript shows). `regions` are the genuine-overlap windows.
    func derivedRows(for seg: TranscriptSegment,
                             regions: [(start: Double, end: Double)]) -> [SpeakerUtterance] {
        // Debug: raw MossFormer2 separated-track ASR, shown verbatim as its own
        // "MossFormer2 Index N" row (no rename, no attribution, never replaces).
        if seg.isSeparationDebug {
            let w = seg.window
            return [SpeakerUtterance(id: seg.id.uuidString,
                                     speaker: seg.pinnedSpeakerName, speakerID: nil,
                                     start: w?.lowerBound, end: w?.upperBound,
                                     text: seg.text, confirmed: true, overlapped: true)]
        }
        // Overlap-repair (pinned) segments belong entirely to one separated speaker —
        // a single row. Tag it orange only if it genuinely sits over an overlap
        // region, so PRESERVED bystander rows aren't all flagged.
        if let pid = seg.pinnedSpeakerID {
            let w = seg.window
            let overlapped = regions.contains {
                min($0.end, seg.window?.upperBound ?? 0) - max($0.start, seg.window?.lowerBound ?? 0) > 0
            }
            return [SpeakerUtterance(id: seg.id.uuidString,
                                     speaker: seg.pinnedSpeakerName, speakerID: pid,
                                     start: w?.lowerBound, end: w?.upperBound,
                                     text: seg.text, confirmed: true, overlapped: overlapped)]
        }
        // Neither branch above carries `asrConf`: a repaired or separation-debug
        // row's text came from a SEPARATED track, so the chunk confidence would
        // be describing audio those words did not come from.
        guard let window = seg.window else {
            return [SpeakerUtterance(id: seg.id.uuidString, speaker: nil,
                                     speakerID: nil, start: nil, end: nil,
                                     text: seg.text, confirmed: seg.confirmed,
                                     asrConf: seg.asrConf)]
        }
        let ranges = speakerRanges(in: window)
        // The one switch on the display source (see `PositionSource.plan`):
        //   both       → pyannote shows, ATND fills pyannote's own complement
        //   atnd       → nothing pyannote shows, empty coverage ⇒ ATND tiles it all
        //   pyannote   → pyannote shows, no gap-fill runs at all
        //   atndTiming → ATND tiles it all, then each span takes the identity of
        //                the pyannote turn it overlaps most (`relabelFromPyannote`)
        // `ranges` itself is untouched in every mode, and the position ids the
        // fills carry never leave `filled` — liveTurns/overlapRegions/
        // speakerCount/SpeakerProfileStore stay pure pyannote throughout.
        // Under the MOSS engine the position plan is forced to pure-pyannote —
        // i.e. "show the ranges, run no gap-fill, relabel nothing". The name is
        // now a misnomer for what `ranges` holds (they are MOSS spans), but the
        // PLAN is what matters: the ATND layer exists to fill holes in pyannote's
        // coverage and to lend pyannote its boundaries, and neither question is
        // meaningful against turns that came from an ASR model rather than from a
        // voice-embedding pass over the room. Mixing beam-derived spans into MOSS
        // rows would also put position ids on rows this engine labels, which is
        // exactly the id-space collision the ranges are kept disjoint to prevent.
        // Logged once per session in `configureMoss`, not per rebuild.
        let source = mossDiarizationActive ? PositionSource.pyannote : positionSource
        let plan = source.plan(pyannoteRanges: ranges)
        // Off/silent ATND → fills is [] regardless of the source.
        let fills = plan.gapFillCoverage.map { positionGapFill(window: window, covered: $0) } ?? []
        var filled = (plan.displayRanges + fills).sorted { $0.start < $1.start }
        // Timing from ATND, identity from pyannote: rename in place, boundaries
        // untouched. Only ids move here, and only pyannote → display, never back.
        if plan.relabelFromPyannote {
            filled = PositionRelabel.fromPyannote(filled, pyannote: ranges)
        }

        // Unconfirmed (realtime) segments stay a single provisional row — the text
        // isn't final, so it isn't sentence-split — but it must still carry the
        // speaker the live view already showed. Label it with whoever dominates
        // the window (pyannote if it has a turn there, else the ATND position
        // fill), so it doesn't drop back to UNKNOWN the moment it commits.
        if !seg.confirmed {
            // A beam change split this window into multiple fills → split the live
            // (unconfirmed) text into per-speaker rows too, so the switch shows in
            // real time. A single fill (or none) stays one provisional row.
            if filled.count > 1 {
                return Self.assignSentences(seg.text, window: window, ranges: filled,
                                       segID: seg.id.uuidString, regions: regions,
                                       confirmed: false, asrConf: seg.asrConf)
            }
            func overlap(_ r: (start: Double, end: Double, id: Int, name: String)) -> Double {
                max(0, min(r.end, window.upperBound) - max(r.start, window.lowerBound))
            }
            let best = filled.max { overlap($0) < overlap($1) }
            return [SpeakerUtterance(id: seg.id.uuidString, speaker: best?.name,
                                     speakerID: best?.id, start: window.lowerBound,
                                     end: window.upperBound, text: seg.text,
                                     confirmed: false, asrConf: seg.asrConf)]
        }

        if filled.isEmpty {
            // No pyannote turn overlaps this window and ATND left no span — a
            // short tail at Stop is the usual case: the final stretch of speech
            // is too brief for its own diarization turn, and position skips runs
            // under 0.75 s. Rather than drop it to UNKNOWN, hand it to whoever
            // was speaking NEAREST in time. In a single-speaker recording that
            // is simply that speaker; across speakers it is the last/next turn,
            // which is a better guess than "unknown" for a sub-second seam.
            // Only a recording with NO turns at all stays UNKNOWN.
            let nearest = displayTurns.min { a, b in
                Self.timeDistance(from: a, to: window) < Self.timeDistance(from: b, to: window)
            }
            return [SpeakerUtterance(id: seg.id.uuidString, speaker: nearest?.name,
                                     speakerID: nearest?.id, start: window.lowerBound,
                                     end: window.upperBound, text: seg.text,
                                     confirmed: true, asrConf: seg.asrConf)]
        }
        // Word-exact path: when the aligner ran, each word goes to the turn that
        // covers it in time instead of to the turn its character offset guesses.
        // Any failed sanity gate returns nil → the estimate below, unchanged.
        if let words = seg.words,
           let pieces = WordAttribution.attribute(text: seg.text, words: words,
                                                  chunkDuration: seg.alignedChunkDuration,
                                                  window: window, ranges: filled,
                                                  log: { self.positionLog($0) }) {
            return pieces.enumerated().map { i, p in
                let overlapped = regions.contains { max($0.start, p.start) < min($0.end, p.end) }
                return SpeakerUtterance(id: "\(seg.id.uuidString)-\(i)", speaker: p.name,
                                        speakerID: p.id, start: p.start, end: p.end,
                                        text: p.text, confirmed: true, overlapped: overlapped,
                                        asrConf: seg.asrConf)
            }
        }
        return Self.assignSentences(seg.text, window: window, ranges: filled,
                               segID: seg.id.uuidString, regions: regions,
                               asrConf: seg.asrConf)
    }

    /// Assign whole sentences to speakers. Each sentence is placed in time by its
    /// character position in the chunk, then handed to the speaker turn it most
    /// overlaps — so text is never cut mid-sentence onto the wrong speaker.
    /// Consecutive sentences by the same speaker merge into one row.
    ///
    /// Pure and `static` so the Remote stream can reuse it verbatim with its own
    /// `ranges` (built from `remoteLiveTurns`) — the split logic is identical,
    /// only the identity space differs.
    nonisolated static func assignSentences(_ text: String,
                                            window: ClosedRange<Double>,
                                            ranges: [(start: Double, end: Double, id: Int, name: String)],
                                            segID: String,
                                            regions: [(start: Double, end: Double)],
                                            confirmed: Bool = true,
                                            asrConf: Double? = nil) -> [SpeakerUtterance] {
        let sentences = splitSentences(text)
        guard !sentences.isEmpty else { return [] }
        let totalChars = max(1, sentences.reduce(0) { $0 + $1.count })
        let span = max(0, window.upperBound - window.lowerBound)

        struct Piece { var id: Int; var name: String; var start: Double; var end: Double; var text: String }
        var pieces: [Piece] = []
        var charsSoFar = 0
        for sentence in sentences {
            let sStart = window.lowerBound + span * Double(charsSoFar) / Double(totalChars)
            charsSoFar += sentence.count
            let sEnd = window.lowerBound + span * Double(charsSoFar) / Double(totalChars)

            var chosen = ranges[0]
            var bestOverlap = -1.0
            for r in ranges {
                let ov = max(0, min(r.end, sEnd) - max(r.start, sStart))
                if ov > bestOverlap { bestOverlap = ov; chosen = r }
            }
            if bestOverlap <= 0 {
                let mid = (sStart + sEnd) / 2
                chosen = ranges.min { abs(($0.start + $0.end) / 2 - mid) < abs(($1.start + $1.end) / 2 - mid) } ?? ranges[0]
            }
            pieces.append(Piece(id: chosen.id, name: chosen.name, start: sStart, end: sEnd, text: sentence))
        }

        var merged: [Piece] = []
        for p in pieces {
            if var last = merged.last, last.id == p.id {
                last.end = p.end
                last.text += " " + p.text
                merged[merged.count - 1] = last
            } else {
                merged.append(p)
            }
        }

        return merged.enumerated().map { i, p in
            // Flag only if this row's time genuinely sits over a simultaneous-
            // speech region (two different speakers active at once).
            let overlapped = regions.contains { max($0.start, p.start) < min($0.end, p.end) }
            // Every row a chunk splits into carries that CHUNK's confidence: the
            // model scored the chunk as a whole and reports nothing finer, so
            // splitting the text does not split the evidence.
            return SpeakerUtterance(id: "\(segID)-\(i)", speaker: p.name, speakerID: p.id,
                                    start: p.start, end: p.end, text: p.text,
                                    confirmed: confirmed, overlapped: overlapped,
                                    asrConf: asrConf)
        }
    }

    /// Windows where two DIFFERENT speakers are active at the same time
    /// (genuine overlap), each at least 0.4s long. Empty in exclusive mode.
    ///
    /// UNDER DIARIZEN THIS IS THE DETECTOR, so the Detect overlap switch gates it
    /// here — at the ONE point that feeds both consumers, `derivedRows` (the
    /// tags) and `repairWindows` (where repair may work). Gating them separately
    /// is how a transcript ends up with no tags and repair running anyway.
    ///
    /// The guard is deliberately narrow. Every other engine reaches this function
    /// with `diarizenDiarizationActive == false` and is untouched: pyannote keeps
    /// marking overlap from its own turns whatever that switch says, which is the
    /// behaviour it has always had and which the owner asked to leave alone.
    func overlapRegions() -> [(start: Double, end: Double)] {
        if diarizenDiarizationActive && !diarizenOverlapMarking { return [] }
        let turns = Self.officeTurnsOnly(liveTurns, "overlapRegions")
        guard turns.count > 1 else { return [] }
        var regions: [(start: Double, end: Double)] = []
        for i in 0..<turns.count {
            for j in (i + 1)..<turns.count where turns[i].id != turns[j].id {
                let s = max(turns[i].start, turns[j].start)
                let e = min(turns[i].end, turns[j].end)
                if e - s >= 0.4 { regions.append((s, e)) }
            }
        }
        return regions
    }

    /// Split text into sentences on . ? ! and line breaks, keeping punctuation.
    /// Fragments with no letters or digits (e.g. ". .") are dropped.
    nonisolated static func splitSentences(_ text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        func hasContent(_ s: String) -> Bool { s.contains { $0.isLetter || $0.isNumber } }

        var result: [String] = []
        var current = ""
        for ch in trimmed {
            current.append(ch)
            if ch == "." || ch == "?" || ch == "!" || ch == "\n" {
                let s = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if hasContent(s) { result.append(s) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if hasContent(tail) { result.append(tail) }
        if result.isEmpty { return hasContent(trimmed) ? [trimmed] : [] }
        return result
    }

    /// Speaker turns overlapping a window, clipped to it and merged when the
    /// same speaker continues (small gaps bridged), sorted by start time.
    /// The office view: `displayTurns`, never the remote ones.
    func speakerRanges(in window: ClosedRange<Double>)
        -> [(start: Double, end: Double, id: Int, name: String)] {
        Self.speakerRanges(in: window, turns: displayTurns)
    }

    /// Which turn collection the OFFICE display draws its labels from: `mossTurns`
    /// under the MOSS engine, `liveTurns` under pyannote.
    ///
    /// A read-only switch on one flag, and only the display path takes it. The
    /// collections stay separate everywhere else on purpose — `overlapRegions`,
    /// `applyFinalSpeakers`, `speakerCount`, the profile stores and every
    /// `officeTurnsOnly` assert keep reading `liveTurns` directly, so under the
    /// MOSS engine they see an EMPTY pyannote layer rather than foreign ids. That
    /// is why no assert had to be relaxed for this engine, and why overlap tagging
    /// simply produces nothing: MOSS makes no overlap claim, so the transcript
    /// shows none.
    var displayTurns: [SpeakerTurn] {
        mossDiarizationActive ? mossTurns : liveTurns
    }

    /// The same clipping/merging, parameterised by the turn set — so the Remote
    /// stream can reuse it against `remoteLiveTurns` without any chance of the
    /// two identity spaces meeting (each call sees exactly one of them).
    nonisolated static func speakerRanges(in window: ClosedRange<Double>,
                                          turns: [SpeakerTurn])
        -> [(start: Double, end: Double, id: Int, name: String)] {
        let clipped = turns.compactMap { t -> (start: Double, end: Double, id: Int, name: String)? in
            let s = max(t.start, window.lowerBound)
            let e = min(t.end, window.upperBound)
            return e > s ? (s, e, t.id, t.name) : nil
        }.sorted { $0.start < $1.start }

        var merged: [(start: Double, end: Double, id: Int, name: String)] = []
        for c in clipped {
            if var last = merged.last, last.id == c.id, c.start - last.end < 1.0 {
                last.end = max(last.end, c.end)
                merged[merged.count - 1] = last
            } else {
                merged.append(c)
            }
        }
        return merged
    }

    /// The display rows the remote segments render as, split by the REMOTE
    /// diarization turns (`remoteLiveTurns`) — never by `liveTurns`.
    ///
    /// The splitting machinery is the office one (`speakerRanges` →
    /// `assignSentences`), parameterised by the remote turns; what is
    /// deliberately absent is everything position-shaped: no `positionGapFill`,
    /// no `PositionSource`, no `PositionDiarizer`, no ATND. The beam describes
    /// the ROOM, so it says nothing whatsoever about the conferencing stream.
    /// `regions: []` for the same reason overlap repair is Office-only — remote
    /// rows are never tagged from office overlap windows.
    ///
    /// With no remote turns (feature idle, or the remote pass has not landed
    /// yet) each segment stays ONE row labelled `remoteSpeakerLabel` with no
    /// speaker id — exactly the phase-3 behaviour.
    ///
    /// Sorted by start time here rather than relying on append order: segments
    /// are appended when their transcription lands, which is chronological today
    /// (one sidecar, one queue) but is not something the merge should depend on.
    nonisolated static func remoteRows(_ segments: [RemoteSegment],
                                       turns: [SpeakerTurn] = [])
        -> [SpeakerUtterance] {
        segments.sorted { $0.window.lowerBound < $1.window.lowerBound }
            .flatMap { seg -> [SpeakerUtterance] in
                let text = seg.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { return [] }
                let ranges = speakerRanges(in: seg.window, turns: turns)
                guard !ranges.isEmpty else {
                    return [SpeakerUtterance(id: seg.id.uuidString,
                                             speaker: remoteSpeakerLabel, speakerID: nil,
                                             start: seg.window.lowerBound,
                                             end: seg.window.upperBound,
                                             text: text, confirmed: true, overlapped: false,
                                             isRemote: true, asrConf: seg.conf)]
                }
                return assignSentences(text, window: seg.window, ranges: ranges,
                                       segID: seg.id.uuidString, regions: [],
                                       asrConf: seg.conf)
                    .map { row in
                        var row = row
                        row.speaker = row.speaker.map(remoteDisplayName)
                        row.isRemote = true
                        return row
                    }
            }
    }

    /// Interleave office and remote rows by start time into one transcript.
    ///
    /// A merge walk, NOT a sort of the combined list: office rows keep their own
    /// order exactly as `rebuildDisplayRows` produced it, and each remote row is
    /// placed before the first office row that starts later. Re-sorting the whole
    /// list would silently reorder office rows the office pipeline placed
    /// deliberately — separation-debug segments, for instance, carry real windows
    /// but are pinned to the end on purpose.
    ///
    /// A tie puts Office first (the room is the primary record). A row with no
    /// start time — an unconfirmed realtime segment with no window yet — sorts as
    /// +∞, the same convention `insertPinnedSorted` uses, so remote rows go ahead
    /// of it; it still keeps its place among the office rows.
    ///
    /// With no remote rows this returns `office` untouched: the single-stream
    /// transcript is not re-sorted, re-ordered or otherwise disturbed by this
    /// feature existing. `remote` is expected sorted (see `remoteRows`).
    nonisolated static func mergeRowsByStartTime(office: [SpeakerUtterance],
                                                 remote: [SpeakerUtterance]) -> [SpeakerUtterance] {
        guard !remote.isEmpty else { return office }
        func key(_ r: SpeakerUtterance) -> Double { r.start ?? .greatestFiniteMagnitude }
        var out: [SpeakerUtterance] = []
        out.reserveCapacity(office.count + remote.count)
        var i = 0, j = 0
        while i < office.count, j < remote.count {
            if key(remote[j]) < key(office[i]) {
                out.append(remote[j]); j += 1
            } else {
                out.append(office[i]); i += 1
            }
        }
        out.append(contentsOf: office[i...])
        out.append(contentsOf: remote[j...])
        return out
    }
}
