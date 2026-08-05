import Foundation

/// Pure text-merge logic for overlap repair: given the EXISTING transcript text
/// for a speaker (the anchor) and a freshly separated TRACK's ASR text, decide
/// whether the track adds new words (COMBINE), is entirely new (REPLACE), or is
/// already fully contained in the existing text (NO-OP) — and produce the merged
/// text without ever duplicating the shared portion.
///
/// Foundation-only and free of any app types, so it compiles/tests standalone.
enum TranscriptMerge {

    enum Kind: Equatable { case combine, replace, noop }

    struct Result: Equatable {
        let kind: Kind
        let text: String
        let longestRun: Int
        /// How many TRACK tokens the merge actually inserted. Zero on `.noop` and
        /// `.replace`. Logged as `inserted=N/M` so a COMBINE that added one word
        /// and one that rewrote half the row are distinguishable in the decisions
        /// log — the alternative (a post-merge inserted-fraction gate) would
        /// silently discard genuine recovery and leave no trace in the transcript.
        let inserted: Int
    }

    // MARK: - Public API

    /// Merge a separated `track` against the `existing` anchor text.
    /// - A significant shared run (≥ `minRunTokens` matched tokens) anchors the
    ///   two texts together:
    ///   * track inserts no new tokens → `.noop` (keep existing verbatim)
    ///   * track inserts new tokens    → `.combine` (existing text verbatim, plus
    ///     the track's words at gaps where existing had nothing)
    /// - No significant shared run → `.replace` (track is verbatim new content).
    static func merge(existing: String, track: String, minRunTokens: Int = 4) -> Result {
        let e = tokenize(existing)
        let t = tokenize(track)

        // A track with NO WORD IN IT can only ever delete, so it is refused here
        // rather than allowed to fall through to `.replace`.
        //
        // THE BUG THIS FIXES (measured 2026-08-05). A punctuation-only track —
        // `"."`, `","`, `"…"` — tokenizes to tokens whose `norm` is empty, and
        // empty norms never match, so no block is found, `significant` is false
        // and the guard below returned `.replace(text: ".")`. `applyRepair` then
        // REPLACED a real sentence with a single full stop. Verified by driving
        // `merge` directly: existing "we should ship it on friday", track "."
        // came back `kind=replace text="."`.
        //
        // It was reachable: BOTH call sites test only
        // `trimmingCharacters(in: .whitespacesAndNewlines).isEmpty`, which `"."`
        // passes, and a re-ASR of a near-silent separated track emitting bare
        // punctuation is ordinary Whisper/mlx-audio behaviour. The canned-caption
        // gate does not cover it either — `"."` is not a canned caption.
        //
        // Placed INSIDE `merge` on purpose. Two callers each remembering a
        // precondition is how this class of bug survives; the type should not be
        // able to delete text no matter who calls it. Over-deletion is the
        // direction this project treats as dangerous, because a deleted sentence
        // leaves no trace in the transcript — only in the decisions log.
        guard t.contains(where: { !$0.norm.isEmpty }) else {
            return Result(kind: .noop, text: existing, longestRun: 0, inserted: 0)
        }

        let blocks = matchingBlocks(e.map(\.norm), t.map(\.norm))

        let longestRun = blocks.map(\.length).max() ?? 0
        let significant = blocks.contains { $0.length >= minRunTokens }

        // No anchoring run → the track is entirely new content; keep it verbatim.
        guard significant else {
            return Result(kind: .replace, text: track, longestRun: longestRun, inserted: 0)
        }

        // Which TRACK token indices are covered by a matched block?
        var coveredT = Set<Int>()
        for b in blocks {
            for k in b.bStart..<(b.bStart + b.length) { coveredT.insert(k) }
        }
        // Track has content the existing text lacks? (empty-norm/punctuation-only
        // tokens don't count — they never participate in matching.)
        let trackHasExtra = t.indices.contains { idx in
            !coveredT.contains(idx) && !t[idx].norm.isEmpty
        }

        if !trackHasExtra {
            return Result(kind: .noop, text: existing, longestRun: longestRun, inserted: 0)
        }

        // COMBINE: walk blocks in order, emitting the EXISTING text verbatim and
        // adding the track's words ONLY where the existing side is silent.
        var out: [String] = []
        var ePrev = 0, tPrev = 0
        var inserted = 0

        // Gap rule — ONE-SIDED. A gap that has words on BOTH sides is mechanically
        // a SUBSTITUTION: the same audio rendered two ways ("didn't"/"did not",
        // "parity"/"parody"). Emitting both produced real damage in the decisions
        // log — "it didn't did not have this", "so that's that is where". So a
        // both-sided gap keeps the EXISTING words alone: that is the confirmed
        // chunked-ASR wording, and the track's job is to fill silence, not to
        // re-word what was already heard. The track's words are inserted only
        // where the existing side has nothing at all (`eLo == eHi`).
        //
        // Invariant: COMBINE output == the existing text verbatim, in order, plus
        // inserted track runs at gaps where existing had nothing.
        func emitGap(_ eLo: Int, _ eHi: Int, _ tLo: Int, _ tHi: Int) {
            if eLo < eHi {
                out.append(contentsOf: e[eLo..<eHi].map(\.original))
            } else if tLo < tHi {
                out.append(contentsOf: t[tLo..<tHi].map(\.original))
                inserted += tHi - tLo
            }
        }

        for b in blocks {
            emitGap(ePrev, b.aStart, tPrev, b.bStart)
            // The matched run is emitted from the EXISTING originals, not the
            // track's: the norms are equal by construction, but casing and
            // punctuation need not be, and existing is the trusted wording.
            for k in b.aStart..<(b.aStart + b.length) { out.append(e[k].original) }
            ePrev = b.aStart + b.length
            tPrev = b.bStart + b.length
        }
        emitGap(ePrev, e.count, tPrev, t.count)

        // A pure-substitution track reaches here having inserted nothing, so the
        // merged text is byte-identical to `existing`. Reporting that as COMBINE
        // would make `applyRepair` perform real segment surgery for a no-change
        // edit and print a misleading COMBINE in the decisions log.
        guard inserted > 0 else {
            return Result(kind: .noop, text: existing, longestRun: longestRun, inserted: 0)
        }

        return Result(kind: .combine, text: out.joined(separator: " "),
                      longestRun: longestRun, inserted: inserted)
    }

    /// Jaccard similarity of the two texts' lowercased alnum word sets.
    /// Same normalization as `AudioRecorder.jaccard`.
    static func jaccard(_ a: String, _ b: String) -> Double {
        func tokens(_ s: String) -> Set<String> {
            Set(s.lowercased().split { !$0.isLetter && !$0.isNumber }
                .map(String.init).filter { !$0.isEmpty })
        }
        let ta = tokens(a), tb = tokens(b)
        if ta.isEmpty && tb.isEmpty { return 1.0 }
        let uni = ta.union(tb).count
        return uni == 0 ? 0.0 : Double(ta.intersection(tb).count) / Double(uni)
    }

    // MARK: - Internals

    private struct Token { let original: String; let norm: String }

    /// Split on whitespace; keep every original word (even pure punctuation, whose
    /// norm is empty) so reconstruction preserves it. `norm` = lowercased alnum.
    private static func tokenize(_ s: String) -> [Token] {
        s.split(whereSeparator: { $0.isWhitespace }).map { piece -> Token in
            let original = String(piece)
            let norm = String(original.lowercased().filter { $0.isLetter || $0.isNumber })
            return Token(original: original, norm: norm)
        }
    }

    struct Block: Equatable { let aStart: Int; let bStart: Int; let length: Int }

    /// difflib-style get_matching_blocks over the two normalized token arrays:
    /// recursively take the longest common contiguous run, recurse left/right;
    /// returns monotonic, non-overlapping blocks sorted by position.
    /// Empty-norm tokens never match (so punctuation can't anchor a run).
    private static func matchingBlocks(_ a: [String], _ b: [String]) -> [Block] {
        // b2j: for each non-empty norm value in b, the sorted list of its indices.
        var b2j: [String: [Int]] = [:]
        for (j, v) in b.enumerated() where !v.isEmpty { b2j[v, default: []].append(j) }

        func longestMatch(_ alo: Int, _ ahi: Int, _ blo: Int, _ bhi: Int)
            -> (i: Int, j: Int, k: Int) {
            var besti = alo, bestj = blo, bestk = 0
            var j2len: [Int: Int] = [:]
            for i in alo..<ahi {
                var newj2len: [Int: Int] = [:]
                if !a[i].isEmpty, let js = b2j[a[i]] {
                    for j in js {
                        if j < blo { continue }
                        if j >= bhi { break }
                        let k = (j2len[j - 1] ?? 0) + 1
                        newj2len[j] = k
                        if k > bestk { besti = i - k + 1; bestj = j - k + 1; bestk = k }
                    }
                }
                j2len = newj2len
            }
            return (besti, bestj, bestk)
        }

        var queue: [(Int, Int, Int, Int)] = [(0, a.count, 0, b.count)]
        var blocks: [Block] = []
        while let (alo, ahi, blo, bhi) = queue.popLast() {
            let (i, j, k) = longestMatch(alo, ahi, blo, bhi)
            guard k > 0 else { continue }
            blocks.append(Block(aStart: i, bStart: j, length: k))
            if alo < i && blo < j { queue.append((alo, i, blo, j)) }
            if i + k < ahi && j + k < bhi { queue.append((i + k, ahi, j + k, bhi)) }
        }
        blocks.sort { $0.aStart < $1.aStart }
        return blocks
    }
}
