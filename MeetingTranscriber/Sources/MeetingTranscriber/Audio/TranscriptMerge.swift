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
    }

    // MARK: - Public API

    /// Merge a separated `track` against the `existing` anchor text.
    /// - A significant shared run (≥ `minRunTokens` matched tokens) anchors the
    ///   two texts together:
    ///   * track adds no new tokens → `.noop` (keep existing verbatim)
    ///   * track adds new tokens    → `.combine` (weave, shared run emitted once)
    /// - No significant shared run → `.replace` (track is verbatim new content).
    static func merge(existing: String, track: String, minRunTokens: Int = 4) -> Result {
        let e = tokenize(existing)
        let t = tokenize(track)
        let blocks = matchingBlocks(e.map(\.norm), t.map(\.norm))

        let longestRun = blocks.map(\.length).max() ?? 0
        let significant = blocks.contains { $0.length >= minRunTokens }

        // No anchoring run → the track is entirely new content; keep it verbatim.
        guard significant else {
            return Result(kind: .replace, text: track, longestRun: longestRun)
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
            return Result(kind: .noop, text: existing, longestRun: longestRun)
        }

        // COMBINE: walk blocks in order, emitting each matched run once (from the
        // track's originals) and filling gaps per the one-sided / both-sided rule.
        var out: [String] = []
        var ePrev = 0, tPrev = 0

        // Gap rule: nothing is ever dropped. The existing text is emitted in full,
        // and the track's un-matched words are added after it — combining, never
        // replacing. Where both sides have a gap they were both heard, so both are
        // kept (existing first, it is the confirmed chunked-ASR wording); the
        // shared runs themselves are still emitted once, so nothing duplicates.
        func emitGap(_ eLo: Int, _ eHi: Int, _ tLo: Int, _ tHi: Int) {
            if eLo < eHi { out.append(contentsOf: e[eLo..<eHi].map(\.original)) }
            if tLo < tHi { out.append(contentsOf: t[tLo..<tHi].map(\.original)) }
        }

        for b in blocks {
            emitGap(ePrev, b.aStart, tPrev, b.bStart)
            for k in b.bStart..<(b.bStart + b.length) { out.append(t[k].original) }
            ePrev = b.aStart + b.length
            tPrev = b.bStart + b.length
        }
        emitGap(ePrev, e.count, tPrev, t.count)

        return Result(kind: .combine, text: out.joined(separator: " "), longestRun: longestRun)
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
