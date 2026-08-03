import XCTest
@testable import MeetingTranscriber

/// Contract tests for word alignment, which became its OWN sidecar on
/// 2026-07-29 (`AlignerService`) and therefore ASYNCHRONOUS: the chunk's text is
/// shown immediately with the estimated split and the words arrive later.
///
/// Three things must hold:
///  * `align.enabled` decides whether the ALIGNER loads — and no longer touches
///    the chunked ASR sidecar's configuration at all;
///  * an ASR `final` no longer carries `words`/`dur`, and a stale sidecar that
///    still sends them decodes without complaint;
///  * a late reply lands on the right segment, or is dropped — never applied to
///    a segment whose text has since been rewritten.
final class ChunkedASRAlignmentTests: XCTestCase {

    private let key = "align.enabled"
    private let modelKey = "chunked.model"

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: key)
        UserDefaults.standard.removeObject(forKey: modelKey)
        super.tearDown()
    }

    // MARK: - Which session loads the aligner

    func testAlignerIsNotLoadedWhenDisabled() {
        XCTAssertFalse(ModelLoader.wantsAligner(alignEnabled: false, chunkedID: "qwen3"))
    }

    func testAlignerIsNotLoadedWhenUnset() {
        UserDefaults.standard.removeObject(forKey: key)
        let stored = UserDefaults.standard.object(forKey: key) as? Bool ?? false
        XCTAssertFalse(ModelLoader.wantsAligner(alignEnabled: stored, chunkedID: "qwen3"))
    }

    func testAlignerIsLoadedWhenEnabled() {
        XCTAssertTrue(ModelLoader.wantsAligner(alignEnabled: true, chunkedID: "qwen3"))
        XCTAssertTrue(ModelLoader.wantsAligner(alignEnabled: true, chunkedID: "whisper"))
    }

    /// MOSS stays excluded even though the split makes aligning its text
    /// possible for the first time — the owner deferred cross-chunk MOSS work,
    /// and the Aligner tab says so.
    func testAlignerIsNeverLoadedForMoss() {
        XCTAssertFalse(ModelLoader.wantsAligner(alignEnabled: true, chunkedID: "moss"))
    }

    func testAlignerConfigCarriesTheAlignerRepo() {
        XCTAssertEqual(AlignerService.Config.fromSettings().repoID,
                       "mlx-community/Qwen3-ForcedAligner-0.6B-bf16")
    }

    // MARK: - The ASR sidecar no longer knows about alignment

    /// The REVERSE of what this file used to assert. Toggling alignment used to
    /// change `ChunkedASRService.Config` (which is Equatable) so the ASR sidecar
    /// was recreated. Now it must NOT: reloading a 1.7–4 B ASR model for a
    /// setting that belongs to a different process would be a pointless minute
    /// of the user's time.
    func testTogglingAlignmentLeavesTheChunkedConfigUnchanged() {
        UserDefaults.standard.set(false, forKey: key)
        let off = ChunkedASRService.Config.fromSettings()
        UserDefaults.standard.set(true, forKey: key)
        let on = ChunkedASRService.Config.fromSettings()
        XCTAssertEqual(off, on)
    }

    /// Mirrors the argument construction in `ChunkedASRService.init`: whatever
    /// the alignment setting says, no `--align-model` can reach an ASR sidecar —
    /// neither of them accepts the flag any more.
    func testArgumentsNeverContainAlignModel() {
        func arguments(for config: ChunkedASRService.Config) -> [String] {
            var arguments = ["--model", config.repoID]
            if config.language != "auto" { arguments += ["--language", config.language] }
            arguments += config.whisper?.processArguments ?? []
            return arguments
        }
        UserDefaults.standard.set(false, forKey: key)
        XCTAssertFalse(arguments(for: .fromSettings()).contains("--align-model"))
        UserDefaults.standard.set(true, forKey: key)
        XCTAssertFalse(arguments(for: .fromSettings()).contains("--align-model"))
    }

    // MARK: - Decoding

    func testFinalWithoutAlignerKeysDecodes() throws {
        let line = #"{"type":"final","text":"hello there"}"#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.type, "final")
        XCTAssertEqual(m.text, "hello there")
    }

    /// A STALE packaged sidecar (`Contents/Resources/scripts` is a copy — see
    /// CLAUDE.md) can still emit `words`/`dur`. `JSONDecoder` ignores unknown
    /// keys, so such a payload must decode and the extra keys simply go unread.
    func testFinalStillCarryingAlignerKeysDecodesAndIsIgnored() throws {
        let line = #"""
        {"type":"final","text":"one-minute break","words":[{"text":"oneminute","start":0.2,"end":0.8,"src":0}],"dur":30.0}
        """#
        let m = try JSONDecoder().decode(ChunkedASRService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.type, "final")
        XCTAssertEqual(m.text, "one-minute break")
    }

    func testAlignerResultDecodes() throws {
        let line = #"""
        {"type":"align_result","id":7,"words":[{"text":"oneminute","start":0.2,"end":0.8,"src":0}],"dur":30.0}
        """#
        let m = try JSONDecoder().decode(AlignerService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.type, "align_result")
        XCTAssertEqual(m.id, 7)
        XCTAssertEqual(m.dur, 30.0)
        XCTAssertEqual(m.words, [AlignerService.AlignedWord(
            text: "oneminute", start: 0.2, end: 0.8, src: 0)])
    }

    /// A rejected alignment: `align_result` with the keys ABSENT, which is a
    /// normal outcome and not an error. `text` is absent too, which is why
    /// `Message.text` is optional.
    func testRejectedAlignmentDecodesWithNoWords() throws {
        let line = #"{"type":"align_result","id":7}"#
        let m = try JSONDecoder().decode(AlignerService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.type, "align_result")
        XCTAssertEqual(m.id, 7)
        XCTAssertNil(m.words)
        XCTAssertNil(m.dur)
        XCTAssertNil(m.text)
    }

    func testAlignErrorDecodes() throws {
        let line = #"{"type":"align_error","id":3,"text":"file not found: /nope.wav"}"#
        let m = try JSONDecoder().decode(AlignerService.Message.self,
                                         from: Data(line.utf8))
        XCTAssertEqual(m.type, "align_error")
        XCTAssertEqual(m.id, 3)
        XCTAssertEqual(m.text, "file not found: /nope.wav")
    }

    // MARK: - Where a late reply lands

    private func segment(_ text: String) -> AudioRecorder.TranscriptSegment {
        AudioRecorder.TranscriptSegment(text: text, confirmed: true,
                                        window: 0...30)
    }

    /// Two chunks in flight: each reply finds its own segment whichever order
    /// they arrive in, because the match is by UUID and not by position.
    func testRepliesMatchTheirOwnSegmentInAnyOrder() {
        let a = segment("first chunk text")
        let b = segment("second chunk text")
        let segments = [a, b]
        XCTAssertEqual(AudioRecorder.indexForAlignedWords(
            segments: segments, id: b.id, requestText: b.text), 1)
        XCTAssertEqual(AudioRecorder.indexForAlignedWords(
            segments: segments, id: a.id, requestText: a.text), 0)
    }

    /// The segment was swept away (session restarted, or the unconfirmed sweep
    /// at `checkLastChunkDone` ran) → drop the reply.
    func testVanishedSegmentIsDropped() {
        let gone = segment("this one is gone")
        XCTAssertNil(AudioRecorder.indexForAlignedWords(
            segments: [segment("something else")], id: gone.id,
            requestText: gone.text))
    }

    /// THE subtle one. `applyRepair`/`TranscriptMerge` COMBINE can rewrite a
    /// segment's text after Stop, and `src` indices are only valid against the
    /// exact `text.split()` they were computed from. An INSERTION shifts every
    /// later index while staying in range, so no gate inside `WordAttribution`
    /// would notice — the words would land silently on the wrong parts of the
    /// sentence. Byte-equality is the only check that can see it.
    func testTextMutatedByCombineIsDropped() {
        let requested = "it have this and it have anything under security"
        var seg = segment(requested)
        // A COMBINE-style insertion: the track filled a gap the existing side
        // had nothing in. Same segment, same id, different words from index 1 on.
        seg.text = "it did not have this and it did not have anything under security"
        XCTAssertNil(AudioRecorder.indexForAlignedWords(
            segments: [seg], id: seg.id, requestText: requested))
        // …and the positive control: unchanged text still matches.
        XCTAssertEqual(AudioRecorder.indexForAlignedWords(
            segments: [seg], id: seg.id, requestText: seg.text), 0)
    }
}
