import XCTest
@testable import MeetingTranscriber

/// Per-model language lists (Settings → Models → Chunked, 2026-07-29).
///
/// What these pin: the picker offers exactly what the SELECTED model published,
/// and a code that model cannot take never reaches a sidecar. The bug they
/// replace was one hard-coded list of six shown for every model — it hid 94 of
/// Whisper's languages AND offered Indonesian/Malay for Voxtral and Granite,
/// which have neither, so picking Indonesian sent `--language id` to a
/// 13-language model.
final class LanguageSupportTests: XCTestCase {

    private let keys = ["chunked.model", "chunked.language", "align.enabled"]

    override func setUp() {
        super.setUp()
        keys.forEach { UserDefaults.standard.removeObject(forKey: $0) }
    }

    override func tearDown() {
        keys.forEach { UserDefaults.standard.removeObject(forKey: $0) }
        super.tearDown()
    }

    private func codes(_ id: String) -> [String] {
        Languages.supported(forModel: id).map(\.code)
    }

    // MARK: - Counts are each model's published roster

    /// The tier boundaries ARE the model rosters — 6 / 13 / 30 cumulative. If a
    /// count moves, either a tier gained an entry it should not have, or a
    /// model's mapping changed.
    func testSupportedCountsMatchEachModelsRoster() {
        XCTAssertEqual(codes("whisper").count, 30)
        XCTAssertEqual(codes("qwen3").count, 30)
        XCTAssertEqual(codes("voxtral").count, 13)
        XCTAssertEqual(codes("granite").count, 5)   // the NAR card, not the family
    }

    /// Each tier is a strict superset of the one below, because each is a
    /// superset of the model roster below it.
    func testTiersAreNested() {
        let granite = Set(codes("granite"))
        let voxtral = Set(codes("voxtral"))
        let qwen3 = Set(codes("qwen3"))
        XCTAssertTrue(granite.isSubset(of: voxtral))
        XCTAssertTrue(voxtral.isSubset(of: qwen3))
        XCTAssertEqual(Set(codes("whisper")), qwen3)
    }

    /// Granite's FIVE are exactly what the checkpoint we ship publishes, and this
    /// number has now been wrong in both directions, so it is pinned to the file
    /// on disk rather than to anyone's memory.
    ///
    /// `models/hub/…granite-speech-4.1-2b-nar-mlx/…/README.md` front-matter says
    /// `language: [en, fr, de, es, pt]`. An earlier note in this repo said five;
    /// a later one "corrected" it to six by reading `ibm-granite`'s *base*
    /// Granite Speech card, which does list Japanese. We do not ship the base
    /// model — we ship the NAR variant. Read the card of the checkpoint that is
    /// actually downloaded.
    ///
    /// Japanese therefore stays out of Granite while remaining in `tier1`, which
    /// turns out to be VOXTRAL's base rather than Granite's list; Voxtral's own
    /// card really does list `ja`.
    func testGraniteIsTheFiveTheNARCardPublishesWithoutJapanese() {
        XCTAssertEqual(Set(codes("granite")), ["en", "de", "fr", "es", "pt"])
        XCTAssertFalse(codes("granite").contains("ja"),
                       "Japanese belongs to the BASE Granite Speech card, not the NAR one we ship")
        XCTAssertTrue(codes("voxtral").contains("ja"),
                      "…and it is still needed here, which is why tier1 keeps it")
    }

    /// No duplicates anywhere: a code appearing in two tiers would inflate a
    /// count and show twice in the menu.
    func testNoDuplicateCodes() {
        let all = Languages.allTiers.map(\.code)
        XCTAssertEqual(Set(all).count, all.count)
        XCTAssertFalse(all.contains(Languages.auto.code),
                       "auto is a UI affordance, not a language in the tiers")
    }

    /// Every offered code exists in Whisper large-v3's own language set — a code
    /// Whisper's tokenizer rejects would fail at decode time, not here. The 30
    /// are hard-coded on purpose: the test must not import mlx to check itself.
    func testEveryOfferedCodeIsAWhisperLanguage() {
        // Verified 2026-07-29 against the INSTALLED
        // `mlx_whisper.tokenizer.LANGUAGES` — all 30 present.
        let whisperCodes: Set<String> = [
            "en", "de", "fr", "es", "pt", "ja",
            "ar", "hi", "it", "nl", "zh", "ko", "ru",
            "id", "ms", "th", "vi", "tr", "yue", "sv", "da", "fi",
            "pl", "cs", "tl", "fa", "el", "hu", "mk", "ro",
        ]
        XCTAssertEqual(whisperCodes.count, 30)
        XCTAssertEqual(Set(codes("whisper")), whisperCodes)
        for code in Languages.allTiers.map(\.code) {
            XCTAssertTrue(whisperCodes.contains(code), "\(code) is not a Whisper language")
        }
    }

    // MARK: - Models that take no language at all

    /// Every picker is selectable now (owner, 2026-07-31) — INCLUDING the two
    /// models that ignore the choice. What must not be lost is the warning: the
    /// note is the only thing left telling the user the setting is inert for
    /// Granite and MOSS, so its presence is asserted harder than before.
    ///
    /// VOXTRAL JOINED THE WARNING GROUP 2026-07-31. It had been in the group
    /// below — "really uses it" — on the strength of a sidecar comment. Measured
    /// instead: its checkpoint is `voxtral_realtime`, whose `generate()` has no
    /// language parameter, and no language / "fr" / "de" / nonsense "xx" all
    /// returned the same 226 characters without raising. Only Whisper and Qwen3
    /// genuinely act on the setting; three of the five only look like they do,
    /// which is exactly why the note is asserted rather than trusted.
    /// Every chunked model's picker is usable, and every one that cannot act on
    /// the choice says so.
    ///
    /// ⚠ THE POPULATION IS DERIVED FROM `ModelCatalog.chunked`, NOT LISTED. It
    /// was a hand-written list of ids until 2026-09-02, and VibeVoice was added
    /// that day without appearing in it — so this test went green while saying
    /// nothing at all about the new model. That is the under-reporting shape
    /// this project keeps finding (the MPS-fuse pin named five files while seven
    /// carried a fuse); a list beside a catalog only ever drifts one way.
    ///
    /// Now a new model must be CLASSIFIED to pass: either it is one of the two
    /// that genuinely honour the flag, or it must carry the note. Silence is no
    /// longer an option.
    func testEveryChunkedModelIsSelectableAndSaysIfTheChoiceIsInert() {
        // The only two the setting actually reaches. Every measurement behind
        // this is in `noLanguageParameterNote` — Granite, MOSS, Voxtral and
        // VibeVoice were each driven directly and returned identical output for
        // different codes.
        let honoursLanguage: Set<String> = ["whisper", "qwen3"]

        XCTAssertFalse(ModelCatalog.chunked.isEmpty, "no models to sweep")
        for model in ModelCatalog.chunked {
            XCTAssertTrue(Languages.acceptsLanguage(model: model.id),
                          "\(model.id): the picker must be usable — a control that "
                          + "vanishes reads as a bug, one with a reason reads as "
                          + "an answer (the 2026-07-31 reversal)")
            let note = Languages.noLanguageParameterNote(forModel: model.id)
            if honoursLanguage.contains(model.id) {
                XCTAssertNil(note,
                             "\(model.id) is one of the two the setting reaches, "
                             + "so a warning here would be a lie in the other "
                             + "direction")
            } else {
                XCTAssertNotNil(note,
                                "\(model.id): a model that ignores the language "
                                + "MUST say so — dropping the note leaves a silent "
                                + "lie, and a NEW model with no note fails here "
                                + "rather than slipping through")
                XCTAssertTrue(note?.contains("will not change the transcript") ?? false,
                              "\(model.id): the note must state the setting has no "
                              + "effect, in those words")
            }
        }

        // The diarization-role MOSS id is not in `chunked` and is checked apart:
        // it reaches the same picker through `diarizationEngineValue`.
        XCTAssertNotNil(Languages.noLanguageParameterNote(forModel: "moss-diar"))
    }

    /// The choice now travels the WHOLE path rather than being dropped at the
    /// Swift boundary — deliberately, so there is exactly one place where it
    /// stops mattering (inside a model with no such concept) instead of two.
    /// `auto` still means "send no flag" for every model.
    func testGraniteAndMossForwardTheCodeTheyWillThenIgnore() {
        XCTAssertEqual(GraniteSpeechModel().languageArgument(for: "en"), "en")
        XCTAssertEqual(MossTranscribeDiarizeModel().languageArgument(for: "en"), "en")
        XCTAssertNil(GraniteSpeechModel().languageArgument(for: "auto"))
        XCTAssertNil(MossTranscribeDiarizeModel().languageArgument(for: "auto"))

        XCTAssertEqual(Qwen3ASRModel().languageArgument(for: "id"), "id")
        XCTAssertEqual(WhisperLargeV3Model().languageArgument(for: "ro"), "ro")
        XCTAssertNil(VoxtralMiniModel().languageArgument(for: "auto"))
    }

    /// Every picker leads with `auto` and now has real rows behind it.
    /// MOSS's rows are this app's shared 30 because upstream publishes no roster
    /// — `supportedSentence` is what says so, and that pairing is asserted here
    /// so the rows can never appear without the caveat.
    func testPickerEntriesLeadWithAuto() {
        XCTAssertEqual(Languages.pickerEntries(forModel: "voxtral").first?.code, "auto")
        XCTAssertEqual(Languages.pickerEntries(forModel: "voxtral").count, 14)
        XCTAssertEqual(Languages.pickerEntries(forModel: "granite").first?.code, "auto")
        XCTAssertEqual(Languages.pickerEntries(forModel: "granite").count, 6)   // auto + 5
        XCTAssertEqual(Languages.pickerEntries(forModel: "moss").count, 31)     // auto + 30
        XCTAssertTrue(Languages.supportedSentence(forModel: "moss")
                        .contains("does not publish the list"),
                      "MOSS's rows are ours, not upstream's — the text must admit it")
        XCTAssertTrue(Languages.supportedSentence(forModel: "granite")
                        .contains("English, French, German, Spanish, Portuguese"))
    }

    // MARK: - resolve()

    func testResolveFallsBackToAutoForAnUnsupportedLanguage() {
        XCTAssertEqual(Languages.resolve(language: "id", forModel: "voxtral"), "auto")
        XCTAssertEqual(Languages.resolve(language: "ms", forModel: "granite"), "auto")
        XCTAssertEqual(Languages.resolve(language: "ro", forModel: "voxtral"), "auto")
    }

    func testResolveKeepsASupportedLanguage() {
        XCTAssertEqual(Languages.resolve(language: "id", forModel: "qwen3"), "id")
        XCTAssertEqual(Languages.resolve(language: "yue", forModel: "whisper"), "yue")
        XCTAssertEqual(Languages.resolve(language: "ja", forModel: "voxtral"), "ja")
    }

    /// A supported code now survives for Granite and MOSS too. `resolve` still
    /// does its real job: dropping a code the model's roster does not contain.
    func testResolveKeepsSupportedCodesForGraniteAndMoss() {
        XCTAssertEqual(Languages.resolve(language: "en", forModel: "granite"), "en")
        XCTAssertEqual(Languages.resolve(language: "en", forModel: "moss"), "en")
        // Japanese is NOT on the NAR card, so Granite must still drop it.
        XCTAssertEqual(Languages.resolve(language: "ja", forModel: "granite"), "auto")
    }

    func testResolveLeavesAutoAlone() {
        for id in ["whisper", "qwen3", "voxtral", "granite", "moss", "nonsense"] {
            XCTAssertEqual(Languages.resolve(language: "auto", forModel: id), "auto")
        }
    }

    /// An unknown id runs as Qwen3 (`ChunkedASRModelFactory`'s fallback), so the
    /// offered list has to be Qwen3's or the menu would describe a different
    /// model than the one that loads.
    func testUnknownModelIDGetsTheFullSetLikeItsFallback() {
        XCTAssertEqual(codes("nonsense"), codes("qwen3"))
        XCTAssertEqual(Languages.resolve(language: "id", forModel: "nonsense"), "id")
    }

    // MARK: - A stale stored value cannot reach a sidecar

    /// THE regression guard. `chunked.language` is deliberately not rewritten on
    /// disk when the model changes, so the read path has to hold the line: a
    /// leftover "id" with Voxtral selected must not become `--language id`.
    func testStaleIndonesianNeverReachesVoxtral() {
        UserDefaults.standard.set("voxtral", forKey: "chunked.model")
        UserDefaults.standard.set("id", forKey: "chunked.language")

        let config = ChunkedASRService.Config.fromSettings()
        XCTAssertEqual(config.language, "auto")

        var arguments = ["--model", config.repoID]
        if config.language != "auto" { arguments += ["--language", config.language] }
        XCTAssertFalse(arguments.contains("--language"))
        XCTAssertFalse(arguments.contains("id"))
    }

    /// The same stored value with a model that DOES support it is still passed —
    /// the guard must not have become "never send a language".
    func testASupportedLanguageIsStillPassed() {
        UserDefaults.standard.set("qwen3", forKey: "chunked.model")
        UserDefaults.standard.set("id", forKey: "chunked.language")
        XCTAssertEqual(ChunkedASRService.Config.fromSettings().language, "id")
    }

    /// End to end: a Granite session now really does carry the picked code into
    /// the sidecar config. The model ignores it — that is measured and recorded
    /// in `Languages.acceptsLanguage` — but nothing in the app silently eats it.
    func testGraniteSessionCarriesThePickedLanguage() {
        UserDefaults.standard.set("granite", forKey: "chunked.model")
        UserDefaults.standard.set("en", forKey: "chunked.language")
        XCTAssertEqual(ChunkedASRService.Config.fromSettings().language, "en")
        // An unsupported code is still resolved away before it can reach argv.
        UserDefaults.standard.set("ja", forKey: "chunked.language")
        XCTAssertEqual(ChunkedASRService.Config.fromSettings().language, "auto")
    }

    // MARK: - The realtime picker is untouched

    /// Nemotron publishes 40 locales without saying which, so its list stays
    /// exactly the six it always had. Applying the tiers here would be a claim
    /// about Nemotron nobody has checked.
    func testRealtimeListIsUnchanged() {
        XCTAssertEqual(Languages.realtime.map(\.code),
                       ["auto", "id", "en", "ms", "zh", "ja"])
    }
}
