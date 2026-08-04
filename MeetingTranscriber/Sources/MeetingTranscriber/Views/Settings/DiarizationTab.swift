import SwiftUI

/// Models → Diarization: pick the engine, then its settings.
///
/// Three engines with genuinely different shapes, so the tab shows only the
/// selected one's blocks. Everything under `pyannoteSettings` is pyannote-specific
/// — a live pass, a cluster threshold and an overlap mode are all things the other
/// two simply do not have — and showing a control that does nothing is the exact
/// complaint the Overlap tab already had to fix once.
///
/// **THE SPECTRAL BLOCK SHOWS ITS INERT CONTROLS RATHER THAN HIDING THEM**, which
/// is the opposite of what the MOSS block does, and the difference is not style.
/// MOSS owns its own `moss.*` stop keys, so pyannote's `diarization.live` /
/// `diarization.continueOnStop` are genuinely not part of a MOSS session and the
/// tab has nothing to say about them. Spectral reads the SAME `diarization.*` keys
/// pyannote does — a user who set them under pyannote still has them set here, and
/// two of them will now be ignored. Hiding those is the failure the language-picker
/// work fixed on 2026-07-31: a control the engine cannot act on stays visible and
/// says so, rather than vanishing or quietly lying.
struct DiarizationTab: View {
    @AppStorage("diarization.engine")      private var engine = "pyannote"
    @AppStorage("diarization.enabled")     private var enabled = true
    @AppStorage("diarization.live")        private var live = true
    @AppStorage("diarization.finalPass")   private var finalPass = true
    @AppStorage("diarization.continueOnStop") private var continueOnStop = false
    @AppStorage("diarization.intervalSec") private var intervalSec = 30
    @AppStorage("diarization.resetOnStart") private var resetOnStart = true
    @AppStorage("diarization.detectOverlap") private var detectOverlap = true
    @AppStorage("diarization.numSpeakers") private var numSpeakers = 0
    @AppStorage("diarization.clusterThreshold") private var clusterThreshold = 0.6
    // The MOSS engine's own stop pair. Named `moss.*` rather than reusing
    // `diarization.*` because the defaults must differ: MOSS's tail-only default
    // is TRUE (today's behaviour), while `diarization.continueOnStop` is FALSE.
    @AppStorage("moss.finalPass")      private var mossFinalPass = true
    @AppStorage("moss.continueOnStop") private var mossTailOnly = true

    private var isMoss: Bool { engine == ModelLoader.mossEngineID }
    private var isSpectral: Bool { engine == ModelLoader.spectralEngineID }
    /// The ASR cadence MOSS rides today, for the comparison in the copy above.
    private var chunkedIntervalSec: Int {
        UserDefaults.standard.object(forKey: "chunked.intervalSec") as? Int ?? 30
    }

    var body: some View {
        Group {
            ForEach(ModelCatalog.diarizationEngines) { m in
                // `diarizationEngineValue`, not `m.id`: the MOSS card's catalog id
                // is "moss-diar" (the chunked entry already owns "moss") while the
                // setting's value is "moss". Writing the id here would store a
                // string neither the loader nor the recorder ever matches, and the
                // engine would silently never switch.
                let value = ModelCatalog.diarizationEngineValue(m)
                Button(action: {
                    withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                        engine = value
                    }
                }) {
                    ModelCardView(model: m, selected: engine == value)
                }
                .buttonStyle(.plain)
            }

            SettingToggle(label: "Enable speaker diarization (who spoke when)", isOn: $enabled)

            if isMoss {
                mossNotes
            } else if isSpectral {
                spectralNotes
            } else {
                pyannoteSettings
            }
        }
    }

    /// What choosing MOSS actually costs, stated plainly. The engine transcribes
    /// and labels in one pass, which means most of pyannote's machinery has no
    /// equivalent here — and the owner should read that as a trade-off list
    /// before a meeting, not discover it afterwards in the transcript.
    @ViewBuilder
    private var mossNotes: some View {
        SettingBlock(title: "How it works") {
            Text("One model returns the words, the speaker labels and the times together, per chunk. It runs on the GPU at about 6.4 s per 30 s chunk. If MOSS is also your chunked model, the same process does both jobs — one load, one pass.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }

        SettingBlock(title: "Labels are per chunk — and they no longer say so") {
            Text("Speakers are labelled inside each chunk only. Rows read Speaker 1, Speaker 2 … but that numbering RESTARTS every chunk: the Speaker 1 of one chunk and the Speaker 1 of the next are not known to be the same person, because the model makes no claim that they are. Rows are still kept apart internally, so two chunks never merge into one — but the names alone cannot be read as one person across the meeting. Linking a voice across chunks is not attempted rather than guessed at; for that, use pyannote, which keeps saved profiles.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }

        // The one lever that actually moves MOSS's diarization quality, and it is
        // not on this tab — so the tab has to say where it is. MEASURED
        // 2026-07-31 on a 3-speaker recording (78.9 s): at 30 s chunks the model
        // returned ['S01'], ['S01'], ['S01','S02'] — the first two chunks are
        // DIFFERENT people both numbered S01, because the boundary fell on the
        // speaker change. One pass over the same audio returned ['S01','S02','S03'],
        // all three separated. MOSS diarizes globally or not at all; chunking is
        // what breaks it. Cost is real too: audio costs ~13.2 context tokens per
        // second, so 120 s uses ~1 638 of the 2 048-token budget.
        SettingBlock(title: "Chunk length is the quality setting — and it lives on the Chunked tab") {
            Text("MOSS labels speakers by looking at one chunk as a whole, so the longer the chunk, the better it separates them. Measured on a three-speaker recording: at 30 s it called two different people “Speaker 1” in consecutive chunks, because the chunk boundary fell on the speaker change. Given the same audio in one pass it separated all three.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)

            Text("Raise Models → Chunked → Chunk interval to 60 s or 120 s for noticeably better speaker separation. It is that setting because MOSS is flushed on the ASR chunk boundary, not on the live diarization interval — identical windows are what let its labels line up with the text. The cost is that rows appear only once per interval, and a 120 s chunk uses about 80% of the model's token budget.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }

        SettingBlock(title: "Not available with this engine") {
            Text("No saved voice profiles and no renaming (a per-chunk label has nothing to save to). No spk confidence and no overlap tags. Remote rows stay “Remote Speaker - Speaker Unknown”. The ATND position layer and overlap repair both work from pyannote turns, so neither runs.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }

        // Only meaningful when a SECOND MOSS process exists (another model does
        // the ASR). In MOSS+MOSS one process does both and its tail is the
        // CHUNKED tail — governed by Models → Chunked → "Run a transcription pass at stop".
        // Saying so here is the whole reason this block is not silently inert.
        SettingBlock(title: "Run a diarization pass at stop") {
            SettingToggle(label: "Run a diarization pass at stop", isOn: $mossFinalPass)

            SettingToggle(label: "Continue from live labels (tail only)", isOn: $mossTailOnly)
                .disabled(!mossFinalPass)
                .opacity(mossFinalPass ? 1 : 0.45)

            Text(mossTailOnly
                 ? "On — at Stop, only the audio since the last chunk is labelled. This is what the app has always done."
                 : "Off — at Stop the whole recording is re-labelled in \(Int(AudioRecorder.mossFullPassWindowSec)) s windows instead of \(chunkedIntervalSec) s ones. THIS IS THE POINT of the full pass: MOSS numbers speakers within whatever it is shown, so a longer window separates them far better. Measured on a three-speaker recording, 30 s chunks called two different people “Speaker 1” in consecutive chunks; one pass over the same audio separated all three.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)

            if !mossTailOnly {
                Text("It re-runs the model over the whole meeting, so expect roughly 6.4 s of processing per 30 s of audio — about 13 minutes after Stop for a 60-minute meeting. Labels still restart per window, so this improves separation WITHIN each two minutes, not across the meeting.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.amber)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if !mossFinalPass {
                Text("Off — no labelling pass runs at Stop, so the last partial chunk keeps whatever the live pass gave it.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("These apply only when another model does the transcription and MOSS only labels. With MOSS as the chunked model too, one process does both jobs and Models → Chunked → “Run a transcription pass at stop” governs it.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
        }

        Text("Switch back to pyannote for voice profiles that persist across meetings, speaker renaming, overlap detection and the ATND position layer.")
            .font(.system(size: 11))
            .foregroundColor(Theme.textDim)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// What choosing the SPECTRAL engine actually costs and actually keeps.
    ///
    /// The shape of this block follows `mossNotes` deliberately — how it works,
    /// what it cannot do, then the controls that remain real — but the trade-off
    /// list is a different one, and in one place it runs the other way: this
    /// engine KEEPS saved voice profiles, renaming and `spk` confidence, which
    /// MOSS cannot. That is the 2026-07-30 pyannote/wespeaker split paying off
    /// (spectral emits run-local spans, so the same WeSpeaker step names them),
    /// and it is worth stating plainly rather than leaving the owner to assume a
    /// third engine costs whatever the second one did.
    ///
    /// **NUMBERS IN THIS COPY ARE MEASURED ON THIS MAC, 2026-08-04**, by driving
    /// `scripts/spectral/spectral-service.py` directly:
    /// `meeting-2026-07-31T08-24-06Z.wav` (396.2 s) → 17.3 s, and
    /// `meeting-2026-07-30T04-53-29Z.wav` (4027.4 s) → 169.3 s, i.e. 22.9x and
    /// 23.8x realtime. The queued-second-job figure is the same run with
    /// `meeting-2026-07-23T05-38-44Z.wav` (3630.7 s) written to the SAME stdin
    /// immediately after: it answered 151.1 s later, 320.4 s from dispatch.
    /// A speech-bearing file stands in for the Remote WAV there on purpose — every
    /// `-remote` capture on disk is digital silence (the dead BlackHole input), and
    /// silence is the cheap case, not the honest one. For the record it was also
    /// measured: the real silent 396 s remote WAV costs 1.6 s and returns 0 turns.
    /// No upstream quality claim is repeated here as a result — see the footer.
    @ViewBuilder
    private var spectralNotes: some View {
        Group {
            SettingBlock(title: "How it works") {
                Text("Silero VAD finds the speech, WeSpeaker embeds it, and the whole recording is clustered in one pass — globally, not window by window. It runs on the CPU in about 0.6 GB, at roughly one second of processing per 24 seconds of audio: measured on this Mac, a 6½-minute recording took 17 s and a 67-minute one took 2 min 49 s.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "No speaker names until you press Stop") {
                Text("This engine counts the speakers and clusters them over the whole file at once, so it has no per-chunk pass and cannot be given one: a 30-second window would be counted and clustered on its own, and its “Speaker 1” would have nothing to do with the next window's. During the meeting the transcript therefore carries no speaker names at all — they all arrive together when the single pass finishes after Stop.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("If the ATND array is your speaker source, its beam positions still label rows live throughout the meeting exactly as they do under pyannote; the clustered names then land on top at Stop.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // finalPass stays ENABLED under this engine even though switching it
            // off is refused at startup. Disabling the toggle would trap a user
            // who had already turned it off under pyannote: the refusal names this
            // control, and a control named by an error message must be reachable.
            SettingBlock(title: "Run a diarization pass at stop") {
                SettingToggle(label: "Run a diarization pass at stop", isOn: $finalPass)

                Text(finalPass
                     ? "Required by this engine, and it is the only pass there is. For pyannote this refines live labels; here it produces them."
                     : "This engine has no live pass, so with this off a recording would produce no speaker labels at all — starting will be refused, before the meeting rather than after it. Turn it back on, choose the pyannote engine (which does label live), or turn speaker diarization off entirely.")
                    .font(.system(size: 11))
                    .foregroundColor(finalPass ? Theme.textFaint : Theme.amber)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Shown, settable, and stated to be inert — the 2026-07-31 language
            // picker rule. Neither is destructive if left on (both are guarded at
            // dispatch and their buffers are dropped rather than accumulated), and
            // both are shared with pyannote, so clearing them here would silently
            // change a setting for the other engine.
            SettingBlock(title: "Settings this engine ignores") {
                SettingToggle(label: "Live labels during recording (chunked)", isOn: $live)
                Text("You can set this, but it will not change anything here — there is no per-chunk pass to run. The value is kept for the pyannote engine.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                SettingToggle(label: "Continue from live labels (tail only)", isOn: $continueOnStop)
                Text("Also settable and also ignored: with no live labels there is nothing for a tail to continue from, so the pass is always over the whole recording — the Remote recording in a dual-stream meeting included.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("The live diarization interval and the voice-separation threshold do not appear at all under this engine, because it has neither: there are no live windows to time, and its clustering has no threshold to set.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Overlapping speech") {
                Text("Every instant is assigned to exactly one speaker — the clustered frames are smoothed with Viterbi, which cannot return two labels at one time. So two of this engine's turns never overlap: rows are never flagged as overlapping, and overlap repair (MossFormer2 / DiCoW) never runs, because there are no overlap regions to find. Neither repair model is even loaded. Choose pyannote if you want overlaps detected and tagged.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Speaker profiles") {
                Text("Saved voice profiles, speaker renaming and the spk confidence number all work here exactly as they do under pyannote. This engine returns spans with run-local labels and the same WeSpeaker step puts names to them, so nothing about identity changes with the engine.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                SettingToggle(label: "Start each recording fresh (reset profiles)", isOn: $resetOnStart)
                Text(resetOnStart
                     ? "Every recording begins with an empty speaker store, so speakers are numbered from 1 each time. Turn off to remember voices across meetings."
                     : "Voices are remembered across meetings — a speaker matched in a past recording keeps the same name.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                Text("spk appears once a voice matches its saved profile; a speaker's first appearance has no match score.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Number of speakers") {
                Picker("", selection: $numSpeakers) {
                    Text("Auto").tag(0)
                    Text("2").tag(2)
                    Text("3").tag(3)
                    Text("4").tag(4)
                    Text("5").tag(5)
                    Text("6").tag(6)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text(numSpeakers == 0
                     ? "Auto — the count is estimated from the audio itself (GMM-BIC over every embedding in the file)."
                     : "Fixed at \(numSpeakers) — the estimate is skipped and the recording is clustered into exactly that many speakers.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Office + Remote (dual stream)") {
                Text("Supported, and not refused. The two recordings are diarized one after the other on a single connection rather than at the same time, so the wait after Stop is the sum of both. Measured on this Mac: a 67-minute recording finished 2 min 49 s after Stop and a second 61-minute one queued behind it a further 2 min 31 s — 5 min 20 s in total. That is a wait, not a backlog: unlike the per-chunk models this engine runs once at the end, so a second stream cannot fall behind the recording.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("Upstream reports a lower diarization error rate than pyannote community-1. That has NOT been measured on this project's recordings, so read it as the author's claim rather than a result here. Switch back to pyannote for speaker names during the meeting, overlap detection and tagging, overlap repair, and a tail pass that keeps numbering stable at Stop.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var pyannoteSettings: some View {
        Group {
            SettingBlock(title: "How it runs") {
                SettingToggle(label: "Live labels during recording (chunked)", isOn: $live)
                Text("Diarizes each chunk as it's recorded and matches voices against your saved speaker profiles, so names appear during the meeting instead of only at the end.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                SettingToggle(label: "Run a diarization pass at stop", isOn: $finalPass)
                Text("After you stop, run one more diarization pass to finalize speaker labels. Turn off to keep only the live labels.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                SettingToggle(label: "Continue from live labels (tail only)", isOn: $continueOnStop)
                    .disabled(!finalPass)
                Text(continueOnStop
                     ? "At stop, only the audio since the last live chunk is diarized and appended — speaker names and numbering stay stable. If live labels are off, the whole recording is diarized as one tail."
                     : "At stop, the entire recording is re-diarized and replaces the live labels — best global clustering, but speaker numbering can change.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Live diarization interval") {
                Picker("", selection: $intervalSec) {
                    Text("15s").tag(15)
                    Text("30s").tag(30)
                    Text("45s").tag(45)
                    Text("60s").tag(60)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(!live)

                Text("How often live diarization runs — separate from the chunked-ASR interval. Longer chunks give steadier voice matching and less CPU load; shorter chunks label speakers sooner. Only used when live labels are on.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Overlapping speech") {
                SettingToggle(label: "Detect overlapping speech (show both speakers)", isOn: $detectOverlap)
                Text(detectOverlap
                     ? "When two people talk at once, both are kept and their rows are flagged as overlapping. Note: on a single mic the words in an overlap are approximate."
                     : "Each moment is assigned to one speaker only (cleaner rows, no overlaps). Best when people rarely talk over each other.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Speaker profiles") {
                SettingToggle(label: "Start each recording fresh (reset profiles)", isOn: $resetOnStart)
                Text(resetOnStart
                     ? "Every recording begins with an empty speaker store, so speakers are numbered from 1 each time. Turn off to remember voices across meetings."
                     : "Voices are remembered across meetings — a speaker matched in a past recording keeps the same name.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)

                // Why the transcript's `spk` number is sometimes absent: it is
                // the profile-match score, and a first appearance matched nothing.
                Text("spk appears once a voice matches its saved profile; a speaker's first appearance has no match score.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            SettingBlock(title: "Number of speakers") {
                Picker("", selection: $numSpeakers) {
                    Text("Auto").tag(0)
                    Text("2").tag(2)
                    Text("3").tag(3)
                    Text("4").tag(4)
                    Text("5").tag(5)
                    Text("6").tag(6)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text(numSpeakers == 0
                     ? "Auto-detect — flexible, but may split or merge voices on hard audio."
                     : "Fixed at \(numSpeakers) — more accurate when you know the participant count.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
            }

            SettingBlock(title: "Voice-separation threshold — \(String(format: "%.2f", clusterThreshold))") {
                Slider(value: $clusterThreshold, in: 0.4...0.8, step: 0.05)
                Text("How readily two turns are judged the same voice. Lower merges more (fewer speakers); higher splits more (more speakers). The default 0.60 is robust for distinct voices — leave it there unless a specific room's voices are so alike they get merged, then nudge it up. Per-room calibration only, like the ATND merge threshold.")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("The pipeline and voice-embedding model load up front (visible in the loading overlay). Speaker profiles are saved locally under models/speaker-profiles and reused across meetings — rename a speaker in the transcript and it's remembered next time.")
                .font(.system(size: 11))
                .foregroundColor(Theme.textDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
