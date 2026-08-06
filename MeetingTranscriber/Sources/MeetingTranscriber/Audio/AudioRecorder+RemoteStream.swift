import Foundation

// The Remote stream (dual-stream): channel/URL resolution, the silence gate,
// chunk transcription and its leg of the stop gate, plus this domain's own log
// writer. Moved verbatim from the core file.
extension AudioRecorder {

    // MARK: - Dual-stream (Office + Remote) helpers

    /// Final validation of the Remote channel against the format the engine is
    /// actually delivering, run after `MicrophoneSettings.resolve` has already
    /// checked it against the device's *advertised* channel count.
    ///
    /// The re-check exists because of Aggregate Devices: the owner's Office array
    /// and the loopback input are combined in Audio MIDI Setup, and the aggregate
    /// can present a different channel layout than the saved selection assumed
    /// (a sub-device unplugged, reordered, or not yet running). Returning nil
    /// degrades to single-stream — the same reject-don't-relocate rule as
    /// `MicrophoneSettings.resolve`: silently moving Remote to another channel
    /// would record whatever happens to be there without telling anyone.
    ///
    /// - Parameters:
    ///   - wanted: the resolved `MicrophoneSettings.remoteChannel`, or nil.
    ///   - officeChannel: the office channel ALREADY clamped to the live format.
    ///   - liveChannelCount: `input.outputFormat(forBus: 0).channelCount`.
    nonisolated static func resolveRemoteChannel(_ wanted: Int?,
                                                 officeChannel: Int,
                                                 liveChannelCount: Int) -> Int? {
        guard let wanted, wanted >= 0 else { return nil }
        guard wanted < liveChannelCount else { return nil }   // aggregate presents fewer channels
        guard wanted != officeChannel else { return nil }     // one channel cannot be both roles
        return wanted
    }

    /// Remote file URL derived from the Office one: `meeting-<stamp>.wav` →
    /// `meeting-<stamp>-remote.wav`, same directory, same stamp. Derived rather
    /// than re-formatted from `Date()` so the pair can never carry two stamps.
    nonisolated static func remoteURL(forOffice office: URL) -> URL {
        let ext = office.pathExtension
        let base = office.deletingPathExtension().lastPathComponent
        let name = ext.isEmpty ? "\(base)-remote" : "\(base)-remote.\(ext)"
        return office.deletingLastPathComponent().appendingPathComponent(name)
    }

    // MARK: Remote stream — transcription, gating and rows

    /// RMS below which a remote chunk is treated as silence and never sent to the
    /// sidecar. An idle conferencing channel is otherwise a standing ~14 % GPU
    /// cost every chunk interval for a guaranteed-empty transcript (measured duty
    /// per 30 s chunk on this M4: Qwen3 4.3 s, Whisper 4.2 s, Granite 5.6 s).
    ///
    /// 0.004 ≈ −48 dBFS: a digital loopback carrying nothing sits at or very near
    /// 0.0, and even room-noise-through-a-codec stays far below this, while
    /// ordinary speech is an order of magnitude above it. Erring low is the safe
    /// direction — a false "not silent" costs one wasted transcription, a false
    /// "silent" would drop real speech.
    nonisolated static let remoteSilenceRMS: Float = 0.004

    /// Why this remote chunk is not worth transcribing, or nil to go ahead.
    /// Pure, so the gate can be tested without audio hardware or a sidecar.
    nonisolated static func remoteChunkSkipReason(_ samples: [Float],
                                                  threshold: Float = remoteSilenceRMS) -> String? {
        // Under half a second there is nothing an ASR model can usefully say, and
        // a stray fragment would still cost a full round trip.
        guard samples.count >= 8_000 else {
            return "under 0.5s (\(samples.count) samples)"
        }
        let level = AudioBufferProcessor.rms(samples)
        guard level >= threshold else {
            return String(format: "near-silent (rms %.5f < %.5f)", level, threshold)
        }
        return nil
    }

    /// Startup refusal for the one dual-stream configuration that cannot work:
    /// Voxtral needs ~27 s to transcribe a 30 s chunk (Qwen3 4.3 s, Whisper 4.2 s,
    /// Granite 5.6 s, all measured on this M4), i.e. ~90 % duty for a SINGLE
    /// stream. A second stream pushes it past 100 %: chunk N+1 arrives before N
    /// has finished and the sidecar's queue grows without bound for the rest of
    /// the meeting. Nil = start normally.
    ///
    /// Refusal rather than a silent fallback to another model: the owner selects
    /// chunked models deliberately, on measured WER, and quietly substituting one
    /// would make the transcript's provenance a lie.
    nonisolated static func dualStreamRefusalMessage(remoteChannel: Int?,
                                                     chunkedModelID: String) -> String? {
        guard remoteChannel != nil, chunkedModelID == "voxtral" else { return nil }
        return "Voxtral cannot transcribe two streams. It needs about 27 s per 30 s chunk "
             + "(Qwen3 4.3 s, Whisper 4.2 s, Granite 5.6 s), so the Remote stream would fall "
             + "permanently behind. Pick another chunked model in Settings → Models → Chunked, "
             + "or turn the Remote channel off in Settings → Microphone."
    }

    /// Startup refusal for a Remote channel with the chunked pass switched OFF.
    ///
    /// The Remote stream's text comes ENTIRELY from chunked ASR — there is no
    /// realtime-only remote transcript — and `prepareAndCapture` already declines
    /// to build a remote resampler when no chunked sidecar exists (`remoteWanted`),
    /// deliberately, so the remote 16 kHz buffer does not grow all meeting for
    /// audio nothing would consume. That is correct, but on its own it is SILENT:
    /// the user would configure two streams, record a hybrid meeting, and get one.
    ///
    /// So it is refused loudly instead, the same rule as the two refusals above.
    /// The single-stream case is deliberately NOT refused: switching the pass off
    /// with one microphone is a legitimate choice (realtime captions only), and
    /// the Chunked tab states its cost before the toggle.
    nonisolated static func chunkedOffRefusalMessage(remoteChannel: Int?,
                                                     chunkedEnabled: Bool) -> String? {
        guard remoteChannel != nil, !chunkedEnabled else { return nil }
        return "The Remote stream is transcribed only by the accurate transcript pass, and "
             + "that pass is switched off in Settings → Models → Chunked. Recording now would "
             + "capture the Remote channel and produce no text for it at all. Switch the pass "
             + "back on, or turn the Remote channel off in Settings → Microphone."
    }

    /// Startup refusal for the one MOSS configuration that cannot keep up:
    /// Voxtral as the chunked ASR while MOSS is the diarization engine.
    ///
    /// The arithmetic, all measured on this M4 per 30 s chunk: Voxtral ~27 s,
    /// MOSS ~6.4 s on MPS. They are two separate processes but contend for the
    /// same GPU and are driven from the same chunk boundary, so the work is
    /// largely serialized — ~33 s against a 30 s budget, i.e. over 100 % duty
    /// before the meeting even starts, and every chunk after that arrives while
    /// the previous one is still running. Every other chunked model leaves room
    /// (Qwen3 4.3 + 6.4, Whisper 4.2 + 6.4, Granite 5.6 + 6.4).
    ///
    /// MOSS in BOTH roles is never refused, however slow the model: that is one
    /// process doing one forward pass (`ModelLoader.needsSecondMossProcess`), so
    /// there is no second cost to add. `remoteChannel` is taken for the same
    /// reason `dualStreamRefusalMessage` takes it — a Remote stream doubles the
    /// MOSS work — but 2 × 6.4 = 12.8 s of 30 s is 43 % duty, which is fine, so
    /// it only sharpens the message rather than causing a refusal of its own.
    ///
    /// Refusal rather than a silent model swap, for exactly the reason the
    /// dual-stream one is: the owner picks chunked models deliberately, on
    /// measured WER. Nil = start normally.
    nonisolated static func mossRefusalMessage(chunkedModelID: String,
                                               diarizationEngine: String,
                                               remoteChannel: Int?) -> String? {
        guard diarizationEngine == ModelLoader.mossEngineID else { return nil }
        guard chunkedModelID == "voxtral" else { return nil }
        let remoteNote = remoteChannel != nil
            ? " The Remote stream doubles the MOSS work on top of that." : ""
        return "Voxtral cannot run alongside the MOSS diarization engine. Voxtral needs about "
             + "27 s per 30 s chunk and MOSS about 6.4 s, and they contend for the same GPU from "
             + "the same chunk boundary — over 100 % of the 30 s budget, so every chunk would "
             + "arrive before the previous one finished.\(remoteNote) Pick another chunked model "
             + "in Settings → Models → Chunked (Qwen3 4.3 s, Whisper 4.2 s, Granite 5.6 s), or "
             + "switch the diarization engine back to pyannote in Settings → Models → Diarization."
    }

    /// Hand the remote audio accumulated since the last boundary to the chunked
    /// sidecar's file-transcribe path, unless the silence gate rejects it.
    /// Always clears `remoteChunkAudio` — a skipped chunk's audio is dropped, not
    /// carried into the next window, so remote windows keep matching office ones.
    func flushRemoteChunk(window: ClosedRange<Double>, chunked: ChunkedASRService?) {
        guard remoteStreamActive else { return }
        let samples = remoteChunkAudio
        remoteChunkAudio = []
        guard let chunked else { return }
        if let reason = Self.remoteChunkSkipReason(samples) {
            dualStreamLog("SKIP remote [\(fmt(window.lowerBound))-\(fmt(window.upperBound))] \(reason)")
            // No confirmed row is coming for this window, so nothing would ever
            // replace the caption — drop it rather than leave it hanging under
            // the transcript for the whole of the next chunk interval.
            remoteCaption.commit()
            return
        }
        remotePendingChunks += 1
        Task { [weak self] in
            guard let self else { return }
            let url = await Task.detached(priority: .utility) {
                Self.writeTempWAV(samples: samples, prefix: "remote-chunk")
            }.value
            guard let url else {
                self.remoteChunkError = "Could not write remote audio for transcription"
                self.dualStreamLog("FAIL remote [\(self.fmt(window.lowerBound))-"
                                   + "\(self.fmt(window.upperBound))] could not write a temp WAV")
                self.remoteCaption.commit()   // nothing will confirm this window
                self.finishRemoteChunk()
                return
            }
            do {
                let result = try await chunked.transcribeFile(path: url.path)
                let trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    self.dualStreamLog("remote [\(self.fmt(window.lowerBound))-"
                                       + "\(self.fmt(window.upperBound))] empty transcript")
                } else {
                    // Remote rows carry the same ASR confidence office rows do:
                    // it is the SAME model over unmixed audio of one real stream,
                    // so the number means exactly what it means for the room.
                    self.remoteSegments.append(RemoteSegment(text: trimmed, window: window,
                                                             conf: result.conf))
                    self.rebuildDisplayRows()
                    self.dualStreamLog("remote [\(self.fmt(window.lowerBound))-"
                                       + "\(self.fmt(window.upperBound))] \(trimmed)")
                }
            } catch {
                // Same philosophy as `onChunkError`: log and carry on. Remote
                // trouble must never cost the user their office transcript.
                self.remoteChunkError = "Some remote audio could not be transcribed — "
                                      + "see logs/dual-stream.log"
                self.dualStreamLog("FAIL remote [\(self.fmt(window.lowerBound))-"
                                   + "\(self.fmt(window.upperBound))] \(error.localizedDescription)")
            }
            try? FileManager.default.removeItem(at: url)
            // This window is settled either way (transcribed, empty or failed):
            // the caption has served its purpose and must not outlive the row —
            // or the absence of one — that answers for the same audio.
            self.remoteCaption.commit()
            self.finishRemoteChunk()
        }
    }

    /// One remote request settled (either way) — maybe complete the stop gate.
    func finishRemoteChunk() {
        remotePendingChunks = max(0, remotePendingChunks - 1)
        checkRemoteChunksDone()
    }

    /// The remote leg of the stop gate, mirroring `checkLastChunkDone`.
    /// Idempotent; every remote exit path calls it.
    func checkRemoteChunksDone() {
        guard stopped, remotePendingChunks == 0, !remoteLastChunkDone else { return }
        remoteLastChunkDone = true
        remoteStopWatchdog?.cancel()
        remoteStopWatchdog = nil
        setStopStep("remote", remoteChunkError.map { .failed($0) } ?? .done)
        checkStopProcessingDone()
    }

    /// Remote can never hold the blocking stop overlay hostage. Each request
    /// already has the sidecar client's own 120 s timeout; this is the backstop
    /// for anything that never resolves at all, on the same pattern as
    /// `chunkWatchdog` / `finalDiarWatchdog`.
    /// `seconds` is scaled up by `stop()` when a full re-transcription runs —
    /// Remote's windows are re-run inside that pass, which takes minutes, so the
    /// fixed 180 s would fire while the pass was still working.
    func startRemoteStopWatchdog(seconds: Double = 180) {
        remoteStopWatchdog?.cancel()
        remoteStopWatchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, !self.remoteLastChunkDone else { return }
                let message = "Remote transcription timed out — see logs/dual-stream.log"
                self.remoteChunkError = message
                self.remotePendingChunks = 0
                self.remoteLastChunkDone = true
                self.remoteStopWatchdog = nil
                self.dualStreamLog(message)
                self.setStopStep("remote", .failed(message))
                self.checkStopProcessingDone()
            }
        }
    }

    /// Append a line to logs/dual-stream.log. Mirrors `overlapLog`: one line per
    /// decision, so a Remote stream that silently degraded to single-stream on the
    /// owner's machine is diagnosable after the fact.
    func dualStreamLog(_ message: String) {
        PythonRuntime.appendAppLog(name: "dual-stream", message: message)
    }
}
