import SwiftUI

/// Diarized transcript: one row per speaker turn — speaker name, time span,
/// and the text spoken. Realtime (Nemotron) text appears instantly and dimmed,
/// then is replaced in place by the accurate, speaker-split chunked rows.
struct TranscriptView: View {
    @ObservedObject var recorder: AudioRecorder

    @State private var renamingID: Int?
    @State private var renameText = ""

    private var isEmpty: Bool {
        recorder.displayRows.isEmpty
            && recorder.partialTranscript.isEmpty
            && recorder.remoteCaption.text.isEmpty
            && recorder.chunkedError == nil
            && !recorder.overlapRepairing
            && recorder.overlapRepairError == nil
    }

    var body: some View {
        if isEmpty {
            emptyState
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        ForEach(recorder.displayRows) { row in
                            rowView(row)
                        }

                        if !recorder.partialTranscript.isEmpty {
                            partialCard(label: recorder.partialSpeakerName ?? "SPEAKER UNKNOWN",
                                        labelColor: Theme.speakerNameText,
                                        text: recorder.partialTranscript)
                        }

                        // The Remote stream's own live caption, below the office
                        // one (the room is the primary record). Amber is the
                        // Remote capture role everywhere in the app, so the two
                        // provisional cards can never be mistaken for each other.
                        // It disappears the moment its confirmed remote row lands.
                        if !recorder.remoteCaption.text.isEmpty {
                            partialCard(label: AudioRecorder.remoteSpeakerLabel,
                                        labelColor: Theme.remoteRole,
                                        text: recorder.remoteCaption.text,
                                        accent: Theme.remoteRoleBorder)
                        }

                        if recorder.chunkedBusy {
                            statusLine("Refining with \(recorder.chunkedModelName)…")
                        }
                        if recorder.diarizing {
                            statusLine("Identifying speakers…")
                        }
                        if recorder.overlapRepairing {
                            statusLine("Recovering overlapped speech \(recorder.overlapRepairProgress ?? "")…")
                        }

                        if let error = recorder.chunkedError ?? recorder.diarizationError ?? recorder.overlapRepairError {
                            Text(error)
                                .font(.system(size: 11))
                                .foregroundColor(Theme.red)
                                .fixedSize(horizontal: false, vertical: true)
                                .textSelection(.enabled)
                        }

                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(24)
                }
                .onChange(of: recorder.displayRows) { _, _ in
                    withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom") }
                }
                .onChange(of: recorder.partialTranscript) { _, _ in
                    proxy.scrollTo("bottom")
                }
                .onChange(of: recorder.remoteCaption) { _, _ in
                    proxy.scrollTo("bottom")
                }
                .alert("Rename speaker", isPresented: Binding(
                    get: { renamingID != nil },
                    set: { if !$0 { renamingID = nil } }
                )) {
                    TextField("Name", text: $renameText)
                    Button("Cancel", role: .cancel) { renamingID = nil }
                    Button("Save") {
                        if let id = renamingID {
                            let trimmed = renameText.trimmingCharacters(in: .whitespaces)
                            if !trimmed.isEmpty { recorder.renameSpeaker(id: id, to: trimmed) }
                        }
                        renamingID = nil
                    }
                } message: {
                    Text("This name is saved to the speaker's voice profile and reused in future meetings.")
                }
            }
        }
    }

    // MARK: - Row

    @ViewBuilder
    private func rowView(_ row: AudioRecorder.SpeakerUtterance) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                // MOSS ids are excluded from the renameable branch: a MOSS label
                // ("Speaker 1" since 2026-07-31, formerly "S1 · #7") is the model's
                // own name for a speaker inside ONE chunk, backed by no voice
                // profile and no cluster, so there is nothing a new name could be
                // saved to and nothing it would apply to in the next chunk.
                // `renameSpeaker` refuses these ids anyway; this is what stops the
                // app offering an action it will not take. Note the label now LOOKS
                // exactly like a pyannote one — this id check is the only thing
                // distinguishing them here, so do not relax it to a name test.
                if let speaker = row.speaker, let id = row.speakerID,
                   id < AudioRecorder.mossIDBase {
                    Button {
                        // Remote rows are renameable too, but what is stored is the
                        // PROFILE name ("R1"), not the composed row label — strip
                        // the "Remote Speaker - " prefix or it would be saved into
                        // the profile and prefixed again on the next rebuild.
                        renameText = row.isRemote ? AudioRecorder.remoteBaseName(speaker) : speaker
                        renamingID = id
                    } label: {
                        HStack(spacing: 4) {
                            Text(speaker.uppercased())
                                .font(.system(size: 14, weight: .heavy))
                                .kerning(0.8)
                            Image(systemName: "pencil")
                                .font(.system(size: 12, weight: .bold))
                                .opacity(0.6)
                        }
                        // Amber is the Remote capture role everywhere in the app,
                        // teal/default is Office — the two identity spaces must
                        // never be mistaken for each other, renameable or not.
                        .foregroundColor(row.isRemote ? Theme.remoteRole : Theme.speakerNameText)
                    }
                    .buttonStyle(.plain)
                    .help(row.isRemote
                          ? "Rename this remote speaker (saved for future meetings)"
                          : "Rename this speaker (saved for future meetings)")
                } else if let speaker = row.speaker {
                    // Undiarized remote rows land here (no speakerID → not
                    // renameable) and keep the same amber colour.
                    Text(speaker.uppercased())
                        .font(.system(size: 14, weight: .heavy))
                        .kerning(0.8)
                        .foregroundColor(row.isRemote ? Theme.remoteRole : Theme.speakerNameText)
                } else {
                    // Not diarized yet — placeholder until speaker turns arrive.
                    Text("SPEAKER UNKNOWN")
                        .font(.system(size: 14, weight: .heavy))
                        .kerning(0.8)
                        .foregroundColor(Theme.speakerNameText)
                }

                if let range = timeRange(row) {
                    Text(range)
                        .font(.system(size: 14, weight: .medium).monospacedDigit())
                        .foregroundColor(Theme.timeRangeText)
                }

                if row.overlapped {
                    HStack(spacing: 3) {
                        Image(systemName: "waveform")
                            .font(.system(size: 12, weight: .bold))
                        Text("overlap")
                            .font(.system(size: 14, weight: .medium))
                    }
                    .foregroundColor(Theme.overlapTagText)
                    .help("Spoken over another speaker — text here is approximate")
                }

                // Confidence, deliberately quiet: smaller and fainter than
                // everything beside it, so a row reads as speech first and as
                // measurement second. LAST in the row so it is what gives way
                // when the overlap tag and a long speaker name compete for width.
                // Absent entirely when neither number was measured — an empty
                // slot says "not measured", which a "0.00" would contradict.
                if let confidence = confidenceText(row) {
                    Text(confidence)
                        .font(.system(size: 11, weight: .regular).monospacedDigit())
                        .foregroundColor(Theme.textFaint)
                        .lineLimit(1)
                        .help(Self.confidenceHelp)
                }
            }

            Text(row.text)
                .font(.system(size: 21))
                .lineSpacing(5)
                .foregroundColor(Theme.bodyTextConfirmed)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
        .modifier(TranscriptCard(accent: row.isRemote ? Theme.remoteRoleBorder : nil))
    }

    /// A provisional (realtime) card: speaker label + italic, not-yet-confirmed
    /// text. Shared by the Office and Remote captions so the two stay visually
    /// identical apart from the Remote accent (label colour + card spine).
    private func partialCard(label: String, labelColor: Color, text: String,
                             accent: Color? = nil) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label.uppercased())
                .font(.system(size: 14, weight: .heavy))
                .kerning(0.8)
                .foregroundColor(labelColor)
            Text(text)
                .font(.system(size: 21))
                .lineSpacing(5)
                .italic()
                .foregroundColor(Theme.bodyTextConfirmed)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .modifier(TranscriptCard(accent: accent))
    }

    private func statusLine(_ text: String) -> some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.mini).tint(Theme.teal)
            Text(text)
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
        }
    }

    // MARK: - Confidence

    /// `spk 0.87 · asr 0.92` — whichever parts were actually measured, or nil
    /// when neither was. Two decimals: the numbers are for comparing rows at a
    /// glance, and more digits would suggest a precision neither measure has.
    private func confidenceText(_ row: AudioRecorder.SpeakerUtterance) -> String? {
        var parts: [String] = []
        if let speaker = row.speakerConf {
            parts.append(String(format: "spk %.2f", speaker))
        }
        if let asr = row.asrConf {
            parts.append(String(format: "asr %.2f", asr))
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// Says what each number is, what it is NOT, and why either can be missing.
    /// The last part matters most: a missing number is a normal, meaningful
    /// state here, and without this the user has no way to tell it from a bug.
    private static let confidenceHelp = """
    spk — how strongly this voice matched its saved speaker profile \
    (cosine similarity, 0.50–1.00; below 0.50 never matches). Where a row spans \
    several turns, the lowest of them is shown.

    asr — how confident the transcription model was in these words: the average \
    per-token probability. It is NOT a percent-correct — a fluent mishearing can \
    still score high.

    A number is left out when it was never measured, which is normal:
    • asr is missing unless the chunked model is Whisper — Qwen3, Granite and \
    Voxtral report no confidence at all.
    • spk is missing on a speaker's first appearance (a new profile matched \
    nothing), on speakers identified by ATND beam position, on live \
    (not yet confirmed) text, and on rows relabelled by the short-island rule.
    • Both are missing on rows produced by overlap repair, whose text comes from \
    separated audio rather than the original recording.
    """

    private func timeRange(_ row: AudioRecorder.SpeakerUtterance) -> String? {
        guard let start = row.start, let end = row.end else { return nil }
        return "\(clock(start)) – \(clock(end))"
    }

    private func clock(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        return String(format: "%02d:%02d", total / 60, total % 60)
    }

    private var emptyState: some View {
        VStack {
            Spacer()
            VStack(spacing: 10) {
                Text("No transcript yet")
                    .font(.system(size: 26, weight: .bold))
                    .foregroundColor(Theme.textPrimary)
                Text("Start recording from the left panel.")
                    .font(.system(size: 13))
                    .foregroundColor(Theme.textDim)
            }
            Spacer()
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }
}

/// Wraps a single transcript turn (label + time + text) in a bordered card.
///
/// `accent` marks the card as Remote: amber border plus an amber spine down the
/// leading edge. When both streams pick up the same speech the transcript shows
/// two rows for it, and the speaker label alone is too easy to skim past — the
/// spine makes "this row came from the call, not the room" readable at a glance
/// without touching the layout, spacing or text of any row. Office rows pass nil
/// and render exactly as before.
private struct TranscriptCard: ViewModifier {
    var accent: Color? = nil

    func body(content: Content) -> some View {
        content
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.rowCardBackground, in: RoundedRectangle(cornerRadius: 12))
            .overlay(alignment: .leading) {
                if let accent {
                    UnevenRoundedRectangle(topLeadingRadius: 12, bottomLeadingRadius: 12)
                        .fill(accent)
                        .frame(width: 3)
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(accent ?? Theme.rowCardBorder, lineWidth: 1)
            )
    }
}
