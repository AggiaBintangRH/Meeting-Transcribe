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
                            VStack(alignment: .leading, spacing: 6) {
                                Text("SPEAKER UNKNOWN")
                                    .font(.system(size: 14, weight: .heavy))
                                    .kerning(0.8)
                                    .foregroundColor(Theme.speakerNameText)
                                Text(recorder.partialTranscript)
                                    .font(.system(size: 21))
                                    .lineSpacing(5)
                                    .italic()
                                    .foregroundColor(Theme.bodyTextConfirmed)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .modifier(TranscriptCard())
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
                if let speaker = row.speaker, let id = row.speakerID {
                    Button {
                        renameText = speaker
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
                        .foregroundColor(Theme.speakerNameText)
                    }
                    .buttonStyle(.plain)
                    .help("Rename this speaker (saved for future meetings)")
                } else if let speaker = row.speaker {
                    Text(speaker.uppercased())
                        .font(.system(size: 14, weight: .heavy))
                        .kerning(0.8)
                        .foregroundColor(Theme.speakerNameText)
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
            }

            Text(row.text)
                .font(.system(size: 21))
                .lineSpacing(5)
                .foregroundColor(Theme.bodyTextConfirmed)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
        .modifier(TranscriptCard())
    }

    private func statusLine(_ text: String) -> some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.mini).tint(Theme.teal)
            Text(text)
                .font(.system(size: 11))
                .foregroundColor(Theme.textFaint)
        }
    }

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
private struct TranscriptCard: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.rowCardBackground, in: RoundedRectangle(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Theme.rowCardBorder, lineWidth: 1)
            )
    }
}
