import SwiftUI

/// Root layout: sidebar | divider | main area. Owns app-level state.
struct ContentView: View {
    @StateObject private var recorder = AudioRecorder()
    @State private var showSettings = false

    var body: some View {
        ZStack {
            HStack(spacing: 0) {
                SidebarView(recorder: recorder,
                            openSettings: { showSettings = true })
                Rectangle()
                    .fill(Theme.divider)
                    .frame(width: 1)
                MainAreaView(recorder: recorder)
            }

            if recorder.state == .preparing || recorder.modelLoader.failureMessage != nil {
                LoadingOverlayView(loader: recorder.modelLoader)
            }

            // Outlives `.processing` when a leg failed, so a red row can be
            // read instead of flashing past as the panel closes.
            if recorder.showsStopOverlay {
                ProcessingOverlayView(recorder: recorder)
            }

            // Last, so it draws ABOVE the others. An audio engine that dies
            // during the stop passes can raise both at once, and the one the
            // user must act on is the error.
            if recorder.showsErrorPopup {
                ErrorOverlayView(recorder: recorder)
            }

            // Position-ID enrollment prompt: only built when the feature is on
            // (diarizer exists); the view itself hides unless a speaker is
            // pending a name. Recording keeps running underneath.
            if let diarizer = recorder.positionDiarizer {
                VStack {
                    Spacer()
                    EnrollmentPromptView(diarizer: diarizer)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                .allowsHitTesting(diarizer.pendingEnrollment != nil)
            }
        }
        .animation(.easeOut(duration: 0.2), value: recorder.state)
        // `state` alone no longer decides either overlay: both can now outlive
        // the transition that used to carry them off screen.
        .animation(.easeOut(duration: 0.2), value: recorder.showsStopOverlay)
        .animation(.easeOut(duration: 0.2), value: recorder.showsErrorPopup)
        .background(Theme.bg)
        .ignoresSafeArea()
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
    }
}
