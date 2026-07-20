import SwiftUI

/// Root layout: sidebar | divider | main area. Owns app-level state.
struct ContentView: View {
    @StateObject private var recorder = AudioRecorder()
    @State private var showSettings = false

    var body: some View {
        ZStack {
            HStack(spacing: 0) {
                SidebarView(recorder: recorder, openSettings: { showSettings = true })
                Rectangle()
                    .fill(Theme.divider)
                    .frame(width: 1)
                MainAreaView(recorder: recorder)
            }

            if recorder.state == .preparing || recorder.modelLoader.failureMessage != nil {
                LoadingOverlayView(loader: recorder.modelLoader)
            }

            if recorder.state == .processing {
                ProcessingOverlayView(recorder: recorder)
            }
        }
        .animation(.easeOut(duration: 0.2), value: recorder.state)
        .background(Theme.bg)
        .ignoresSafeArea()
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
    }
}
