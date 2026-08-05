import SwiftUI

@main
struct MeetingTranscriberApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 1100, minHeight: 700)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {

    /// Signal sources for the DEV path — see `installSignalHandlers`. Held so
    /// they outlive `applicationDidFinishLaunching`; a released
    /// `DispatchSourceSignal` stops firing.
    private var signalSources: [DispatchSourceSignal] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        migrateSettings()
        installSignalHandlers()
        // Start the ATND1061 beam listener if it is enabled. It runs for the
        // whole app lifetime — independent of recording and of the TCP control
        // link — and re-syncs itself on later settings changes.
        // Bring the TCP control link up first when it is already configured: the
        // beam listener is gated on it, so auto-connecting here is what makes the
        // multicast stream arrive without anyone pressing Connect.
        Task { @MainActor in
            ATNDControlService.shared.autoConnectIfConfigured()
            ATNDBeamService.shared.syncWithSettings()
        }
        // Bring window to front when launched via `swift run`
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // Dock icon: a `swift run` executable has no app bundle, so set it at
        // runtime from the AppIcon in Assets.xcassets. Try the compiled catalog
        // first (Xcode builds run actool), then fall back to the raw PNG that
        // `swift build` copies verbatim into Bundle.module.
        //
        // Only do this for the bare `swift run` executable. Inside a real .app
        // bundle the compiled Assets.car already provides the system-themed
        // (light/dark/tinted) icon; forcing a single static NSImage here would
        // override that and defeat the appearance variants.
        if !Bundle.main.bundlePath.hasSuffix(".app") {
            let bundle = Bundle.module
            let icon = bundle.image(forResource: NSImage.Name("AppIcon"))
                ?? bundle.url(forResource: "1024", withExtension: "png",
                              subdirectory: "Assets.xcassets/AppIcon.appiconset")
                    .flatMap { NSImage(contentsOf: $0) }
            if let icon {
                NSApp.applicationIconImage = icon
            }
        }
    }

    // MARK: - Quitting while a recording is open

    /// Quit requested (⌘Q, the Quit menu item, a logout, `osascript quit`).
    ///
    /// An open `AVAudioFile` writes its WAV `data` chunk size only when it is
    /// RELEASED. Until 2026-08-05 nothing closed it on the way out, so quitting
    /// mid-recording left a header saying **0 frames** over a file full of audio —
    /// and every stage in this app then read it as an empty recording. 13 such
    /// files, ~1.7 GB, were found on the owner's machine, one holding 34.1 minutes
    /// of real speech. See `AudioRecorder.finalizeRecordingFiles`.
    ///
    /// `.terminateNow`, deliberately. Closing the files is a handful of bytes and
    /// completes immediately, so there is nothing to wait for — and asking for
    /// `.terminateLater` would make quitting depend on a reply this app would then
    /// have to guarantee. The transcript is NOT rescued here: the user asked to
    /// quit, and the audio is the only thing that cannot be reproduced afterwards.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        AudioRecorder.active?.finalizeRecordingFiles()
        return .terminateNow
    }

    /// The DEV path: `swift run` interrupted with ⌃C, and `kill` (SIGTERM).
    ///
    /// Neither reaches `applicationShouldTerminate` — the process is signalled,
    /// AppKit is never asked — and ⌃C on `swift run` is how this app is stopped
    /// during development, which is the likeliest origin of the 13 broken files.
    ///
    /// `signal(..., SIG_IGN)` first, then a `DispatchSourceSignal`: the default
    /// disposition would kill the process before any Swift ran, and a classic
    /// C signal handler may not call arbitrary Swift at all. A dispatch source
    /// delivers on a normal queue, where closing a file is ordinary work.
    ///
    /// SIGKILL (Force Quit, `kill -9`) is deliberately absent — it cannot be
    /// caught by anyone, so a recording lost that way is only recoverable with
    /// `scripts/tools/repair-wav-header.py`, never preventable.
    private func installSignalHandlers() {
        for sig in [SIGINT, SIGTERM] {
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler {
                MainActor.assumeIsolated {
                    AudioRecorder.active?.finalizeRecordingFiles()
                }
                exit(sig == SIGINT ? 130 : 143)   // conventional 128 + signal
            }
            source.resume()
            signalSources.append(source)
        }
    }

    /// One-time UserDefaults migrations. Idempotent: each only fires while the
    /// new key is still unset, so it is a no-op on every later launch.
    private func migrateSettings() {
        let d = UserDefaults.standard
        // 2026-07-16: overlap repair gained a second engine (DiCoW), so the
        // on/off switch stopped being MossFormer2-specific and was renamed
        // overlap.mossformer.enabled → overlap.repair.enabled. Carry the
        // owner's existing choice over instead of silently resetting it.
        if d.object(forKey: "overlap.repair.enabled") == nil,
           let legacy = d.object(forKey: "overlap.mossformer.enabled") as? Bool {
            d.set(legacy, forKey: "overlap.repair.enabled")
        }
    }
}
