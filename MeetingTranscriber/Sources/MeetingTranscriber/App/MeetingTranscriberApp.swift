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
        enterFullScreenAtLaunch()
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

    // MARK: - Fullscreen at launch

    /// Set once the app has entered fullscreen, so a SECOND window can never
    /// drag the app back through the transition. `WindowGroup` will happily make
    /// another window (⌘N, state restoration), and each one arriving would
    /// otherwise re-enter — which on macOS reads as the app flickering between
    /// Spaces.
    private var didEnterFullScreen = false

    /// Remaining attempts to find the window. Bounded rather than a bare retry:
    /// an unbounded chain that never finds a window would poll the main runloop
    /// for the life of the process, and the symptom (a warm CPU, no visible
    /// fault) is the kind this project keeps having to hunt down.
    ///
    /// ⚠ MEASURED, because I first assumed the opposite and was wrong: on this
    /// Mac the window is already there on **attempt 1**, synchronously inside
    /// `applicationDidFinishLaunching`, so the retry never runs. It is kept as
    /// DEFENCE for the launch paths not measured here (the packaged `.app`,
    /// state restoration, a loaded machine), not because it is known to be
    /// needed — and the log line reports which attempt won, so the real answer
    /// is visible rather than assumed.
    ///
    /// The retry is nonetheless SPACED (`asyncAfter`, not bare `async`): chained
    /// `async` turns would burn all 120 attempts in milliseconds and give up
    /// before a slow window ever appeared, which would defeat the point of
    /// having a retry at all.
    private var fullScreenAttemptsLeft = 120
    private static let fullScreenRetryInterval = 0.05  // 120 x 0.05 s = 6 s ceiling

    /// The one-shot `didEnterFullScreenNotification` observer, released as soon
    /// as it fires.
    private var fullScreenObserver: NSObjectProtocol?

    /// Enter true macOS fullscreen once, at launch (owner, 2026-08-12).
    ///
    /// TRUE fullscreen was the owner's explicit choice over merely maximising:
    /// the app takes its own Space and the menu bar hides. The cost was stated
    /// and accepted — reaching Zoom or Teams during a meeting is now a Space
    /// switch (Ctrl-→ / swipe), not a click. Undo it by removing the call in
    /// `applicationDidFinishLaunching`; the green button still works normally.
    private func enterFullScreenAtLaunch() {
        guard !didEnterFullScreen, fullScreenAttemptsLeft > 0 else { return }
        fullScreenAttemptsLeft -= 1

        // SwiftUI builds the `WindowGroup`'s window on a LATER runloop turn than
        // `applicationDidFinishLaunching`, so there is usually nothing to enter
        // yet on the first call. Retrying on the next turn is deliberate rather
        // than sleeping a guessed interval: it acts the moment the window is
        // real, on a fast Mac and a loaded one alike.
        // `canBecomeMain` ALONE IS NOT ENOUGH, measured 2026-08-12: at launch it
        // matches a window SwiftUI has created but not yet shown, and
        // `toggleFullScreen` on an invisible window is silently ignored — the
        // request logged, `didEnterFullScreenNotification` never posted, the app
        // starting windowed. `isVisible` is what separates the real one.
        guard let window = NSApp.windows.first(where: { $0.canBecomeMain && $0.isVisible }) else {
            if fullScreenAttemptsLeft == 0 {
                // Say so rather than giving up quietly — a silent give-up is
                // exactly how the first version's bug hid.
                logWindow("gave up looking for the window after 6 s — staying windowed")
                return
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.fullScreenRetryInterval) {
                [weak self] in self?.enterFullScreenAtLaunch()
            }
            return
        }
        didEnterFullScreen = true

        // ⚠ THE ACTUAL BLOCKER, and inserting `.fullScreenPrimary` alone did NOT
        // clear it. Measured on this window at launch:
        //
        //     behavior = 66176 = primary(65536) + fullScreenNone(512)
        //                                       + fullScreenPrimary(128)
        //
        // SwiftUI ships the window with **`.fullScreenNone`**, and that flag
        // WINS: `toggleFullScreen` then does nothing at all — no error, no
        // exception, `didEnterFullScreenNotification` never posted, app starts
        // windowed. `insert` is a set-union, so the contradictory pair simply
        // coexisted. It has to be REMOVED, not out-voted.
        window.collectionBehavior.remove(.fullScreenNone)
        window.collectionBehavior.insert(.fullScreenPrimary)
        // The window is visible but not yet key at this point (`key=false`,
        // measured). Fullscreen is a main-window transition, so make it key
        // first rather than hoping AppKit picks it.
        window.makeKeyAndOrderFront(nil)
        guard !window.styleMask.contains(.fullScreen) else {
            logWindow("already fullscreen at launch — nothing to do")
            return
        }
        // OBSERVE THE RESULT, don't just log the request. `toggleFullScreen`
        // returns nothing and a window that refuses the transition simply stays
        // put, so a line printed straight after the call only ever proves the
        // call happened — which is the "component test cannot see a call site"
        // mistake in miniature. macOS posts this notification only on a
        // transition that really completed, so it is the honest evidence.
        // Held as a PROPERTY, not a local: a local `var token` assigned from the
        // `addObserver` result and then read inside its own closure is
        // "mutated after capture by sendable closure" — a warning today and a
        // concurrency error later.
        fullScreenObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didEnterFullScreenNotification,
            object: window, queue: .main) { [weak self] _ in
                guard let self else { return }
                self.logWindow("fullscreen CONFIRMED by macOS")
                if let observer = self.fullScreenObserver {
                    NotificationCenter.default.removeObserver(observer)
                    self.fullScreenObserver = nil
                }
            }
        // `behavior` is kept in the line ON PURPOSE: it is the number that
        // explained the silent failure (512 = `.fullScreenNone` present), and it
        // is the first thing to read if this ever stops working after a SwiftUI
        // update changes the window's defaults again.
        logWindow("requesting fullscreen at launch "
                  + "(attempt \(120 - fullScreenAttemptsLeft), "
                  + "behavior=\(window.collectionBehavior.rawValue))")
        window.toggleFullScreen(nil)
    }

    /// App-level diagnostics, the `[ATND]` precedent — no log file of its own.
    ///
    /// STDERR, not `print`: Swift's stdout is BLOCK-buffered when redirected to a
    /// file, so `swift run > log` swallowed these lines while the app kept
    /// running. That is what actually hid the first verification — an empty log
    /// read as "the code never ran", and sent me rewriting a retry that had been
    /// working the whole time. The existing `print("[ATND] …")` calls have the
    /// same flaw; they are simply never read under redirection.
    private func logWindow(_ message: String) {
        FileHandle.standardError.write(Data("[window] \(message)\n".utf8))
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
