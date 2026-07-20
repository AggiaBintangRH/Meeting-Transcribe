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
    func applicationDidFinishLaunching(_ notification: Notification) {
        migrateSettings()
        // Start the ATND1061 beam listener if it is enabled. It runs for the
        // whole app lifetime — independent of recording and of the TCP control
        // link — and re-syncs itself on later settings changes.
        Task { @MainActor in ATNDBeamService.shared.syncWithSettings() }
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
