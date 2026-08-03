import Foundation

/// Resolves which Python the sidecar processes should use, and where the app
/// reads scripts/models from and writes data to.
///
/// Two modes:
///  • Dev (swift run/build): everything lives under
///    ~/Documents/AI/Meeting Transcribe (unchanged legacy behavior).
///  • Bundled (.app): scripts + models ship inside Contents/Resources, a
///    self-contained interpreter lives at Contents/Resources/python, and all
///    mutable data goes to ~/Library/Application Support/Meeting Transcriber.
enum PythonRuntime {

    /// True when running from a packaged .app that carries bundled scripts.
    static let isBundled: Bool = {
        let bundleURL = Bundle.main.bundleURL
        guard bundleURL.pathExtension == "app" else { return false }
        let scripts = bundleURL
            .appendingPathComponent("Contents/Resources/scripts")
        return FileManager.default.fileExists(atPath: scripts.path)
    }()

    /// Root that contains scripts/, models/, python/, .venv/ (read-only when bundled).
    static let projectDir: URL = {
        if isBundled {
            return Bundle.main.bundleURL.appendingPathComponent("Contents/Resources")
        }
        // Dev: find the repo root by walking up from the working directory,
        // looking for download-best-models.sh (the unambiguous project marker).
        // swift run and RUN-APP.command both set the working directory inside
        // the repo, so this makes the app work no matter WHERE the repo was
        // cloned — not only at the one legacy ~/Documents/AI/Meeting Transcribe
        // path, which broke on a second Mac where `git clone` put it elsewhere
        // (and named it "Meeting-Transcribe" with a hyphen).
        let fm = FileManager.default
        var dir = URL(fileURLWithPath: fm.currentDirectoryPath)
        for _ in 0..<8 {
            if fm.fileExists(atPath: dir.appendingPathComponent("download-best-models.sh").path) {
                return dir
            }
            let parent = dir.deletingLastPathComponent()
            if parent == dir { break }   // reached the filesystem root
            dir = parent
        }
        // Legacy fallback — the original hardcoded location.
        return URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Documents/AI/Meeting Transcribe")
    }()

    /// Root for all mutable data (logs, recordings, profiles, HF caches).
    /// In dev this is the project dir (legacy paths); bundled it is
    /// ~/Library/Application Support/Meeting Transcriber.
    static let dataDir: URL = {
        if isBundled {
            let dir = URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent("Library/Application Support/Meeting Transcriber")
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir
        }
        return projectDir
    }()

    static var scriptsDir: URL { projectDir.appendingPathComponent("scripts") }

    /// Where speaker profiles live. Dev keeps them next to the models; bundled
    /// they are user data under Application Support.
    static var profilesDir: URL {
        if isBundled {
            return dataDir.appendingPathComponent("speaker-profiles")
        }
        return projectDir.appendingPathComponent("models/speaker-profiles")
    }

    /// (executable, argument-prefix) to run a python script.
    /// Prefers the bundled self-contained interpreter, then the dev venv,
    /// then system python3.
    static func command(forScript script: URL) -> (executable: URL, arguments: [String]) {
        let bundledPython = projectDir.appendingPathComponent("python/bin/python3")
        if FileManager.default.isExecutableFile(atPath: bundledPython.path) {
            return (bundledPython, [script.path])
        }
        let venvPython = projectDir.appendingPathComponent(".venv/bin/python3")
        if FileManager.default.isExecutableFile(atPath: venvPython.path) {
            return (venvPython, [script.path])
        }
        return (URL(fileURLWithPath: "/usr/bin/env"), ["python3", script.path])
    }

    /// (executable, argument-prefix) to run a python script under a DEDICATED
    /// venv (e.g. `.venv-dicow`), for sidecars whose dependencies conflict with
    /// the main `.venv` — DiCoW needs transformers 4.55, the MLX stack needs 5.x.
    /// Returns nil when that venv is absent; the caller surfaces a setup error.
    /// No fallback on purpose: running such a sidecar on the wrong interpreter
    /// fails deep inside the model with a far more confusing error.
    static func command(forScript script: URL, venvName: String)
        -> (executable: URL, arguments: [String])? {
        let python = projectDir.appendingPathComponent("\(venvName)/bin/python3")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { return nil }
        return (python, [script.path])
    }

    /// Shared environment for every sidecar process. Starts from the current
    /// process env and forces offline + unbuffered behavior; when bundled it
    /// also redirects the HuggingFace caches and profile dir into the bundle /
    /// Application Support so nothing is written next to the read-only .app.
    static func sidecarEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        if isBundled {
            let hfHome = dataDir.appendingPathComponent("hf-home")
            env["HF_HOME"] = hfHome.path
            env["HF_HUB_CACHE"] = projectDir.appendingPathComponent("models/hub").path
            env["HF_XET_CACHE"] = hfHome.appendingPathComponent("xet").path
            env["MT_PROFILE_DIR"] = profilesDir.path
        }
        return env
    }

    /// File handle for a sidecar's stderr log (logs/<name>.log under the data dir).
    /// Crucial for debugging — Python tracebacks land here instead of vanishing.
    /// Roll a log over at this size, keeping ONE previous generation as
    /// `<name>.log.1`. Sidecar logs are append-only and were never bounded:
    /// `position-diarization.log` had reached 13 MB by 2026-07-31 and nothing
    /// would ever have stopped it. They are the project's primary evidence when
    /// something goes wrong — every dropped chunk, every gate decision, every
    /// truncation warning is only there — so they are ROLLED rather than
    /// truncated: rolling loses the oldest generation, truncating loses the
    /// evidence you are currently reading.
    ///
    /// 16 MB holds many meetings of ordinary logging (a full session writes tens
    /// of KB), so a roll is rare and the previous generation is almost always
    /// still the run before this one.
    static let logRollBytes: UInt64 = 16 * 1024 * 1024

    static func logHandle(name: String) -> FileHandle {
        let dir = dataDir.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("\(name).log")
        // Rolled at OPEN time, not per write: this is the one place every sidecar
        // log is created, and checking a file size once per process start costs
        // nothing, while checking per line would cost a stat on every log call.
        if let size = (try? FileManager.default.attributesOfItem(atPath: file.path)[.size])
                        as? NSNumber, size.uint64Value >= logRollBytes {
            let previous = dir.appendingPathComponent("\(name).log.1")
            try? FileManager.default.removeItem(at: previous)
            try? FileManager.default.moveItem(at: file, to: previous)
        }
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: file) else {
            return FileHandle.nullDevice
        }
        handle.seekToEndOfFile()
        return handle
    }
}
