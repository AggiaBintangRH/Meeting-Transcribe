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
    static func logHandle(name: String) -> FileHandle {
        let dir = dataDir.appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("\(name).log")
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
