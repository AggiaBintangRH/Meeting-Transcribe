import Foundation

/// Client for `scripts/overlap-detect/overlap-detect-service.py` — marks WHERE
/// two people spoke at once, and nothing else.
///
/// Its own process rather than a mode of `PyannoteService`, because the engines
/// that need this are exactly the ones where pyannote is never loaded: MOSS and
/// spectral both assign one speaker per instant, so neither can ever mark overlap
/// itself. Adding a mode to a process those sessions do not start would mean
/// loading the whole 1.17 GB pipeline to use one 32 MB network.
///
/// Naming follows the project rule with no exception:
/// `scripts/overlap-detect/overlap-detect-service.py` writes `logs/overlap-detect.log`.
final class OverlapDetectService {

    enum ServiceError: LocalizedError {
        case scriptMissing
        case launchFailed(String)
        case startupFailed(String)

        var errorDescription: String? {
            switch self {
            case .scriptMissing:
                return "overlap-detect-service.py is missing from scripts/overlap-detect."
            case .launchFailed(let why):
                return "The overlap detector could not start: \(why)"
            case .startupFailed(let why):
                return why
            }
        }
    }

    struct Config: Equatable {
        static let scriptName = "overlap-detect/overlap-detect-service.py"
        static let logName = "overlap-detect"
    }

    /// `(regions, audioPath)` — regions in RECORDING seconds.
    var onResult: (([(start: Double, end: Double)], String) -> Void)?
    var onError: ((String) -> Void)?

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "overlap-detect.write")
    private var buffer = Data()

    init() throws {
        let script = PythonRuntime.scriptsDir.appendingPathComponent(Config.scriptName)
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw ServiceError.scriptMissing
        }
        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: Config.logName)
        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch {
            throw ServiceError.launchFailed(error.localizedDescription)
        }
        // 32 MB and CPU-only: measured at ~0 s. The generous budget is for a cold
        // filesystem on a client machine, not for the model.
        let startup = waitUntilLoaded(timeout: 120)
        guard startup.ready else {
            process.terminate()
            throw ServiceError.startupFailed(
                startup.errorReason ?? "The overlap detector did not become ready (timeout).")
        }
        startResultReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        terminate()
    }

    func detect(audio: URL) {
        guard process.isRunning else {
            onError?("The overlap detector is not running — restart the session.")
            return
        }
        guard var data = try? JSONSerialization.data(
            withJSONObject: ["cmd": "detect", "audio": audio.path]) else { return }
        data.append(0x0A)
        writeQueue.async { [stdinPipe] in
            try? stdinPipe.fileHandleForWriting.write(contentsOf: data)
        }
    }

    func terminate() {
        if process.isRunning {
            try? stdinPipe.fileHandleForWriting.close()
            process.terminate()
        }
    }

    // MARK: - Output

    private struct Message: Decodable {
        let type: String
        let text: String?
        let audio: String?
        let regions: [[Double]]?
    }

    private func waitUntilLoaded(timeout: TimeInterval) -> (ready: Bool, errorReason: String?) {
        let handle = stdoutPipe.fileHandleForReading
        let deadline = Date().addingTimeInterval(timeout)
        var pending = Data()
        while Date() < deadline {
            let chunk = handle.availableData
            if chunk.isEmpty { continue }
            pending.append(chunk)
            while let nl = pending.firstIndex(of: 0x0A) {
                let line = pending.subdata(in: pending.startIndex..<nl)
                pending.removeSubrange(pending.startIndex...nl)
                guard let m = try? JSONDecoder().decode(Message.self, from: line) else { continue }
                if m.type == "status", m.text == "LOADED" { return (true, nil) }
                if m.type == "error" { return (false, m.text) }
            }
        }
        return (false, nil)
    }

    private func startResultReader() {
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self else { return }
            self.buffer.append(handle.availableData)
            while let nl = self.buffer.firstIndex(of: 0x0A) {
                let line = self.buffer.subdata(in: self.buffer.startIndex..<nl)
                self.buffer.removeSubrange(self.buffer.startIndex...nl)
                guard let m = try? JSONDecoder().decode(Message.self, from: line) else { continue }
                switch m.type {
                case "result":
                    // A pair that is not exactly [start, end] is dropped rather
                    // than guessed at — a malformed region would silently mark the
                    // wrong rows.
                    let regions = (m.regions ?? []).compactMap { pair -> (start: Double, end: Double)? in
                        pair.count == 2 ? (start: pair[0], end: pair[1]) : nil
                    }
                    self.onResult?(regions, m.audio ?? "")
                case "error":
                    self.onError?(m.text ?? "Overlap detection failed")
                default:
                    break
                }
            }
        }
    }
}
