import Foundation

/// Client for the Silero VAD v6 Python sidecar (scripts/vad/vad-service.py).
///
/// Feed 16 kHz mono samples from the audio thread; read `latestProbability`
/// (0..1, speech likelihood). Silero is trained to distinguish speech from
/// music/instruments — this is what makes piano/guitar read as SILENT.
///
/// Failable init: returns nil if Python or the model isn't available,
/// callers then fall back to the heuristic VAD.
final class SileroVADService: @unchecked Sendable {

    static let frameSize = 512  // samples per frame @ 16 kHz

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()

    private let lock = NSLock()
    private var pendingSamples: [Float] = []
    private var probability: Float = 0

    /// Most recent speech probability reported by Silero.
    var latestProbability: Float {
        lock.lock(); defer { lock.unlock() }
        return probability
    }

    init?() {
        let script = PythonRuntime.scriptsDir.appendingPathComponent("vad/vad-service.py")
        guard FileManager.default.fileExists(atPath: script.path) else { return nil }

        let command = PythonRuntime.command(forScript: script)
        process.executableURL = command.executable
        process.arguments = command.arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = PythonRuntime.logHandle(name: "vad")

        process.environment = PythonRuntime.sidecarEnvironment()

        do { try process.run() } catch { return nil }
        guard waitUntilReady(timeout: 30) else {
            process.terminate()
            return nil
        }
        startProbabilityReader()
    }

    deinit {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        if process.isRunning { process.terminate() }
    }

    // MARK: - Feeding audio (called from the audio tap thread)

    /// Buffer incoming samples and ship complete 512-sample frames to Python.
    func feed(_ samples: [Float]) {
        guard process.isRunning, !samples.isEmpty else { return }

        var frames: [Data] = []
        lock.lock()
        pendingSamples.append(contentsOf: samples)
        while pendingSamples.count >= Self.frameSize {
            let frame = Array(pendingSamples.prefix(Self.frameSize))
            pendingSamples.removeFirst(Self.frameSize)
            frame.withUnsafeBufferPointer { frames.append(Data(buffer: $0)) }
        }
        lock.unlock()

        let writer = stdinPipe.fileHandleForWriting
        for frame in frames {
            try? writer.write(contentsOf: frame)
        }
    }

    // MARK: - Reading results

    /// Block until the sidecar prints READY (model loaded) or timeout.
    private func waitUntilReady(timeout: TimeInterval) -> Bool {
        let semaphore = DispatchSemaphore(value: 0)
        var ready = false
        var accumulated = Data()

        let reader = stdoutPipe.fileHandleForReading
        reader.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { // EOF — process died
                semaphore.signal()
                return
            }
            accumulated.append(data)
            if let text = String(data: accumulated, encoding: .utf8), text.contains("READY") {
                ready = true
                semaphore.signal()
            }
        }
        _ = semaphore.wait(timeout: .now() + timeout)
        reader.readabilityHandler = nil
        return ready
    }

    /// Continuously parse probability lines into `probability`.
    private func startProbabilityReader() {
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            guard let self,
                  let text = String(data: handle.availableData, encoding: .utf8)
            else { return }
            for line in text.split(separator: "\n") {
                if let value = Float(line.trimmingCharacters(in: .whitespaces)) {
                    self.lock.lock()
                    self.probability = value
                    self.lock.unlock()
                }
            }
        }
    }
}
