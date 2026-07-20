import Combine
import Foundation

/// Listens to the ATND1061's UDP multicast notice stream (default
/// 239.0.0.100:17000, pushed every 100 ms) and publishes the currently
/// speaking beam — channel, elevation angle, rotation.
///
/// A single shared instance: the status chip is rebuilt constantly and a
/// per-view socket would rejoin the multicast group on every redraw.
///
/// **Requires the TCP control link**, matching the array: it broadcasts nothing
/// until its notification switches are on, and those live on the device and are
/// only reachable over IP control. So listening without a control link would be
/// pretending — the socket would be open and forever silent. The listener runs
/// only while `atnd.enabled` is true *and* `ATNDControlService` is connected,
/// and it drops the socket the moment that link goes away.
///
/// Testing therefore needs both sidecars together:
///   python3 scripts/atnd1061-tcp-simulator.py
///   python3 scripts/atnd1061-multicast-simulator.py
///
/// Transport is a raw BSD socket rather than `NWMulticastGroup`: this Mac is
/// multi-homed (the array sits on its own interface), so `imr_interface` has to
/// be set explicitly; `SO_REUSEPORT` lets the Python receiver/simulator bind the
/// same port alongside us; and the first-join quirk below needs a plain retry.
@MainActor
final class ATNDBeamService: ObservableObject {
    static let shared = ATNDBeamService()

    /// A speaking beam. `channel` is already the human-facing Beam Channel
    /// (1...6) — the wire value (0...5) is converted at parse time.
    struct Talker: Equatable {
        let channel: Int
        let elevation: Int
        let rotation: Int
    }

    enum State: Equatable {
        case off
        /// Enabled, but the TCP control link is down — the array would not be
        /// sending anything, so no socket is open. Distinct from `off` so the
        /// UI can say which of the two it is.
        case waitingForControl
        case listening
        case failed(String)
    }

    @Published private(set) var state: State = .off
    /// nil = nobody speaking, stream gone stale, or listener off.
    @Published private(set) var talker: Talker?

    private var source: DispatchSourceRead?
    private let queue = DispatchQueue(label: "atnd.multicast")

    /// Snapshot of the settings the running socket was started with, so the
    /// app-wide `didChangeNotification` only restarts us on a real change.
    private var currentSnapshot: String?

    private var silenceClearTask: Task<Void, Never>?
    private var staleTimer: Timer?
    private var lastPacketAt: Date = .distantPast

    /// Status flips 0/1 between words at 10 Hz; clearing the chip instantly
    /// would make it flicker. Hold the last talker briefly on silence.
    private let silenceHoldSeconds: TimeInterval = 1.0
    /// No packet at all for this long = the array is gone; drop the talker
    /// rather than freezing the chip on whoever spoke last.
    private let staleAfterSeconds: TimeInterval = 3.0

    private var controlObserver: AnyCancellable?

    private init() {
        NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification,
            object: UserDefaults.standard,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.syncWithSettings() }
        }
        // The control link coming up or going down starts and stops us.
        controlObserver = ATNDControlService.shared.$state
            .removeDuplicates()
            .sink { [weak self] _ in
                Task { @MainActor in self?.syncWithSettings() }
            }
    }

    // MARK: - Lifecycle

    /// Starts, stops, or restarts the listener to match the current settings.
    /// Safe to call often — a no-op unless the relevant defaults changed.
    func syncWithSettings() {
        let d = UserDefaults.standard
        let enabled = d.bool(forKey: "atnd.enabled")
        let group = (d.string(forKey: "atnd.multicastGroup") ?? "239.0.0.100")
            .trimmingCharacters(in: .whitespaces)
        let port = d.object(forKey: "atnd.multicastPort") as? Int ?? 17000
        let interfaceIP = (d.string(forKey: "atnd.interfaceIP") ?? "")
            .trimmingCharacters(in: .whitespaces)

        let controlUp = ATNDControlService.shared.isConnected

        let snapshot = "\(enabled)|\(controlUp)|\(group)|\(port)|\(interfaceIP)"
        guard snapshot != currentSnapshot else { return }
        currentSnapshot = snapshot

        stop()
        guard enabled else {
            state = .off
            return
        }
        guard controlUp else {
            state = .waitingForControl
            return
        }
        start(group: group, port: port, interfaceIP: interfaceIP)
    }

    private func start(group: String, port: Int, interfaceIP: String) {
        guard !group.isEmpty, inet_addr(group) != INADDR_NONE else {
            state = .failed("Multicast group must be a valid IPv4 address like 239.0.0.100.")
            return
        }
        guard let rawPort = UInt16(exactly: port), rawPort > 0 else {
            state = .failed("Multicast port must be between 1 and 65535.")
            return
        }
        guard interfaceIP.isEmpty || inet_addr(interfaceIP) != INADDR_NONE else {
            state = .failed("Interface IP must be a valid IPv4 address, or blank for the default one.")
            return
        }

        let fd = socket(AF_INET, SOCK_DGRAM, 0)
        guard fd >= 0 else {
            state = .failed("Could not open a UDP socket (\(errnoText())).")
            return
        }

        var one: Int32 = 1
        let optLen = socklen_t(MemoryLayout<Int32>.size)
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, optLen)
        // Lets the Python receiver/simulator hold the same port at the same time.
        setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, optLen)

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = rawPort.bigEndian
        addr.sin_addr = in_addr(s_addr: INADDR_ANY)
        let bound = withUnsafePointer(to: &addr) { raw in
            raw.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else {
            let text = errnoText()
            close(fd)
            state = .failed("Could not bind UDP port \(rawPort) (\(text)).")
            return
        }

        var mreq = ip_mreq(
            imr_multiaddr: in_addr(s_addr: inet_addr(group)),
            imr_interface: in_addr(s_addr: interfaceIP.isEmpty ? INADDR_ANY : inet_addr(interfaceIP)))

        // macOS clones the multicast route lazily, so the very first join right
        // after launch can fail with EHOSTUNREACH even though the interface is
        // fine. One short retry clears it; a second failure is real.
        var joined = setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                                &mreq, socklen_t(MemoryLayout<ip_mreq>.size)) == 0
        if !joined {
            Thread.sleep(forTimeInterval: 0.3)
            joined = setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                                &mreq, socklen_t(MemoryLayout<ip_mreq>.size)) == 0
        }
        guard joined else {
            let text = errnoText()
            close(fd)
            state = .failed("Could not join multicast group \(group) (\(text)). Check the interface IP and this Mac's network.")
            return
        }

        _ = fcntl(fd, F_SETFL, O_NONBLOCK)

        let src = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
        src.setEventHandler { [weak self] in
            self?.readPending(fd)
        }
        // The fd is closed here and nowhere else: closing it directly would let
        // the read handler race the close and read a recycled descriptor.
        src.setCancelHandler { close(fd) }
        source = src
        src.resume()

        lastPacketAt = .now
        state = .listening
        startStaleTimer()
    }

    private func stop() {
        source?.cancel()
        source = nil
        silenceClearTask?.cancel()
        silenceClearTask = nil
        staleTimer?.invalidate()
        staleTimer = nil
        talker = nil
        lastPacketAt = .distantPast
    }

    // MARK: - Socket queue

    /// Runs on `queue`. Drains every pending datagram, parses off the main
    /// actor, and hands only immutable values back.
    private nonisolated func readPending(_ fd: Int32) {
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let n = buffer.withUnsafeMutableBytes { recvfrom(fd, $0.baseAddress, $0.count, 0, nil, nil) }
            if n <= 0 { return }  // EWOULDBLOCK on an empty queue, or a dead socket

            let text = String(decoding: buffer[0..<n], as: UTF8.self)
            // One datagram may carry more than one CR-terminated message.
            for line in text.split(separator: "\r") {
                guard let notice = Self.parse(String(line)) else { continue }
                Task { @MainActor [weak self] in self?.apply(notice) }
            }
        }
    }

    // MARK: - Parsing

    enum ParsedNotice: Equatable {
        case talking(Talker)
        case silence
    }

    /// Parses one CR-stripped notice. Returns nil for anything we don't care
    /// about — level_meter_notice arrives on this same socket at 10 Hz, so an
    /// unknown message is never an error and is never logged.
    /// `nonisolated` because the class is `@MainActor`: parsing must run on the
    /// socket queue, and it is pure — only the immutable result crosses over.
    nonisolated static func parse(_ line: String) -> ParsedNotice? {
        let tokens = line.split(separator: " ", omittingEmptySubsequences: false)
        guard tokens.count == 6, tokens[0] == "MD", tokens[4] == "NC" else { return nil }
        guard tokens[1] == "camera_control_notice" else { return nil }

        let fields = tokens[5].split(separator: ",", omittingEmptySubsequences: false)
        guard fields.count == 5 else { return nil }

        // Silence is literally "0,,,," — fields 1...4 are empty by spec, so
        // they must not be validated.
        if fields[0] == "0" { return .silence }
        guard fields[0] == "1" else { return nil }

        guard let ch = Int(fields[1]), (0...5).contains(ch),
              let angle = Int(fields[2]),
              let rotate = Int(fields[3]) else { return nil }

        return .talking(Talker(channel: ch + 1, elevation: angle, rotation: rotate))
    }

    // MARK: - Main actor state

    private func apply(_ notice: ParsedNotice) {
        lastPacketAt = .now
        switch notice {
        case .talking(let t):
            silenceClearTask?.cancel()
            silenceClearTask = nil
            if t != talker { talker = t }
        case .silence:
            guard talker != nil, silenceClearTask == nil else { return }
            silenceClearTask = Task { [weak self] in
                guard let self else { return }
                try? await Task.sleep(for: .seconds(self.silenceHoldSeconds))
                guard !Task.isCancelled else { return }
                self.talker = nil
                self.silenceClearTask = nil
            }
        }
    }

    private func startStaleTimer() {
        staleTimer?.invalidate()
        staleTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.state == .listening, self.talker != nil else { return }
                if Date.now.timeIntervalSince(self.lastPacketAt) > self.staleAfterSeconds {
                    self.silenceClearTask?.cancel()
                    self.silenceClearTask = nil
                    self.talker = nil
                }
            }
        }
    }

    private func errnoText() -> String {
        String(cString: strerror(errno))
    }
}
