import Foundation
import Network

/// One reply from the array (spec Tables 2-4, 2-5, 2-7).
enum ATNDReply: Equatable {
    case ack
    case nak(code: String)
    case answer(fields: [String])
}

enum ATNDError: Error {
    case notConnected
    case timeout
    case linkDropped
    case malformed(String)
    case nak(String)
}

extension ATNDError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .notConnected:        return "Not connected to the array."
        case .timeout:             return "The array did not answer in time."
        case .linkDropped:         return "The connection to the array dropped."
        case .malformed(let why):  return "Unexpected reply from the array — \(why)."
        case .nak(let code):       return "The array refused the command — NAK \(code) (\(ATNDCommands.nakText(code)))."
        }
    }
}

/// TCP control link to the ATND1061 (IP Control spec v9.0, default port 17300).
///
/// Opens and holds the connection, and carries one request at a time over it.
///
/// **Requests are strictly serialized, and that is the correctness mechanism,
/// not a simplification.** The protocol has no request IDs: a reply is matched
/// to its request by command name and by nothing else. With two requests in
/// flight there is no way to tell which `g_output_mute` answer belongs to which
/// channel, so a second request waits its turn in a FIFO queue.
///
/// A single shared instance, because the settings tab is rebuilt on every tab
/// switch and a per-view connection would drop each time.
@MainActor
final class ATNDControlService: ObservableObject {
    static let shared = ATNDControlService()

    enum State: Equatable {
        case disconnected
        case connecting
        case connected
        case failed(String)
    }

    @Published private(set) var state: State = .disconnected

    private var connection: NWConnection?
    private var timeout: Task<Void, Never>?
    private var retry: Task<Void, Never>?
    private let queue = DispatchQueue(label: "atnd.control")

    /// How long to wait before calling an unanswered connect attempt dead.
    /// NWConnection will otherwise sit in `.preparing` for a long time on a
    /// wrong address, which reads as a hang.
    private let timeoutSeconds: UInt64 = 5

    /// Gap between reconnect attempts once a live link drops.
    private let retrySeconds: UInt64 = 5

    /// How long one request may wait for its reply.
    private let requestTimeout: Duration = .seconds(3)

    // MARK: Request plumbing

    /// Bytes received but not yet split at a CR.
    private var rxBuffer = Data()

    private struct Pending {
        let command: String
        let continuation: CheckedContinuation<ATNDReply, Error>
        let timeoutTask: Task<Void, Never>
    }

    /// The one request currently on the wire, if any.
    private var pending: Pending?

    /// FIFO turn queue. A ticket identifies its holder so that a request whose
    /// turn was revoked by a teardown cannot later hand the turn on to someone
    /// else.
    private var turnHolder: UInt64?
    private var nextTicket: UInt64 = 0
    private var waiters: [(ticket: UInt64, continuation: CheckedContinuation<Void, Error>)] = []

    /// True while the user wants the link up — set by `connect`, cleared only
    /// by `disconnect`. A drop retries; a deliberate disconnect stays down.
    @Published private(set) var wantsConnection = false

    private var lastHost = ""
    private var lastPort = 0

    private init() {}

    var isConnected: Bool { state == .connected }

    /// A failure that auto-reconnect is working on, rather than a dead end.
    var isRetrying: Bool {
        if case .failed = state { return wantsConnection }
        return false
    }

    func connect(host: String, port: Int) {
        let host = host.trimmingCharacters(in: .whitespaces)
        // Config errors are NOT retried — the answer will never change on its
        // own, and a retry loop would bury the message under "Connecting…".
        guard !host.isEmpty else {
            stopWanting()
            state = .failed("Enter the array's IP address first.")
            return
        }
        // NWEndpoint.Port accepts 0, so the range has to be checked here or an
        // out-of-range port silently becomes port 0 and hangs in .preparing.
        guard UInt16(exactly: port) != nil, port > 0 else {
            stopWanting()
            state = .failed("Port must be between 1 and 65535.")
            return
        }

        lastHost = host
        lastPort = port
        wantsConnection = true
        open(host: host, port: port)
    }

    /// Open the link at launch when the settings already describe a valid array,
    /// so the owner does not have to press Connect every session. Auto-reconnect
    /// then keeps it up, exactly as it does after a manual connect.
    ///
    /// Deliberately SILENT when ATND is off or the address has not been filled in
    /// yet: `connect` reports those as `.failed`, which is right when a person
    /// just pressed the button, but on a first launch — where nothing is
    /// configured and nothing was asked for — it would greet them with a
    /// connection error for a feature they have not turned on.
    func autoConnectIfConfigured() {
        let d = UserDefaults.standard
        guard d.bool(forKey: "atnd.enabled"), !wantsConnection else { return }
        let host = (d.string(forKey: "atnd.deviceIP") ?? "")
            .trimmingCharacters(in: .whitespaces)
        let port = d.object(forKey: "atnd.controlPort") as? Int ?? 17300
        guard !host.isEmpty, UInt16(exactly: port) != nil, port > 0 else { return }
        connect(host: host, port: port)
    }

    func disconnect() {
        stopWanting()
        teardown()
        state = .disconnected
    }

    // MARK: - Internals

    private func stopWanting() {
        wantsConnection = false
        retry?.cancel()
        retry = nil
    }

    private func open(host: String, port: Int) {
        guard let raw = UInt16(exactly: port), let port = NWEndpoint.Port(rawValue: raw) else {
            state = .failed("Port must be between 1 and 65535.")
            return
        }

        teardown()
        state = .connecting

        let conn = NWConnection(host: NWEndpoint.Host(host), port: port, using: Self.parameters)
        connection = conn

        conn.stateUpdateHandler = { [weak self] newState in
            Task { @MainActor in
                self?.handle(newState, of: conn)
            }
        }
        // Fires when the path underneath dies (cable pulled, Wi-Fi dropped) —
        // faster than waiting for keepalive probes to run out.
        conn.viabilityUpdateHandler = { [weak self] viable in
            Task { @MainActor in
                guard let self, conn === self.connection, self.state == .connected else { return }
                if !viable {
                    self.fail("Lost the network path to the array — check the cable and the network.")
                }
            }
        }
        conn.start(queue: queue)
        receiveLoop(conn)

        timeout = Task { [weak self] in
            try? await Task.sleep(for: .seconds(self?.timeoutSeconds ?? 5))
            guard !Task.isCancelled, let self, self.state == .connecting else { return }
            self.fail("No response from \(host):\(port.rawValue). Check the address and that the array is on this network.")
        }
    }

    /// Drops the socket without touching `state` or `wantsConnection`, so it
    /// can serve both a deliberate disconnect and a retry cycle.
    ///
    /// Anything waiting on the old socket is failed here rather than left to
    /// hang: a request whose link is gone will never be answered.
    private func teardown() {
        timeout?.cancel()
        timeout = nil
        connection?.stateUpdateHandler = nil
        connection?.viabilityUpdateHandler = nil
        connection?.cancel()
        connection = nil

        rxBuffer.removeAll()

        if let p = pending {
            pending = nil
            p.timeoutTask.cancel()
            p.continuation.resume(throwing: ATNDError.linkDropped)
        }
        let stranded = waiters
        waiters = []
        turnHolder = nil
        for w in stranded { w.continuation.resume(throwing: ATNDError.linkDropped) }
    }

    /// TCP with keepalive, because this link is otherwise silent: the app sends
    /// nothing and the array answers nothing unless asked. Without probes, an
    /// array that vanished without a FIN (unplugged, powered off) would leave
    /// the connection sitting in `.ready` for the system default of ~2 hours,
    /// and the UI would keep claiming "Connected". Probing after 5s idle and
    /// giving up after 3 failures detects a dead array in roughly 11 seconds.
    private static var parameters: NWParameters {
        let tcp = NWProtocolTCP.Options()
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = 5
        tcp.keepaliveInterval = 2
        tcp.keepaliveCount = 3
        return NWParameters(tls: nil, tcp: tcp)
    }

    /// The single reader on this socket. It does two jobs at once and must stay
    /// one loop: a pending read is the only way to learn the peer closed cleanly
    /// (with no receive posted, a FIN goes unnoticed and the state stays
    /// `.ready`), and it is also where every reply arrives.
    private func receiveLoop(_ conn: NWConnection) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, isComplete, error in
            Task { @MainActor in
                guard let self, conn === self.connection else { return }
                if let data, !data.isEmpty { self.ingest(data) }
                if let error {
                    self.fail(Self.describe(error))
                } else if isComplete {
                    self.fail("The array closed the connection.")
                } else {
                    self.receiveLoop(conn)
                }
            }
        }
    }

    /// Splits the byte stream at CRs. TCP gives no message boundaries, so a
    /// reply can arrive in pieces or two replies in one read.
    private func ingest(_ data: Data) {
        rxBuffer.append(data)
        while let cr = rxBuffer.firstIndex(of: ATNDCommands.terminator) {
            let line = String(decoding: rxBuffer[rxBuffer.startIndex..<cr], as: UTF8.self)
            rxBuffer.removeSubrange(rxBuffer.startIndex...cr)
            route(line.trimmingCharacters(in: .whitespaces))
        }
        // Past the spec's own maximum message length with no CR in sight, the
        // buffer is not holding a message — it is holding garbage, and keeping
        // it would only corrupt the next real reply.
        if rxBuffer.count > ATNDCommands.maxMessageBytes { rxBuffer.removeAll() }
    }

    /// Hands one complete reply to whoever is waiting for it — or drops it.
    ///
    /// Dropping is the safe default: with no request IDs, a reply that does not
    /// match the request in flight cannot be attributed to anything, and
    /// guessing would put a wrong value on screen.
    private func route(_ line: String) {
        guard !line.isEmpty else { return }
        guard let p = pending else {
            log("dropped an unexpected reply (nothing was waiting): \(line)")
            return
        }
        switch ATNDCommands.parse(line, expecting: p.command) {
        case .reply(let reply):
            finish(.success(reply))
        case .mismatch:
            log("dropped a reply for \(line.split(separator: " ").first.map(String.init) ?? "?") while waiting on \(p.command)")
        case .malformed(let why):
            finish(.failure(ATNDError.malformed(why)))
        }
    }

    private func finish(_ result: Result<ATNDReply, Error>) {
        guard let p = pending else { return }
        pending = nil
        p.timeoutTask.cancel()
        p.continuation.resume(with: result)
    }

    private func log(_ message: String) {
        #if DEBUG
        print("[ATND] \(message)")
        #endif
    }

    // MARK: - Requests

    /// Sends a Set and reports what the array said.
    ///
    /// A NAK is returned rather than thrown: it is the array answering, not a
    /// failure to talk to it, and the caller needs the code to show it.
    @discardableResult
    func set(_ command: String, params: String = "") async throws -> ATNDReply {
        try await request(command, handshake: .set, params: params)
    }

    /// Sends a Get and returns the answer's parameter fields.
    func get(_ command: String, params: String = "") async throws -> [String] {
        switch try await request(command, handshake: .get, params: params) {
        case .answer(let fields):
            return fields
        case .nak(let code):
            throw ATNDError.nak(code)
        case .ack:
            throw ATNDError.malformed("ACK where an answer was expected")
        }
    }

    private func request(_ command: String,
                         handshake: ATNDCommands.Handshake,
                         params: String) async throws -> ATNDReply {
        let ticket = try await acquireTurn()
        defer { releaseTurn(ticket) }

        guard state == .connected, let conn = connection else { throw ATNDError.notConnected }

        let data = ATNDCommands.frame(ATNDCommands.action(command: command,
                                                          handshake: handshake,
                                                          params: params))
        guard data.count <= ATNDCommands.maxMessageBytes else {
            throw ATNDError.malformed("the message is longer than the spec's \(ATNDCommands.maxMessageBytes)-byte limit")
        }

        return try await withCheckedThrowingContinuation { continuation in
            let timeoutTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: self?.requestTimeout ?? .seconds(3))
                guard !Task.isCancelled else { return }
                self?.timedOut(command)
            }
            pending = Pending(command: command, continuation: continuation, timeoutTask: timeoutTask)
            conn.send(content: data, completion: .contentProcessed { [weak self] error in
                guard let error else { return }
                Task { @MainActor in
                    guard let self, self.pending?.command == command else { return }
                    self.finish(.failure(ATNDError.malformed(Self.describe(error))))
                }
            })
        }
    }

    /// A request that ran out of time takes the whole link with it, on purpose.
    ///
    /// Without request IDs, a reply that arrives after its request gave up would
    /// be matched to the NEXT request of the same command name — the array's
    /// answer to an old question shown as the answer to a new one, silently and
    /// wrongly. The stream cannot be resynchronised once that is possible, so
    /// the only honest move is to drop the socket and let auto-reconnect build
    /// a clean one.
    private func timedOut(_ command: String) {
        guard let p = pending, p.command == command else { return }
        pending = nil
        p.timeoutTask.cancel()
        p.continuation.resume(throwing: ATNDError.timeout)
        fail("The array did not answer \(command) within 3 seconds.")
    }

    // MARK: - Turn queue

    private func acquireTurn() async throws -> UInt64 {
        nextTicket += 1
        let ticket = nextTicket
        if turnHolder == nil {
            turnHolder = ticket
            return ticket
        }
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            waiters.append((ticket, c))
        }
        return ticket
    }

    private func releaseTurn(_ ticket: UInt64) {
        // Not the holder any more: a teardown revoked this turn and already
        // dealt with the queue. Handing the turn on now would grant it twice.
        guard turnHolder == ticket else { return }
        guard !waiters.isEmpty else {
            turnHolder = nil
            return
        }
        let next = waiters.removeFirst()
        turnHolder = next.ticket
        next.continuation.resume()
    }

    /// Ignores updates from a superseded connection (a second Connect tap
    /// while the first was still preparing).
    private func handle(_ newState: NWConnection.State, of conn: NWConnection) {
        guard conn === connection else { return }

        switch newState {
        case .ready:
            timeout?.cancel()
            timeout = nil
            state = .connected
        case .waiting(let error):
            // Reachability problems (host down, no route) surface here and
            // NWConnection keeps retrying — the array is not reachable now.
            fail(Self.describe(error))
        case .failed(let error):
            fail(Self.describe(error))
        case .cancelled:
            break // disconnect() already set the state
        case .setup, .preparing:
            break
        @unknown default:
            break
        }
    }

    /// Reports why the link is down and, if the user still wants it up, keeps
    /// trying — an array that was unplugged usually comes back.
    private func fail(_ message: String) {
        teardown()
        state = .failed(message)

        guard wantsConnection else { return }
        retry?.cancel()
        retry = Task { [weak self] in
            try? await Task.sleep(for: .seconds(self?.retrySeconds ?? 5))
            guard !Task.isCancelled, let self, self.wantsConnection else { return }
            self.open(host: self.lastHost, port: self.lastPort)
        }
    }

    private static func describe(_ error: NWError) -> String {
        switch error {
        case .posix(.ECONNREFUSED):
            return "Connection refused — something answered, but not on this port."
        case .posix(.EHOSTUNREACH), .posix(.ENETUNREACH):
            return "Host unreachable — check the array's IP and this Mac's network."
        case .posix(.ETIMEDOUT):
            return "Timed out — no reply from the array."
        default:
            return error.localizedDescription
        }
    }
}
