import Foundation

/// Streams the backend's live calculation log over the /ws/logs WebSocket,
/// with automatic reconnection and tqdm carriage-return handling (a "\r…"
/// chunk overwrites the last line, matching the web console behaviour).
@MainActor
final class LogStreamClient: ObservableObject {
    @Published private(set) var lines: [String] = []
    @Published private(set) var isConnected = false

    private var task: URLSessionWebSocketTask?
    private var reconnectTask: Task<Void, Never>?
    private var heartbeatTask: Task<Void, Never>?
    private var baseURL: URL?
    private let maxLines = 1000

    func connect(to baseURL: URL) {
        self.baseURL = baseURL
        reconnect()
    }

    func disconnect() {
        reconnectTask?.cancel()
        heartbeatTask?.cancel()
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        baseURL = nil
        isConnected = false
    }

    func clear() {
        lines = []
    }

    private func reconnect() {
        guard let baseURL else { return }
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.scheme = components.scheme == "https" ? "wss" : "ws"
        components.path = "/ws/logs"
        guard let wsURL = components.url else { return }

        let task = URLSession.shared.webSocketTask(with: wsURL)
        self.task = task
        task.resume()
        isConnected = true
        receiveLoop(on: task)
        startHeartbeat(on: task)
    }

    private func startHeartbeat(on task: URLSessionWebSocketTask) {
        heartbeatTask?.cancel()
        heartbeatTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(30))
                guard let self, self.task === task else { return }
                task.send(.string("ping")) { _ in }
            }
        }
    }

    private func receiveLoop(on task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            Task { @MainActor [weak self] in
                guard let self, self.task === task else { return }
                switch result {
                case .success(let message):
                    if case .string(let text) = message,
                       text != "ping", text != "pong" {
                        self.append(text)
                    }
                    self.receiveLoop(on: task)
                case .failure:
                    self.isConnected = false
                    self.scheduleReconnect()
                }
            }
        }
    }

    private func scheduleReconnect() {
        guard baseURL != nil else { return }
        reconnectTask?.cancel()
        reconnectTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(5))
            guard let self, !Task.isCancelled else { return }
            self.reconnect()
        }
    }

    private func append(_ raw: String) {
        if raw.hasPrefix("\r"), !lines.isEmpty {
            lines[lines.count - 1] = String(raw.dropFirst())
        } else {
            lines.append(raw)
            if lines.count > maxLines {
                lines.removeFirst(lines.count - maxLines)
            }
        }
    }
}
