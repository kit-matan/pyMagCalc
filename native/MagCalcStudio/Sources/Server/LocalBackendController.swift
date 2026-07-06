#if os(macOS)
import Foundation

/// Launches and supervises the Python FastAPI backend (gui/server.py) as a
/// child process, so the macOS app is fully self-contained: no terminal
/// session or start script needed.
@MainActor
final class LocalBackendController: ObservableObject {
    enum State: Equatable {
        case stopped
        case starting
        case running
        case failed(String)
    }

    @Published private(set) var state: State = .stopped

    /// Absolute path to the pyMagCalc checkout (the directory containing gui/server.py).
    @Published var projectRoot: String {
        didSet { UserDefaults.standard.set(projectRoot, forKey: "backend.projectRoot") }
    }

    /// Python interpreter used to run the server. Keyed "v2" so installs that
    /// persisted the pre-pyenv-aware guess re-run detection once.
    @Published var pythonPath: String {
        didSet { UserDefaults.standard.set(pythonPath, forKey: "backend.pythonPath.v2") }
    }

    @Published var port: Int {
        didSet { UserDefaults.standard.set(port, forKey: "backend.port") }
    }

    private var process: Process?

    init() {
        let defaults = UserDefaults.standard
        projectRoot = defaults.string(forKey: "backend.projectRoot") ?? Self.guessProjectRoot()
        pythonPath = defaults.string(forKey: "backend.pythonPath.v2") ?? Self.guessPython()
        let savedPort = defaults.integer(forKey: "backend.port")
        port = savedPort == 0 ? 8000 : savedPort
    }

    var serverScript: String { projectRoot + "/gui/server.py" }

    var isConfigured: Bool {
        FileManager.default.fileExists(atPath: serverScript)
            && FileManager.default.fileExists(atPath: pythonPath)
    }

    func start() {
        guard state != .running, state != .starting else { return }
        guard isConfigured else {
            state = .failed("Server script or Python interpreter not found. Set paths in Settings.")
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonPath)
        proc.arguments = [serverScript]
        proc.currentDirectoryURL = URL(fileURLWithPath: projectRoot)
        var env = ProcessInfo.processInfo.environment
        env["MAGCALC_HOST"] = "127.0.0.1"
        env["MAGCALC_PORT"] = String(port)
        proc.environment = env
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        proc.terminationHandler = { [weak self] p in
            Task { @MainActor [weak self] in
                guard let self, self.process === p else { return }
                self.process = nil
                if case .failed = self.state { return }
                self.state = p.terminationStatus == 0
                    ? .stopped
                    : .failed("Backend exited with status \(p.terminationStatus)")
            }
        }

        do {
            try proc.run()
            process = proc
            state = .starting
            waitUntilReachable()
        } catch {
            state = .failed("Failed to launch backend: \(error.localizedDescription)")
        }
    }

    func stop() {
        process?.terminate()
        process = nil
        state = .stopped
    }

    private func waitUntilReachable() {
        let client = APIClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!)
        Task { [weak self] in
            // The backend imports pymatgen/spglib/sympy at startup; allow a
            // generous budget (cold starts from cloud-synced folders are slow).
            for _ in 0..<120 {
                try? await Task.sleep(for: .seconds(1))
                guard let self, self.state == .starting else { return }
                if await client.health() {
                    self.state = .running
                    return
                }
            }
            guard let self, self.state == .starting else { return }
            self.state = .failed("Backend did not respond within 120 s. Check the Python environment.")
            self.stop()
        }
    }

    // MARK: Path guessing

    private static func guessProjectRoot() -> String {
        // When the app is built from within the repo, walk up from the bundle
        // looking for gui/server.py; otherwise fall back to a sensible default.
        let fm = FileManager.default
        var dir = Bundle.main.bundleURL.deletingLastPathComponent()
        for _ in 0..<8 {
            if fm.fileExists(atPath: dir.appendingPathComponent("gui/server.py").path) {
                return dir.path
            }
            dir.deleteLastPathComponent()
        }
        return NSHomeDirectory()
    }

    private static func guessPython() -> String {
        let fm = FileManager.default
        let home = NSHomeDirectory()

        // Prefer pyenv: it is typically the environment that actually has the
        // magcalc dependencies installed. The shim resolves to the version
        // selected by pyenv; fall back to the newest installed version.
        var candidates = ["\(home)/.pyenv/shims/python3", "\(home)/.pyenv/shims/python"]
        let versionsDir = "\(home)/.pyenv/versions"
        if let versions = try? fm.contentsOfDirectory(atPath: versionsDir) {
            for v in versions.sorted(by: >) {
                candidates.append("\(versionsDir)/\(v)/bin/python3")
                candidates.append("\(versionsDir)/\(v)/bin/python")
            }
        }
        candidates += [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        return candidates.first { fm.fileExists(atPath: $0) } ?? "/usr/bin/python3"
    }
}
#endif
