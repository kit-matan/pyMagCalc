import Foundation
import SwiftUI

/// Navigation destinations, mirroring the web app's sidebar tabs.
enum StudioTab: String, CaseIterable, Identifiable {
    case structure, interactions, environment, tasks, magneticStructure, fitting, run

    var id: String { rawValue }

    var title: String {
        switch self {
        case .structure: return "Structure"
        case .interactions: return "Interactions"
        case .environment: return "Environment"
        case .tasks: return "Tasks & Plotting"
        case .magneticStructure: return "Mag. Structure"
        case .fitting: return "Data Fitting"
        case .run: return "Run & Analyze"
        }
    }

    var systemImage: String {
        switch self {
        case .structure: return "cube"
        case .interactions: return "link"
        case .environment: return "slider.horizontal.3"
        case .tasks: return "chart.xyaxis.line"
        case .magneticStructure: return "wind"
        case .fitting: return "scope"
        case .run: return "play.circle"
        }
    }
}

struct Notification: Identifiable, Equatable {
    enum Kind { case success, error, info }
    let id = UUID()
    let message: String
    let kind: Kind
}

/// Root observable state for MagCalc Studio: the model configuration, the
/// backend connection, live visualizer data, and calculation lifecycle.
@MainActor
final class AppModel: ObservableObject {
    // MARK: Config + persistence

    @Published var config: MagCalcConfig {
        didSet { scheduleSave() }
    }

    @Published var selectedTab: StudioTab = .structure
    @Published var notification: Notification?

    // MARK: Backend connection

    @Published var serverURLString: String {
        didSet {
            UserDefaults.standard.set(serverURLString, forKey: "server.url")
            connectLogStream()
            refreshVisualizer()
        }
    }
    @Published private(set) var serverReachable = false

    let logStream = LogStreamClient()
    #if os(macOS)
    let backend = LocalBackendController()
    #endif

    // MARK: Derived / transient state

    @Published var visualizerData: VisualizerData?
    @Published var neighborShells: [NeighborShell] = []
    @Published var neighborsLoading = false
    @Published var hiddenBondKeys: Set<String> = []
    @Published var selectedBond: VisualizerBond?

    @Published var calcRunning = false
    @Published var calcStopping = false
    @Published var calcResults: CalculationResults?
    @Published var calcError: String?
    @Published var magStructure: MagStructureResult?

    private var saveTask: Task<Void, Never>?
    private var visualizerTask: Task<Void, Never>?
    private var stopRequested = false
    private static let configKey = "magcalc.config.v1"

    var api: APIClient? {
        guard let url = URL(string: serverURLString), url.host != nil else { return nil }
        return APIClient(baseURL: url)
    }

    init() {
        serverURLString = UserDefaults.standard.string(forKey: "server.url") ?? "http://127.0.0.1:8000"
        if let data = UserDefaults.standard.data(forKey: Self.configKey),
           let saved = try? JSONDecoder().decode(MagCalcConfig.self, from: data) {
            config = saved
        } else {
            config = .demo
        }
        connectLogStream()
        startHealthPolling()
        refreshVisualizer()
    }

    // MARK: Notifications

    func notify(_ message: String, _ kind: Notification.Kind = .success) {
        notification = Notification(message: message, kind: kind)
        let current = notification
        Task {
            try? await Task.sleep(for: .seconds(5))
            if notification == current { notification = nil }
        }
    }

    // MARK: Persistence

    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(400))
            guard let self, !Task.isCancelled else { return }
            if let data = try? JSONEncoder().encode(self.config) {
                UserDefaults.standard.set(data, forKey: Self.configKey)
            }
        }
        scheduleVisualizerRefresh()
    }

    func resetToDemo() {
        config = .demo
        notify("Reset to demo model (α-Cu₂V₂O₇)", .info)
    }

    // MARK: Project files (.magcalcproj JSON)

    func exportProjectData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(config)
    }

    func importProject(from data: Data) throws {
        config = try JSONDecoder().decode(MagCalcConfig.self, from: data)
        calcResults = nil
        calcError = nil
        magStructure = nil
        notify("Project loaded", .success)
    }

    // MARK: Backend health

    private func startHealthPolling() {
        Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                if let api = self.api {
                    let ok = await api.health()
                    if ok != self.serverReachable {
                        self.serverReachable = ok
                        if ok { self.refreshVisualizer() }
                    }
                } else {
                    self.serverReachable = false
                }
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    private func connectLogStream() {
        logStream.disconnect()
        if let url = URL(string: serverURLString), url.host != nil {
            logStream.connect(to: url)
        }
    }

    // MARK: Visualizer + neighbors

    private func scheduleVisualizerRefresh() {
        visualizerTask?.cancel()
        visualizerTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            self?.refreshVisualizer()
        }
    }

    func refreshVisualizer() {
        guard let api, !config.wyckoffAtoms.isEmpty else { return }
        let snapshot = config
        Task { [weak self] in
            do {
                let data = try await api.visualizerData(for: snapshot)
                self?.visualizerData = data
            } catch {
                // Silent: preview refresh runs on every edit.
            }
        }
    }

    func fetchNeighbors() {
        guard let api, !config.wyckoffAtoms.isEmpty else { return }
        neighborsLoading = true
        let snapshot = config
        Task { [weak self] in
            defer { self?.neighborsLoading = false }
            do {
                let shells = try await api.neighbors(for: snapshot)
                self?.neighborShells = shells
            } catch {
                self?.notify("Failed to fetch neighbor shells: \(error.localizedDescription)", .error)
            }
        }
    }

    #if os(macOS)
    /// Start the embedded Python backend and point the app at it.
    func startEmbeddedBackend() {
        backend.start()
        let url = "http://127.0.0.1:\(backend.port)"
        if serverURLString != url { serverURLString = url }
        notify("Starting embedded backend…", .info)
    }
    #endif

    // MARK: CIF / fit-data import

    func importCIF(from url: URL) {
        guard let api else { return notify("Connect to a backend first", .error) }
        Task { [weak self] in
            do {
                let parsed = try await api.parseCIF(fileURL: url)
                guard let self else { return }
                self.config.lattice = parsed.lattice
                self.config.wyckoffAtoms = parsed.wyckoffAtoms
                self.notify("CIF loaded: \(parsed.international) (SG \(parsed.lattice.spaceGroup))")
            } catch {
                self?.notify("CIF import failed: \(error.localizedDescription)", .error)
            }
        }
    }

    func uploadFitData(from url: URL) {
        guard let api else { return notify("Connect to a backend first", .error) }
        Task { [weak self] in
            do {
                let result = try await api.uploadFitData(fileURL: url)
                guard let self else { return }
                self.config.fitting.dataFile = result.dataFile
                self.config.fitting.dataLabel = result.originalName ?? url.lastPathComponent
                self.notify("Loaded data: \(self.config.fitting.dataLabel)")
            } catch {
                self?.notify("Data upload failed: \(error.localizedDescription)", .error)
            }
        }
    }

    // MARK: Calculation lifecycle

    func runCalculation(taskOverrides: [String: JSONValue]? = nil) {
        guard let api else { return notify("Connect to a backend first", .error) }
        guard !calcRunning else { return }
        calcRunning = true
        calcStopping = false
        stopRequested = false
        calcError = nil
        calcResults = nil
        magStructure = nil
        logStream.clear()
        selectedTab = .run
        let snapshot = config

        Task { [weak self] in
            do {
                let results = try await api.runCalculation(snapshot, taskOverrides: taskOverrides)
                guard let self else { return }
                self.calcResults = results
                if results.plots.contains("/files/mag_structure.json") {
                    self.magStructure = try? await api.fetchMagStructure()
                }
                self.notify("Calculation completed")
                self.postCompletionNotification(success: true)
            } catch {
                guard let self else { return }
                if self.stopRequested {
                    self.notify("Calculation stopped", .info)
                } else {
                    self.calcError = error.localizedDescription
                    self.notify("Calculation failed", .error)
                    self.postCompletionNotification(success: false)
                }
            }
            self?.calcRunning = false
            self?.calcStopping = false
        }
    }

    func stopCalculation() {
        guard let api, calcRunning, !calcStopping else { return }
        calcStopping = true
        stopRequested = true
        Task {
            try? await api.stopCalculation()
        }
    }

    func runFit() {
        guard !config.fitting.dataFile.isEmpty else {
            return notify("Upload an experimental data file first", .error)
        }
        guard !config.fitting.vary.isEmpty else {
            return notify("Select at least one parameter to vary", .error)
        }
        runCalculation(taskOverrides: ["fit": .bool(true), "plot_fit": .bool(true)])
    }

    /// Adopt best-fit parameter values returned by a fit run.
    func applyFitParameters() {
        guard let fitParams = calcResults?.fitParams else { return }
        for (name, value) in fitParams {
            if let d = value.doubleValue {
                config.parameters[name] = .number(d)
            }
        }
        notify("Applied \(fitParams.count) best-fit parameter values")
    }

    /// Import the minimized spin structure as a fixed manual structure
    /// (mirrors importMinimizedStructure in the web app).
    func importMinimizedStructure() {
        guard let vectors = magStructure?.vectors, !vectors.isEmpty else {
            return notify("No structure vectors found to import", .error)
        }
        config.magneticStructure.enabled = true
        config.magneticStructure.type = "pattern"
        config.magneticStructure.patternType = "generic"
        config.magneticStructure.directions = vectors
        config.tasks.minimization = false
        notify("Imported \(vectors.count) spins into Manual Structure (minimization disabled)")
        selectedTab = .magneticStructure
    }

    /// Long-run completion alert (user may have switched apps during a
    /// multi-minute calculation).
    private func postCompletionNotification(success: Bool) {
        #if os(macOS)
        NSSound(named: success ? "Glass" : "Basso")?.play()
        #endif
    }

    // MARK: Interaction helpers

    /// Adds a Heisenberg rule J{n} from a neighbor-shell suggestion, exactly
    /// like the web app's "Add J{i+1}" button (n = 1-based shell index).
    func addNeighborRule(shellIndex: Int, pair: [String], offset: [Int], distance: Double) {
        let name = "J\(shellIndex + 1)"
        var rule = SymmetryInteraction()
        rule.type = .heisenberg
        rule.refPair = pair
        rule.distance = (distance * 100000).rounded() / 100000
        rule.offset = offset
        rule.value = .string(name)
        config.symmetryInteractions.append(rule)
        if config.parameters[name] == nil || config.parameters[name]?.doubleValue == nil {
            config.parameters[name] = .number(0)
        }
        notify("Added Interaction Rule \(name)")
    }

    /// "Add Rule" button: blank heisenberg J1 at 3 Å (ensures the parameter).
    func addBlankRule() {
        config.symmetryInteractions.append(SymmetryInteraction())
        if config.parameters["J1"] == nil { config.parameters["J1"] = .number(0) }
    }

    /// Add an interaction on the bond selected in the 3D visualizer,
    /// mirroring addRuleFromVisualizer in App.jsx.
    func addRuleFromVisualizer(type: InteractionType) {
        guard let bond = selectedBond, let atoms = visualizerData?.atoms else { return }

        if config.interactionMode == "explicit" {
            var inter = ExplicitInteraction()
            inter.type = type == .dm ? "dm_manual" : type.rawValue
            inter.atomI = bond.atomI
            inter.atomJ = bond.atomJ
            inter.offsetJ = bond.offset
            inter.distance = bond.distance ?? 0
            inter.value = type.defaultValue
            config.explicitInteractions.append(inter)
        } else {
            var rule = SymmetryInteraction()
            rule.type = type
            rule.refPair = [
                atoms.first { $0.idx == bond.atomI }?.label ?? "?",
                atoms.first { $0.idx == bond.atomJ }?.label ?? "?",
            ]
            rule.offset = bond.offset
            rule.distance = ((bond.distance ?? 0) * 100000).rounded() / 100000
            rule.value = type.defaultValue
            config.symmetryInteractions.append(rule)
        }
        registerParameters(of: type.defaultValue)
        notify("Added \(type.rawValue) interaction")
    }

    /// Adds an interaction_matrix rule from a symmetry orbit + its allowed
    /// matrix form; free parameters get created at 0 (web parity).
    func addOrbitMatrixRule(orbit: BondOrbit, constraints: BondConstraints) {
        var rule = SymmetryInteraction()
        rule.type = .interactionMatrix
        rule.refPair = [orbit.representative.atomIText, orbit.representative.atomJText]
        rule.offset = orbit.representative.offset
        rule.distance = (orbit.distance * 100000).rounded() / 100000
        rule.value = .array(constraints.symbolicMatrix.map { row in
            .array(row.map { .string($0) })
        })
        config.symmetryInteractions.append(rule)
        for p in constraints.freeParameters where config.parameters[p] == nil {
            config.parameters[p] = .number(0)
        }
        notify("Added Symmetry Matrix Interaction")
    }

    /// Toggle 3D visibility of all bonds produced by a rule, keyed like the
    /// web app's getBondKey (arrays join with ",").
    func toggleBondVisibility(_ value: JSONValue?) {
        let key = VisualizerBond.bondKey(value)
        if hiddenBondKeys.contains(key) {
            hiddenBondKeys.remove(key)
        } else {
            hiddenBondKeys.insert(key)
        }
    }

    func isBondHidden(_ value: JSONValue?) -> Bool {
        hiddenBondKeys.contains(VisualizerBond.bondKey(value))
    }

    // MARK: YAML config exchange (web-app format)

    func exportYAML() throws -> String {
        try YAMLConfig.export(config)
    }

    func importYAML(from url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let text = try String(contentsOf: url, encoding: .utf8)
            config = try YAMLConfig.importConfig(from: text)
            neighborShells = []
            visualizerData = nil
            selectedBond = nil
            calcResults = nil
            calcError = nil
            magStructure = nil
            logStream.clear()
            refreshVisualizer()
            notify("Configuration imported successfully! Previous state cleared.")
        } catch {
            notify("Error parsing YAML: \(error.localizedDescription)", .error)
        }
    }

    /// Ensure symbolic names used in an interaction value exist as parameters.
    func registerParameters(of value: JSONValue) {
        switch value {
        case .string(let s):
            let name = s.hasPrefix("-") ? String(s.dropFirst()) : s
            guard !name.isEmpty, Double(name) == nil else { return }
            let identifier = name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
            if identifier, config.parameters[name] == nil {
                config.parameters[name] = .number(0)
            }
        case .array(let items):
            items.forEach { registerParameters(of: $0) }
        default:
            break
        }
    }
}

#if os(macOS)
import AppKit
#endif
