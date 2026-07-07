import Foundation

// MARK: - Config sections
// These mirror the state shape of the web app (gui/src/App.jsx) so that
// projects and backend payloads stay wire-compatible with the FastAPI server.

struct LatticeParameters: Codable, Hashable {
    var a: Double = 5.0
    var b: Double = 5.0
    var c: Double = 5.0
    var alpha: Double = 90
    var beta: Double = 90
    var gamma: Double = 90
    var spaceGroup: Int = 1

    enum CodingKeys: String, CodingKey {
        case a, b, c, alpha, beta, gamma
        case spaceGroup = "space_group"
    }

    /// Cartesian lattice vectors (rows) from the cell parameters, using the
    /// standard crystallographic convention (a along x, b in the x-y plane).
    var latticeVectors: [SIMD3<Double>] {
        let d2r = Double.pi / 180
        let ca = cos(alpha * d2r), cb = cos(beta * d2r), cg = cos(gamma * d2r)
        let sg = sin(gamma * d2r)
        let v1 = SIMD3<Double>(a, 0, 0)
        let v2 = SIMD3<Double>(b * cg, b * sg, 0)
        let cx = c * cb
        let cy = c * (ca - cb * cg) / sg
        let czSq = c * c - cx * cx - cy * cy
        let cz = czSq > 0 ? sqrt(czSq) : 0
        let v3 = SIMD3<Double>(cx, cy, cz)
        return [v1, v2, v3]
    }

    func cartesian(fromFractional f: [Double]) -> SIMD3<Double> {
        let v = latticeVectors
        return v[0] * (f.count > 0 ? f[0] : 0)
             + v[1] * (f.count > 1 ? f[1] : 0)
             + v[2] * (f.count > 2 ? f[2] : 0)
    }
}

struct WyckoffAtom: Codable, Hashable, Identifiable {
    var id = UUID()
    var label: String = "Atom"
    var pos: [Double] = [0, 0, 0]
    var spinS: Double = 0.5
    var ion: String?
    var element: String?

    enum CodingKeys: String, CodingKey {
        case label, pos, ion, element
        case spinS = "spin_S"
    }
}

enum InteractionType: String, Codable, CaseIterable, Identifiable {
    case heisenberg
    case dm
    case anisotropicExchange = "anisotropic_exchange"
    case interactionMatrix = "interaction_matrix"
    case kitaev

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .heisenberg: return "Heisenberg"
        case .dm: return "DM (Dzyaloshinskii–Moriya)"
        case .anisotropicExchange: return "Anisotropic Exchange"
        case .interactionMatrix: return "Interaction Matrix"
        case .kitaev: return "Kitaev"
        }
    }

    /// Default symbolic value, matching the web app's getInitValue().
    var defaultValue: JSONValue {
        switch self {
        case .heisenberg: return .string("J1")
        case .kitaev: return .string("K1")
        case .dm: return .array([.string("D1"), .string("D2"), .string("D3")])
        case .anisotropicExchange: return .array([.string("G1"), .string("G2"), .string("G3")])
        case .interactionMatrix:
            let zeroRow = JSONValue.array([.string("0"), .string("0"), .string("0")])
            return .array([zeroRow, zeroRow, zeroRow])
        }
    }
}

struct SymmetryInteraction: Codable, Hashable, Identifiable {
    var id = UUID()
    var type: InteractionType = .heisenberg
    /// nil ⇒ distance-only rule ("Auto-detected" in the web UI).
    var refPair: [String]?
    var distance: Double = 3.0
    var value: JSONValue = .string("J1")
    var offset: [Int]?
    /// Kitaev only: bond direction x/y/z.
    var bondDirection: String?

    enum CodingKeys: String, CodingKey {
        case type, distance, value, offset
        case refPair = "ref_pair"
        case bondDirection = "bond_direction"
    }
}

/// Manual per-bond interaction used in "Explicit Interactions" mode
/// (interaction type heisenberg | dm_manual | anisotropic_exchange).
struct ExplicitInteraction: Codable, Hashable, Identifiable {
    var id = UUID()
    var type = "heisenberg"
    var atomI = 0
    var atomJ = 1
    var offsetJ: [Int] = [0, 0, 0]
    var distance: Double = 3.0
    var value: JSONValue = .string("J1")

    enum CodingKeys: String, CodingKey {
        case type, distance, value
        case atomI = "atom_i"
        case atomJ = "atom_j"
        case offsetJ = "offset_j"
    }
}

struct SingleIonAnisotropy: Codable, Hashable, Identifiable {
    var id = UUID()
    var type = "sia"
    var atomLabel = "Cu"
    var value: JSONValue = .string("D")
    var axis: [Double] = [0, 0, 1]

    enum CodingKeys: String, CodingKey {
        case type, value, axis
        case atomLabel = "atom_label"
    }
}

struct QPath: Codable, Hashable {
    /// Named q-points; `path` lists names in traversal order.
    var points: [String: [Double]] = ["Start": [0, 1, 0], "End": [0, 3, 0]]
    var path: [String] = ["Start", "End"]
    var pointsPerSegment: Int = 200

    enum CodingKeys: String, CodingKey {
        case points, path
        case pointsPerSegment = "points_per_segment"
    }
}

struct Tasks: Codable, Hashable {
    var minimization = true
    var dispersion = true
    var plotDispersion = true
    var sqwMap = true
    var plotSqwMap = true
    var exportCSV = false
    var powderAverage = false
    var plotStructure = false

    enum CodingKeys: String, CodingKey {
        case minimization, dispersion
        case plotDispersion = "plot_dispersion"
        case sqwMap = "sqw_map"
        case plotSqwMap = "plot_sqw_map"
        case exportCSV = "export_csv"
        case powderAverage = "powder_average"
        case plotStructure = "plot_structure"
    }
}

struct PlottingSettings: Codable, Hashable {
    var energyMin: Double = 0
    var energyMax: Double = 10
    var broadening: Double = 0.2
    var energyResolution: Double = 0.05
    var momentumMax: Double = 4.0
    var savePlot = false
    var showPlot = false
    var plotStructure = false
    var dispPlotFilename = "disp_plot.png"
    var sqwPlotFilename = "sqw_plot.png"

    enum CodingKeys: String, CodingKey {
        case energyMin = "energy_min"
        case energyMax = "energy_max"
        case broadening
        case energyResolution = "energy_resolution"
        case momentumMax = "momentum_max"
        case savePlot = "save_plot"
        case showPlot = "show_plot"
        case plotStructure = "plot_structure"
        case dispPlotFilename = "disp_plot_filename"
        case sqwPlotFilename = "sqw_plot_filename"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = PlottingSettings()
        energyMin = try c.decodeIfPresent(Double.self, forKey: .energyMin) ?? d.energyMin
        energyMax = try c.decodeIfPresent(Double.self, forKey: .energyMax) ?? d.energyMax
        broadening = try c.decodeIfPresent(Double.self, forKey: .broadening) ?? d.broadening
        energyResolution = try c.decodeIfPresent(Double.self, forKey: .energyResolution) ?? d.energyResolution
        momentumMax = try c.decodeIfPresent(Double.self, forKey: .momentumMax) ?? d.momentumMax
        savePlot = try c.decodeIfPresent(Bool.self, forKey: .savePlot) ?? d.savePlot
        showPlot = try c.decodeIfPresent(Bool.self, forKey: .showPlot) ?? d.showPlot
        plotStructure = try c.decodeIfPresent(Bool.self, forKey: .plotStructure) ?? d.plotStructure
        dispPlotFilename = try c.decodeIfPresent(String.self, forKey: .dispPlotFilename) ?? d.dispPlotFilename
        sqwPlotFilename = try c.decodeIfPresent(String.self, forKey: .sqwPlotFilename) ?? d.sqwPlotFilename
    }
}

struct MagneticStructureSettings: Codable, Hashable {
    var enabled = false
    var type = "pattern"
    var patternType = "antiferromagnetic"
    var directions: [[Double]] = []

    enum CodingKeys: String, CodingKey {
        case enabled, type, directions
        case patternType = "pattern_type"
    }
}

struct MinimizationSettings: Codable, Hashable {
    var numStarts = 1000
    var nWorkers = 8
    var earlyStopping = 10
    var method = "L-BFGS-B"

    enum CodingKeys: String, CodingKey {
        case method
        case numStarts = "num_starts"
        case nWorkers = "n_workers"
        case earlyStopping = "early_stopping"
    }
}

struct CalculationSettings: Codable, Hashable {
    var cacheMode = "auto"
    var backend = "numpy"

    enum CodingKeys: String, CodingKey {
        case backend
        case cacheMode = "cache_mode"
    }
}

struct PowderAverageSettings: Codable, Hashable {
    var qMin: Double = 0.1
    var qMax: Double = 4.0
    var qCount = 50
    var numSamples = 50

    enum CodingKeys: String, CodingKey {
        case qMin = "q_min"
        case qMax = "q_max"
        case qCount = "q_count"
        case numSamples = "num_samples"
    }
}

struct FitScalar: Codable, Hashable {
    var value: Double
    var vary: Bool
}

struct FittingSettings: Codable, Hashable {
    var type = "dispersion" // dispersion | sqw | powder
    var dataFile = ""
    var dataLabel = ""
    var method = "leastsq"
    var vary: [String] = []
    var bounds: [String: [Double?]] = [:]
    var match = "nearest"
    var scale = FitScalar(value: 1.0, vary: true)
    var background = FitScalar(value: 0.0, vary: true)
    var energyBroadening = FitScalar(value: 0.3, vary: false)

    enum CodingKeys: String, CodingKey {
        case type, method, vary, bounds, match, scale, background
        case dataFile = "data_file"
        case dataLabel = "data_label"
        case energyBroadening = "energy_broadening"
    }
}

struct OutputSettings: Codable, Hashable {
    var dispCSVFilename = "disp_data.csv"
    var sqwCSVFilename = "sqw_data.csv"
    var dispDataFilename = "disp_data.npz"
    var sqwDataFilename = "sqw_data.npz"
    var saveData = true

    enum CodingKeys: String, CodingKey {
        case dispCSVFilename = "disp_csv_filename"
        case sqwCSVFilename = "sqw_csv_filename"
        case dispDataFilename = "disp_data_filename"
        case sqwDataFilename = "sqw_data_filename"
        case saveData = "save_data"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = OutputSettings()
        dispCSVFilename = try c.decodeIfPresent(String.self, forKey: .dispCSVFilename) ?? d.dispCSVFilename
        sqwCSVFilename = try c.decodeIfPresent(String.self, forKey: .sqwCSVFilename) ?? d.sqwCSVFilename
        dispDataFilename = try c.decodeIfPresent(String.self, forKey: .dispDataFilename) ?? d.dispDataFilename
        sqwDataFilename = try c.decodeIfPresent(String.self, forKey: .sqwDataFilename) ?? d.sqwDataFilename
        saveData = try c.decodeIfPresent(Bool.self, forKey: .saveData) ?? d.saveData
    }
}

// MARK: - Root config

struct MagCalcConfig: Codable, Hashable {
    var lattice = LatticeParameters()
    var wyckoffAtoms: [WyckoffAtom] = []
    var magneticElements: [String] = ["Cu"]
    var symmetryInteractions: [SymmetryInteraction] = []
    var explicitInteractions: [ExplicitInteraction] = []
    var singleIonAnisotropy: [SingleIonAnisotropy] = []
    var parameters: [String: JSONValue] = ["H_mag": .number(0), "H_dir": .array([0, 0, 1])]
    var tasks = Tasks()
    var qPath = QPath()
    var plotting = PlottingSettings()
    var magneticStructure = MagneticStructureSettings()
    var minimization = MinimizationSettings()
    var calculation = CalculationSettings()
    var powderAverage = PowderAverageSettings()
    var fitting = FittingSettings()
    var output = OutputSettings()
    var atomMode = "symmetry"          // "symmetry" | "explicit"
    var interactionMode = "symmetry"   // "symmetry" | "explicit"

    /// Raw crystal_structure / interactions / magnetic_structure captured when an
    /// example config is imported. Sent to the backend verbatim so features the
    /// designer cannot model -- lattice_vectors, interaction_matrix, single-ion
    /// anisotropy, DM, spiral/generic magnetic orders -- run exactly as
    /// `python -m magcalc run` does. Not part of CodingKeys, so it is neither
    /// encoded nor decoded (custom init(from:) leaves it nil).
    var rawImport: JSONValue? = nil

    enum CodingKeys: String, CodingKey {
        case lattice, parameters, tasks, plotting, minimization, calculation, fitting, output
        case wyckoffAtoms = "wyckoff_atoms"
        case magneticElements = "magnetic_elements"
        case symmetryInteractions = "symmetry_interactions"
        case explicitInteractions = "explicit_interactions"
        case singleIonAnisotropy = "single_ion_anisotropy"
        case qPath = "q_path"
        case magneticStructure = "magnetic_structure"
        case powderAverage = "powder_average"
        case atomMode = "atom_mode"
        case interactionMode = "interaction_mode"
    }

    // Decode with defaults for keys added after older saved configs (mirrors
    // the web app's merge-over-defaults on load).
    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = MagCalcConfig()
        lattice = try c.decodeIfPresent(LatticeParameters.self, forKey: .lattice) ?? d.lattice
        wyckoffAtoms = try c.decodeIfPresent([WyckoffAtom].self, forKey: .wyckoffAtoms) ?? d.wyckoffAtoms
        magneticElements = try c.decodeIfPresent([String].self, forKey: .magneticElements) ?? d.magneticElements
        symmetryInteractions = try c.decodeIfPresent([SymmetryInteraction].self, forKey: .symmetryInteractions) ?? d.symmetryInteractions
        explicitInteractions = try c.decodeIfPresent([ExplicitInteraction].self, forKey: .explicitInteractions) ?? d.explicitInteractions
        singleIonAnisotropy = try c.decodeIfPresent([SingleIonAnisotropy].self, forKey: .singleIonAnisotropy) ?? d.singleIonAnisotropy
        parameters = try c.decodeIfPresent([String: JSONValue].self, forKey: .parameters) ?? d.parameters
        tasks = try c.decodeIfPresent(Tasks.self, forKey: .tasks) ?? d.tasks
        qPath = try c.decodeIfPresent(QPath.self, forKey: .qPath) ?? d.qPath
        plotting = try c.decodeIfPresent(PlottingSettings.self, forKey: .plotting) ?? d.plotting
        magneticStructure = try c.decodeIfPresent(MagneticStructureSettings.self, forKey: .magneticStructure) ?? d.magneticStructure
        minimization = try c.decodeIfPresent(MinimizationSettings.self, forKey: .minimization) ?? d.minimization
        calculation = try c.decodeIfPresent(CalculationSettings.self, forKey: .calculation) ?? d.calculation
        powderAverage = try c.decodeIfPresent(PowderAverageSettings.self, forKey: .powderAverage) ?? d.powderAverage
        fitting = try c.decodeIfPresent(FittingSettings.self, forKey: .fitting) ?? d.fitting
        output = try c.decodeIfPresent(OutputSettings.self, forKey: .output) ?? d.output
        atomMode = try c.decodeIfPresent(String.self, forKey: .atomMode) ?? d.atomMode
        interactionMode = try c.decodeIfPresent(String.self, forKey: .interactionMode) ?? d.interactionMode
    }

    /// Scalar parameter names available for fitting (vectors like H_dir excluded).
    var fittableParameterNames: [String] {
        parameters
            .filter { $0.key != "S" && !$0.value.isVector }
            .keys.sorted()
    }

    /// The demo model shipped with the web app: α-Cu2V2O7 (aCVO).
    static var demo: MagCalcConfig {
        var c = MagCalcConfig()
        c.lattice = LatticeParameters(a: 20.645, b: 8.383, c: 6.442,
                                      alpha: 90, beta: 90, gamma: 90, spaceGroup: 43)
        c.wyckoffAtoms = [WyckoffAtom(label: "Cu", pos: [0.16572, 0.3646, 0.7545], spinS: 0.5)]
        c.magneticElements = ["Cu"]
        c.symmetryInteractions = [
            SymmetryInteraction(type: .heisenberg, refPair: ["Cu0", "Cu2"], distance: 3.1325, value: .string("J1"), offset: [0, 0, 0]),
            SymmetryInteraction(type: .dm, refPair: ["Cu0", "Cu2"], distance: 3.1325,
                                value: .array([.string("Dx"), .string("0"), .string("0")]), offset: [0, 0, 0]),
            SymmetryInteraction(type: .anisotropicExchange, refPair: ["Cu0", "Cu2"], distance: 3.1325,
                                value: .array([.string("G1"), .string("-G1"), .string("-G1")]), offset: [0, 0, 0]),
            SymmetryInteraction(type: .heisenberg, refPair: ["Cu0", "Cu13"], distance: 3.9751, value: .string("J2"), offset: [0, 0, 0]),
            SymmetryInteraction(type: .heisenberg, refPair: ["Cu0", "Cu9"], distance: 5.2572, value: .string("J3"), offset: [0, 0, 0]),
        ]
        c.parameters = [
            "J1": .number(2.49), "J2": .number(2.79), "J3": .number(5.05),
            "G1": .number(0.28), "Dx": .number(2.67), "D": .number(0.0),
            "H_mag": .number(20.0), "H_dir": .array([0, 0, 1]),
        ]
        c.qPath = QPath(points: ["Start": [0, 1, 0], "End": [0, 3, 0]],
                        path: ["Start", "End"], pointsPerSegment: 200)
        return c
    }
}

// MARK: - Backend payload construction

extension MagCalcConfig {
    /// Builds the `data` payload sent to /run-calculation and /expand-config,
    /// mirroring runCalculation() in the web app's App.jsx.
    func backendInput(taskOverrides: [String: JSONValue]? = nil) -> JSONValue {
        var qPoints: [String: JSONValue] = [:]
        for (name, coords) in qPath.points {
            qPoints[name] = .array(coords.map { .number($0) })
        }
        qPoints["path"] = .array(qPath.path.map { .string($0) })
        qPoints["points_per_segment"] = .number(Double(qPath.pointsPerSegment))

        var plottingDict = (try? JSONValue(encoding: plotting)) ?? .object([:])
        if var p = plottingDict.objectValue {
            p["energy_limits_disp"] = .array([.number(plotting.energyMin), .number(plotting.energyMax)])
            p["broadening_width"] = .number(plotting.broadening)
            plottingDict = .object(p)
        }

        var minimizationDict = (try? JSONValue(encoding: minimization)) ?? .object([:])
        if var m = minimizationDict.objectValue {
            m["enabled"] = .bool(tasks.minimization)
            minimizationDict = .object(m)
        }

        let tasksValue: JSONValue
        if let overrides = taskOverrides {
            tasksValue = .object(overrides)
        } else {
            tasksValue = (try? JSONValue(encoding: tasks)) ?? .object([:])
        }

        let interactionsValue: JSONValue = interactionMode == "explicit"
            ? .object(["list": .array(explicitInteractions.map { $0.payloadValue })])
            : .object(["symmetry_rules": .array(symmetryInteractions.map { $0.payloadValue })])

        // Prefer the verbatim structure/interactions from an imported example so
        // lattice_vectors, interaction_matrix, SIA and spiral/generic orders
        // reach the backend intact (the designer model cannot represent them).
        let rawObj = rawImport?.objectValue
        let crystalValue: JSONValue = rawObj?["crystal_structure"] ?? .object([
            "lattice_parameters": (try? JSONValue(encoding: lattice)) ?? .object([:]),
            "wyckoff_atoms": .array(wyckoffAtoms.map { $0.payloadValue }),
            "atom_mode": .string(atomMode),
            "dimensionality": .number(3),
            "magnetic_elements": .array(magneticElements.map { .string($0) }),
        ])
        let interactionsFinal: JSONValue = rawObj?["interactions"] ?? interactionsValue
        let magStructFinal: JSONValue = rawObj?["magnetic_structure"]
            ?? ((try? JSONValue(encoding: magneticStructure)) ?? .object([:]))

        var input: [String: JSONValue] = [
            "crystal_structure": crystalValue,
            "interactions": interactionsFinal,
            "magnetic_structure": magStructFinal,
            "parameters": .object(parameters),
            "tasks": tasksValue,
            "q_path": .object(qPoints),
            "plotting": plottingDict,
            "minimization": minimizationDict,
            "powder_average": (try? JSONValue(encoding: powderAverage)) ?? .object([:]),
            "calculation": (try? JSONValue(encoding: calculation)) ?? .object([:]),
            "output": (try? JSONValue(encoding: output)) ?? .object([:]),
        ]
        input["fitting"] = (try? JSONValue(encoding: fitting)) ?? .object([:])
        return .object(input)
    }

    /// Payload for structure-only endpoints (/get-neighbors, /analyze-bonds,
    /// /get-visualizer-data).
    func structurePayload(includeInteractions: Bool = false) -> JSONValue {
        let rawObj = rawImport?.objectValue
        var data: [String: JSONValue] = [
            "crystal_structure": rawObj?["crystal_structure"] ?? .object([
                "lattice_parameters": (try? JSONValue(encoding: lattice)) ?? .object([:]),
                "wyckoff_atoms": .array(wyckoffAtoms.map { $0.payloadValue }),
                "atom_mode": .string(atomMode),
            ]),
        ]
        if let ms = rawObj?["magnetic_structure"] { data["magnetic_structure"] = ms }
        if includeInteractions {
            if let it = rawObj?["interactions"] {
                data["interactions"] = it
            } else {
                // Mirrors the web app's preview payload: symmetry rules always,
                // explicit list only in explicit mode, SIA alongside.
                var inter: [String: JSONValue] = [
                    "symmetry_rules": .array(symmetryInteractions.map { $0.payloadValue }),
                    "single_ion_anisotropy": .array(singleIonAnisotropy.map {
                        (try? JSONValue(encoding: $0)) ?? .object([:])
                    }),
                ]
                if interactionMode == "explicit" {
                    inter["list"] = .array(explicitInteractions.map { $0.payloadValue })
                }
                data["interactions"] = .object(inter)
            }
            data["parameters"] = .object(parameters)
        }
        return .object(data)
    }
}

extension WyckoffAtom {
    var payloadValue: JSONValue {
        var o: [String: JSONValue] = [
            "label": .string(label),
            "pos": .array(pos.map { .number($0) }),
            "spin_S": .number(spinS),
        ]
        if let ion { o["ion"] = .string(ion) }
        if let element { o["element"] = .string(element) }
        return .object(o)
    }
}

extension SymmetryInteraction {
    var payloadValue: JSONValue {
        var o: [String: JSONValue] = [
            "type": .string(type.rawValue),
            "distance": .number(distance),
            "value": value,
        ]
        if let refPair { o["ref_pair"] = .array(refPair.map { .string($0) }) }
        if let offset { o["offset"] = .array(offset.map { .number(Double($0)) }) }
        if let bondDirection { o["bond_direction"] = .string(bondDirection) }
        return .object(o)
    }
}

extension ExplicitInteraction {
    var payloadValue: JSONValue {
        .object([
            "type": .string(type),
            "atom_i": .number(Double(atomI)),
            "atom_j": .number(Double(atomJ)),
            "offset_j": .array(offsetJ.map { .number(Double($0)) }),
            "distance": .number(distance),
            "value": value,
        ])
    }
}

extension JSONValue {
    /// Round-trips any Encodable through JSON into a JSONValue tree.
    init<T: Encodable>(encoding value: T) throws {
        let data = try JSONEncoder().encode(value)
        self = try JSONDecoder().decode(JSONValue.self, from: data)
    }
}
