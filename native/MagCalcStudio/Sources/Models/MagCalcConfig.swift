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
    /// 1/S corrections: zero-point energy + ordered-moment reduction.
    var corrections = false
    /// Beyond-LSWT tasks (see TUTORIAL 4h): SCGA diffuse S(q), thermal Monte-Carlo,
    /// classical-dynamics S(q,w), and the KPM spectral solver.
    var scga = false
    var thermalMC = false
    var sampledCorrelations = false
    var kpmSqw = false

    enum CodingKeys: String, CodingKey {
        case minimization, dispersion, corrections, scga
        case plotDispersion = "plot_dispersion"
        case sqwMap = "sqw_map"
        case plotSqwMap = "plot_sqw_map"
        case exportCSV = "export_csv"
        case powderAverage = "powder_average"
        case plotStructure = "plot_structure"
        case thermalMC = "thermal_mc"
        case sampledCorrelations = "sampled_correlations"
        case kpmSqw = "kpm_sqw"
    }

    init() {}

    // Decode with defaults so project files saved before a flag existed still open.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = Tasks()
        minimization = try c.decodeIfPresent(Bool.self, forKey: .minimization) ?? d.minimization
        dispersion = try c.decodeIfPresent(Bool.self, forKey: .dispersion) ?? d.dispersion
        plotDispersion = try c.decodeIfPresent(Bool.self, forKey: .plotDispersion) ?? d.plotDispersion
        sqwMap = try c.decodeIfPresent(Bool.self, forKey: .sqwMap) ?? d.sqwMap
        plotSqwMap = try c.decodeIfPresent(Bool.self, forKey: .plotSqwMap) ?? d.plotSqwMap
        exportCSV = try c.decodeIfPresent(Bool.self, forKey: .exportCSV) ?? d.exportCSV
        powderAverage = try c.decodeIfPresent(Bool.self, forKey: .powderAverage) ?? d.powderAverage
        plotStructure = try c.decodeIfPresent(Bool.self, forKey: .plotStructure) ?? d.plotStructure
        corrections = try c.decodeIfPresent(Bool.self, forKey: .corrections) ?? d.corrections
        scga = try c.decodeIfPresent(Bool.self, forKey: .scga) ?? d.scga
        thermalMC = try c.decodeIfPresent(Bool.self, forKey: .thermalMC) ?? d.thermalMC
        sampledCorrelations = try c.decodeIfPresent(Bool.self, forKey: .sampledCorrelations) ?? d.sampledCorrelations
        kpmSqw = try c.decodeIfPresent(Bool.self, forKey: .kpmSqw) ?? d.kpmSqw
    }
}

// MARK: - Beyond-LSWT task settings (mirror the web app's blocks; TUTORIAL 4h)

struct SCGASettings: Codable, Hashable {
    /// kT in meV (SCGA is classical: temperature IS the energy scale).
    var temperature: Double = 1.0
    var meshDensity: Int = 12

    enum CodingKeys: String, CodingKey {
        case temperature
        case meshDensity = "mesh_density"
    }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = SCGASettings()
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature) ?? d.temperature
        meshDensity = try c.decodeIfPresent(Int.self, forKey: .meshDensity) ?? d.meshDensity
    }
}

struct ThermalMCSettings: Codable, Hashable {
    /// Comma-separated kT list in meV, parsed at payload build.
    var temperatures = "0.5, 1.0, 2.0, 4.0"
    var supercell = "4, 4, 1"
    var nSweeps: Int = 4000
    var nEquil: Int = 1500

    enum CodingKeys: String, CodingKey {
        case temperatures, supercell
        case nSweeps = "n_sweeps"
        case nEquil = "n_equil"
    }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = ThermalMCSettings()
        temperatures = try c.decodeIfPresent(String.self, forKey: .temperatures) ?? d.temperatures
        supercell = try c.decodeIfPresent(String.self, forKey: .supercell) ?? d.supercell
        nSweeps = try c.decodeIfPresent(Int.self, forKey: .nSweeps) ?? d.nSweeps
        nEquil = try c.decodeIfPresent(Int.self, forKey: .nEquil) ?? d.nEquil
    }
}

struct SampledCorrelationsSettings: Codable, Hashable {
    var temperature: Double = 0.5
    var supercell = "8, 1, 1"
    var dt: Double = 0.02
    var nSteps: Int = 2048
    var nTraj: Int = 8

    enum CodingKeys: String, CodingKey {
        case temperature, supercell, dt
        case nSteps = "n_steps"
        case nTraj = "n_traj"
    }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = SampledCorrelationsSettings()
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature) ?? d.temperature
        supercell = try c.decodeIfPresent(String.self, forKey: .supercell) ?? d.supercell
        dt = try c.decodeIfPresent(Double.self, forKey: .dt) ?? d.dt
        nSteps = try c.decodeIfPresent(Int.self, forKey: .nSteps) ?? d.nSteps
        nTraj = try c.decodeIfPresent(Int.self, forKey: .nTraj) ?? d.nTraj
    }
}

struct KPMSettings: Codable, Hashable {
    var eMin: Double = 0.0
    var eMax: Double = 10.0
    var eStep: Double = 0.05
    var fwhm: Double = 0.1

    enum CodingKeys: String, CodingKey {
        case fwhm
        case eMin = "e_min"
        case eMax = "e_max"
        case eStep = "e_step"
    }

    init() {}
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = KPMSettings()
        eMin = try c.decodeIfPresent(Double.self, forKey: .eMin) ?? d.eMin
        eMax = try c.decodeIfPresent(Double.self, forKey: .eMax) ?? d.eMax
        eStep = try c.decodeIfPresent(Double.self, forKey: .eStep) ?? d.eStep
        fwhm = try c.decodeIfPresent(Double.self, forKey: .fwhm) ?? d.fwhm
    }
}

struct PlottingSettings: Codable, Hashable {
    var energyMin: Double = 0
    var energyMax: Double = 10
    var broadening: Double = 0.2
    var energyResolution: Double = 0.05
    var momentumMax: Double = 4.0
    var autoScaleDisp = true
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
        case autoScaleDisp = "auto_scale_disp"
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
        autoScaleDisp = try c.decodeIfPresent(Bool.self, forKey: .autoScaleDisp) ?? d.autoScaleDisp
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
    /// Monte-Carlo annealing (SpinW `anneal` / Sunny LocalSampler) is the default: it
    /// crosses barriers, so it does not get trapped the way multistart gradient descent
    /// does. On SW20-in-field the gradient path reached the true minimum in only 3 of
    /// 200 starts; anneal finds it in a single run.
    var method = "anneal"
    /// Annealing runs (a handful) for `anneal`/`steep`; random restarts (hundreds) for
    /// the gradient methods. See `applyMethodDefaults`.
    var numStarts = 4
    /// Cooling steps for `anneal`; each attempts one move per spin.
    var nSweeps = 2000
    var nWorkers = 8
    var earlyStopping = 10

    var isAnnealMethod: Bool {
        method == "anneal" || method == "monte_carlo" || method == "steep"
    }

    /// The two families take completely different budgets, and carrying one method's
    /// numbers over to the other gives either an absurdly slow run or a silently wrong
    /// ground state. Retune whenever the method changes.
    mutating func applyMethodDefaults(previousMethod: String) {
        let wasAnneal = previousMethod == "anneal" || previousMethod == "monte_carlo"
            || previousMethod == "steep"
        guard wasAnneal != isAnnealMethod else { return }
        if isAnnealMethod {
            numStarts = 4
        } else {
            numStarts = 1000
            earlyStopping = 10
        }
    }

    enum CodingKeys: String, CodingKey {
        case method
        case numStarts = "num_starts"
        case nSweeps = "n_sweeps"
        case nWorkers = "n_workers"
        case earlyStopping = "early_stopping"
    }
}

struct CalculationSettings: Codable, Hashable {
    var cacheMode = "auto"
    var backend = "numpy"
    /// LSWT is an expansion about a classical energy MINIMUM. If the magnetic structure
    /// is not one, the spectrum is meaningless — so the run FAILS by default rather than
    /// drawing a plausible-looking plot. Use "warn" only for structures that are
    /// knowingly metastable (e.g. a commensurate approximation to an incommensurate
    /// spiral).
    var onImaginary = "error"
    /// LSWT engine: "dipole" (default) or "SUN". SU(N) captures single-ion (multipolar)
    /// excitations that dipole LSWT structurally cannot represent (e.g. FeI2's bound state).
    var mode = "dipole"
    /// Sample temperature in Kelvin. nil => T -> 0 (bare LSWT). Applies the Bose factor
    /// to S(Q,w)/powder intensities.
    var temperature: Double? = nil
    /// Neutron cross-section contraction: "perp" (default) | "trace" | "chiral" | a
    /// tensor component like "xx"/"yy"/"zz".
    var crossSection = "perp"
    /// Entangled-mode extras: dimer series order (0 = harmonic bond-operator) and the
    /// units partition as JSON text, e.g. "[[0,1],[2,3]]" (blank: from the config).
    var seriesOrder: Int = 0
    var unitsText = ""

    enum CodingKeys: String, CodingKey {
        case backend, mode, temperature
        case cacheMode = "cache_mode"
        case onImaginary = "on_imaginary"
        case crossSection = "cross_section"
        case seriesOrder = "series_order"
        case unitsText = "units_text"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = CalculationSettings()
        cacheMode = try c.decodeIfPresent(String.self, forKey: .cacheMode) ?? d.cacheMode
        backend = try c.decodeIfPresent(String.self, forKey: .backend) ?? d.backend
        onImaginary = try c.decodeIfPresent(String.self, forKey: .onImaginary) ?? d.onImaginary
        mode = try c.decodeIfPresent(String.self, forKey: .mode) ?? d.mode
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature) ?? d.temperature
        crossSection = try c.decodeIfPresent(String.self, forKey: .crossSection) ?? d.crossSection
        seriesOrder = try c.decodeIfPresent(Int.self, forKey: .seriesOrder) ?? d.seriesOrder
        unitsText = try c.decodeIfPresent(String.self, forKey: .unitsText) ?? d.unitsText
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
    var scga = SCGASettings()
    var thermalMC = ThermalMCSettings()
    var sampledCorrelations = SampledCorrelationsSettings()
    var kpm = KPMSettings()
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
        case scga, kpm
        case thermalMC = "thermal_mc"
        case sampledCorrelations = "sampled_correlations"
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
        scga = try c.decodeIfPresent(SCGASettings.self, forKey: .scga) ?? d.scga
        thermalMC = try c.decodeIfPresent(ThermalMCSettings.self, forKey: .thermalMC) ?? d.thermalMC
        sampledCorrelations = try c.decodeIfPresent(SampledCorrelationsSettings.self,
                                                    forKey: .sampledCorrelations) ?? d.sampledCorrelations
        kpm = try c.decodeIfPresent(KPMSettings.self, forKey: .kpm) ?? d.kpm
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

        // calculation: strip UI-only fields; only send series_order when the
        // entangled series is actually requested (mirrors the web app).
        var calcDict = ((try? JSONValue(encoding: calculation)) ?? .object([:])).objectValue ?? [:]
        calcDict.removeValue(forKey: "units_text")
        if !(calculation.mode == "entangled" && calculation.seriesOrder > 0) {
            calcDict.removeValue(forKey: "series_order")
        }

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
            "calculation": .object(calcDict),
            "output": (try? JSONValue(encoding: output)) ?? .object([:]),
        ]
        input["fitting"] = (try? JSONValue(encoding: fitting)) ?? .object([:])

        // Beyond-LSWT blocks: emitted only for enabled tasks, so the config the
        // backend receives matches the CLI form (TUTORIAL 4h).
        if tasks.scga {
            input["scga"] = (try? JSONValue(encoding: scga)) ?? .object([:])
        }
        if tasks.thermalMC {
            input["thermal_mc"] = .object([
                "temperatures": .array(Self.parseNumberList(thermalMC.temperatures,
                                                            fallback: [0.5, 1, 2, 4]).map { .number($0) }),
                "supercell": .array(Self.parseNumberList(thermalMC.supercell,
                                                         fallback: [4, 4, 1]).map { .number($0) }),
                "n_sweeps": .number(Double(thermalMC.nSweeps)),
                "n_equil": .number(Double(thermalMC.nEquil)),
            ])
        }
        if tasks.sampledCorrelations {
            input["sampled_correlations"] = .object([
                "temperature": .number(sampledCorrelations.temperature),
                "supercell": .array(Self.parseNumberList(sampledCorrelations.supercell,
                                                         fallback: [8, 1, 1]).map { .number($0) }),
                "dt": .number(sampledCorrelations.dt),
                "n_steps": .number(Double(sampledCorrelations.nSteps)),
                "n_traj": .number(Double(sampledCorrelations.nTraj)),
            ])
        }
        if tasks.kpmSqw {
            input["kpm"] = (try? JSONValue(encoding: kpm)) ?? .object([:])
        }
        if calculation.mode == "entangled", !calculation.unitsText.isEmpty,
           let data = calculation.unitsText.data(using: .utf8),
           let units = try? JSONDecoder().decode(JSONValue.self, from: data),
           case .array(let arr) = units, !arr.isEmpty {
            input["units"] = units
        }
        return .object(input)
    }

    /// Parses "0.5, 1, 2" / "4,4,1" style strings into a number list.
    static func parseNumberList(_ text: String, fallback: [Double]) -> [Double] {
        let parts = text.split(whereSeparator: { ", \t".contains($0) })
            .compactMap { Double($0) }
        return parts.isEmpty ? fallback : parts
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
