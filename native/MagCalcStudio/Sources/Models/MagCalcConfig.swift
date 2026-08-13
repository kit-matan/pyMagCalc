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
    /// Every OTHER per-site key the file carried — `g` (the per-site g-tensor),
    /// `charge`, `wyckoff`, `species`, and whatever the engine gains next. An atom
    /// is physics input, not a fixed five-field record: 32 sites across the shipped
    /// examples declare a `g`, and re-emitting the app's five known fields alone
    /// would silently drop it and change the Zeeman term. The web app keeps them by
    /// spreading the parsed atom (`{...clone(a)}`); this is the Swift equivalent.
    var extras: [String: JSONValue] = [:]

    enum CodingKeys: String, CodingKey {
        case label, pos, ion, element, extras
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
    /// nil ⇒ the file gave no `distance`, which is legal for a `ref_pair` rule
    /// with an explicit `offset`. Defaulting it to 0 and emitting it added a
    /// `distance: 0.0` the CLI never sees.
    var distance: Double? = 3.0
    var value: JSONValue = .string("J1")
    var offset: [Int]?
    /// Kitaev only: bond direction x/y/z.
    var bondDirection: String?
    /// Any other key the rule carried (`name`, and whatever the engine gains).
    /// A rule is engine input, not a fixed six-field record; the web app edits
    /// the parsed object in place and so keeps these for free.
    var extras: [String: JSONValue] = [:]

    enum CodingKeys: String, CodingKey {
        case type, distance, value, offset, extras
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

    /// The WHOLE config document captured when a file is imported. `backendInput`
    /// starts from it and overwrites only the blocks this app models, so features
    /// the designer cannot represent -- lattice_vectors, interaction_matrix,
    /// single-ion anisotropy, DM, spiral/generic orders -- and blocks it has no UI
    /// for at all -- `from_mcif`, `units`, `energy_cut`, `static_correlations`,
    /// `experiment` -- run exactly as `magcalc run <file>` does. It used to hold
    /// only three keys, so everything else was silently rebuilt from the app's
    /// defaults. Not part of CodingKeys, so it is neither encoded nor decoded
    /// (custom init(from:) leaves it nil).
    var rawImport: JSONValue? = nil

    /// `tasks` flags the imported file carried that this app has no checkbox for
    /// (fit, plot_fit, energy_cut, wang_landau, static_correlations, ...). Merged
    /// back into the emitted `tasks` block so opening a config cannot switch off
    /// the task it exists to run.
    var extraTasks: [String: JSONValue] = [:]

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
    /// Top-level blocks this app models and re-emits itself. Everything else the
    /// imported file carried is passed through untouched. Mirrors
    /// EDITOR_OWNED_BLOCKS in gui/src/lib/configIO.js.
    static let editorOwnedBlocks: Set<String> = [
        "crystal_structure", "interactions", "magnetic_structure",
        "parameters", "parameter_order", "tasks", "q_path", "plotting",
        "minimization", "calculation", "output", "powder_average", "fitting",
        "scga", "thermal_mc", "sampled_correlations", "kpm",
    ]

    /// The file's block, updated with the edits the user actually made.
    ///
    /// The app models each settings block as a struct, so encoding one emits ALL
    /// its fields — including defaults the opened file never mentioned. That is
    /// not cosmetic: `minimization` carries method-specific keys, and adding the
    /// anneal-only `n_sweeps` to a `method: TNC` config makes the minimizer throw
    /// ("unexpected keyword argument"), after which the run dies at the
    /// ground-state guard blaming the magnetic structure. So a key is written
    /// only when the file declared it, or when the editor's value has moved off
    /// the blank default (i.e. the user touched it).
    ///
    /// Mirrors `mergeEdits` in gui/src/lib/configIO.js — keep the two in step.
    /// A block the file did NOT declare is `[:]`, not "emit the whole editor
    /// struct". That distinction is the whole point: encoding the struct writes
    /// every field, so opening a config with no `minimization:` block used to add
    /// the app's `method: anneal, n_sweeps: 2000, num_starts: 4, …` to the run, and
    /// one with no `fitting:` block gained a placeholder fit. The web app's
    /// mergeEdits starts from `clone(fileBlock) || {}` and skips untouched
    /// defaults; treating a missing block as "no merge" is what made the two
    /// implementations disagree on 54 of the 58 shipped configs.
    static func mergeEdits(_ editor: JSONValue, over fileBlock: JSONValue?,
                           defaults: JSONValue) -> JSONValue {
        let file = fileBlock?.objectValue ?? [:]
        var out = file
        let e = editor.objectValue ?? [:]
        let d = defaults.objectValue ?? [:]
        for (k, v) in e {
            if file[k] == nil, d[k] == v { continue }   // untouched UI default
            out[k] = v
        }
        return .object(out)
    }

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
            var t = ((try? JSONValue(encoding: tasks)) ?? .object([:])).objectValue ?? [:]
            for (k, v) in extraTasks { t[k] = v }
            // The runner reads `calculate_dispersion`/`calculate_sqw_map` as
            // "recompute" flags (both default TRUE when absent), and the web app
            // emits them explicitly. Same rule, same keys.
            t["calculate_dispersion"] = .bool(tasks.dispersion)
            t["calculate_sqw_map"] = .bool(tasks.sqwMap)
            tasksValue = .object(t)
        }

        let interactionsValue: JSONValue = interactionMode == "explicit"
            ? .object(["list": .array(explicitInteractions.map { $0.payloadValue })])
            : .object(["symmetry_rules": .array(symmetryInteractions.map { $0.payloadValue })])

        let rawObj = rawImport?.objectValue
        // A `from_mcif` config has no crystal_structure of its own: the runner
        // fills it from the mCIF. Inventing the designer's default 5 A cube there
        // silently replaced an experimentally determined magnetic cell.
        let mcifDriven = rawObj?["crystal_structure"] == nil && rawObj?["from_mcif"] != nil

        // Crystal structure: the file's block is the base, but the ATOMS come from
        // the editor -- WyckoffAtom now carries every per-site key the file had
        // (see `extras`), so emitting the editor's list is loss-free AND an edit
        // made in the Structure panel actually reaches the run. Re-emitting the
        // file's atoms verbatim (what this used to do) made the whole panel
        // read-only after opening a config: changing a spin_S did nothing.
        var crystalValue: JSONValue?
        if let cs = rawObj?["crystal_structure"]?.objectValue {
            var out = cs
            let atoms: [JSONValue] = wyckoffAtoms.isEmpty
                ? (cs["atoms_uc"]?.arrayValue ?? cs["wyckoff_atoms"]?.arrayValue ?? [])
                : wyckoffAtoms.map { $0.payloadValue }
            if cs["lattice_vectors"] == nil {
                // Only the keys the file actually declared, so the app's
                // placeholder defaults (space_group: 1) are never injected into a
                // cell that did not ask for them.
                if let fileLat = cs["lattice_parameters"]?.objectValue {
                    let edited = ((try? JSONValue(encoding: lattice)) ?? .object([:])).objectValue ?? [:]
                    var latOut: [String: JSONValue] = [:]
                    for (k, v) in fileLat { latOut[k] = edited[k] ?? v }
                    out["lattice_parameters"] = .object(latOut)
                } else {
                    out["lattice_parameters"] = (try? JSONValue(encoding: lattice)) ?? .object([:])
                }
            }
            // A file that lists `atoms_uc` is already the full cell (explicit);
            // only `wyckoff_atoms` needs symmetry expansion. Emit ONLY the key
            // matching the mode, so the runner does its own (CLI-identical) one.
            let mode = cs["atom_mode"]?.stringValue
                ?? (cs["atoms_uc"] != nil ? "explicit"
                    : (cs["wyckoff_atoms"] != nil ? "symmetry"
                       : (cs["lattice_vectors"] != nil ? "explicit" : "symmetry")))
            out["atom_mode"] = .string(mode)
            out[mode == "explicit" ? "atoms_uc" : "wyckoff_atoms"] = .array(atoms)
            out.removeValue(forKey: mode == "explicit" ? "wyckoff_atoms" : "atoms_uc")
            out["magnetic_elements"] = cs["magnetic_elements"]
                ?? .array(magneticElements.map { .string($0) })
            out["dimensionality"] = cs["dimensionality"] ?? .number(3)
            crystalValue = .object(out)
        } else if !mcifDriven {
            crystalValue = .object([
                "lattice_parameters": (try? JSONValue(encoding: lattice)) ?? .object([:]),
                atomMode == "explicit" ? "atoms_uc" : "wyckoff_atoms":
                    .array(wyckoffAtoms.map { $0.payloadValue }),
                "atom_mode": .string(atomMode),
                "dimensionality": .number(3),
                "magnetic_elements": .array(magneticElements.map { .string($0) }),
            ])
        }

        // Interactions: symmetry rules are modelled by the designer, so the
        // editor's version wins -- merged INTO the file's block so its siblings
        // (single_ion_anisotropy, stevens, biquadratic, dipole_dipole, ...) stay.
        // The designer's `{list: [...]}` wrapper is unwrapped to the bare list
        // config_loader expects.
        var interactionsFinal: JSONValue = rawObj?["interactions"] ?? interactionsValue
        if var obj = interactionsFinal.objectValue {
            // Only for a file that declares its own crystal: an mCIF-driven config
            // has no editor-side structure to have edited the rules against.
            if obj["symmetry_rules"] != nil, rawObj?["crystal_structure"] != nil {
                obj["symmetry_rules"] = .array(symmetryInteractions.map { $0.payloadValue })
                interactionsFinal = .object(obj)
            }
            if let list = interactionsFinal.objectValue?["list"] { interactionsFinal = list }
        }

        // A magnetic structure the file supplied is physics input and always
        // ships; only a designer-built one is gated on the editor's toggle.
        var magStructFinal: JSONValue?
        if let fileMS = rawObj?["magnetic_structure"] {
            var merged = Self.mergeEdits(
                (try? JSONValue(encoding: magneticStructure)) ?? .object([:]),
                over: fileMS,
                defaults: (try? JSONValue(encoding: MagneticStructureSettings())) ?? .object([:])
            ).objectValue ?? [:]
            // Set explicitly rather than left to mergeEdits: turning the structure
            // OFF lands back on the app's default (false), which mergeEdits reads
            // as "untouched" and would not write -- and the runner treats a MISSING
            // `enabled` as true, so the toggle would do nothing.
            merged["enabled"] = .bool(magneticStructure.enabled)
            magStructFinal = .object(merged)
        } else if !mcifDriven, magneticStructure.enabled {
            magStructFinal = (try? JSONValue(encoding: magneticStructure)) ?? .object([:])
        }

        // calculation: strip UI-only fields; only send series_order when the
        // entangled series is actually requested (mirrors the web app).
        var calcDict = ((try? JSONValue(encoding: calculation)) ?? .object([:])).objectValue ?? [:]
        calcDict.removeValue(forKey: "units_text")
        if !(calculation.mode == "entangled" && calculation.seriesOrder > 0) {
            calcDict.removeValue(forKey: "series_order")
        }

        // Start from the imported file: any block this app does not model
        // (from_mcif/mcif, units, energy_cut, corrections, static_correlations,
        // experiment, ...) is carried through verbatim rather than dropped.
        var input: [String: JSONValue] = [:]
        for (k, v) in rawObj ?? [:] where !Self.editorOwnedBlocks.contains(k) {
            input[k] = v
        }
        // Settings blocks: when a file was opened it is the base and only real
        // edits go over it (see mergeEdits); a config built in the app has no
        // file, so its own defaults are the source and are emitted in full.
        let blank = MagCalcConfig.blankDefault
        func settings(_ key: String, _ built: JSONValue, _ defaults: JSONValue) -> JSONValue {
            rawObj == nil ? built : Self.mergeEdits(built, over: rawObj?[key], defaults: defaults)
        }

        input["interactions"] = interactionsFinal
        // `parameters` takes the same rule: the app always carries H_mag/H_dir,
        // so an opened file with no field used to gain a zero-magnitude Zeeman
        // term the CLI never adds (worth 6e-14 on CoRh2O4's bands). The global `S`
        // is dropped in both directions -- spin lives on the atoms.
        var runParams = parameters
        runParams.removeValue(forKey: "S")
        runParams = (settings("parameters", .object(runParams),
                              .object(blank.parameters)).objectValue ?? [:])
        runParams.removeValue(forKey: "S")
        // Interaction symbols sorted, then the field pair -- the order the web app
        // writes and the order `parameter_order` has to agree with.
        let fieldKeys = ["H_mag", "H_dir"]
        let sortedParamKeys = runParams.keys.filter { !fieldKeys.contains($0) }.sorted()
            + fieldKeys.filter { runParams[$0] != nil }
        input["parameters"] = .object(runParams)
        input["parameter_order"] = rawObj?["parameter_order"]
            ?? .array(sortedParamKeys.map { .string($0) })
        input["tasks"] = tasksValue
        input["q_path"] = .object(qPoints)
        input["plotting"] = settings("plotting", plottingDict,
                                     (try? JSONValue(encoding: blank.plotting)) ?? .object([:]))
        input["minimization"] = settings("minimization", minimizationDict,
                                         (try? JSONValue(encoding: blank.minimization)) ?? .object([:]))
        input["powder_average"] = settings("powder_average",
                                           (try? JSONValue(encoding: powderAverage)) ?? .object([:]),
                                           (try? JSONValue(encoding: blank.powderAverage)) ?? .object([:]))
        // `cache_mode` is a pure performance knob (the symbolic matrix is
        // deterministic per model topology), so 'auto' is always worth sending
        // when the file did not pick one.
        var calcOut = settings("calculation", .object(calcDict),
                               (try? JSONValue(encoding: blank.calculation))
                                ?? .object([:])).objectValue ?? [:]
        if calcOut["cache_mode"] == nil { calcOut["cache_mode"] = .string("auto") }
        input["calculation"] = .object(calcOut)
        input["output"] = settings("output", (try? JSONValue(encoding: output)) ?? .object([:]),
                                   (try? JSONValue(encoding: blank.output)) ?? .object([:]))
        if let crystalValue { input["crystal_structure"] = crystalValue }
        if let magStructFinal { input["magnetic_structure"] = magStructFinal }
        // The blocks the app models are written OVER the file's, key by key, so
        // settings it has no widget for (fitting's data_file/vary/bounds, scga's
        // cross_section, thermal_mc's disorder/periodic, ...) are not replaced by
        // the app's placeholders.
        func overFile(_ key: String, _ built: JSONValue) -> JSONValue {
            guard var base = rawObj?[key]?.objectValue, let new = built.objectValue else { return built }
            for (k, v) in new { base[k] = v }
            return .object(base)
        }
        // `fitting` is a full struct encode too, so it takes the same rule --
        // `overFile` would have written the app's empty `data_file`/`vary` over
        // the real ones and fitted the wrong thing.
        input["fitting"] = settings("fitting", (try? JSONValue(encoding: fitting)) ?? .object([:]),
                                    (try? JSONValue(encoding: blank.fitting)) ?? .object([:]))

        // Beyond-LSWT blocks: an enabled task emits the app's values merged over
        // the file's; a disabled task keeps whatever the file had, so saving does
        // not delete a block the user simply did not switch on this session.
        if tasks.scga {
            input["scga"] = overFile("scga", (try? JSONValue(encoding: scga)) ?? .object([:]))
        } else if let v = rawObj?["scga"] { input["scga"] = v }
        if tasks.thermalMC {
            input["thermal_mc"] = overFile("thermal_mc", .object([
                "temperatures": .array(Self.parseNumberList(thermalMC.temperatures,
                                                            fallback: [0.5, 1, 2, 4]).map { .number($0) }),
                "supercell": .array(Self.parseNumberList(thermalMC.supercell,
                                                         fallback: [4, 4, 1]).map { .number($0) }),
                "n_sweeps": .number(Double(thermalMC.nSweeps)),
                "n_equil": .number(Double(thermalMC.nEquil)),
            ]))
        } else if let v = rawObj?["thermal_mc"] { input["thermal_mc"] = v }
        if tasks.sampledCorrelations {
            input["sampled_correlations"] = overFile("sampled_correlations", .object([
                "temperature": .number(sampledCorrelations.temperature),
                "supercell": .array(Self.parseNumberList(sampledCorrelations.supercell,
                                                         fallback: [8, 1, 1]).map { .number($0) }),
                "dt": .number(sampledCorrelations.dt),
                "n_steps": .number(Double(sampledCorrelations.nSteps)),
                "n_traj": .number(Double(sampledCorrelations.nTraj)),
            ]))
        } else if let v = rawObj?["sampled_correlations"] { input["sampled_correlations"] = v }
        if tasks.kpmSqw {
            input["kpm"] = overFile("kpm", (try? JSONValue(encoding: kpm)) ?? .object([:]))
        } else if let v = rawObj?["kpm"] { input["kpm"] = v }
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
        // `extras` first, so the modelled fields win if a file ever carried both.
        var o: [String: JSONValue] = extras
        o["label"] = .string(label)
        o["pos"] = .array(pos.map { .number($0) })
        o["spin_S"] = .number(spinS)
        if let ion { o["ion"] = .string(ion) }
        if let element { o["element"] = .string(element) }
        return .object(o)
    }
}

extension SymmetryInteraction {
    var payloadValue: JSONValue {
        var o: [String: JSONValue] = extras
        o["type"] = .string(type.rawValue)
        o["value"] = value
        if let distance { o["distance"] = .number(distance) }
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
