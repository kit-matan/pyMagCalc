import Foundation

// MARK: - Server response models

struct ParsedCIF: Codable {
    var lattice: LatticeParameters
    var international: String
    var wyckoffAtoms: [WyckoffAtom]

    enum CodingKeys: String, CodingKey {
        case lattice, international
        case wyckoffAtoms = "wyckoff_atoms"
    }
}

struct NeighborShell: Codable, Identifiable {
    var distance: Double
    var refPair: [String]
    var offset: [Int]
    var multiplicity: Int
    var shellLabel: String?
    var rank: Int?

    var id: String { "\(refPair.joined(separator: "-"))@\(distance)@\(offset.map(String.init).joined(separator: ","))" }

    enum CodingKeys: String, CodingKey {
        case distance, offset, multiplicity, rank
        case refPair = "ref_pair"
        case shellLabel = "shell_label"
    }
}

struct VisualizerAtom: Codable, Identifiable {
    var label: String
    var pos: [Double]
    var spinS: Double
    var idx: Int

    var id: Int { idx }

    enum CodingKeys: String, CodingKey {
        case label, pos, idx
        case spinS = "spin_S"
    }
}

struct VisualizerBond: Codable, Identifiable, Hashable {
    var atomI: Int
    var atomJ: Int
    var offset: [Int]
    var type: String
    var value: JSONValue?
    var label: String?
    var dmVector: [Double]?
    var distance: Double?
    var exchangeMatrix: [[Double]]?

    var id: String { "\(atomI)-\(atomJ)@\(offset.map(String.init).joined(separator: ","))-\(type)-\(value?.displayString ?? "")" }

    /// Grouping key used for per-rule visibility toggles (same idea as the
    /// web app's getBondKey).
    var valueKey: String { value?.displayString ?? label ?? type }

    enum CodingKeys: String, CodingKey {
        case offset, type, value, label, distance
        case atomI = "atom_i"
        case atomJ = "atom_j"
        case dmVector = "dm_vector"
        case exchangeMatrix = "exchange_matrix"
    }
}

struct VisualizerData: Codable {
    var atoms: [VisualizerAtom]
    var bonds: [VisualizerBond]
}

struct CalculationResults: Codable {
    var message: String
    var plots: [String]
    var fitParams: [String: JSONValue]?

    enum CodingKeys: String, CodingKey {
        case message, plots
        case fitParams = "fit_params"
    }
}

struct FitDataUploadResult: Codable {
    var dataFile: String
    var originalName: String?

    enum CodingKeys: String, CodingKey {
        case dataFile = "data_file"
        case originalName = "original_name"
    }
}

/// mag_structure.json written by the runner: minimized spin configuration.
struct MagStructureResult: Codable {
    var atoms: [[Double]]
    var vectors: [[Double]]
    var energy: Double?
}
