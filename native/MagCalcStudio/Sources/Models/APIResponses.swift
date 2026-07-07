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

struct EquivalentBond: Codable, Hashable {
    var pair: [String]
    var offset: [Int]
}

struct NeighborShell: Codable, Identifiable {
    var distance: Double
    var refPair: [String]
    var offset: [Int]
    var multiplicity: Int
    var shellLabel: String?
    var rank: Int?
    var equivalentBonds: [EquivalentBond]?

    var id: String { "\(refPair.joined(separator: "-"))@\(distance)@\(offset.map(String.init).joined(separator: ","))" }

    enum CodingKeys: String, CodingKey {
        case distance, offset, multiplicity, rank
        case refPair = "ref_pair"
        case shellLabel = "shell_label"
        case equivalentBonds = "equivalent_bonds"
    }
}

/// /analyze-bonds response entry.
struct BondOrbit: Codable, Identifiable {
    struct Representative: Codable {
        var atomI: JSONValue   // label (symmetry mode) or index (explicit)
        var atomJ: JSONValue
        var offset: [Int]

        enum CodingKeys: String, CodingKey {
            case atomI = "atom_i"
            case atomJ = "atom_j"
            case offset
        }

        var atomIText: String { atomI.stringValue ?? "?" }
        var atomJText: String { atomJ.stringValue ?? "?" }
    }

    var distance: Double
    var multiplicity: Int
    var representative: Representative

    var id: String {
        "\(representative.atomIText)-\(representative.atomJText)@\(distance)@\(representative.offset.map(String.init).joined(separator: ","))"
    }
}

/// /bond-constraints response: symmetry-allowed exchange-matrix form.
struct BondConstraints: Codable {
    var symbolicMatrix: [[String]]
    var freeParameters: [String]
    var isCentrosymmetric: Bool?

    enum CodingKeys: String, CodingKey {
        case symbolicMatrix = "symbolic_matrix"
        case freeParameters = "free_parameters"
        case isCentrosymmetric = "is_centrosymmetric"
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
    var ruleValue: JSONValue?
    var label: String?
    var dmVector: [Double]?
    var distance: Double?
    var exchangeMatrix: [[Double]]?

    var id: String { "\(atomI)-\(atomJ)@\(offset.map(String.init).joined(separator: ","))-\(type)-\(value?.displayString ?? "")" }

    /// Visibility-toggle key, matching the web app's
    /// getBondKey(bond.rule_value || bond.value): arrays join with ",",
    /// everything else stringifies.
    var valueKey: String { Self.bondKey(ruleValue ?? value) }

    static func bondKey(_ value: JSONValue?) -> String {
        guard let value else { return "undefined" }
        if let arr = value.arrayValue {
            return arr.map { $0.stringValue ?? $0.displayString }.joined(separator: ",")
        }
        return value.stringValue ?? value.displayString
    }

    enum CodingKeys: String, CodingKey {
        case offset, type, value, label, distance
        case atomI = "atom_i"
        case atomJ = "atom_j"
        case ruleValue = "rule_value"
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
