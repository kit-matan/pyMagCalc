import Foundation
import Yams

// MARK: - JSONValue <-> Any bridging (for Yams)

extension JSONValue {
    var anyValue: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let b): return b
        case .number(let n):
            if n.truncatingRemainder(dividingBy: 1) == 0, abs(n) < 1e15 { return Int(n) }
            return n
        case .string(let s): return s
        case .array(let a): return a.map { $0.anyValue }
        case .object(let o): return o.mapValues { $0.anyValue }
        }
    }

    init(any: Any) {
        switch any {
        case is NSNull: self = .null
        case let b as Bool: self = .bool(b)
        case let i as Int: self = .number(Double(i))
        case let d as Double: self = .number(d)
        case let s as String: self = .string(s)
        case let a as [Any]: self = .array(a.map { JSONValue(any: $0) })
        case let o as [String: Any]: self = .object(o.mapValues { JSONValue(any: $0) })
        case let o as [AnyHashable: Any]:
            var dict: [String: JSONValue] = [:]
            for (k, v) in o { dict[String(describing: k)] = JSONValue(any: v) }
            self = .object(dict)
        default: self = .string(String(describing: any))
        }
    }
}

/// YAML config import/export, matching gui/src/App.jsx handleExportYaml /
/// handleImport byte-for-byte in structure (section order, parameter_order,
/// 5-decimal rounding, inline [x, y, z] vectors, task aliases).
enum YAMLConfig {

    // MARK: Export

    static func export(_ config: MagCalcConfig) throws -> String {
        func round5(_ v: Double) -> Any {
            let r = (v * 100000).rounded() / 100000
            if r.truncatingRemainder(dividingBy: 1) == 0, abs(r) < 1e15 { return Int(r) }
            return r
        }
        func cleanValue(_ v: JSONValue) -> Any {
            if let n = v.doubleValue, case .number = v { return round5(n) }
            if let arr = v.arrayValue {
                return arr.map { item -> Any in
                    if case .number(let n) = item { return round5(n) }
                    return item.anyValue
                }
            }
            return v.anyValue
        }

        // 1. Parameters: interaction symbols sorted, then H_dir, H_mag; no S.
        var rawParams = config.parameters
        rawParams.removeValue(forKey: "S")
        let fieldKeys = ["H_mag", "H_dir"]
        let interactionKeys = rawParams.keys.filter { !fieldKeys.contains($0) }.sorted()
        let sortedParamKeys = (interactionKeys + fieldKeys).filter { rawParams[$0] != nil }
        var cleanParams: [(String, Any)] = []
        for key in sortedParamKeys {
            cleanParams.append((key, cleanValue(rawParams[key]!)))
        }

        // 2. Lattice + atoms rounded to 5 decimals.
        let lat = config.lattice
        let cleanLattice: [(String, Any)] = [
            ("a", round5(lat.a)), ("b", round5(lat.b)), ("c", round5(lat.c)),
            ("alpha", round5(lat.alpha)), ("beta", round5(lat.beta)), ("gamma", round5(lat.gamma)),
            ("space_group", lat.spaceGroup),
        ]
        let cleanAtoms: [Any] = config.wyckoffAtoms.map { atom -> Any in
            var dict: [(String, Any)] = [
                ("label", atom.label),
                ("pos", atom.pos.map { round5($0) }),
                ("spin_S", round5(atom.spinS)),
            ]
            if let ion = atom.ion, !ion.isEmpty { dict.append(("ion", ion)) }
            if let element = atom.element, !element.isEmpty { dict.append(("element", element)) }
            return orderedDict(dict)
        }

        // 3. Full document in the web app's key order.
        var doc: [(String, Any)] = []
        doc.append(("parameter_order", sortedParamKeys))
        doc.append(("parameters", orderedDict(cleanParams)))
        doc.append(("crystal_structure", orderedDict([
            ("lattice_parameters", orderedDict(cleanLattice)),
            ("wyckoff_atoms", cleanAtoms),
            ("atom_mode", config.atomMode),
            ("magnetic_elements", config.magneticElements),
            ("dimensionality", 3),
        ])))
        if config.interactionMode == "explicit" {
            doc.append(("interactions", orderedDict([
                ("list", config.explicitInteractions.map { $0.payloadValue.anyValue }),
            ])))
        } else {
            doc.append(("interactions", orderedDict([
                ("symmetry_rules", config.symmetryInteractions.map { $0.payloadValue.anyValue }),
            ])))
        }
        if config.magneticStructure.enabled {
            doc.append(("magnetic_structure", ((try? JSONValue(encoding: config.magneticStructure)) ?? .object([:])).anyValue))
        }
        var tasksAny = ((try? JSONValue(encoding: config.tasks)) ?? .object([:])).objectValue ?? [:]
        tasksAny["calculate_dispersion"] = .bool(config.tasks.dispersion)
        tasksAny["calculate_sqw_map"] = .bool(config.tasks.sqwMap)
        doc.append(("tasks", JSONValue.object(tasksAny).anyValue))
        var qPath: [(String, Any)] = config.qPath.points
            .sorted { config.qPath.path.firstIndex(of: $0.key) ?? .max < config.qPath.path.firstIndex(of: $1.key) ?? .max }
            .map { ($0.key, $0.value.map { round5($0) }) }
        qPath.append(("path", config.qPath.path))
        qPath.append(("points_per_segment", config.qPath.pointsPerSegment))
        doc.append(("q_path", orderedDict(qPath)))
        var plottingAny = ((try? JSONValue(encoding: config.plotting)) ?? .object([:])).objectValue ?? [:]
        plottingAny["energy_limits_disp"] = .array([.number(config.plotting.energyMin), .number(config.plotting.energyMax)])
        plottingAny["broadening_width"] = .number(config.plotting.broadening)
        doc.append(("plotting", JSONValue.object(plottingAny).anyValue))
        var minAny: [(String, Any)] = [("enabled", config.tasks.minimization)]
        minAny.append(("num_starts", config.minimization.numStarts))
        minAny.append(("n_workers", config.minimization.nWorkers))
        minAny.append(("early_stopping", config.minimization.earlyStopping))
        minAny.append(("method", config.minimization.method))
        doc.append(("minimization", orderedDict(minAny)))
        doc.append(("output", ((try? JSONValue(encoding: config.output)) ?? .object([:])).anyValue))

        let yaml = try Yams.serialize(node: yamlNode(doc))
        return collapseVectors(yaml)
    }

    /// Marker wrapper: an ordered key–value list to be emitted as a mapping.
    private static func orderedDict(_ pairs: [(String, Any)]) -> Any { OrderedPairs(pairs: pairs) }

    private struct OrderedPairs {
        var pairs: [(String, Any)]
    }

    /// Converts the export tree into a Yams Node, preserving mapping order.
    private static func yamlNode(_ any: Any) -> Node {
        switch any {
        case let ordered as OrderedPairs:
            return Node.mapping(Node.Mapping(ordered.pairs.map { (Node($0.0), yamlNode($0.1)) }))
        case let pairs as [(String, Any)]:
            return Node.mapping(Node.Mapping(pairs.map { (Node($0.0), yamlNode($0.1)) }))
        case let dict as [String: Any]:
            return Node.mapping(Node.Mapping(dict.sorted { $0.key < $1.key }.map { (Node($0.key), yamlNode($0.value)) }))
        case let arr as [Any]:
            return Node.sequence(Node.Sequence(arr.map { yamlNode($0) }))
        case let arr as [String]:
            return Node.sequence(Node.Sequence(arr.map { Node($0) }))
        case let b as Bool:
            return Node(b ? "true" : "false", Tag(.bool))
        case let i as Int:
            return Node(String(i), Tag(.int))
        case let d as Double:
            return Node(String(d), Tag(.float))
        case let s as String:
            return Node(s)
        case is NSNull:
            return Node("null", Tag(.null))
        default:
            return Node(String(describing: any))
        }
    }

    /// Post-processes YAML so short numeric/string lists render inline as
    /// [x, y, z] — same regex walk as the web app's collapseVectors().
    static func collapseVectors(_ yaml: String) -> String {
        let lines = yaml.components(separatedBy: "\n")
        var out: [String] = []
        var i = 0
        let keyRegex = try! NSRegularExpression(pattern: #"^(\s*)([\w\d_]+):\s*$"#)

        while i < lines.count {
            let line = lines[i]
            let ns = line as NSString
            if let m = keyRegex.firstMatch(in: line, range: NSRange(location: 0, length: ns.length)) {
                let indent = ns.substring(with: m.range(at: 1))
                let key = ns.substring(with: m.range(at: 2))
                var items: [String] = []
                var j = i + 1
                var valid = true
                let itemPrefix = indent + "- "
                while j < lines.count {
                    let next = lines[j]
                    guard next.hasPrefix(itemPrefix) else { break }
                    let val = String(next.dropFirst(itemPrefix.count)).trimmingCharacters(in: .whitespaces)
                    if val.contains(":") && !(val.hasPrefix("'") || val.hasPrefix("\"")) {
                        valid = false
                        break
                    }
                    items.append(val)
                    j += 1
                }
                if valid, items.count >= 2, items.count <= 8 {
                    out.append("\(indent)\(key): [\(items.joined(separator: ", "))]")
                    i = j
                    continue
                }
            }
            out.append(line)
            i += 1
        }
        return out.joined(separator: "\n")
    }

    // MARK: Import

    /// Mirrors handleImport in App.jsx: start from the web app's blank
    /// DEFAULT_CONFIG, then overlay whatever the YAML defines.
    static func importConfig(from text: String) throws -> MagCalcConfig {
        guard let loaded = try Yams.load(yaml: text) else {
            throw NSError(domain: "YAMLConfig", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Empty YAML document"])
        }
        let doc = JSONValue(any: loaded)
        guard let root = doc.objectValue else {
            throw NSError(domain: "YAMLConfig", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "YAML root is not a mapping"])
        }

        var config = MagCalcConfig.blankDefault
        var labelMap: [String: Int] = [:]

        // 1. Crystal structure
        if let cs = root["crystal_structure"]?.objectValue {
            if let lat = cs["lattice_parameters"]?.objectValue {
                config.lattice.a = lat["a"]?.doubleValue ?? config.lattice.a
                config.lattice.b = lat["b"]?.doubleValue ?? config.lattice.b
                config.lattice.c = lat["c"]?.doubleValue ?? config.lattice.c
                config.lattice.alpha = lat["alpha"]?.doubleValue ?? config.lattice.alpha
                config.lattice.beta = lat["beta"]?.doubleValue ?? config.lattice.beta
                config.lattice.gamma = lat["gamma"]?.doubleValue ?? config.lattice.gamma
                if let sg = lat["space_group"]?.doubleValue { config.lattice.spaceGroup = Int(sg) }
            }

            var atomsSource: [JSONValue]?
            if let wy = cs["wyckoff_atoms"]?.arrayValue {
                atomsSource = wy
                config.atomMode = "symmetry"
            } else if let uc = cs["atoms_uc"]?.arrayValue {
                atomsSource = uc
                config.atomMode = "explicit"
            }
            if let atomsSource {
                config.wyckoffAtoms = atomsSource.enumerated().map { idx, a in
                    let o = a.objectValue ?? [:]
                    let label = o["label"]?.stringValue ?? "Atom"
                    labelMap[label] = idx
                    return WyckoffAtom(
                        label: label,
                        pos: o["pos"]?.arrayValue?.compactMap { $0.doubleValue } ?? [0, 0, 0],
                        spinS: o["spin_S"]?.doubleValue ?? 0.5,
                        ion: o["ion"]?.stringValue,
                        element: o["element"]?.stringValue
                    )
                }
            }
            if let mags = cs["magnetic_elements"]?.arrayValue {
                config.magneticElements = mags.compactMap { $0.stringValue }
            } else if let atomsSource {
                var unique: [String] = []
                for a in atomsSource {
                    let raw = a.objectValue?["label"]?.stringValue
                        ?? a.objectValue?["species"]?.stringValue ?? ""
                    let stripped = raw.replacingOccurrences(of: #"[0-9]+$"#, with: "",
                                                            options: .regularExpression)
                    if !stripped.isEmpty, !unique.contains(stripped) { unique.append(stripped) }
                }
                if !unique.isEmpty { config.magneticElements = unique }
            }
        }

        // 2. Interactions
        func explicitFrom(_ list: [JSONValue]) -> [ExplicitInteraction] {
            list.map { item in
                let o = item.objectValue ?? [:]
                var inter = ExplicitInteraction()
                inter.type = o["type"]?.stringValue ?? "heisenberg"
                inter.distance = o["distance"]?.doubleValue ?? 0
                inter.value = o["value"] ?? .string("J1")
                if let pair = o["pair"]?.arrayValue, o["atom_i"] == nil,
                   let i = labelMap[pair.first?.stringValue ?? ""],
                   let j = labelMap[pair.last?.stringValue ?? ""] {
                    inter.atomI = i
                    inter.atomJ = j
                } else {
                    inter.atomI = Int(o["atom_i"]?.doubleValue ?? 0)
                    inter.atomJ = Int(o["atom_j"]?.doubleValue ?? 0)
                }
                let off = o["offset_j"]?.arrayValue ?? o["rij_offset"]?.arrayValue
                inter.offsetJ = off?.compactMap { $0.doubleValue.map { Int($0) } } ?? [0, 0, 0]
                return inter
            }
        }

        if let list = root["interactions"]?.arrayValue {
            config.explicitInteractions = explicitFrom(list)
            config.interactionMode = "explicit"
        } else if let inter = root["interactions"]?.objectValue {
            if let list = inter["list"]?.arrayValue {
                config.explicitInteractions = explicitFrom(list)
                config.interactionMode = "explicit"
            } else if let rules = inter["symmetry_rules"]?.arrayValue {
                config.symmetryInteractions = rules.map { rule in
                    let o = rule.objectValue ?? [:]
                    var si = SymmetryInteraction()
                    si.type = InteractionType(rawValue: o["type"]?.stringValue ?? "heisenberg") ?? .heisenberg
                    si.refPair = o["ref_pair"]?.arrayValue?.compactMap { $0.stringValue }
                    si.distance = o["distance"]?.doubleValue ?? 0
                    si.value = o["value"] ?? .string("J1")
                    si.offset = o["offset"]?.arrayValue?.compactMap { $0.doubleValue.map { Int($0) } }
                    si.bondDirection = o["bond_direction"]?.stringValue
                    return si
                }
                config.interactionMode = "symmetry"
            }
            if let sia = inter["single_ion_anisotropy"]?.arrayValue {
                config.singleIonAnisotropy = sia.map { item in
                    let o = item.objectValue ?? [:]
                    var s = SingleIonAnisotropy()
                    s.atomLabel = o["atom_label"]?.stringValue ?? s.atomLabel
                    s.value = o["value"] ?? s.value
                    s.axis = o["axis"]?.arrayValue?.compactMap { $0.doubleValue } ?? s.axis
                    return s
                }
            }
        }

        // 3. Other sections
        if let params = root["parameters"]?.objectValue {
            for (k, v) in params { config.parameters[k] = v }
        }
        if let t = root["tasks"]?.objectValue {
            func flag(_ keys: String...) -> Bool? {
                for k in keys {
                    if case .bool(let b)? = t[k] { return b }
                }
                return nil
            }
            config.tasks.minimization = flag("minimization", "run_minimization") ?? config.tasks.minimization
            config.tasks.dispersion = flag("dispersion", "run_dispersion", "calculate_dispersion") ?? config.tasks.dispersion
            config.tasks.sqwMap = flag("sqw_map", "run_sqw_map", "calculate_sqw_map") ?? config.tasks.sqwMap
            config.tasks.powderAverage = flag("powder_average", "run_powder_average") ?? config.tasks.powderAverage
            config.tasks.exportCSV = flag("export_csv") ?? config.tasks.exportCSV
            config.tasks.plotDispersion = flag("plot_dispersion") ?? config.tasks.plotDispersion
            config.tasks.plotSqwMap = flag("plot_sqw_map") ?? config.tasks.plotSqwMap
            config.tasks.plotStructure = flag("plot_structure") ?? config.tasks.plotStructure
        }
        if let p = root["plotting"]?.objectValue {
            config.plotting.energyMin = p["energy_min"]?.doubleValue
                ?? p["energy_limits_disp"]?.arrayValue?.first?.doubleValue ?? config.plotting.energyMin
            config.plotting.energyMax = p["energy_max"]?.doubleValue
                ?? p["energy_limits_disp"]?.arrayValue?.last?.doubleValue ?? config.plotting.energyMax
            config.plotting.broadening = p["broadening"]?.doubleValue
                ?? p["broadening_width"]?.doubleValue ?? config.plotting.broadening
            config.plotting.energyResolution = p["energy_resolution"]?.doubleValue ?? config.plotting.energyResolution
            config.plotting.momentumMax = p["momentum_max"]?.doubleValue ?? config.plotting.momentumMax
            if case .bool(let b)? = p["save_plot"] { config.plotting.savePlot = b }
            if case .bool(let b)? = p["show_plot"] { config.plotting.showPlot = b }
            if case .bool(let b)? = p["plot_structure"] { config.plotting.plotStructure = b }
            if let f = p["disp_plot_filename"]?.stringValue { config.plotting.dispPlotFilename = f }
            if let f = p["sqw_plot_filename"]?.stringValue { config.plotting.sqwPlotFilename = f }
        }
        if let m = root["minimization"]?.objectValue {
            if let v = m["num_starts"]?.doubleValue { config.minimization.numStarts = Int(v) }
            if let v = m["n_workers"]?.doubleValue { config.minimization.nWorkers = Int(v) }
            if let v = m["early_stopping"]?.doubleValue { config.minimization.earlyStopping = Int(v) }
            if let v = m["method"]?.stringValue { config.minimization.method = v }
        }
        if let pa = root["powder_average"]?.objectValue {
            if let v = pa["q_min"]?.doubleValue { config.powderAverage.qMin = v }
            if let v = pa["q_max"]?.doubleValue { config.powderAverage.qMax = v }
            if let v = pa["q_count"]?.doubleValue { config.powderAverage.qCount = Int(v) }
            if let v = pa["num_samples"]?.doubleValue { config.powderAverage.numSamples = Int(v) }
        }
        if let calc = root["calculation"]?.objectValue {
            if let v = calc["cache_mode"]?.stringValue { config.calculation.cacheMode = v }
            if let v = calc["backend"]?.stringValue { config.calculation.backend = v }
        }
        if let ms = root["magnetic_structure"]?.objectValue {
            if case .bool(let b)? = ms["enabled"] { config.magneticStructure.enabled = b }
            if let v = ms["type"]?.stringValue { config.magneticStructure.type = v }
            if let v = ms["pattern_type"]?.stringValue { config.magneticStructure.patternType = v }
            if let dirs = ms["directions"]?.arrayValue {
                config.magneticStructure.directions = dirs.compactMap {
                    $0.arrayValue?.compactMap { $0.doubleValue }
                }
            }
        }
        if let qp = root["q_path"]?.objectValue {
            var points: [String: [Double]] = [:]
            for (k, v) in qp where k != "path" && k != "points_per_segment" {
                if let coords = v.arrayValue?.compactMap({ $0.doubleValue }), coords.count == 3 {
                    points[k] = coords
                }
            }
            config.qPath.points = points
            config.qPath.path = qp["path"]?.arrayValue?.compactMap { $0.stringValue } ?? []
            config.qPath.pointsPerSegment = Int(qp["points_per_segment"]?.doubleValue ?? 100)
        }
        if let out = root["output"]?.objectValue {
            if let v = out["disp_csv_filename"]?.stringValue { config.output.dispCSVFilename = v }
            if let v = out["sqw_csv_filename"]?.stringValue { config.output.sqwCSVFilename = v }
            if let v = out["disp_data_filename"]?.stringValue { config.output.dispDataFilename = v }
            if let v = out["sqw_data_filename"]?.stringValue { config.output.sqwDataFilename = v }
            if case .bool(let b)? = out["save_data"] { config.output.saveData = b }
        }
        if let fit = root["fitting"]?.objectValue {
            if let v = fit["type"]?.stringValue { config.fitting.type = v }
            if let v = fit["method"]?.stringValue { config.fitting.method = v }
            if let v = fit["match"]?.stringValue { config.fitting.match = v }
        }

        return config
    }
}

extension MagCalcConfig {
    /// The web app's DEFAULT_CONFIG (used as the base when importing YAML).
    static var blankDefault: MagCalcConfig {
        var c = MagCalcConfig()
        c.lattice = LatticeParameters(a: 5, b: 5, c: 5, alpha: 90, beta: 90, gamma: 90, spaceGroup: 1)
        c.wyckoffAtoms = []
        c.magneticElements = ["Cu"]
        c.parameters = ["H_mag": .number(0), "H_dir": .array([0, 0, 1])]
        c.tasks = Tasks(minimization: true, dispersion: true, plotDispersion: true,
                        sqwMap: false, plotSqwMap: false, exportCSV: false,
                        powderAverage: false, plotStructure: false)
        c.qPath = QPath(points: ["Gamma": [0, 0, 0]], path: ["Gamma"], pointsPerSegment: 100)
        c.plotting.energyMax = 20
        c.plotting.savePlot = true
        c.plotting.showPlot = true
        return c
    }
}
