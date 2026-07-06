import SwiftUI

struct InteractionsView: View {
    @EnvironmentObject var model: AppModel
    @State private var showOrbitSheet = false
    @State private var bondOrbits: [BondOrbit] = []
    @State private var selectedBondIdxs: [String: Int] = [:]   // shell id → equivalent-bond index

    var body: some View {
        HSplitViewCompat {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text(model.config.interactionMode == "symmetry"
                         ? "Bonding Rules" : "Explicit Interactions (Manual)")
                        .font(.title2.bold())

                    modeToggle

                    if model.config.interactionMode == "symmetry" {
                        symmetryRules
                        siaSection
                        symmetryAnalysisSection
                    } else {
                        explicitInteractions
                    }
                }
                .padding()
            }
            .task {
                if model.neighborShells.isEmpty { model.fetchNeighbors() }
            }
            .sheet(isPresented: $showOrbitSheet) {
                OrbitAnalysisSheet(orbits: bondOrbits)
                    .environmentObject(model)
            }
        } trailing: {
            CrystalVisualizerPanel()
        }
    }

    // MARK: Mode toggle + add button

    private var modeToggle: some View {
        HStack {
            Picker("", selection: $model.config.interactionMode) {
                Label("Symmetry Rules", systemImage: "wind").tag("symmetry")
                Label("Explicit Interactions", systemImage: "waveform.path.ecg").tag("explicit")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 360)

            if model.config.interactionMode == "symmetry" {
                Button {
                    model.addBlankRule()
                } label: {
                    Label("Add Rule", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
            } else {
                Button {
                    model.config.explicitInteractions.append(ExplicitInteraction())
                } label: {
                    Label("Add Interaction", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    // MARK: Symmetry rules

    private var symmetryRules: some View {
        VStack(spacing: 10) {
            ForEach($model.config.symmetryInteractions) { $rule in
                InteractionRuleCard(rule: $rule) {
                    model.config.symmetryInteractions.removeAll { $0.id == rule.id }
                }
            }
        }
    }

    // MARK: Single-ion anisotropy

    private var siaSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Single-Ion Anisotropy").font(.headline)
                Spacer()
                Button {
                    var sia = SingleIonAnisotropy()
                    sia.atomLabel = model.config.wyckoffAtoms.first?.label ?? "Cu"
                    model.config.singleIonAnisotropy.append(sia)
                    if model.config.parameters["D"] == nil {
                        model.config.parameters["D"] = .number(0)
                    }
                } label: {
                    Label("Add SIA", systemImage: "plus")
                }
            }
            ForEach($model.config.singleIonAnisotropy) { $sia in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Label("Single-Ion Anisotropy  ·  Atom: \(sia.atomLabel)", systemImage: "bolt")
                            .font(.callout.weight(.medium))
                        Spacer()
                        Button(role: .destructive) {
                            model.config.singleIonAnisotropy.removeAll { $0.id == sia.id }
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                    }
                    HStack(alignment: .top, spacing: 10) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Atom Label").font(.caption).foregroundStyle(.secondary)
                            Picker("", selection: $sia.atomLabel) {
                                ForEach(model.config.wyckoffAtoms.map(\.label), id: \.self) { label in
                                    Text(label).tag(label)
                                }
                            }
                            .labelsHidden()
                        }
                        VStack(alignment: .leading, spacing: 3) {
                            Text("K / D Constant").font(.caption).foregroundStyle(.secondary)
                            TextField("D", text: Binding(
                                get: { sia.value.stringValue ?? "" },
                                set: { sia.value = .string($0); model.registerParameters(of: sia.value) }
                            ))
                            .textFieldStyle(.roundedBorder)
                            .font(.body.monospaced())
                            .frame(width: 100)
                        }
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Anisotropy Axis").font(.caption).foregroundStyle(.secondary)
                            VectorEditor(labels: ["", "", ""], values: $sia.axis)
                                .frame(maxWidth: 220)
                        }
                    }
                }
                .padding(10)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding(.top, 8)
    }

    // MARK: Symmetry analysis + neighbor suggestions

    private var symmetryAnalysisSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Symmetry Analysis").font(.headline)
            HStack {
                Button {
                    analyzeOrbits()
                } label: {
                    Label("Analyze Symmetry Orbits", systemImage: "waveform.path.ecg")
                }
                .disabled(!model.serverReachable)
                Button {
                    model.fetchNeighbors()
                } label: {
                    Label("Find Neighbors", systemImage: "magnifyingglass")
                }
                .disabled(!model.serverReachable || model.neighborsLoading)
            }

            if model.neighborsLoading {
                ProgressView().frame(maxWidth: .infinity)
            } else if model.neighborShells.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "info.circle").font(.title2).foregroundStyle(.tertiary)
                    Text("Click the button above to analyze the crystal structure and find neighbor shell distances.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding()
            } else {
                neighborSuggestions
            }
        }
        .padding(.top, 8)
    }

    private var neighborSuggestions: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 250), spacing: 10)], spacing: 10) {
            ForEach(Array(model.neighborShells.enumerated()), id: \.element.id) { i, shell in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("\(String(format: "%.4f", shell.distance)) Å")
                            .font(.callout.monospaced().weight(.bold))
                            .foregroundStyle(.tint)
                        Spacer()
                        Label("\(shell.multiplicity) Pairs", systemImage: "cube")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Reference Bond").font(.caption2).foregroundStyle(.secondary)
                        if let equivalents = shell.equivalentBonds, equivalents.count > 1 {
                            Picker("", selection: equivalentBinding(shell)) {
                                ForEach(Array(equivalents.enumerated()), id: \.offset) { idx, b in
                                    Text("\(b.pair.joined(separator: " → ")) [\(b.offset.map(String.init).joined(separator: ","))]")
                                        .tag(idx)
                                }
                            }
                            .labelsHidden()
                            .font(.caption)
                        } else {
                            Text(shell.refPair.joined(separator: " → "))
                                .font(.caption.monospaced())
                        }
                    }

                    Button {
                        let bidx = selectedBondIdxs[shell.id] ?? 0
                        let chosen = shell.equivalentBonds?[safe: bidx]
                        model.addNeighborRule(shellIndex: i,
                                              pair: chosen?.pair ?? shell.refPair,
                                              offset: chosen?.offset ?? shell.offset,
                                              distance: shell.distance)
                    } label: {
                        Label("Add J\(i + 1)", systemImage: "plus")
                            .font(.caption)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
                .padding(10)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private func equivalentBinding(_ shell: NeighborShell) -> Binding<Int> {
        Binding(
            get: { selectedBondIdxs[shell.id] ?? 0 },
            set: { selectedBondIdxs[shell.id] = $0 }
        )
    }

    private func analyzeOrbits() {
        guard let api = model.api else { return }
        model.notify("Analyzing bond symmetry...", .info)
        let snapshot = model.config
        Task {
            do {
                let orbits = try await api.analyzeBonds(for: snapshot)
                bondOrbits = orbits
                showOrbitSheet = true
                model.notify("Found \(orbits.count) bond orbits.")
            } catch {
                model.notify("Failed to analyze symmetry. Check server logs.", .error)
            }
        }
    }

    // MARK: Explicit interactions

    private var explicitInteractions: some View {
        VStack(spacing: 10) {
            ForEach($model.config.explicitInteractions) { $inter in
                ExplicitInteractionCard(inter: $inter) {
                    model.config.explicitInteractions.removeAll { $0.id == inter.id }
                }
            }
        }
    }
}

// MARK: - Symmetry rule card

struct InteractionRuleCard: View {
    @Binding var rule: SymmetryInteraction
    var onDelete: () -> Void
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(typeTitle)
                        .font(.callout.weight(.semibold))
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    model.toggleBondVisibility(rule.value)
                } label: {
                    Image(systemName: model.isBondHidden(rule.value) ? "eye.slash" : "eye")
                }
                .buttonStyle(.borderless)
                .help(model.isBondHidden(rule.value) ? "Show bonds" : "Hide bonds")
                Button(role: .destructive, action: onDelete) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
            }

            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Distance (Å)").font(.caption).foregroundStyle(.secondary)
                    NumberField(label: "", value: $rule.distance)
                        .frame(width: 100)
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text(rule.type == .kitaev ? "Coupling (K)" : "Value")
                        .font(.caption).foregroundStyle(.secondary)
                    valueEditor
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("Type").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $rule.type) {
                        Text("Heisenberg").tag(InteractionType.heisenberg)
                        Text("DM Interaction").tag(InteractionType.dm)
                        Text("Anisotropic").tag(InteractionType.anisotropicExchange)
                        Text("Interaction Matrix").tag(InteractionType.interactionMatrix)
                        Text("Kitaev").tag(InteractionType.kitaev)
                    }
                    .labelsHidden()
                    .onChange(of: rule.type) { _, newType in
                        // Match web: matrix gets zero matrix; switching back to
                        // heisenberg resets array values to a scalar.
                        if newType == .interactionMatrix {
                            rule.value = newType.defaultValue
                        } else if newType == .heisenberg, rule.value.isVector {
                            rule.value = .string("J1")
                        } else if (newType == .dm || newType == .anisotropicExchange), !rule.value.isVector {
                            rule.value = newType.defaultValue
                        }
                        model.registerParameters(of: rule.value)
                    }
                }
            }

            if let matrix = ExchangeMatrix.symbolic(type: rule.type.rawValue, value: rule.value) {
                ExchangeTensorGrid(matrix: matrix, title: "Exchange Tensor (Jij)")
            }
        }
        .padding(12)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }

    private var typeTitle: String {
        switch rule.type {
        case .dm: return "Dzyaloshinskii–Moriya"
        default:
            return rule.type.rawValue
                .split(separator: "_")
                .map { $0.prefix(1).uppercased() + $0.dropFirst() }
                .joined(separator: " ")
        }
    }

    private var subtitle: String {
        var s = rule.refPair.map { "Ref: \($0.joined(separator: "-"))" } ?? "Auto-detected"
        if let off = rule.offset, off != [0, 0, 0] {
            s += " [\(off.map(String.init).joined(separator: ","))]"
        }
        return s
    }

    @ViewBuilder
    private var valueEditor: some View {
        switch rule.type {
        case .heisenberg:
            TextField("J1", text: scalarBinding)
                .textFieldStyle(.roundedBorder)
                .font(.body.monospaced())
                .frame(width: 140)
        case .kitaev:
            HStack(spacing: 6) {
                TextField("K1", text: scalarBinding)
                    .textFieldStyle(.roundedBorder)
                    .font(.body.monospaced())
                    .frame(width: 100)
                Picker("", selection: Binding(
                    get: { rule.bondDirection ?? "x" },
                    set: { rule.bondDirection = $0 }
                )) {
                    Text("X").tag("x")
                    Text("Y").tag("y")
                    Text("Z").tag("z")
                }
                .labelsHidden()
                .frame(width: 64)
            }
        case .dm, .anisotropicExchange:
            HStack(spacing: 4) {
                ForEach(0..<3, id: \.self) { i in
                    TextField("0", text: vectorBinding(i))
                        .textFieldStyle(.roundedBorder)
                        .font(.body.monospaced())
                        .frame(width: 64)
                }
            }
        case .interactionMatrix:
            VStack(spacing: 3) {
                ForEach(0..<3, id: \.self) { r in
                    HStack(spacing: 3) {
                        ForEach(0..<3, id: \.self) { c in
                            TextField("0", text: matrixBinding(r, c))
                                .textFieldStyle(.roundedBorder)
                                .font(.caption.monospaced())
                                .frame(width: 64)
                        }
                    }
                }
            }
        }
    }

    private var scalarBinding: Binding<String> {
        Binding(
            get: { rule.value.stringValue ?? "" },
            set: {
                rule.value = .string($0)
                model.registerParameters(of: rule.value)
            }
        )
    }

    private func vectorBinding(_ i: Int) -> Binding<String> {
        Binding(
            get: { rule.value.arrayValue?[safe: i]?.stringValue ?? "0" },
            set: { newVal in
                var arr = rule.value.arrayValue ?? [.string("0"), .string("0"), .string("0")]
                while arr.count < 3 { arr.append(.string("0")) }
                arr[i] = .string(newVal)
                rule.value = .array(arr)
                model.registerParameters(of: rule.value)
            }
        )
    }

    private func matrixBinding(_ r: Int, _ c: Int) -> Binding<String> {
        Binding(
            get: { rule.value.arrayValue?[safe: r]?.arrayValue?[safe: c]?.stringValue ?? "0" },
            set: { newVal in
                let zeroRow = JSONValue.array([.string("0"), .string("0"), .string("0")])
                var rows = rule.value.arrayValue ?? [zeroRow, zeroRow, zeroRow]
                while rows.count < 3 { rows.append(zeroRow) }
                var row = rows[r].arrayValue ?? [.string("0"), .string("0"), .string("0")]
                while row.count < 3 { row.append(.string("0")) }
                row[c] = .string(newVal)
                rows[r] = .array(row)
                rule.value = .array(rows)
                model.registerParameters(of: rule.value)
            }
        )
    }
}

// MARK: - Explicit interaction card

struct ExplicitInteractionCard: View {
    @Binding var inter: ExplicitInteraction
    var onDelete: () -> Void
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(typeTitle).font(.callout.weight(.semibold))
                    Text("Atoms: \(inter.atomI) → \(inter.atomJ)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    model.toggleBondVisibility(inter.value)
                } label: {
                    Image(systemName: model.isBondHidden(inter.value) ? "eye.slash" : "eye")
                }
                .buttonStyle(.borderless)
                Button(role: .destructive, action: onDelete) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
            }

            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Type").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $inter.type) {
                        Text("Heisenberg").tag("heisenberg")
                        Text("DM Manual").tag("dm_manual")
                        Text("Anisotropic").tag("anisotropic_exchange")
                    }
                    .labelsHidden()
                    .onChange(of: inter.type) { _, newType in
                        inter.value = newType == "heisenberg"
                            ? .string("J1")
                            : .array([.string("0"), .string("0"), .string("0")])
                    }
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("Distance").font(.caption).foregroundStyle(.secondary)
                    NumberField(label: "", value: $inter.distance)
                        .frame(width: 90)
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("Value / Vector").font(.caption).foregroundStyle(.secondary)
                    if inter.value.isVector {
                        HStack(spacing: 4) {
                            ForEach(0..<3, id: \.self) { i in
                                TextField(placeholder(i), text: vectorBinding(i))
                                    .textFieldStyle(.roundedBorder)
                                    .font(.body.monospaced())
                                    .frame(width: 60)
                            }
                        }
                    } else {
                        TextField("J1", text: Binding(
                            get: { inter.value.stringValue ?? "" },
                            set: { inter.value = .string($0); model.registerParameters(of: inter.value) }
                        ))
                        .textFieldStyle(.roundedBorder)
                        .font(.body.monospaced())
                        .frame(width: 120)
                    }
                }
            }

            HStack(spacing: 16) {
                HStack(spacing: 4) {
                    Text("OFFSET J:").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                    ForEach(0..<3, id: \.self) { k in
                        IntField(label: "", value: Binding(
                            get: { inter.offsetJ[safe: k] ?? 0 },
                            set: { newVal in
                                while inter.offsetJ.count < 3 { inter.offsetJ.append(0) }
                                inter.offsetJ[k] = newVal
                            }
                        ))
                        .frame(width: 44)
                    }
                }
                HStack(spacing: 4) {
                    Text("INDICES:").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                    IntField(label: "", value: $inter.atomI).frame(width: 52)
                    Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
                    IntField(label: "", value: $inter.atomJ).frame(width: 52)
                }
            }
        }
        .padding(12)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }

    private var typeTitle: String {
        if inter.type == "heisenberg" { return "Heisenberg" }
        if inter.type.contains("anisotropic") { return "Anisotropic" }
        return "DM Interaction"
    }

    private func placeholder(_ i: Int) -> String {
        inter.type.contains("anisotropic") ? ["Jx", "Jy", "Jz"][i] : ["Dx", "Dy", "Dz"][i]
    }

    private func vectorBinding(_ i: Int) -> Binding<String> {
        Binding(
            get: { inter.value.arrayValue?[safe: i]?.stringValue ?? "0" },
            set: { newVal in
                var arr = inter.value.arrayValue ?? [.string("0"), .string("0"), .string("0")]
                while arr.count < 3 { arr.append(.string("0")) }
                arr[i] = .string(newVal)
                inter.value = .array(arr)
                model.registerParameters(of: inter.value)
            }
        )
    }
}

// MARK: - Shared tensor grid + orbit sheet

struct ExchangeTensorGrid: View {
    let matrix: [[String]]
    var title: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let title {
                Text(title)
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
            }
            Grid(horizontalSpacing: 3, verticalSpacing: 3) {
                ForEach(0..<matrix.count, id: \.self) { r in
                    GridRow {
                        ForEach(0..<matrix[r].count, id: \.self) { c in
                            let cell = matrix[r][c]
                            Text(cell)
                                .font(.caption.monospaced())
                                .foregroundStyle(cell == "0" || cell == "0.0" ? .tertiary : .primary)
                                .frame(minWidth: 56)
                                .padding(.vertical, 4)
                                .background(.background, in: RoundedRectangle(cornerRadius: 4))
                        }
                    }
                }
            }
        }
    }
}

/// Bond-orbit browser (web app's "Analyze Symmetry Orbits" modal): pick an
/// orbit, view its symmetry-allowed matrix form, add an interaction_matrix rule.
struct OrbitAnalysisSheet: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    let orbits: [BondOrbit]

    @State private var selectedOrbit: BondOrbit?
    @State private var constraints: BondConstraints?
    @State private var loading = false

    var body: some View {
        NavigationStack {
            Group {
                if let orbit = selectedOrbit {
                    constraintsView(orbit)
                } else {
                    orbitList
                }
            }
            .navigationTitle("Bond Orbits")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
        .frame(minWidth: 480, minHeight: 420)
    }

    private var orbitList: some View {
        List(orbits) { orbit in
            Button {
                fetchConstraints(orbit)
            } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(String(format: "%.4f", orbit.distance)) Å")
                            .font(.callout.monospaced().weight(.bold))
                            .foregroundStyle(.tint)
                        Text("Multiplicity: \(orbit.multiplicity)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("\(orbit.representative.atomIText) → \(orbit.representative.atomJText)")
                            .font(.caption.weight(.semibold))
                        Text("Offset: [\(orbit.representative.offset.map(String.init).joined(separator: ","))]")
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    Image(systemName: "chevron.right")
                        .foregroundStyle(.tertiary)
                }
            }
            .buttonStyle(.plain)
        }
    }

    @ViewBuilder
    private func constraintsView(_ orbit: BondOrbit) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Button {
                    selectedOrbit = nil
                    constraints = nil
                } label: {
                    Label("Back to Orbits", systemImage: "chevron.left")
                }

                HStack {
                    Text("Selected Bond: \(orbit.representative.atomIText) → \(orbit.representative.atomJText)")
                        .font(.headline)
                        .foregroundStyle(.tint)
                    Spacer()
                    Text("[\(orbit.representative.offset.map(String.init).joined(separator: ","))]")
                        .font(.caption.monospaced())
                }

                if loading {
                    ProgressView().frame(maxWidth: .infinity)
                } else if let constraints {
                    ExchangeTensorGrid(matrix: constraints.symbolicMatrix, title: "Allowed Matrix Form")

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Free Parameters")
                            .font(.caption2.weight(.bold))
                            .foregroundStyle(.secondary)
                            .textCase(.uppercase)
                        if constraints.freeParameters.isEmpty {
                            Text("None (Fixed by symmetry)")
                                .font(.caption.italic())
                                .foregroundStyle(.secondary)
                        } else {
                            HStack {
                                ForEach(constraints.freeParameters, id: \.self) { p in
                                    Text(p)
                                        .font(.caption.monospaced().weight(.semibold))
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 3)
                                        .background(.tint.opacity(0.15), in: Capsule())
                                }
                            }
                        }
                    }

                    if constraints.isCentrosymmetric == true {
                        Label("Bond has inversion symmetry (No DM allowed).", systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.blue)
                    }

                    Button {
                        model.addOrbitMatrixRule(orbit: orbit, constraints: constraints)
                        dismiss()
                    } label: {
                        Label("Add Interaction Rule (Matrix)", systemImage: "plus")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
        }
    }

    private func fetchConstraints(_ orbit: BondOrbit) {
        guard let api = model.api else { return }
        selectedOrbit = orbit
        loading = true
        let snapshot = model.config
        Task {
            defer { loading = false }
            do {
                constraints = try await api.bondConstraints(for: snapshot, bond: orbit.representative)
            } catch {
                model.notify("Failed to fetch bond constraints.", .error)
                selectedOrbit = nil
            }
        }
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
