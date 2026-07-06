import SwiftUI

struct StructureView: View {
    @EnvironmentObject var model: AppModel
    @Binding var showCIFImporter: Bool
    @State private var sgSearch = ""

    var body: some View {
        HSplitViewCompat {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    header
                    latticeCard
                    spaceGroupCard
                    atomsCard
                }
                .padding()
            }
        } trailing: {
            CrystalVisualizerPanel()
        }
    }

    private var header: some View {
        HStack {
            Text("Crystal Structure").font(.title2.bold())
            Spacer()
            Button {
                showCIFImporter = true
            } label: {
                Label("Import CIF", systemImage: "doc.badge.arrow.up")
            }
            .disabled(!model.serverReachable)
        }
    }

    private var latticeCard: some View {
        SectionCard(title: "Lattice Parameters",
                    subtitle: "Cell lengths in Å, angles in degrees") {
            Grid(horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    NumberField(label: "a (Å)", value: $model.config.lattice.a)
                    NumberField(label: "b (Å)", value: $model.config.lattice.b)
                    NumberField(label: "c (Å)", value: $model.config.lattice.c)
                }
                GridRow {
                    NumberField(label: "α (°)", value: $model.config.lattice.alpha)
                    NumberField(label: "β (°)", value: $model.config.lattice.beta)
                    NumberField(label: "γ (°)", value: $model.config.lattice.gamma)
                }
            }
        }
    }

    private var spaceGroupCard: some View {
        SectionCard(title: "Space Group") {
            HStack {
                Picker("Space group", selection: $model.config.lattice.spaceGroup) {
                    ForEach(filteredGroups) { sg in
                        Text(sg.display).tag(sg.number)
                    }
                }
                .labelsHidden()
            }
            TextField("Search symbol or number…", text: $sgSearch)
                .textFieldStyle(.roundedBorder)
            Text("Current: #\(model.config.lattice.spaceGroup) \(SpaceGroups.symbol(for: model.config.lattice.spaceGroup))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var filteredGroups: [SpaceGroup] {
        let query = sgSearch.trimmingCharacters(in: .whitespaces).lowercased()
        var groups = SpaceGroups.all
        if !query.isEmpty {
            groups = groups.filter {
                $0.symbol.lowercased().contains(query) || String($0.number).contains(query)
            }
        }
        // Keep the currently selected group in the list so the Picker
        // selection stays valid while filtering.
        if !groups.contains(where: { $0.number == model.config.lattice.spaceGroup }),
           let current = SpaceGroups.all.first(where: { $0.number == model.config.lattice.spaceGroup }) {
            groups.insert(current, at: 0)
        }
        return groups
    }

    private var atomsCard: some View {
        SectionCard(title: "Basis Atoms (Wyckoff Positions)",
                    subtitle: "Fractional coordinates of the asymmetric unit; symmetry expansion happens on the backend") {
            ForEach($model.config.wyckoffAtoms) { $atom in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        TextField("Label", text: $atom.label)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 120)
                        NumberField(label: "Spin S", value: $atom.spinS)
                            .frame(maxWidth: 90)
                        Spacer()
                        Button(role: .destructive) {
                            model.config.wyckoffAtoms.removeAll { $0.id == atom.id }
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                    }
                    VectorEditor(labels: ["x", "y", "z"], values: $atom.pos)
                }
                .padding(10)
                .background(.background, in: RoundedRectangle(cornerRadius: 8))
            }

            Button {
                model.config.wyckoffAtoms.append(WyckoffAtom(label: "Cu", pos: [0, 0, 0], spinS: 0.5))
            } label: {
                Label("Add Atom", systemImage: "plus")
            }

            Divider()
            HStack {
                Text("Magnetic elements").font(.caption).foregroundStyle(.secondary)
                TextField("Cu, Fe, …", text: Binding(
                    get: { model.config.magneticElements.joined(separator: ", ") },
                    set: { model.config.magneticElements = $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty } }
                ))
                .textFieldStyle(.roundedBorder)
            }
        }
    }
}

/// Side-by-side layout on regular widths (macOS/iPad), stacked on compact (iPhone).
struct HSplitViewCompat<Leading: View, Trailing: View>: View {
    @ViewBuilder var leading: Leading
    @ViewBuilder var trailing: Trailing

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 0) {
                leading.frame(minWidth: 420, maxWidth: .infinity)
                Divider()
                trailing.frame(minWidth: 360, idealWidth: 460, maxWidth: 560)
            }
            ScrollView {
                VStack(spacing: 0) {
                    leading
                    trailing.frame(minHeight: 380)
                }
            }
        }
    }
}

/// The live 3D crystal + bonds preview shown next to editor tabs.
struct CrystalVisualizerPanel: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("3D Preview", systemImage: "rotate.3d")
                    .font(.headline)
                Spacer()
                Button {
                    model.refreshVisualizer()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .disabled(!model.serverReachable)
            }
            .padding(10)

            if let data = model.visualizerData {
                CrystalSceneView(lattice: model.config.lattice,
                                 atoms: data.atoms,
                                 bonds: data.bonds.filter { !model.hiddenBondKeys.contains($0.valueKey) })
                bondLegend(data: data)
            } else {
                ContentUnavailableView(
                    model.serverReachable ? "No structure yet" : "Backend offline",
                    systemImage: "cube.transparent",
                    description: Text(model.serverReachable
                        ? "Add basis atoms to see the unit cell."
                        : "Connect to a MagCalc backend in Settings to enable the 3D preview.")
                )
            }
        }
    }

    @ViewBuilder
    private func bondLegend(data: VisualizerData) -> some View {
        let keys = Array(Set(data.bonds.map(\.valueKey))).sorted()
        if !keys.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack {
                    ForEach(keys, id: \.self) { key in
                        Button {
                            if model.hiddenBondKeys.contains(key) {
                                model.hiddenBondKeys.remove(key)
                            } else {
                                model.hiddenBondKeys.insert(key)
                            }
                        } label: {
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(BondPalette.color(for: key))
                                    .frame(width: 8, height: 8)
                                Text(key).font(.caption)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.background.secondary, in: Capsule())
                            .opacity(model.hiddenBondKeys.contains(key) ? 0.35 : 1)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(10)
            }
        }
    }
}
