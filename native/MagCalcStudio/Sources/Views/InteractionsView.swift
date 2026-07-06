import SwiftUI

struct InteractionsView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        HSplitViewCompat {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Text("Exchange Interactions").font(.title2.bold())
                        Spacer()
                        Button {
                            model.fetchNeighbors()
                        } label: {
                            Label("Find Neighbors", systemImage: "magnifyingglass")
                        }
                        .disabled(!model.serverReachable || model.neighborsLoading)
                    }

                    rulesCard
                    neighborsCard
                }
                .padding()
            }
            .task {
                if model.neighborShells.isEmpty { model.fetchNeighbors() }
            }
        } trailing: {
            CrystalVisualizerPanel()
        }
    }

    private var rulesCard: some View {
        SectionCard(title: "Symmetry Rules",
                    subtitle: "Each rule is expanded to all symmetry-equivalent bonds by the backend") {
            if model.config.symmetryInteractions.isEmpty {
                Text("No interactions defined. Use “Find Neighbors” below to add bonds by coordination shell.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach($model.config.symmetryInteractions) { $rule in
                InteractionRuleRow(rule: $rule) {
                    model.config.symmetryInteractions.removeAll { $0.id == rule.id }
                }
            }
        }
    }

    private var neighborsCard: some View {
        SectionCard(title: "Neighbor Shells",
                    subtitle: "Symmetry-unique bond distances; tap + to add an interaction on a shell") {
            if model.neighborsLoading {
                ProgressView().frame(maxWidth: .infinity)
            } else if model.neighborShells.isEmpty {
                Text("No neighbor data. Check the backend connection and structure.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                ForEach(model.neighborShells.prefix(20)) { shell in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(shell.shellLabel ?? "") neighbor · d = \(String(format: "%.4f", shell.distance)) Å")
                                .font(.callout.weight(.medium))
                            Text("\(shell.refPair.joined(separator: " – "))  offset \(shell.offset.map(String.init).joined(separator: ","))  ×\(shell.multiplicity)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Menu {
                            ForEach(InteractionType.allCases) { type in
                                Button(type.displayName) {
                                    model.addInteraction(from: shell, type: type)
                                }
                            }
                        } label: {
                            Image(systemName: "plus.circle.fill")
                        }
                        .menuStyle(.borderlessButton)
                        .frame(width: 44)
                    }
                    .padding(.vertical, 3)
                    Divider()
                }
            }
        }
    }
}

struct InteractionRuleRow: View {
    @Binding var rule: SymmetryInteraction
    var onDelete: () -> Void
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Picker("", selection: $rule.type) {
                    ForEach(InteractionType.allCases) { t in
                        Text(t.displayName).tag(t)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 220)
                .onChange(of: rule.type) { _, newType in
                    rule.value = newType.defaultValue
                    model.registerParameters(of: rule.value)
                }
                Spacer()
                Text("d = \(String(format: "%.4f", rule.distance)) Å")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button(role: .destructive, action: onDelete) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
            }

            HStack {
                Text("\(rule.refPair.first ?? "?") – \(rule.refPair.last ?? "?")")
                    .font(.caption.monospaced())
                Text("offset \(rule.offset.map(String.init).joined(separator: ","))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            valueEditor
        }
        .padding(10)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var valueEditor: some View {
        switch rule.type {
        case .heisenberg, .kitaev:
            TextField("Symbol or value (e.g. J1)", text: scalarBinding)
                .textFieldStyle(.roundedBorder)
                .font(.body.monospaced())
        case .dm, .anisotropicExchange:
            HStack(spacing: 6) {
                ForEach(0..<3, id: \.self) { i in
                    TextField(rule.type == .dm ? "D\(["x","y","z"][i])" : "G\(["x","y","z"][i])",
                              text: vectorBinding(i))
                        .textFieldStyle(.roundedBorder)
                        .font(.body.monospaced())
                }
            }
        case .interactionMatrix:
            VStack(spacing: 4) {
                ForEach(0..<3, id: \.self) { r in
                    HStack(spacing: 4) {
                        ForEach(0..<3, id: \.self) { c in
                            TextField("0", text: matrixBinding(r, c))
                                .textFieldStyle(.roundedBorder)
                                .font(.caption.monospaced())
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

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
