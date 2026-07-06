import SwiftUI

/// Hamiltonian parameters and the applied magnetic field ("Environment" tab).
struct EnvironmentView: View {
    @EnvironmentObject var model: AppModel
    @State private var newParamName = ""

    private var scalarNames: [String] {
        model.config.parameters
            .filter { !$0.value.isVector && $0.key != "H_mag" }
            .keys.sorted()
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Environment & Parameters").font(.title2.bold())

                SectionCard(title: "Hamiltonian Parameters",
                            subtitle: "Values in meV. Symbols referenced by interaction rules (J1, Dx, …) are defined here.") {
                    ForEach(scalarNames, id: \.self) { name in
                        HStack {
                            Text(name)
                                .font(.body.monospaced().weight(.semibold))
                                .frame(width: 90, alignment: .leading)
                            NumberField(label: "", value: parameterBinding(name))
                            Button(role: .destructive) {
                                model.config.parameters.removeValue(forKey: name)
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                    HStack {
                        TextField("New parameter name", text: $newParamName)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 200)
                        Button {
                            let name = newParamName.trimmingCharacters(in: .whitespaces)
                            guard !name.isEmpty, model.config.parameters[name] == nil else { return }
                            model.config.parameters[name] = .number(0)
                            newParamName = ""
                        } label: {
                            Label("Add", systemImage: "plus")
                        }
                    }
                }

                SectionCard(title: "Applied Magnetic Field") {
                    NumberField(label: "Field magnitude H (T)", value: parameterBinding("H_mag"))
                        .frame(maxWidth: 200)
                    Text("Field direction (crystal frame)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    VectorEditor(labels: ["Hx", "Hy", "Hz"], values: fieldDirectionBinding)
                        .frame(maxWidth: 340)
                }

                SectionCard(title: "Calculation Backend") {
                    Picker("Cache mode", selection: $model.config.calculation.cacheMode) {
                        Text("Auto (recommended)").tag("auto")
                        Text("None").tag("none")
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 340)
                    Text("Auto reuses the cached symbolic Hamiltonian across runs (~79× faster startup on cached models).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding()
        }
    }

    private func parameterBinding(_ name: String) -> Binding<Double> {
        Binding(
            get: { model.config.parameters[name]?.doubleValue ?? 0 },
            set: { model.config.parameters[name] = .number($0) }
        )
    }

    private var fieldDirectionBinding: Binding<[Double]> {
        Binding(
            get: {
                model.config.parameters["H_dir"]?.arrayValue?.compactMap { $0.doubleValue } ?? [0, 0, 1]
            },
            set: { newVal in
                model.config.parameters["H_dir"] = .array(newVal.map { .number($0) })
            }
        )
    }
}
