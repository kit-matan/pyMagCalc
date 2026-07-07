import SwiftUI

struct FittingView: View {
    @EnvironmentObject var model: AppModel
    @State private var showDataImporter = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Data Fitting").font(.title2.bold())
                    Spacer()
                    Button {
                        model.runFit()
                    } label: {
                        Label(model.calcRunning ? "Fitting…" : "Run Fit", systemImage: "scope")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.calcRunning || !model.serverReachable)
                }

                Text("Fit the spin Hamiltonian to inelastic-neutron-scattering data. Best-fit values, the lmfit report, and the data-vs-model comparison plot appear in Run & Analyze.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                dataCard
                parametersCard
                if model.config.fitting.type != "dispersion" { intensityCard }
            }
            .padding()
        }
        .fileImporter(isPresented: $showDataImporter,
                      allowedContentTypes: [.plainText, .commaSeparatedText, .data]) { result in
            if case .success(let url) = result { model.uploadFitData(from: url) }
        }
    }

    private var dataCard: some View {
        SectionCard(title: "Experimental Data") {
            Picker("Data type", selection: $model.config.fitting.type) {
                Text("Dispersion E(Q) — peak positions").tag("dispersion")
                Text("Single-crystal S(Q,ω) — intensities").tag("sqw")
                Text("Powder S(|Q|,ω) — intensities").tag("powder")
            }

            HStack {
                Button {
                    showDataImporter = true
                } label: {
                    Label("Choose Data File…", systemImage: "doc.badge.plus")
                }
                .disabled(!model.serverReachable)
                if !model.config.fitting.dataLabel.isEmpty {
                    Text(model.config.fitting.dataLabel)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.tint)
                }
            }
            Text(columnHint)
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Picker("Method", selection: $model.config.fitting.method) {
                    Text("Levenberg–Marquardt (leastsq)").tag("leastsq")
                    Text("Trust Region (least_squares)").tag("least_squares")
                    Text("Nelder–Mead").tag("nelder")
                    Text("Differential Evolution").tag("differential_evolution")
                }
                if model.config.fitting.type == "dispersion" {
                    Picker("Band assignment", selection: $model.config.fitting.match) {
                        Text("Nearest band").tag("nearest")
                        Text("Use mode column").tag("mode")
                    }
                }
            }
        }
    }

    private var columnHint: String {
        switch model.config.fitting.type {
        case "dispersion": return "Columns: h, k, l, E, sigma [, mode]  (comma-separated, # comments)"
        case "sqw": return "Columns: h, k, l, energy, intensity, error"
        default: return "Columns: |Q|, energy, intensity, error"
        }
    }

    private var parametersCard: some View {
        SectionCard(title: "Parameters to Fit") {
            let names = model.config.fittableParameterNames
            if names.isEmpty {
                Text("No scalar parameters defined.").font(.caption).foregroundStyle(.secondary)
            } else {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                    GridRow {
                        Text("Vary").font(.caption.bold())
                        Text("Name").font(.caption.bold())
                        Text("Start").font(.caption.bold())
                        Text("Min").font(.caption.bold())
                        Text("Max").font(.caption.bold())
                    }
                    ForEach(names, id: \.self) { name in
                        let varied = model.config.fitting.vary.contains(name)
                        GridRow {
                            Toggle("", isOn: varyBinding(name)).labelsHidden()
                            Text(name).font(.body.monospaced().weight(.semibold))
                            Text(model.config.parameters[name]?.displayString ?? "0")
                            TextField("min", text: boundBinding(name, 0))
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                                .disabled(!varied)
                            TextField("max", text: boundBinding(name, 1))
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                                .disabled(!varied)
                        }
                    }
                }
            }
        }
    }

    private func varyBinding(_ name: String) -> Binding<Bool> {
        Binding(
            get: { model.config.fitting.vary.contains(name) },
            set: { on in
                var vary = Set(model.config.fitting.vary)
                if on { vary.insert(name) } else { vary.remove(name) }
                model.config.fitting.vary = Array(vary).sorted()
            }
        )
    }

    private func boundBinding(_ name: String, _ idx: Int) -> Binding<String> {
        Binding(
            get: {
                guard let b = model.config.fitting.bounds[name], idx < b.count,
                      let v = b[idx] else { return "" }
                return String(format: "%g", v)
            },
            set: { newVal in
                var bounds = model.config.fitting.bounds
                var pair = bounds[name] ?? [nil, nil]
                while pair.count < 2 { pair.append(nil) }
                pair[idx] = Double(newVal.replacingOccurrences(of: ",", with: "."))
                bounds[name] = pair
                model.config.fitting.bounds = bounds
            }
        )
    }

    private var intensityCard: some View {
        SectionCard(title: "Intensity Model") {
            intensityRow("Scale", scalar: $model.config.fitting.scale)
            intensityRow("Background", scalar: $model.config.fitting.background)
            intensityRow("Energy broadening (FWHM)", scalar: $model.config.fitting.energyBroadening)
        }
    }

    private func intensityRow(_ label: String, scalar: Binding<FitScalar>) -> some View {
        HStack {
            Text(label)
                .font(.callout.weight(.medium))
                .frame(width: 190, alignment: .leading)
            NumberField(label: "", value: scalar.value)
                .frame(width: 100)
            Toggle("vary", isOn: scalar.vary)
                .font(.caption)
        }
    }
}
