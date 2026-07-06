import SwiftUI

/// Tasks & Plotting tab: what to compute, the q-path, plot ranges, powder
/// averaging and minimization settings.
struct TasksView: View {
    @EnvironmentObject var model: AppModel
    @State private var newPointName = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Tasks & Plotting").font(.title2.bold())

                tasksCard
                qPathCard
                plottingCard
                if model.config.tasks.powderAverage { powderCard }
                if model.config.tasks.minimization { minimizationCard }
            }
            .padding()
        }
    }

    private var tasksCard: some View {
        SectionCard(title: "Calculation Tasks") {
            Toggle("Energy minimization (find ground state)", isOn: $model.config.tasks.minimization)
            Toggle("Spin-wave dispersion E(Q)", isOn: $model.config.tasks.dispersion)
            Toggle("Plot dispersion", isOn: $model.config.tasks.plotDispersion)
                .disabled(!model.config.tasks.dispersion)
            Toggle("S(Q,ω) intensity map", isOn: $model.config.tasks.sqwMap)
            Toggle("Plot S(Q,ω) map", isOn: $model.config.tasks.plotSqwMap)
                .disabled(!model.config.tasks.sqwMap)
            Toggle("Powder average S(|Q|,ω)", isOn: $model.config.tasks.powderAverage)
            Toggle("Plot magnetic structure", isOn: $model.config.tasks.plotStructure)
            Toggle("Export CSV data", isOn: $model.config.tasks.exportCSV)
        }
    }

    private var qPathCard: some View {
        SectionCard(title: "Momentum Path (q-path)",
                    subtitle: "Reciprocal-lattice points in r.l.u.; the path visits them in order") {
            ForEach(model.config.qPath.path, id: \.self) { name in
                HStack {
                    Text(name)
                        .font(.body.monospaced().weight(.semibold))
                        .frame(width: 80, alignment: .leading)
                    VectorEditor(labels: ["h", "k", "l"], values: qPointBinding(name))
                    Button(role: .destructive) {
                        removeQPoint(name)
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                }
            }
            HStack {
                TextField("Point name (e.g. Gamma, M, K)", text: $newPointName)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 220)
                Button {
                    let name = newPointName.trimmingCharacters(in: .whitespaces)
                    guard !name.isEmpty, model.config.qPath.points[name] == nil else { return }
                    model.config.qPath.points[name] = [0, 0, 0]
                    model.config.qPath.path.append(name)
                    newPointName = ""
                } label: {
                    Label("Add Point", systemImage: "plus")
                }
            }
            IntField(label: "Points per segment", value: $model.config.qPath.pointsPerSegment)
                .frame(maxWidth: 180)
        }
    }

    private func qPointBinding(_ name: String) -> Binding<[Double]> {
        Binding(
            get: { model.config.qPath.points[name] ?? [0, 0, 0] },
            set: { model.config.qPath.points[name] = $0 }
        )
    }

    private func removeQPoint(_ name: String) {
        model.config.qPath.points.removeValue(forKey: name)
        model.config.qPath.path.removeAll { $0 == name }
    }

    private var plottingCard: some View {
        SectionCard(title: "Plot Ranges") {
            Grid(horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    NumberField(label: "Energy min (meV)", value: $model.config.plotting.energyMin)
                    NumberField(label: "Energy max (meV)", value: $model.config.plotting.energyMax)
                }
                GridRow {
                    NumberField(label: "Broadening (meV)", value: $model.config.plotting.broadening)
                    NumberField(label: "Energy resolution (meV)", value: $model.config.plotting.energyResolution)
                }
            }
            .frame(maxWidth: 420)
        }
    }

    private var powderCard: some View {
        SectionCard(title: "Powder Average") {
            Grid(horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    NumberField(label: "|Q| min (Å⁻¹)", value: $model.config.powderAverage.qMin)
                    NumberField(label: "|Q| max (Å⁻¹)", value: $model.config.powderAverage.qMax)
                }
                GridRow {
                    IntField(label: "Q points", value: $model.config.powderAverage.qCount)
                    IntField(label: "Orientation samples", value: $model.config.powderAverage.numSamples)
                }
            }
            .frame(maxWidth: 420)
        }
    }

    private var minimizationCard: some View {
        SectionCard(title: "Energy Minimization",
                    subtitle: "Multi-start local optimization of the classical ground state") {
            Grid(horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    IntField(label: "Random starts", value: $model.config.minimization.numStarts)
                    IntField(label: "Parallel workers", value: $model.config.minimization.nWorkers)
                }
                GridRow {
                    IntField(label: "Early stopping", value: $model.config.minimization.earlyStopping)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Method").font(.caption).foregroundStyle(.secondary)
                        Picker("", selection: $model.config.minimization.method) {
                            Text("L-BFGS-B").tag("L-BFGS-B")
                            Text("Nelder-Mead").tag("Nelder-Mead")
                            Text("SLSQP").tag("SLSQP")
                        }
                        .labelsHidden()
                    }
                }
            }
            .frame(maxWidth: 420)
        }
    }
}
