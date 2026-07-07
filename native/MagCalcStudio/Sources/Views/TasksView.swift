import SwiftUI

/// Tasks & Plotting tab — mirrors the web app's layout: task cards, display
/// parameters, minimization, powder settings, calculation settings, data
/// export, high-symmetry points and the q-path sequence.
struct TasksView: View {
    @EnvironmentObject var model: AppModel
    @State private var newPointName = ""
    @State private var pointToAdd = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Tasks & Plotting").font(.title2.bold())

                tasksCard
                displayCard
                minimizationCard
                if model.config.tasks.powderAverage { powderCard }
                calculationCard
                dataExportCard
                qPointsCard
                qPathCard
            }
            .padding()
        }
    }

    // MARK: Calculation tasks (4 toggle cards, like the web grid)

    private var tasksCard: some View {
        SectionCard(title: "Calculation Tasks") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 8)], spacing: 8) {
                TaskToggleCard(title: "Run Minimization", subtitle: "Calculate results",
                               systemImage: "wand.and.stars", isOn: $model.config.tasks.minimization)
                TaskToggleCard(title: "Dispersion", subtitle: "Calculate & Plot",
                               systemImage: "waveform.path.ecg", isOn: $model.config.tasks.dispersion)
                TaskToggleCard(title: "S(Q,ω) Map", subtitle: "Full spectral map",
                               systemImage: "chart.bar", isOn: $model.config.tasks.sqwMap)
                TaskToggleCard(title: "Powder Average", subtitle: "S(Q,ω) Sphere Sampl.",
                               systemImage: "wind", isOn: $model.config.tasks.powderAverage)
            }
        }
    }

    // MARK: Display parameters

    private var displayCard: some View {
        SectionCard(title: "Display Parameters") {
            Grid(horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    NumberField(label: "Energy Min (meV)", value: $model.config.plotting.energyMin)
                    NumberField(label: "Energy Max (meV)", value: $model.config.plotting.energyMax)
                }
                GridRow {
                    NumberField(label: "Broadening (meV)", value: $model.config.plotting.broadening)
                    NumberField(label: "Energy Res. (meV)", value: $model.config.plotting.energyResolution)
                }
            }
            .frame(maxWidth: 420)
            NumberField(label: "Momentum Max (Å⁻¹)", value: $model.config.plotting.momentumMax)
                .frame(maxWidth: 200)

            Text("Visualization Targets").font(.caption).foregroundStyle(.secondary)
            TaskToggleCard(title: "Show Plot", subtitle: "Energy dispersion / Sq(w)",
                           systemImage: "eye", isOn: $model.config.plotting.showPlot)
            TaskToggleCard(title: "Show Structure", subtitle: "3D Crystal View",
                           systemImage: "cube", isOn: $model.config.plotting.plotStructure)
        }
    }

    // MARK: Minimization

    private var minimizationCard: some View {
        SectionCard(title: "Minimization Parameters") {
            Grid(horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    IntField(label: "Num Starts", value: $model.config.minimization.numStarts)
                    IntField(label: "N Workers", value: $model.config.minimization.nWorkers)
                }
                GridRow {
                    IntField(label: "Early Stopping", value: $model.config.minimization.earlyStopping)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Method").font(.caption).foregroundStyle(.secondary)
                        Picker("", selection: $model.config.minimization.method) {
                            Text("L-BFGS-B").tag("L-BFGS-B")
                            Text("TNC").tag("TNC")
                            Text("SLSQP").tag("SLSQP")
                        }
                        .labelsHidden()
                    }
                }
            }
            .frame(maxWidth: 420)
            Text("≥ 10 to ensure accurate magnetic structure.")
                .font(.caption)
                .foregroundStyle(.orange)
        }
    }

    // MARK: Powder average

    private var powderCard: some View {
        SectionCard(title: "Powder Average Settings") {
            Grid(horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    NumberField(label: "Q Min (Å⁻¹)", value: $model.config.powderAverage.qMin)
                    NumberField(label: "Q Max (Å⁻¹)", value: $model.config.powderAverage.qMax)
                }
                GridRow {
                    IntField(label: "Q Points", value: $model.config.powderAverage.qCount)
                    IntField(label: "Num Samples", value: $model.config.powderAverage.numSamples)
                }
            }
            .frame(maxWidth: 420)
        }
    }

    // MARK: Calculation settings

    private var calculationCard: some View {
        SectionCard(title: "Calculation Settings") {
            HStack(alignment: .top, spacing: 14) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Cache Mode").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $model.config.calculation.cacheMode) {
                        Text("None (No Caching)").tag("none")
                        Text("Auto (Smart Caching)").tag("auto")
                        Text("Read (Force Read Cache)").tag("r")
                        Text("Write (Force Regeneration)").tag("w")
                    }
                    .labelsHidden()
                    Text("'None' is recommended for small systems or when debugging.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("Compute Backend").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $model.config.calculation.backend) {
                        Text("NumPy (default)").tag("numpy")
                        Text("Fortran (fMagCalc)").tag("fortran")
                    }
                    .labelsHidden()
                    Text("'Fortran' uses the fMagCalc backend for S(Q,ω) and powder (much faster); falls back to NumPy if it isn't installed.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: Data export

    private var dataExportCard: some View {
        SectionCard(title: "Data Export") {
            Toggle(isOn: $model.config.output.saveData) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Export Numeric Results (.npz)").font(.callout.weight(.semibold))
                    Text("Save raw eigenvalues and intensities to binary files.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            if model.config.output.saveData {
                Grid(horizontalSpacing: 10, verticalSpacing: 6) {
                    GridRow {
                        labeledText("Dispersion NPZ", text: $model.config.output.dispDataFilename)
                        labeledText("S(Q,w) NPZ", text: $model.config.output.sqwDataFilename)
                    }
                }
                .frame(maxWidth: 460)
            }

            Toggle(isOn: $model.config.plotting.savePlot) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Export Visual Plots (.png)").font(.callout.weight(.semibold))
                    Text("Save dispersion and S(Q,w) maps as image files.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            if model.config.plotting.savePlot {
                Grid(horizontalSpacing: 10, verticalSpacing: 6) {
                    GridRow {
                        labeledText("Dispersion Plot", text: $model.config.plotting.dispPlotFilename)
                        labeledText("S(Q,w) Plot", text: $model.config.plotting.sqwPlotFilename)
                    }
                }
                .frame(maxWidth: 460)
            }

            Toggle(isOn: $model.config.tasks.exportCSV) {
                Text("Export results to CSV").font(.callout.weight(.semibold))
            }
            if model.config.tasks.exportCSV {
                Grid(horizontalSpacing: 10, verticalSpacing: 6) {
                    GridRow {
                        labeledText("Dispersion CSV", text: $model.config.output.dispCSVFilename)
                        labeledText("S(Q,w) CSV", text: $model.config.output.sqwCSVFilename)
                    }
                }
                .frame(maxWidth: 460)
            }
        }
    }

    private func labeledText(_ label: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            TextField(label, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.body.monospaced())
        }
    }

    // MARK: High-symmetry points

    private var qPointsCard: some View {
        SectionCard(title: "High Symmetry Points") {
            ForEach(model.config.qPath.points.keys.sorted(), id: \.self) { name in
                HStack {
                    Text(name)
                        .font(.body.monospaced().weight(.semibold))
                        .frame(width: 80, alignment: .leading)
                    VectorEditor(labels: ["H", "K", "L"], values: qPointBinding(name))
                    Button(role: .destructive) {
                        model.config.qPath.points.removeValue(forKey: name)
                        model.config.qPath.path.removeAll { $0 == name }
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                }
            }
            HStack {
                TextField("Point name (e.g. L)", text: $newPointName)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 200)
                    .onSubmit(addPoint)
                Button(action: addPoint) {
                    Label("Add Point", systemImage: "plus")
                }
            }
        }
    }

    private func addPoint() {
        let name = newPointName.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty, model.config.qPath.points[name] == nil else { return }
        model.config.qPath.points[name] = [0, 0, 0]
        newPointName = ""
    }

    private func qPointBinding(_ name: String) -> Binding<[Double]> {
        Binding(
            get: { model.config.qPath.points[name] ?? [0, 0, 0] },
            set: { model.config.qPath.points[name] = $0 }
        )
    }

    // MARK: Q-path sequence

    private var qPathCard: some View {
        SectionCard(title: "Q-Path Sequence") {
            if model.config.qPath.path.isEmpty {
                Text("Add points below to build your calculation path...")
                    .font(.caption.italic())
                    .foregroundStyle(.secondary)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        ForEach(Array(model.config.qPath.path.enumerated()), id: \.offset) { idx, name in
                            HStack(spacing: 4) {
                                Text("\(idx + 1)")
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(.secondary)
                                Text(name).font(.callout.monospaced().weight(.semibold))
                                Button {
                                    model.config.qPath.path.remove(at: idx)
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.caption)
                                        .foregroundStyle(.tertiary)
                                }
                                .buttonStyle(.plain)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(.background, in: Capsule())
                            if idx < model.config.qPath.path.count - 1 {
                                Image(systemName: "chevron.right")
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
            }

            HStack {
                Picker("", selection: $pointToAdd) {
                    Text("Select Point...").tag("")
                    ForEach(model.config.qPath.points.keys.sorted(), id: \.self) { p in
                        Text(p).tag(p)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 180)
                Button {
                    guard !pointToAdd.isEmpty else { return }
                    model.config.qPath.path.append(pointToAdd)
                    pointToAdd = ""
                } label: {
                    Label("Add to Path", systemImage: "plus")
                }
                Spacer()
                Text("Points per segment:")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                IntField(label: "", value: $model.config.qPath.pointsPerSegment)
                    .frame(width: 80)
            }
        }
    }
}

/// Toggleable "task card" matching the web app's card-style checkboxes.
struct TaskToggleCard: View {
    let title: String
    let subtitle: String
    let systemImage: String
    @Binding var isOn: Bool

    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: systemImage)
                    .frame(width: 30, height: 30)
                    .background(.tint.opacity(isOn ? 0.18 : 0.07), in: RoundedRectangle(cornerRadius: 7))
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.callout.weight(.semibold))
                    Text(subtitle).font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: isOn ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isOn ? Color.accentColor : Color.secondary.opacity(0.4))
            }
            .padding(10)
            .background(.background, in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(isOn ? Color.accentColor.opacity(0.5) : Color.secondary.opacity(0.15))
            )
        }
        .buttonStyle(.plain)
    }
}
