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
                TaskToggleCard(title: "1/S Corrections", subtitle: "Zero-point energy + moment reduction",
                               systemImage: "target", isOn: $model.config.tasks.corrections)
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
            TaskToggleCard(title: "Auto-scale Y-Axis", subtitle: "Dispersion fits energy range (ignores Energy Min/Max)",
                           systemImage: "arrow.up.and.down", isOn: $model.config.plotting.autoScaleDisp)
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
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Method").font(.caption).foregroundStyle(.secondary)
                        Picker("", selection: Binding(
                            get: { model.config.minimization.method },
                            set: { newValue in
                                let previous = model.config.minimization.method
                                model.config.minimization.method = newValue
                                model.config.minimization.applyMethodDefaults(previousMethod: previous)
                            }
                        )) {
                            Text("Monte-Carlo annealing (recommended)").tag("anneal")
                            Text("Steepest descent (local field)").tag("steep")
                            Text("L-BFGS-B (gradient multistart)").tag("L-BFGS-B")
                            Text("TNC (gradient multistart)").tag("TNC")
                            Text("SLSQP (gradient multistart)").tag("SLSQP")
                        }
                        .labelsHidden()
                    }
                    IntField(
                        label: model.config.minimization.isAnnealMethod ? "Runs" : "Num Starts",
                        value: $model.config.minimization.numStarts
                    )
                }
                if model.config.minimization.method == "anneal" {
                    GridRow {
                        IntField(label: "Sweeps", value: $model.config.minimization.nSweeps)
                        Color.clear.frame(height: 0)
                    }
                }
                if !model.config.minimization.isAnnealMethod {
                    GridRow {
                        IntField(label: "N Workers", value: $model.config.minimization.nWorkers)
                        IntField(label: "Early Stopping", value: $model.config.minimization.earlyStopping)
                    }
                }
            }
            .frame(maxWidth: 420)
            if model.config.minimization.isAnnealMethod {
                Text(model.config.minimization.method == "steep"
                     ? "Aligns each spin with its local field (SpinW optmagsteep). Fast, but it only goes downhill — it cannot escape a local minimum."
                     : "Metropolis + cooling (SpinW anneal / Sunny LocalSampler), then a gradient polish. Crosses barriers, so it does not get trapped.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("Random multistart in (θ, φ). Gets trapped on frustrated systems — prefer annealing. Early Stopping should be ≥ 2 × the number of magnetic sites; too low silently returns a LOCAL minimum.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
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
                VStack(alignment: .leading, spacing: 3) {
                    Text("Ground-State Check").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $model.config.calculation.onImaginary) {
                        Text("Fail the run (default)").tag("error")
                        Text("Warn only (metastable structure)").tag("warn")
                        Text("Disable").tag("off")
                    }
                    .labelsHidden()
                    Text("Spin waves are an expansion about a classical energy minimum; about anything else the spectrum is meaningless. The run is checked for imaginary magnon energies and for a lower-energy relaxation, and fails if either fires.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if model.config.calculation.onImaginary == "warn" {
                        Text("Use only when the structure is knowingly metastable — e.g. a commensurate approximation to an incommensurate spiral. Otherwise you are silently computing the wrong physics.")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    } else if model.config.calculation.onImaginary == "off" {
                        Text("Both ground-state guards are disabled. A wrong ground state will now produce a plausible-looking but meaningless spectrum, with no warning.")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("LSWT Engine").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $model.config.calculation.mode) {
                        Text("Dipole (default)").tag("dipole")
                        Text("SU(N) — single-ion / multipolar").tag("SUN")
                    }
                    .labelsHidden()
                    Text("SU(N) captures single-ion (multipolar) excitations — e.g. FeI₂'s bound state — that dipole LSWT cannot represent. Use it for S ≥ 1 with strong single-ion anisotropy.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if model.config.calculation.mode == "SUN" {
                        Text("SU(N)'s ground state differs from the dipole one — enable Run Minimization so it is found in SU(N), not inherited. Powder/domain averaging not yet supported.")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("Temperature (K)").font(.caption).foregroundStyle(.secondary)
                    TextField("0 (T → 0)", value: $model.config.calculation.temperature,
                              format: .number)
                        .textFieldStyle(.roundedBorder)
                    Text("Applies the Bose thermal factor to S(Q,ω)/powder intensities. Blank = T → 0 (bare LSWT).")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("Cross-section").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $model.config.calculation.crossSection) {
                        Text("Unpolarized ⊥ (default)").tag("perp")
                        Text("Trace (full)").tag("trace")
                        Text("Chiral").tag("chiral")
                        Text("Sˣˣ").tag("xx")
                        Text("Sʸʸ").tag("yy")
                        Text("Sᶻᶻ").tag("zz")
                    }
                    .labelsHidden()
                    Text("Neutron cross-section contraction for S(Q,ω) intensities.")
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
