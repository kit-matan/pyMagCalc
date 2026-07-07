import SwiftUI

/// Run & Analyze tab: run controls, streaming log console, result plots.
struct RunView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                if let error = model.calcError {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                }
                resultsSection
                logConsole
            }
            .padding()
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Run Calculation & Analysis").font(.title2.bold())
                Text("Execute the simulation with current settings and inspect results.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if model.calcRunning {
                ProgressView()
                    .controlSize(.small)
                    .padding(.trailing, 6)
                Button(role: .destructive) {
                    model.stopCalculation()
                } label: {
                    Label(model.calcStopping ? "Stopping…" : "Stop", systemImage: "stop.fill")
                }
                .disabled(model.calcStopping)
            } else {
                Button {
                    model.runCalculation()
                } label: {
                    Label("Run Calculation", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.serverReachable)
            }
        }
    }

    @ViewBuilder
    private var resultsSection: some View {
        if let results = model.calcResults {
            Label("Calculation Completed Successfully", systemImage: "checkmark.circle.fill")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.green)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.green.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))

            if let fitParams = results.fitParams, !fitParams.isEmpty {
                SectionCard(title: "Best-Fit Parameters") {
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                        ForEach(fitParams.keys.sorted(), id: \.self) { name in
                            GridRow {
                                Text(name).font(.body.monospaced().weight(.semibold))
                                Text(fitParams[name]?.displayString ?? "—")
                                    .font(.body.monospaced())
                            }
                        }
                    }
                    Button {
                        model.applyFitParameters()
                    } label: {
                        Label("Apply to Model Parameters", systemImage: "arrow.down.circle")
                    }
                }
            }

            // Match the web app: if mag_structure.json is present, show the
            // interactive 3D viewer and skip the static mag_structure.png.
            let hasStructureJSON = results.plots.contains { $0.hasSuffix("mag_structure.json") }
            let imagePlots = results.plots.filter {
                $0.hasSuffix(".png") && !(hasStructureJSON && $0.hasSuffix("mag_structure.png"))
            }
            ForEach(imagePlots, id: \.self) { path in
                SectionCard(title: plotTitle(path)) {
                    ResultPlotView(serverPath: path)
                }
            }

            if hasStructureJSON, let structure = model.magStructure {
                SectionCard(title: "Interactive Magnetic Structure") {
                    SpinStructureSceneView(structure: structure)
                        .frame(height: 420)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    if let e = structure.energy {
                        Text("Ground-state energy: \(String(format: "%.6f", e)) meV")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Button {
                        model.importMinimizedStructure()
                    } label: {
                        Label("Use as Manual Structure", systemImage: "wind")
                    }
                    .help("Save this minimized structure into the Manual Structure tab for reuse (disables minimization)")
                }
            }

            if results.plots.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "info.circle").font(.title2).foregroundStyle(.tertiary)
                    Text("No plots generated. Enable plotting in \"Tasks & Plotting\" tab.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding()
            }
        } else if !model.calcRunning {
            ContentUnavailableView(
                "No results yet",
                systemImage: "chart.xyaxis.line",
                description: Text("Run a calculation to see dispersion, S(Q,ω) and structure plots here.")
            )
            .frame(maxWidth: .infinity)
        }
    }

    private func plotTitle(_ path: String) -> String {
        switch (path as NSString).lastPathComponent {
        case "disp_plot.png": return "Spin-Wave Dispersion"
        case "sqw_plot.png": return "S(Q,ω) Intensity Map"
        case "powder_plot.png": return "Powder-Averaged S(|Q|,ω)"
        case "fit_comparison.png": return "Fit: Data vs. Model"
        case "mag_structure.png": return "Magnetic Structure"
        default: return (path as NSString).lastPathComponent
        }
    }

    private var logConsole: some View {
        SectionCard(title: "Live Log",
                    subtitle: model.logStream.isConnected ? "Streaming from backend" : "Log stream disconnected") {
            LogConsoleView(lines: model.logStream.lines)
                .frame(height: 260)
        }
    }
}

/// Loads a result plot from the backend and offers export/share.
struct ResultPlotView: View {
    @EnvironmentObject var model: AppModel
    let serverPath: String
    @State private var imageData: Data?

    var body: some View {
        Group {
            if let data = imageData, let image = PlatformImage.from(data: data) {
                VStack(alignment: .trailing, spacing: 6) {
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 480)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    ShareLink(item: exportURL(data: data))
                        .font(.caption)
                }
            } else {
                ProgressView().frame(maxWidth: .infinity, minHeight: 120)
            }
        }
        .task(id: serverPath + (model.calcResults?.message ?? "")) {
            imageData = try? await model.api?.fetchFile(serverPath)
        }
    }

    /// Writes the plot to a temp file so ShareLink can share a real PNG.
    private func exportURL(data: Data) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent((serverPath as NSString).lastPathComponent)
        try? data.write(to: url)
        return url
    }
}

enum PlatformImage {
    static func from(data: Data) -> Image? {
        #if os(macOS)
        guard let ns = NSImage(data: data) else { return nil }
        return Image(nsImage: ns)
        #else
        guard let ui = UIImage(data: data) else { return nil }
        return Image(uiImage: ui)
        #endif
    }
}

struct LogConsoleView: View {
    let lines: [String]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(lines.enumerated()), id: \.offset) { idx, line in
                        Text(line.trimmingCharacters(in: .newlines))
                            .font(.caption.monospaced())
                            .foregroundStyle(color(for: line))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(idx)
                    }
                }
                .padding(8)
            }
            .background(Color.black.opacity(0.85), in: RoundedRectangle(cornerRadius: 8))
            .onChange(of: lines.count) { _, count in
                if count > 0 {
                    proxy.scrollTo(count - 1, anchor: .bottom)
                }
            }
        }
    }

    private func color(for line: String) -> Color {
        if line.contains("ERROR") || line.contains("Error") { return .red }
        if line.contains("WARNING") { return .yellow }
        if line.contains("INFO") { return .green.opacity(0.9) }
        return .white.opacity(0.85)
    }
}
