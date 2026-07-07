import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @State private var showCIFImporter = false
    @State private var showYAMLImporter = false
    @State private var showYAMLExporter = false
    @State private var showProjectImporter = false
    @State private var showProjectExporter = false
    @State private var showSettings = false
    @State private var showResetConfirm = false

    var body: some View {
        NavigationSplitView {
            List(StudioTab.allCases, selection: tabSelection) { tab in
                Label(tab.title, systemImage: tab.systemImage)
                    .tag(tab)
            }
            .navigationTitle("MagCalc Studio")
            #if os(macOS)
            .navigationSplitViewColumnWidth(min: 190, ideal: 210)
            #endif
            .safeAreaInset(edge: .bottom) {
                ConnectionStatusView()
                    .padding(.vertical, 8)
            }
        } detail: {
            detailView
                .toolbar { toolbarContent }
        }
        .overlay(alignment: .top) {
            if let notification = model.notification {
                NotificationBanner(notification: notification)
                    .padding(.top, 8)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.spring(duration: 0.3), value: model.notification)
        .confirmationDialog(
            "Are you sure you want to load the default example (aCVO)?\nCurrent changes will be lost.",
            isPresented: $showResetConfirm, titleVisibility: .visible
        ) {
            Button("Load Defaults", role: .destructive) { model.resetToDemo() }
            Button("Cancel", role: .cancel) {}
        }
        // Each file importer/exporter is isolated on its own hidden view.
        // SwiftUI does not reliably present more than one .fileImporter /
        // .fileExporter attached to the same view -- stacking them makes all
        // but the first silently no-op (which is why "Load YAML…" did nothing).
        .background {
            Color.clear.fileImporter(isPresented: $showCIFImporter,
                                     allowedContentTypes: [.data, .text]) { result in
                if case .success(let url) = result { model.importCIF(from: url) }
            }
        }
        .background {
            Color.clear.fileImporter(isPresented: $showYAMLImporter,
                                     allowedContentTypes: [.yaml, .data, .text]) { result in
                if case .success(let url) = result { model.importYAML(from: url) }
            }
        }
        .background {
            Color.clear.fileExporter(isPresented: $showYAMLExporter,
                                     document: YAMLDocument(text: (try? model.exportYAML()) ?? ""),
                                     contentType: .yaml,
                                     defaultFilename: "config_designer") { result in
                if case .success = result { model.notify("Success! Configuration exported.") }
            }
        }
        .background {
            Color.clear.fileImporter(isPresented: $showProjectImporter,
                                     allowedContentTypes: [.json]) { result in
                if case .success(let url) = result {
                    let scoped = url.startAccessingSecurityScopedResource()
                    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                    do {
                        try model.importProject(from: Data(contentsOf: url))
                    } catch {
                        model.notify("Failed to load project: \(error.localizedDescription)", .error)
                    }
                }
            }
        }
        .background {
            Color.clear.fileExporter(isPresented: $showProjectExporter,
                                     document: ProjectDocument(data: (try? model.exportProjectData()) ?? Data()),
                                     contentType: .json,
                                     defaultFilename: "MagCalcProject") { result in
                if case .success = result { model.notify("Project saved") }
            }
        }
        #if os(iOS)
        .sheet(isPresented: $showSettings) {
            NavigationStack {
                SettingsView()
                    .navigationTitle("Settings")
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showSettings = false }
                        }
                    }
            }
        }
        #endif
    }

    private var tabSelection: Binding<StudioTab?> {
        Binding(get: { model.selectedTab }, set: { model.selectedTab = $0 ?? .structure })
    }

    @ViewBuilder
    private var detailView: some View {
        switch model.selectedTab {
        case .structure: StructureView(showCIFImporter: $showCIFImporter)
        case .interactions: InteractionsView()
        case .environment: EnvironmentView()
        case .tasks: TasksView()
        case .magneticStructure: MagneticStructureView()
        case .fitting: FittingView()
        case .run: RunView()
        }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup {
            #if os(iOS)
            Button {
                showSettings = true
            } label: {
                Label("Settings", systemImage: "gearshape")
            }
            #endif
            Menu {
                Button {
                    showCIFImporter = true
                } label: {
                    Label("Load CIF…", systemImage: "square.and.arrow.up")
                }
                Button {
                    showYAMLImporter = true
                } label: {
                    Label("Load YAML…", systemImage: "chevron.left.forwardslash.chevron.right")
                }
                Button {
                    showYAMLExporter = true
                } label: {
                    Label("Export YAML…", systemImage: "square.and.arrow.down")
                }
                Divider()
                Button("Open Project (JSON)…") { showProjectImporter = true }
                Button("Save Project (JSON)…") { showProjectExporter = true }
                Divider()
                Button("Load Defaults (aCVO)", role: .destructive) { showResetConfirm = true }
            } label: {
                Label("Project", systemImage: "folder")
            }

            if model.calcRunning {
                Button {
                    model.stopCalculation()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .tint(.red)
            } else {
                Button {
                    model.runCalculation()
                } label: {
                    Label("Run", systemImage: "play.fill")
                }
                .disabled(!model.serverReachable)
            }
        }
    }
}

/// FileDocument wrapper for exporting the web-app-format YAML config.
struct YAMLDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.yaml, .plainText] }

    var text: String

    init(text: String) { self.text = text }

    init(configuration: ReadConfiguration) throws {
        text = String(data: configuration.file.regularFileContents ?? Data(), encoding: .utf8) ?? ""
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

/// FileDocument wrapper for saving the project JSON via fileExporter.
struct ProjectDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }

    var data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
