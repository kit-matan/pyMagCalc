import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @State private var showCIFImporter = false
    @State private var showProjectImporter = false
    @State private var showProjectExporter = false
    @State private var showSettings = false

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
        .fileImporter(isPresented: $showCIFImporter,
                      allowedContentTypes: [.data, .text]) { result in
            if case .success(let url) = result { model.importCIF(from: url) }
        }
        .fileImporter(isPresented: $showProjectImporter,
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
        .fileExporter(isPresented: $showProjectExporter,
                      document: ProjectDocument(data: (try? model.exportProjectData()) ?? Data()),
                      contentType: .json,
                      defaultFilename: "MagCalcProject") { result in
            if case .success = result { model.notify("Project saved") }
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
                Button("Open Project…") { showProjectImporter = true }
                Button("Save Project…") { showProjectExporter = true }
                Divider()
                Button("Import CIF…") { showCIFImporter = true }
                Divider()
                Button("Reset to Demo (aCVO)", role: .destructive) { model.resetToDemo() }
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
