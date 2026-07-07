import SwiftUI

/// Backend connection settings. On macOS this also manages the embedded
/// Python server; on iOS you point the app at a Mac running the backend.
struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @State private var urlDraft = ""

    var body: some View {
        Form {
            Section("Backend Connection") {
                TextField("Server URL", text: $urlDraft, prompt: Text("http://127.0.0.1:8000"))
                    #if os(iOS)
                    .keyboardType(.URL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    #endif
                    .onSubmit { applyURL() }
                HStack {
                    Button("Apply") { applyURL() }
                    Spacer()
                    ConnectionStatusView()
                }
                #if os(iOS)
                Text("Run the MagCalc backend on your Mac (python gui/server.py with MAGCALC_HOST=0.0.0.0) and enter its address here, e.g. http://your-mac.local:8000.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                #endif
            }

            #if os(macOS)
            EmbeddedBackendSection(backend: model.backend) { port in
                model.serverURLString = "http://127.0.0.1:\(port)"
            }
            #endif
        }
        .formStyle(.grouped)
        .frame(minWidth: 420)
        .onAppear { urlDraft = model.serverURLString }
    }

    #if os(macOS)
    struct EmbeddedBackendSection: View {
        @ObservedObject var backend: LocalBackendController
        var onStart: (Int) -> Void

        var body: some View {
            Section("Embedded Backend (macOS)") {
                LabeledContent("Status") {
                    switch backend.state {
                    case .stopped: Text("Stopped")
                    case .starting: Text("Starting…").foregroundStyle(.orange)
                    case .running: Text("Running").foregroundStyle(.green)
                    case .failed(let msg): Text(msg).foregroundStyle(.red)
                    }
                }
                TextField("pyMagCalc directory", text: $backend.projectRoot)
                TextField("Python interpreter", text: $backend.pythonPath)
                LabeledContent("Port") {
                    TextField("Port", value: $backend.port, format: .number.grouping(.never))
                        .frame(width: 90)
                }
                HStack {
                    Button("Start Backend") {
                        backend.start()
                        onStart(backend.port)
                    }
                    .disabled(backend.state == .running || backend.state == .starting)
                    Button("Stop Backend") { backend.stop() }
                        .disabled(backend.state == .stopped)
                }
                Text("Requires a Python environment with the magcalc package and its dependencies installed (see pyMagCalc/requirements.txt).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
    #endif

    private func applyURL() {
        var text = urlDraft.trimmingCharacters(in: .whitespaces)
        if !text.isEmpty, !text.contains("://") {
            text = "http://" + text
        }
        guard URL(string: text)?.host != nil else {
            model.notify("Invalid server URL", .error)
            return
        }
        model.serverURLString = text
        urlDraft = text
        model.notify("Backend URL updated", .info)
    }
}
