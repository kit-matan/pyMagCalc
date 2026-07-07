import SwiftUI

@main
struct MagCalcStudioApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        #if os(macOS)
        WindowGroup {
            ContentView()
                .environmentObject(model)
        }
        .defaultSize(width: 1280, height: 840)
        .commands {
            CommandGroup(after: .newItem) {
                Divider()
                Button("Run Calculation") { model.runCalculation() }
                    .keyboardShortcut("r", modifiers: .command)
                    .disabled(model.calcRunning || !model.serverReachable)
                Button("Stop Calculation") { model.stopCalculation() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.calcRunning)
            }
        }

        Settings {
            SettingsView()
                .environmentObject(model)
        }
        #else
        WindowGroup {
            ContentView()
                .environmentObject(model)
        }
        #endif
    }
}
