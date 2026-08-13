// Print the config the NATIVE Studio would RUN for a given config file.
//
//   magcalc-emit-config <config.yaml> [out.json]
//
// This is exactly what the native app does when you open a file and press Run:
// YAMLConfig.importConfig() (Open) then MagCalcConfig.backendInput() (Run),
// which APIClient posts to /run-calculation and the server writes verbatim to
// .config_gui_run.yaml before handing it to magcalc.runner.
//
// It exists to make the Swift emitter TESTABLE from outside Xcode. The web app
// has `node gui/tests/emit_run_config.mjs`, whose output is the oracle: the two
// apps are two implementations of one rule ("the file is the base, write only
// real edits over it"), and nothing checked that they agreed until this tool and
// tests/test_native_emitter_parity.py existed. Output is JSON with sorted keys
// so a diff is meaningful.

import Foundation

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write(
        Data("usage: magcalc-emit-config <config.yaml> [out.json]\n".utf8))
    exit(2)
}

do {
    let text = try String(contentsOfFile: args[1], encoding: .utf8)
    let config = try YAMLConfig.importConfig(from: text)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
    let data = try encoder.encode(config.backendInput())
    if args.count >= 3 {
        try data.write(to: URL(fileURLWithPath: args[2]))
    } else {
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
