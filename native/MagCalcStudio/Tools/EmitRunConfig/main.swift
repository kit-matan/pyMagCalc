// Print the config the NATIVE Studio would RUN for a given config file, or the
// payload it would POST to the structure endpoints.
//
//   magcalc-emit-config [--structure] <config.yaml> [out.json]
//
// Without a flag this is exactly what the native app does when you open a file
// and press Run: YAMLConfig.importConfig() (Open) then MagCalcConfig.backendInput()
// (Run), which APIClient posts to /run-calculation and the server writes verbatim
// to .config_gui_run.yaml before handing it to magcalc.runner.
//
// With `--structure` it is what the app POSTs to /get-visualizer-data on every
// edit -- MagCalcConfig.structurePayload(includeInteractions: true) -- i.e. the
// input to the 3D bond preview. That path is emitted separately from the run
// config and had drifted from it: it sent `LatticeParameters` straight out of the
// editor, whose `spaceGroup` defaults to 1, and a declared `space_group: 1` (P1,
// the identity alone) stops the backend detecting the real group, so every
// `ref_pair` rule expanded to its own bond and nothing else -- 18 bonds in the
// native preview where the web app drew 72 for the same file.
//
// It exists to make the Swift emitters TESTABLE from outside Xcode. The web app
// has `node gui/tests/emit_run_config.mjs` and `emit_struct_payload.mjs`, whose
// output is the oracle: the two apps are two implementations of one rule ("the
// file is the base, write only real edits over it"), and nothing checked that
// they agreed until this tool and tests/test_native_emitter_parity.py existed.
// Output is JSON with sorted keys so a diff is meaningful.

import Foundation

var args = Array(CommandLine.arguments.dropFirst())
var structureMode = false
if let i = args.firstIndex(of: "--structure") {
    structureMode = true
    args.remove(at: i)
}
// Drop the captured document, so the payload is built from the EDITOR MODEL
// alone -- what the app holds for a config it authored itself, or after any
// path that does not set `rawImport`. Both emitters have a separate branch for
// it, and it is the branch that broke (`lattice_parameters` + the placeholder
// `space_group: 1`); with the raw document present they take the other one, so
// nothing without this flag can reach it.
var noRaw = false
if let i = args.firstIndex(of: "--no-raw") {
    noRaw = true
    args.remove(at: i)
}

guard args.count >= 1 else {
    FileHandle.standardError.write(
        Data("usage: magcalc-emit-config [--structure] [--no-raw] <config.yaml> [out.json]\n".utf8))
    exit(2)
}

do {
    let text = try String(contentsOfFile: args[0], encoding: .utf8)
    var config = try YAMLConfig.importConfig(from: text)
    if noRaw { config.rawImport = nil }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
    let payload = structureMode
        ? config.structurePayload(includeInteractions: true)
        : config.backendInput()
    let data = try encoder.encode(payload)
    if args.count >= 2 {
        try data.write(to: URL(fileURLWithPath: args[1]))
    } else {
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
