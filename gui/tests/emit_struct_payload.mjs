// Print the payload the Studio POSTs to /get-visualizer-data for a given config.
//
//   node gui/tests/emit_struct_payload.mjs <config.yaml> [out.yaml]
//
// This is exactly what the app sends on every edit to draw the 3D bond preview:
// importConfigDoc() (Open) then buildStructPayload(), whose crystal_structure /
// interactions / magnetic_structure the App.jsx effect wraps with `parameters`.
//
// Its counterpart is `magcalc-emit-config --structure` in the native app. The two
// previews had diverged on the same file -- see that tool's header, and
// tests/test_native_visualizer_parity.py, which drives BOTH through the real
// endpoint and compares the bond networks they draw.

import fs from 'node:fs'
import yaml from 'js-yaml'
import { DEFAULT_CONFIG } from '../src/lib/defaultConfig.js'
import { importConfigDoc, buildStructPayload } from '../src/lib/configIO.js'

const argv = process.argv.slice(2)
// Build the payload from the EDITOR STATE alone, dropping the captured document
// -- what the app holds for a config it authored itself. Both apps have a
// separate branch for it, and nothing without this flag reaches it.
const noRaw = argv.includes('--no-raw')
const [inPath, outPath] = argv.filter(a => a !== '--no-raw')
if (!inPath) {
  console.error('usage: node emit_struct_payload.mjs [--no-raw] <config.yaml> [out.yaml]')
  process.exit(2)
}

const doc = yaml.load(fs.readFileSync(inPath, 'utf8'))
const imported = importConfigDoc(doc, DEFAULT_CONFIG)
if (noRaw) imported.raw = undefined
const sp = buildStructPayload(imported)

// The wrapper App.jsx builds around buildStructPayload for the preview fetch.
const payload = {
  crystal_structure: sp.crystal_structure,
  interactions: sp.interactions,
  magnetic_structure: sp.magnetic_structure,
  parameters: imported.config.parameters,
}

const text = yaml.dump(payload)
if (outPath) fs.writeFileSync(outPath, text)
else process.stdout.write(text)
