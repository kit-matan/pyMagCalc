// Print the config the Studio would RUN for a given config file.
//
//   node gui/tests/emit_run_config.mjs <config.yaml> [out.yaml]
//
// This is exactly what the app does when you open a file and press Run:
// importConfigDoc() (Open) then buildRunConfig() (Run), which the server writes
// verbatim to .config_gui_run.yaml and hands to magcalc.runner. Having it as a
// CLI lets tests/test_gui_roundtrip.py run the app's config through the real
// engine and compare the spectrum with the file's own.

import fs from 'node:fs'
import yaml from 'js-yaml'
import { DEFAULT_CONFIG } from '../src/lib/defaultConfig.js'
import { importConfigDoc, buildRunConfig } from '../src/lib/configIO.js'

const [, , inPath, outPath] = process.argv
if (!inPath) {
  console.error('usage: node emit_run_config.mjs <config.yaml> [out.yaml]')
  process.exit(2)
}

const doc = yaml.load(fs.readFileSync(inPath, 'utf8'))
const text = yaml.dump(buildRunConfig(importConfigDoc(doc, DEFAULT_CONFIG)))
if (outPath) fs.writeFileSync(outPath, text)
else process.stdout.write(text)
