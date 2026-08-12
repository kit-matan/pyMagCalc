// Opening a config file in the Studio and pressing Run must be the same run as
// `magcalc run <that file>`.
//
// This drives the browser-side half of that promise with no browser: every YAML
// config shipped in examples/ goes through importConfigDoc() (what "Open"/"Load
// YAML" does) and then buildRunConfig() (what Run/Save/Export emit, and what the
// server writes verbatim to .config_gui_run.yaml), and the result is compared
// with the file it came from.
//
// It exists because the editor used to REBUILD the config from a whitelist of
// keys it knew about, so a config could run perfectly from the CLI and, opened
// in the app, run a quietly different model: `tasks: {fit: true}` never fitting,
// `from_mcif` losing the entire crystal, a `magnetic_structure` without an
// explicit `enabled: true` being deleted, `output`/`fitting` replaced by the
// UI's placeholders. Nothing caught any of it, because nothing tested this path.
//
// Run: node gui/tests/roundtrip.test.mjs   (also driven by tests/test_gui_roundtrip.py)

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import yaml from 'js-yaml'

import { DEFAULT_CONFIG } from '../src/lib/defaultConfig.js'
import { importConfigDoc, buildRunConfig } from '../src/lib/configIO.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PROJECT = path.resolve(HERE, '../..')

/** Every config under examples/, minus outputs that are not configs. */
function exampleConfigs(dir = path.join(PROJECT, 'examples'), out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      if (entry.name !== 'archive') exampleConfigs(p, out)
    } else if (/\.ya?ml$/.test(entry.name)) {
      out.push(p)
    }
  }
  return out
}

const failures = []
const fail = (file, msg) => failures.push(`${path.relative(PROJECT, file)}: ${msg}`)

/** Deep equality that ignores key order (YAML mappings are unordered). */
const canon = (v) => {
  if (Array.isArray(v)) return v.map(canon)
  if (v && typeof v === 'object') {
    return Object.fromEntries(Object.keys(v).sort().map(k => [k, canon(v[k])]))
  }
  return v
}
const eq = (a, b) => JSON.stringify(canon(a)) === JSON.stringify(canon(b))

/** Round-trip one config file and check nothing physics-bearing changed. */
function check(file) {
  const doc = yaml.load(fs.readFileSync(file, 'utf8'))
  // Outputs (fit_params.yaml) and fragments are not run configs.
  if (!doc || typeof doc !== 'object' || !doc.tasks) return false

  const state = importConfigDoc(doc, DEFAULT_CONFIG)
  const out = buildRunConfig(state)

  // 1. No top-level block may vanish.
  for (const k of Object.keys(doc)) {
    if (!(k in out)) fail(file, `top-level block '${k}' was dropped`)
  }

  // 2. The crystal is the file's crystal. The emitted block keeps exactly one
  // of atoms_uc/wyckoff_atoms (whichever matches atom_mode), so compare the
  // atom list under whichever key the file used.
  if (doc.crystal_structure) {
    const cs = doc.crystal_structure, os_ = out.crystal_structure || {}
    for (const [k, v] of Object.entries(cs)) {
      if (k === 'atoms_uc' || k === 'wyckoff_atoms') {
        const got = os_.atoms_uc || os_.wyckoff_atoms
        if (!eq(got, v)) fail(file, `crystal_structure.${k} changed`)
      } else if (!eq(os_[k], v)) {
        fail(file, `crystal_structure.${k} changed: ${JSON.stringify(v)} -> ${JSON.stringify(os_[k])}`)
      }
    }
  } else if (doc.from_mcif) {
    // The structure comes from the mCIF at run time; the editor must not
    // fabricate one (that silently replaced the experimental magnetic cell).
    if (out.crystal_structure) fail(file, 'invented a crystal_structure over from_mcif')
    if (!eq(out.from_mcif, doc.from_mcif)) fail(file, 'from_mcif changed')
  }

  // 3. Interactions ARE the Hamiltonian: byte-for-byte or the model differs.
  if (doc.interactions !== undefined && !eq(out.interactions, doc.interactions)) {
    fail(file, 'interactions changed')
  }

  // 4. Magnetic structure: physics input, never dropped. `enabled: true` may be
  // added (the runner's own default when the key is absent).
  if (doc.magnetic_structure) {
    const want = { enabled: true, ...doc.magnetic_structure }
    if (!eq(out.magnetic_structure, want)) {
      fail(file, `magnetic_structure changed: ${JSON.stringify(out.magnetic_structure)}`)
    }
  }

  // 5. Every task the file switched on must still be on (under its canonical
  // name or the alias the runner also accepts).
  const TASK_ALIASES = {
    dispersion: ['dispersion', 'calculate_dispersion', 'run_dispersion'],
    calculate_dispersion: ['dispersion', 'calculate_dispersion'],
    run_dispersion: ['dispersion', 'calculate_dispersion'],
    sqw_map: ['sqw_map', 'calculate_sqw_map', 'run_sqw_map'],
    calculate_sqw_map: ['sqw_map', 'calculate_sqw_map'],
    run_sqw_map: ['sqw_map', 'calculate_sqw_map'],
    minimization: ['minimization', 'run_minimization'],
    run_minimization: ['minimization'],
    powder_average: ['powder_average', 'run_powder_average'],
    run_powder_average: ['powder_average'],
  }
  for (const [k, v] of Object.entries(doc.tasks || {})) {
    const names = TASK_ALIASES[k] || [k]
    const got = names.some(n => out.tasks?.[n])
    if (Boolean(v) !== Boolean(got)) {
      fail(file, `tasks.${k}: ${JSON.stringify(v)} -> ${JSON.stringify(names.map(n => out.tasks?.[n]))}`)
    }
  }

  // 6. Parameters: values are the Hamiltonian's numbers. `S` is dropped on
  // purpose (it lives on the atoms).
  for (const [k, v] of Object.entries(doc.parameters || {})) {
    if (k === 'S') continue
    if (!eq(out.parameters?.[k], v)) {
      fail(file, `parameters.${k} changed: ${JSON.stringify(v)} -> ${JSON.stringify(out.parameters?.[k])}`)
    }
  }

  // 7. q_path, and the blocks that used to be replaced by UI defaults.
  for (const [k, v] of Object.entries(doc.q_path || {})) {
    if (!eq(out.q_path?.[k], v)) fail(file, `q_path.${k} changed`)
  }
  for (const block of ['output', 'fitting', 'calculation', 'minimization',
                       'powder_average', 'scga', 'thermal_mc', 'kpm',
                       'energy_cut', 'units', 'corrections']) {
    for (const [k, v] of Object.entries(doc[block] || {})) {
      // thermal_mc/sampled_correlations lists round-trip through a text field.
      const got = out[block]?.[k]
      if (!eq(got, v) && !(Array.isArray(v) && eq(got, v.map(Number)))) {
        fail(file, `${block}.${k} changed: ${JSON.stringify(v)} -> ${JSON.stringify(got)}`)
      }
    }
  }
  return true
}

/** A config BUILT in the app (never opened from a file) still emits in full.
 *
 * The "file is the base" rule strips editor defaults a file did not declare --
 * that must not leak into the designer path, where the editor IS the source and
 * dropping its defaults would emit a config with no plotting/output settings. */
function checkDesignerPath() {
  const fresh = {
    config: JSON.parse(JSON.stringify(DEFAULT_CONFIG)),
    atomMode: 'symmetry',
    interactionMode: 'symmetry',
    raw: null,
  }
  fresh.config.wyckoff_atoms = [{ label: 'Cu', pos: [0, 0, 0], spin_S: 0.5 }]
  const out = buildRunConfig(fresh)
  for (const k of ['crystal_structure', 'interactions', 'parameters', 'tasks',
                   'q_path', 'plotting', 'minimization', 'calculation', 'output']) {
    if (out[k] === undefined) failures.push(`designer path: '${k}' missing`)
  }
  if (!out.plotting?.disp_plot_filename) {
    failures.push("designer path: plotting defaults were stripped")
  }
  if (!out.crystal_structure?.wyckoff_atoms?.length) {
    failures.push("designer path: the atoms did not survive")
  }
}
checkDesignerPath()

let n = 0
for (const f of exampleConfigs()) {
  try {
    if (check(f)) n++
  } catch (err) {
    fail(f, `threw: ${err.message}`)
  }
}

if (failures.length) {
  console.error(`\nFAILED: ${failures.length} problem(s) across ${n} configs\n`)
  for (const f of failures) console.error('  ' + f)
  process.exit(1)
}
console.log(`OK: ${n} example configs survive the app's open -> run round-trip unchanged.`)
