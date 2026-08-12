// The editor's config <-> file bridge, as pure functions.
//
// The apps are EDITORS OF THE CONFIG FILE: whatever `magcalc run <file>` reads,
// opening that file in the app and pressing Run must produce the same run. The
// two directions live here so they can be tested against every shipped example
// config outside the browser (see gui/tests/roundtrip.test.mjs), instead of
// being buried in App.jsx where nothing could exercise them.
//
// The rule both directions obey: THE FILE IS THE BASE. `importConfigDoc` keeps
// the parsed document whole in `raw`; `buildRunConfig` starts from `raw` and
// overwrites only the blocks the editor actually models. Anything the UI has no
// widget for -- `from_mcif`, `units`, `static_correlations`, `experiment`, a
// task flag added after this code was written -- therefore survives a load/run
// untouched. Reconstructing the config from a whitelist of known keys (which is
// what this used to do) silently dropped whole physics blocks: a config ran from
// the CLI and, opened in the app, ran a DIFFERENT model without complaining.

import { DEFAULT_CONFIG } from './defaultConfig.js'

// Blocks the editor models and re-emits itself. Everything else in the file is
// passed through verbatim. Keep this list in sync with buildRunConfig.
export const EDITOR_OWNED_BLOCKS = new Set([
  'crystal_structure', 'interactions', 'magnetic_structure',
  'parameters', 'parameter_order', 'tasks', 'q_path', 'plotting',
  'minimization', 'calculation', 'output', 'powder_average', 'fitting',
  'scga', 'thermal_mc', 'sampled_correlations', 'kpm',
])

// `tasks` keys the UI has checkboxes for. Any OTHER key found in the file's
// `tasks` block (energy_cut, fit, wang_landau, static_correlations, ...) is
// carried through unchanged rather than dropped.
const UI_TASK_ALIASES = {
  minimization: ['minimization', 'run_minimization'],
  dispersion: ['dispersion', 'run_dispersion', 'calculate_dispersion'],
  sqw_map: ['sqw_map', 'run_sqw_map', 'calculate_sqw_map'],
  powder_average: ['powder_average', 'run_powder_average'],
}

const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)))

/**
 * The file's block, updated with the edits the user actually made.
 *
 * The editor hydrates a block as `{...uiDefaults, ...fileBlock}` so its widgets
 * always have something to bind to. Emitting that union back would inject the
 * UI's placeholders into the file -- harmless for `plotting`, but not for
 * physics: a `single_k` structure would come back carrying the pattern panel's
 * `pattern_type: antiferromagnetic` and an empty `directions: []`. So a key is
 * written only when the file declared it, or when the editor's value has moved
 * off the default (i.e. the user touched it).
 */
function mergeEdits(editorBlock, fileBlock, defaultBlock) {
  const out = clone(fileBlock) || {}
  for (const [k, v] of Object.entries(editorBlock || {})) {
    const untouched = JSON.stringify(v) === JSON.stringify(defaultBlock?.[k])
    if (!(fileBlock && k in fileBlock) && untouched) continue
    out[k] = clone(v)
  }
  return out
}

export const parseNumList = (v, fallback) => {
  if (Array.isArray(v)) return v.map(Number)
  if (typeof v !== 'string') return fallback
  const out = v.split(/[\s,]+/).filter(Boolean).map(Number).filter(x => !isNaN(x))
  return out.length ? out : fallback
}

const joinList = (v) => (Array.isArray(v) ? v.join(', ') : v)

const norm = (v) => (typeof v === 'number' ? Number(v.toFixed(5)) : v)

/**
 * Parsed YAML document -> editor state.
 *
 * Returns `{ config, atomMode, interactionMode, raw }`. `raw` is the whole
 * document; it is the base `buildRunConfig` writes over.
 */
export function importConfigDoc(doc, defaultConfig) {
  const newConfig = clone(defaultConfig)
  let atomMode = 'symmetry'
  let interactionMode = 'symmetry'

  // 1. Crystal structure & atoms (first: it builds the label map).
  const labelMap = {}
  if (doc.crystal_structure) {
    const cs = doc.crystal_structure
    if (cs.lattice_parameters) {
      newConfig.lattice = { ...newConfig.lattice, ...cs.lattice_parameters }
    }
    if (cs.lattice_vectors) {
      const lv = cs.lattice_vectors
      newConfig.lattice.lattice_vectors = lv
      // Derive a/b/c/angles from the vectors so the Lattice Constants panel
      // shows the real cell instead of the UI defaults (the 3D view uses the
      // vectors directly).
      if (!cs.lattice_parameters && Array.isArray(lv) && lv.length === 3) {
        const nrm = v => Math.hypot(v[0], v[1], v[2])
        const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        const ang = (u, v) => {
          const d = nrm(u) * nrm(v)
          return d ? Math.acos(Math.max(-1, Math.min(1, dot(u, v) / d))) * 180 / Math.PI : 90
        }
        const r5 = x => Number(x.toFixed(5))
        newConfig.lattice = {
          ...newConfig.lattice,
          a: r5(nrm(lv[0])), b: r5(nrm(lv[1])), c: r5(nrm(lv[2])),
          alpha: r5(ang(lv[1], lv[2])), beta: r5(ang(lv[0], lv[2])), gamma: r5(ang(lv[0], lv[1])),
          lattice_vectors: lv,
        }
      }
    }

    let atomsSource = null
    if (cs.wyckoff_atoms) {
      atomsSource = cs.wyckoff_atoms
      atomMode = 'symmetry'
    } else if (cs.atoms_uc) {
      atomsSource = cs.atoms_uc
      atomMode = 'explicit'
    }
    if (cs.atom_mode) atomMode = cs.atom_mode
    if (atomsSource) atomsSource.forEach((a, i) => { labelMap[a.label || 'Atom'] = i })

    if (atomsSource) {
      // Preserve every per-site key (ion, element, g-tensor, and anything added
      // later) -- an atom is physics input, not a fixed 3-field record.
      newConfig.wyckoff_atoms = atomsSource.map(a => ({
        ...clone(a),
        label: a.label || 'Atom',
        pos: a.pos || [0, 0, 0],
        spin_S: a.spin_S !== undefined ? a.spin_S : 0.5,
      }))
    }
    if (cs.magnetic_elements) {
      newConfig.magnetic_elements = cs.magnetic_elements
    } else if (atomsSource) {
      const uniqueLabels = [...new Set(atomsSource.map(
        a => (a.label || a.species || '').replace(/[0-9]+$/, '')))].filter(x => x)
      if (uniqueLabels.length > 0) newConfig.magnetic_elements = uniqueLabels
    }
  }

  // 2. Interactions (normalize pair -> atom_i/atom_j for the designer table).
  const withIndices = (list) => list.map(inter => {
    if (inter.pair && inter.atom_i === undefined) {
      const idxI = labelMap[inter.pair[0]]
      const idxJ = labelMap[inter.pair[1]]
      if (idxI !== undefined && idxJ !== undefined) return { ...inter, atom_i: idxI, atom_j: idxJ }
    }
    return inter
  })
  if (Array.isArray(doc.interactions)) {
    newConfig.explicit_interactions = withIndices(doc.interactions)
    interactionMode = 'explicit'
  } else if (doc.interactions && doc.interactions.list) {
    newConfig.explicit_interactions = withIndices(doc.interactions.list)
    interactionMode = 'explicit'
  } else if (doc.interactions && doc.interactions.symmetry_rules) {
    newConfig.symmetry_interactions = doc.interactions.symmetry_rules
    interactionMode = 'symmetry'
  }

  // 3. The blocks with UI editors: merge over the defaults, never replace the
  // default set with a hand-picked subset of the file's keys.
  if (doc.parameters) newConfig.parameters = { ...newConfig.parameters, ...doc.parameters }

  if (doc.tasks) {
    // Canonicalise the run_*/calculate_* aliases onto the UI's names, then keep
    // EVERY other key the file carried (fit, plot_fit, energy_cut, wang_landau,
    // static_correlations, sun_sampled_correlations, ...). Dropping them meant a
    // config's headline task simply never ran when launched from the app.
    const t = { ...doc.tasks }
    const merged = { ...newConfig.tasks }
    for (const [canonical, aliases] of Object.entries(UI_TASK_ALIASES)) {
      for (const a of aliases) {
        if (t[a] !== undefined) { merged[canonical] = t[a]; delete t[a] }
      }
    }
    // `calculate_dispersion` / `calculate_sqw_map` are emitted as aliases on the
    // way out, so don't keep stale copies here.
    delete t.calculate_dispersion
    delete t.calculate_sqw_map
    newConfig.tasks = { ...merged, ...t }
  }

  if (doc.plotting) {
    newConfig.plotting = { ...newConfig.plotting, ...doc.plotting }
    // The UI edits energy_min/energy_max and broadening; the file writes the
    // runner's spellings. Without this the panel showed the UI defaults and then
    // emitted them back, so opening a config with `energy_limits_disp: [0, 12]`
    // silently re-plotted it over 0-20 meV.
    const lim = doc.plotting.energy_limits_disp
    if (Array.isArray(lim) && lim.length === 2) {
      newConfig.plotting.energy_min = lim[0]
      newConfig.plotting.energy_max = lim[1]
    }
    if (doc.plotting.broadening_width !== undefined && doc.plotting.broadening === undefined) {
      newConfig.plotting.broadening = doc.plotting.broadening_width
    }
  }
  if (doc.minimization) newConfig.minimization = { ...newConfig.minimization, ...doc.minimization }
  if (doc.powder_average) newConfig.powder_average = { ...newConfig.powder_average, ...doc.powder_average }
  if (doc.calculation) newConfig.calculation = { ...newConfig.calculation, ...doc.calculation }
  // `output` and `fitting` had no import branch at all: the editor's defaults
  // were emitted over the file's, so an opened fitting config ran the UI's
  // placeholder fit (no data file, default vary list) instead of the real one.
  if (doc.output) newConfig.output = { ...newConfig.output, ...doc.output }
  if (doc.fitting) newConfig.fitting = { ...newConfig.fitting, ...doc.fitting }

  // Top-level `units:` (entangled mode) -> the UI field, so the run payload
  // re-emits it.
  if (doc.units) {
    newConfig.calculation = { ...newConfig.calculation, units_text: JSON.stringify(doc.units) }
  }

  if (doc.scga) newConfig.scga = { ...newConfig.scga, ...doc.scga }
  if (doc.thermal_mc) newConfig.thermal_mc = {
    ...newConfig.thermal_mc, ...doc.thermal_mc,
    temperatures: joinList(doc.thermal_mc.temperatures) ?? newConfig.thermal_mc.temperatures,
    supercell: joinList(doc.thermal_mc.supercell) ?? newConfig.thermal_mc.supercell,
  }
  if (doc.sampled_correlations) newConfig.sampled_correlations = {
    ...newConfig.sampled_correlations, ...doc.sampled_correlations,
    supercell: joinList(doc.sampled_correlations.supercell) ?? newConfig.sampled_correlations.supercell,
  }
  if (doc.kpm) newConfig.kpm = { ...newConfig.kpm, ...doc.kpm }

  if (doc.magnetic_structure) {
    // The runner reads `enabled` as TRUE when absent (runner.py: `ms_cfg.get(
    // 'enabled', True)`). The editor's default is false, so merging the file
    // over it used to turn a structure the CLI honours into one the app deletes
    // -- the run then fell back to a minimised/default order.
    newConfig.magnetic_structure = {
      ...newConfig.magnetic_structure,
      enabled: true,
      ...doc.magnetic_structure,
    }
  }

  if (doc.q_path) {
    const { path, points_per_segment, ...points } = doc.q_path
    newConfig.q_path = {
      points: points || {},
      path: path || [],
      points_per_segment: points_per_segment || 100,
    }
  }

  // The file itself is the base for the run config. Keep it whole.
  return { config: newConfig, atomMode, interactionMode, raw: clone(doc) }
}

/**
 * The crystal_structure / interactions / magnetic_structure portion of any
 * backend payload. When a config file was loaded, its blocks pass through
 * verbatim (lattice_vectors, interaction_matrix, single_ion_anisotropy, spiral
 * and generic orders, magnetic_supercell, ... all reach the runner exactly as
 * `magcalc run` sees them); otherwise the designer state is serialised.
 */
export function buildStructPayload({ config, atomMode, interactionMode, raw }) {
  const cs = raw?.crystal_structure
  if (cs) {
    // Atoms come from the EDITOR, not the file: importConfigDoc copies every
    // per-site key through, so emitting the editor's list is loss-free and an
    // edit made in the Structure panel actually reaches the run. (It used to
    // re-emit the file's atoms verbatim, which made the whole structure panel
    // read-only after opening a config -- changing a spin_S did nothing.)
    const atoms = config.wyckoff_atoms?.length
      ? config.wyckoff_atoms
      : (cs.atoms_uc || cs.wyckoff_atoms || [])
    // Lattice likewise, but only the keys the file actually declared, so the
    // editor's placeholder defaults (space_group: 1) are never injected into a
    // cell that did not ask for them.
    const latticeParams = cs.lattice_parameters
      ? Object.fromEntries(Object.keys(cs.lattice_parameters).map(
        k => [k, config.lattice?.[k] ?? cs.lattice_parameters[k]]))
      : null
    // symmetry_rules are modelled by the designer, so the editor's version wins
    // -- merged INTO the file's interactions block so sibling keys
    // (single_ion_anisotropy, stevens, biquadratic, dipole_dipole, ...) stay.
    let interactions = raw.interactions
    if (interactions && !Array.isArray(interactions) && interactions.symmetry_rules) {
      interactions = { ...clone(interactions), symmetry_rules: config.symmetry_interactions }
    }
    return {
      crystal_structure: {
        // Spread the file's block FIRST so keys with no UI (magnetic_supercell,
        // dimensionality, and whatever the engine gains next) survive.
        ...clone(cs),
        ...(cs.lattice_vectors
          ? { lattice_vectors: cs.lattice_vectors }
          : { lattice_parameters: latticeParams || config.lattice }),
        atoms_uc: atoms,
        wyckoff_atoms: atoms,
        // A raw config that lists `atoms_uc` is already the full cell (explicit);
        // only `wyckoff_atoms` needs symmetry expansion.
        atom_mode: cs.atom_mode
          || (cs.atoms_uc ? 'explicit' : (cs.wyckoff_atoms ? 'symmetry'
            : (cs.lattice_vectors ? 'explicit' : 'symmetry'))),
        magnetic_elements: cs.magnetic_elements || config.magnetic_elements || ['Cu'],
        dimensionality: cs.dimensionality ?? 3,
      },
      interactions,
      // `enabled` is set explicitly rather than left to mergeEdits: the user
      // turning the structure OFF lands back on the UI's default (false), which
      // mergeEdits reads as "untouched" and would not write -- and the runner
      // treats a MISSING `enabled` as true, so the toggle would do nothing.
      magnetic_structure: raw.magnetic_structure
        ? { ...mergeEdits(config.magnetic_structure, raw.magnetic_structure,
                          DEFAULT_CONFIG.magnetic_structure),
            enabled: config.magnetic_structure?.enabled !== false }
        : config.magnetic_structure,
      parameter_order: raw.parameter_order,
    }
  }
  if (raw?.from_mcif) {
    // The structure comes from the mCIF at run time -- there is nothing for the
    // editor to emit, and emitting its default cell would silently replace the
    // experimental magnetic structure with a 5 A cube.
    return {
      crystal_structure: null,
      interactions: raw.interactions,
      magnetic_structure: raw.magnetic_structure ?? null,
      parameter_order: raw.parameter_order,
    }
  }
  return {
    crystal_structure: {
      ...(config.lattice.lattice_vectors
        ? { lattice_vectors: config.lattice.lattice_vectors }
        : { lattice_parameters: config.lattice }),
      wyckoff_atoms: config.wyckoff_atoms,
      atom_mode: atomMode,
      magnetic_elements: config.magnetic_elements || ['Cu'],
      dimensionality: 3,
    },
    interactions: interactionMode === 'explicit'
      ? { list: config.explicit_interactions || [] }
      : { symmetry_rules: config.symmetry_interactions, single_ion_anisotropy: config.single_ion_anisotropy || [] },
    magnetic_structure: config.magnetic_structure,
    parameter_order: undefined,
  }
}

// calculation block: strip UI-only fields; only send series_order/series_resum
// when the entangled series is actually requested.
function buildCalcPayload(config) {
  const calc = { ...config.calculation }
  delete calc.units_text
  if (!(calc.mode === 'entangled' && Number(calc.series_order) > 0)) {
    delete calc.series_order
    delete calc.series_resum
  }
  return calc
}

// Config blocks for the beyond-LSWT tasks (SCGA, thermal MC, classical
// dynamics, KPM) plus entangled-mode extras. An enabled task emits the UI's
// values merged OVER the file's block, so settings with no widget yet
// (`cross_section`, `disorder`, `periodic`, `integrator`, ...) are kept; a
// disabled task whose block came from the file keeps that block verbatim.
function buildBeyondLswtBlocks({ config, raw }) {
  const out = {}
  const base = (k) => clone(raw?.[k]) || {}

  if (config.tasks.scga) {
    out.scga = {
      ...base('scga'),
      temperature: Number(config.scga.temperature) || 1.0,
      mesh_density: Number(config.scga.mesh_density) || 12,
    }
  } else if (raw?.scga) out.scga = base('scga')

  if (config.tasks.thermal_mc) {
    out.thermal_mc = {
      ...base('thermal_mc'),
      temperatures: parseNumList(config.thermal_mc.temperatures, [0.5, 1, 2, 4]),
      supercell: parseNumList(config.thermal_mc.supercell, [4, 4, 1]),
      n_sweeps: Number(config.thermal_mc.n_sweeps) || 4000,
      n_equil: Number(config.thermal_mc.n_equil) || 1500,
    }
  } else if (raw?.thermal_mc) out.thermal_mc = base('thermal_mc')

  if (config.tasks.sampled_correlations) {
    out.sampled_correlations = {
      ...base('sampled_correlations'),
      temperature: Number(config.sampled_correlations.temperature) || 0.5,
      supercell: parseNumList(config.sampled_correlations.supercell, [8, 1, 1]),
      dt: Number(config.sampled_correlations.dt) || 0.02,
      n_steps: Number(config.sampled_correlations.n_steps) || 2048,
      n_traj: Number(config.sampled_correlations.n_traj) || 8,
    }
  } else if (raw?.sampled_correlations) out.sampled_correlations = base('sampled_correlations')

  if (config.tasks.kpm_sqw) {
    out.kpm = {
      ...base('kpm'),
      e_min: Number(config.kpm.e_min) || 0,
      e_max: Number(config.kpm.e_max) || 10,
      e_step: Number(config.kpm.e_step) || 0.05,
      fwhm: Number(config.kpm.fwhm) || 0.1,
    }
  } else if (raw?.kpm) out.kpm = base('kpm')

  if (config.calculation.mode === 'entangled' && config.calculation.units_text) {
    try {
      const u = JSON.parse(config.calculation.units_text)
      if (Array.isArray(u) && u.length) out.units = u
    } catch { /* leave units for the runner to derive/complain about */ }
  }
  return out
}

/**
 * Editor state -> the config dict that Run, Save and Export all use. This IS
 * the file `magcalc run` reads: the backend writes it verbatim and hands it to
 * magcalc.runner, so there is no separate run-payload dialect that can drift.
 */
export function buildRunConfig({ config, atomMode, interactionMode, raw }) {
  // Parameters: drop the misleading global S (it lives on the atoms) and order
  // interaction params before field params (H_dir before H_mag).
  //
  // The editor always carries H_mag/H_dir, so an opened file that declares no
  // field used to gain `H_mag: 0, H_dir: [0, 0, 1]` -- a zero-magnitude Zeeman
  // term. Numerically it is nothing, but it is a term the CLI does not add, and
  // it moved CoRh2O4's bands by 6e-14. Same rule as the settings blocks: keep
  // what the file declared plus what the user actually changed.
  let rawParams = clone(config.parameters)
  delete rawParams.S
  if (raw) {
    rawParams = mergeEdits(rawParams, raw.parameters || {}, DEFAULT_CONFIG.parameters)
    delete rawParams.S
  }
  const fieldKeys = ['H_mag', 'H_dir']
  const interactionKeys = Object.keys(rawParams).filter(k => !fieldKeys.includes(k)).sort()
  const sortedParamKeys = [...interactionKeys, ...fieldKeys].filter(k => rawParams[k] !== undefined)
  // Values are NOT rounded. They used to be clamped to 5 decimals for a tidy
  // file, which quietly degraded every fitted parameter the moment a config was
  // opened and run: J11 = 6.89141591 came back as 6.89142, and an H_dir of
  // 1/sqrt(2) lost 11 digits. A number in the file is the model's number.
  const cleanParams = {}
  sortedParamKeys.forEach(key => { cleanParams[key] = rawParams[key] })

  const sp = buildStructPayload({ config, atomMode, interactionMode, raw })

  // Crystal structure: emit ONLY the atom key that matches the mode, so the
  // runner does its own (CLI-identical) symmetry expansion.
  let crystal_structure = null
  if (sp.crystal_structure && raw?.crystal_structure) {
    crystal_structure = { ...sp.crystal_structure }
    if (crystal_structure.atom_mode === 'explicit') delete crystal_structure.wyckoff_atoms
    else delete crystal_structure.atoms_uc
  } else if (sp.crystal_structure) {
    const cleanLattice = { ...config.lattice }
    ;['a', 'b', 'c', 'alpha', 'beta', 'gamma'].forEach(k => { cleanLattice[k] = norm(cleanLattice[k]) })
    const cleanAtoms = config.wyckoff_atoms.map(a => ({
      ...a,
      pos: a.pos.map(norm),
      spin_S: norm(a.spin_S),
    }))
    crystal_structure = {
      ...(config.lattice.lattice_vectors
        ? { lattice_vectors: config.lattice.lattice_vectors }
        : { lattice_parameters: cleanLattice }),
      [atomMode === 'explicit' ? 'atoms_uc' : 'wyckoff_atoms']: cleanAtoms,
      atom_mode: atomMode,
      magnetic_elements: config.magnetic_elements || ['Cu'],
      dimensionality: 3,
    }
  }

  // Interactions: unwrap the designer "explicit" wrapper {list:[...]} to a bare
  // list (config_loader's expected form); leave symmetry_rules/dict forms alone
  // so the runner expands them.
  let interactions = sp.interactions
  if (interactions && !Array.isArray(interactions) && interactions.list) {
    interactions = interactions.list
  }

  // Blocks the file carried that the editor does not model are passed through
  // verbatim -- from_mcif/mcif, experiment, static_correlations, hamiltonian
  // extras, and anything the engine gains before this file is next touched.
  const passthrough = {}
  for (const k of Object.keys(raw || {})) {
    if (!EDITOR_OWNED_BLOCKS.has(k)) passthrough[k] = clone(raw[k])
  }

  // When a file was opened it is the base: a settings block is emitted as the
  // file's block plus the edits the user made, NOT as the editor's full default
  // set. Injecting an untouched default is not cosmetic -- `minimization` alone
  // carries method-specific keys, and adding the anneal-only `n_sweeps: 2000` to
  // a `method: TNC` config crashed the minimizer
  // ("minimize() got an unexpected keyword argument 'n_sweeps'"), after which the
  // run died at the ground-state guard with a message about the magnetic
  // structure. A config that ran from the CLI, opened and run in the app, failed.
  const block = (editorBlock, key) =>
    (raw ? mergeEdits(editorBlock, raw[key], DEFAULT_CONFIG[key]) : editorBlock)

  const cfg = {
    ...passthrough,
    parameter_order: sp.parameter_order || sortedParamKeys,
    parameters: cleanParams,
    ...(crystal_structure ? { crystal_structure } : {}),
    interactions,
    magnetic_structure: sp.magnetic_structure,
    tasks: {
      ...config.tasks,
      calculate_dispersion: config.tasks.dispersion,
      calculate_sqw_map: config.tasks.sqw_map,
    },
    q_path: {
      ...config.q_path.points,
      path: config.q_path.path,
      points_per_segment: config.q_path.points_per_segment,
    },
    plotting: {
      ...block(config.plotting, 'plotting'),
      energy_limits_disp: [config.plotting.energy_min, config.plotting.energy_max],
      broadening_width: config.plotting.broadening,
    },
    minimization: {
      ...block(config.minimization, 'minimization'),
      enabled: config.tasks.minimization,
    },
    // cache_mode is a pure performance knob (the symbolic matrix is deterministic
    // per model topology), so the app's 'auto' is always worth sending.
    calculation: { cache_mode: 'auto', ...block(buildCalcPayload(config), 'calculation') },
    output: block(config.output, 'output'),
    powder_average: block(config.powder_average, 'powder_average'),
    fitting: block(config.fitting, 'fitting'),
    ...buildBeyondLswtBlocks({ config, raw }),
  }

  // A magnetic structure the file supplied is physics input and always ships.
  // Only a designer-built one is gated on the editor's enabled toggle.
  const msFromFile = Boolean(raw?.magnetic_structure)
  if (!msFromFile && !config.magnetic_structure.enabled) delete cfg.magnetic_structure
  if (cfg.magnetic_structure == null) delete cfg.magnetic_structure

  return cfg
}
