# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## Where a run's output files go

Both Studio clients (web and native) run through `/run-calculation`, and where the
outputs land follows the config, not the app:

* **A config you opened from disk** — the client sends `config_dir`, the run happens
  in that directory exactly as `magcalc run <file>` would, and the plots and `.npz`
  land **beside the config**. This is what keeps a figure directory self-contained.
* **A config with no file behind it** (built in the editor, or opened through the
  browser's file picker, which hands over only a name) — the run falls back to the
  project root, where nothing owns the outputs, so everything it produces goes into
  **`app_runs/`** (`GUI_OUTPUT_SUBDIR` in `gui/server.py`; gitignored). Delete that
  folder whenever you like: it is entirely regenerable.

The run still *executes* in the project root either way — only the output paths move
— so a config's own relative references (`from_mcif:`, `fitting.data_file:`,
`cif_file:`, `python_model_file:`) resolve exactly as they do for the CLI. The
runnable record of the run, `.config_gui_run.yaml`, stays at the run root for the
same reason; `magcalc run .config_gui_run.yaml` reproduces the run.

Custom output names are kept, not overridden: a config asking for `G1_h00_d.npz` gets
that file, inside the folder.

## Minimization (ground state)

The **Method** selector in *Tasks & Plotting → Minimization Parameters* now offers:

* **Monte-Carlo annealing** (`anneal`) — *the default and recommended choice.*
  Metropolis + cooling (SpinW `anneal` / Sunny `LocalSampler`), then a gradient
  polish. Crosses energy barriers, so it does not get trapped. Controls: **Runs**
  and **Sweeps**.
* **Steepest descent** (`steep`) — aligns each spin with its local field (SpinW
  `optmagsteep`). Fast, but monotone: it cannot escape a local minimum.
* **L-BFGS-B / TNC / SLSQP** — the legacy random multistart in (θ, φ). Kept for
  compatibility, but it gets trapped on frustrated systems. Controls: **Num Starts**,
  **N Workers**, **Early Stopping**.

Switching method retunes the budget automatically (a handful of annealing *runs* vs.
hundreds of gradient *restarts*) — the two are not interchangeable.

> **Why this matters.** LSWT is an expansion about a classical energy *minimum*; about
> anything else the spectrum is meaningless. A run whose magnetic structure is not the
> ground state now **fails with an error** instead of drawing a plausible-looking plot.
> If you hit that, switch to annealing and/or raise the budget.

## Ground-State Check (`calculation.on_imaginary`)

*Tasks & Plotting → Calculation Settings → **Ground-State Check***

Spin-wave theory is an expansion about a classical energy **minimum**; about anything
else the spectrum is meaningless — and it will still *look* like a spectrum. Two guards
run before any task: one for imaginary magnon energies, one that nudges the structure
and relaxes it to see whether a lower energy exists. (Neither alone is sufficient: a
ferromagnetic structure supplied for an antiferromagnet returns a perfectly plausible
real, positive spectrum, and is only caught by the energy guard.)

| Setting | Behaviour |
|---|---|
| **Fail the run** (`error`, default) | A structure that is not a minimum aborts the calculation with an actionable message. |
| **Warn only** (`warn`) | For **knowingly metastable** structures — a commensurate approximation to an incommensurate spiral (SW03), or a state the reference calculation itself treats as non-minimal (SW23, where SpinW uses `hermit=false`). |
| **Disable** (`off`) | Both guards off. A wrong ground state then produces a plausible-looking but meaningless spectrum, silently. |

If a run fails this check, the usual fix is not to switch to *Warn* — it is to switch
the minimization **Method** to *Monte-Carlo annealing* (see above), which is built to
escape the local minima that cause it.

## Open / Save go through the backend

**Open File** lists recently-opened configs and lets you walk the filesystem
(`/recent-configs`, `/browse-configs`), then loads the file with `/load-config`.
That route exists for one reason: it is the only one that yields a **path**. A
browser's own file picker hands over `handle.name` and nothing else, so the app
could not tell the server where the file lived — and `from_mcif:`,
`fitting.data_file:`, `cif_file:` and `python_model_file:` are all resolved
against the config file's own directory. A config that ran fine as `magcalc run
<file>` therefore died on FileNotFoundError in the web app, which could only ever
run in the project root.

With a server-opened file the app sends `config_dir` on every run, so the run
happens in that directory and those references resolve exactly as the CLI's do.
**Save** writes back through `/save-config` (there is no writable browser handle
for such a file), using the same canonical serializer as *Export YAML* — the
saved file is directly runnable from the command line.

The browser routes are still there: **Load YAML** imports a file with no path,
and **Open File** falls back to the browser dialog when the backend is
unreachable. Both clear the directory, so a run stays in the project root — where
a relative reference fails loudly rather than silently resolving against the
previous file's directory.

Pinned by `tests/test_gui_relative_paths.py`, whose oracle is the CLI.
