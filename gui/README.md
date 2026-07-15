# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

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
