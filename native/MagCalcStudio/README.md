# MagCalc Studio — Native macOS & iOS Apps

Native SwiftUI clients for the MagCalc spin-wave calculator, based on the web
app in `gui/`. Both apps talk to the same FastAPI backend (`gui/server.py`)
using the same JSON protocol, so projects and results stay interchangeable
with the web UI.

```text
┌──────────────────────┐        HTTP + WebSocket        ┌──────────────────────┐
│ MagCalc Studio       │  /run-calculation, /ws/logs …  │ FastAPI backend      │
│ (SwiftUI, macOS/iOS) │ ─────────────────────────────► │ gui/server.py        │
└──────────────────────┘                                │ magcalc (Python)     │
                                                        └──────────────────────┘
```

- **macOS** — full studio experience. Can launch and supervise the Python
  backend itself (Settings → Embedded Backend), so no terminal is needed.
- **iOS / iPadOS** — same editor and run console; connects to a Mac on your
  network running the backend (`MAGCALC_HOST=0.0.0.0 python gui/server.py`).

## Building

Requirements: Xcode 16+ and [XcodeGen](https://github.com/yonaskolb/XcodeGen)
(`brew install xcodegen`).

```sh
cd native/MagCalcStudio
xcodegen generate          # produces MagCalcStudio.xcodeproj
open MagCalcStudio.xcodeproj
```

Build/run the `MagCalcStudio-macOS` or `MagCalcStudio-iOS` scheme. CLI builds:

```sh
xcodebuild -project MagCalcStudio.xcodeproj -scheme MagCalcStudio-macOS build
xcodebuild -project MagCalcStudio.xcodeproj -scheme MagCalcStudio-iOS \
           -destination 'generic/platform=iOS Simulator' build
```

## Feature parity with the web app

| Web tab | Native view |
| --- | --- |
| Structure (lattice, space group, Wyckoff atoms, CIF import) | `StructureView` |
| Interactions (symmetry rules, neighbor shells, DM/aniso/matrix editors) | `InteractionsView` |
| Environment (parameters, applied field, cache mode) | `EnvironmentView` |
| Tasks & Plotting (tasks, q-path, plot ranges, powder, minimization) | `TasksView` |
| Mag. Structure (manual patterns, import minimized structure) | `MagneticStructureView` |
| Data Fitting (data upload, vary/bounds, intensity model) | `FittingView` |
| Run & Analyze (live logs, result plots, fit report) | `RunView` |
| three.js 3D visualizer | SceneKit `CrystalSceneView` / `SpinStructureSceneView` |

## Native-only improvements over the web app

Implemented in this version:

- **Embedded backend management (macOS)** — start/stop the Python server from
  Settings; the app finds the repo and a Python interpreter automatically.
- **Metal-backed 3D rendering** — SceneKit with proper lighting, MSAA and
  trackpad/touch orbit controls instead of WebGL in a browser tab.
- **Project files** — save/open the entire model as a JSON project via the
  native file panel (works with iCloud Drive), instead of browser localStorage.
- **Keyboard shortcuts** — ⌘R runs, ⌘. stops a calculation (macOS menu bar).
- **Share sheet export** — share/AirDrop result plots directly from the app.
- **Adaptive layout** — sidebar + editor + 3D preview on Mac/iPad, stacked
  compact layout on iPhone.
- **Completion chime** — audible notification when a long calculation
  finishes (macOS).
- **Auto-registered parameters** — typing a new symbol (e.g. `J4`) in an
  interaction automatically creates the parameter in Environment.

Good candidates for future iterations:

- Bonjour advertisement in `server.py` + auto-discovery on iOS (zero-config
  pairing with the Mac backend).
- Swift Charts rendering of dispersion/S(Q,ω) from the exported `.npz`/CSV
  data instead of static PNGs (zoom, hover readout).
- Document-based app (`.magcalcproj` UTI) with recents, iCloud sync and
  multiple open projects.
- Push/local notification when a calculation finishes while the app is in
  the background (the run itself continues server-side).
- Native bond-symmetry inspector (the web app's /analyze-bonds and
  /bond-constraints endpoints are already covered by `APIClient`).
- Metal shader instancing for very large supercells in the visualizer.

## Layout

```text
native/MagCalcStudio/
├── project.yml               # XcodeGen project definition
├── SupportFiles/             # Info.plists (ATS local-networking exceptions)
├── Resources/space_groups.json
└── Sources/
    ├── App/                  # @main scene, menu commands
    ├── Models/               # MagCalcConfig (Codable), JSONValue, API DTOs
    ├── Networking/           # APIClient, WebSocket log stream
    ├── Server/               # LocalBackendController (macOS)
    ├── ViewModels/           # AppModel (state, persistence, run lifecycle)
    ├── Views/                # One view per sidebar tab + settings/components
    └── Visualizer/           # SceneKit crystal & spin-structure scenes
```
