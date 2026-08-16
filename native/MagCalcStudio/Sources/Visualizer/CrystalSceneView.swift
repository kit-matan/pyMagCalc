import SceneKit
import SwiftUI

// MARK: - Colors matching the web visualizer (Visualizer.jsx)

private enum VizColor {
    static func atom(label: String, dark: Bool) -> PlatformColor {
        if label.contains("Cu") {
            return PlatformColor(red: dark ? 0.97 : 0.94, green: dark ? 0.44 : 0.27, blue: dark ? 0.44 : 0.27, alpha: 1)
        }
        return PlatformColor(red: dark ? 0.22 : 0.05, green: dark ? 0.74 : 0.65, blue: dark ? 0.97 : 0.91, alpha: 1)
    }

    static func bond(type: String) -> PlatformColor {
        switch type {
        case "dm": return PlatformColor(red: 0.96, green: 0.45, blue: 0.71, alpha: 1)          // #f472b6
        case "anisotropic": return PlatformColor(red: 0.98, green: 0.75, blue: 0.14, alpha: 1) // #fbbf24
        default: return PlatformColor(red: 0.58, green: 0.64, blue: 0.72, alpha: 1)            // #94a3b8
        }
    }

    static let selected = PlatformColor(red: 0.98, green: 0.75, blue: 0.14, alpha: 1)  // #fbbf24
    static let spin = PlatformColor(red: 0.06, green: 0.73, blue: 0.51, alpha: 1)      // #10b981
    static let dmVector = PlatformColor(red: 0.91, green: 0.47, blue: 0.98, alpha: 1)  // #e879f9
    static let ghost = PlatformColor(white: 0.55, alpha: 0.6)
}

/// Interactive unit-cell view matching the web app's three.js Visualizer:
/// atoms with labels + spin arrows, bonds colored/dashed by interaction type,
/// DM vector arrows, ghost atoms for out-of-cell partners, tap selection and
/// camera presets. Z is up, like the web scene.
struct CrystalSceneView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.colorScheme) private var colorScheme
    let lattice: LatticeParameters
    let atoms: [VisualizerAtom]
    let bonds: [VisualizerBond]

    @State private var cameraPreset: CameraPreset = .iso
    @State private var presetTick = 0

    var body: some View {
        ZStack(alignment: .topTrailing) {
            CrystalSceneRepresentable(
                lattice: lattice,
                atoms: atoms,
                bonds: bonds,
                selectedBond: model.selectedBond,
                dark: colorScheme == .dark,
                preset: cameraPreset,
                presetTick: presetTick
            ) { bond in
                model.selectedBond = bond
            }

            // Camera toolbar (Reset / XY / XZ / YZ), like the web overlay.
            HStack(spacing: 4) {
                Button { trigger(.iso) } label: { Image(systemName: "arrow.counterclockwise") }
                Divider().frame(height: 14)
                Button("XY") { trigger(.top) }
                Button("XZ") { trigger(.front) }
                Button("YZ") { trigger(.side) }
            }
            .font(.caption.weight(.bold))
            .buttonStyle(.borderless)
            .padding(6)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
            .padding(8)
        }
        .overlay(alignment: .topLeading) {
            if let bond = model.selectedBond {
                SelectedBondPanel(bond: bond)
                    .padding(8)
            }
        }
    }

    private func trigger(_ preset: CameraPreset) {
        cameraPreset = preset
        presetTick += 1
    }
}

enum CameraPreset {
    case iso, top, front, side

    /// Viewing direction, from the web app's CameraRig (z-up).
    var direction: SIMD3<Double> {
        switch self {
        case .iso: return simd_normalize(SIMD3(12, 12, 10))
        case .top: return SIMD3(0, 0, 1)
        case .front: return simd_normalize(SIMD3(0, -15, 2))
        case .side: return simd_normalize(SIMD3(15, 0, 2))
        }
    }

    /// Camera position that frames a sphere of `radius` about `center`.
    ///
    /// The distances were copied from the web app as CONSTANTS (12,12,10 etc.),
    /// which only ever framed a ~5 A cell: at a = 20.6 A (alpha-CVO) the camera
    /// sits inside the structure and the preview shows a wall of spheres. Same
    /// rule as SceneBuilder.standardScene — fit the content, don't assume a size.
    func position(center: SIMD3<Double>, radius: Double, fieldOfView: Double = 50) -> SIMD3<Double> {
        let halfFOV = fieldOfView * .pi / 360
        // 1.1: the fieldOfView applies to the viewport's larger dimension, so
        // the other one is tighter than the nominal 50 degrees. `radius` is a
        // bounding SPHERE, already generous for an elongated cell.
        let distance = max(radius / tan(halfFOV) * 1.1, 6)
        return center + direction * distance
    }
}

// MARK: - SceneKit representable with tap-to-select

private struct CrystalSceneRepresentable: PlatformViewRepresentable {
    let lattice: LatticeParameters
    let atoms: [VisualizerAtom]
    let bonds: [VisualizerBond]
    let selectedBond: VisualizerBond?
    let dark: Bool
    let preset: CameraPreset
    let presetTick: Int
    var onBondTap: (VisualizerBond?) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onBondTap: onBondTap) }

    #if os(macOS)
    func makeNSView(context: Context) -> SCNView { makeView(context: context) }
    func updateNSView(_ view: SCNView, context: Context) { update(view, context: context) }
    #else
    func makeUIView(context: Context) -> SCNView { makeView(context: context) }
    func updateUIView(_ view: SCNView, context: Context) { update(view, context: context) }
    #endif

    private func makeView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = false
        view.antialiasingMode = .multisampling4X
        view.backgroundColor = .clear

        #if os(macOS)
        let click = NSClickGestureRecognizer(target: context.coordinator,
                                             action: #selector(Coordinator.handleTap(_:)))
        view.addGestureRecognizer(click)
        #else
        let tap = UITapGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handleTap(_:)))
        view.addGestureRecognizer(tap)
        #endif
        return view
    }

    private func update(_ view: SCNView, context: Context) {
        context.coordinator.onBondTap = onBondTap
        context.coordinator.bonds = bonds

        // Rebuilding the scene resets the user's orbit; preserve the camera
        // transform across data updates, but honor explicit preset clicks.
        let previousTransform = view.pointOfView?.worldTransform
        // ... and re-frame when the model changes SIZE (another config opened,
        // a lattice parameter edited): an orbit kept from a 5 A cell leaves a
        // 20 A one filling the whole frame.
        let radius = contentBounds().radius
        let resized = abs(radius - context.coordinator.lastRadius)
            > 0.2 * max(radius, context.coordinator.lastRadius)
        context.coordinator.lastRadius = radius

        view.scene = buildScene()
        let cameraNode = view.scene?.rootNode.childNode(withName: "camera", recursively: false)
        if context.coordinator.lastPresetTick != presetTick || context.coordinator.lastPresetTick == -1 {
            context.coordinator.lastPresetTick = presetTick
        } else if let previousTransform, !resized {
            cameraNode?.transform = previousTransform
        }
        view.pointOfView = cameraNode
    }

    final class Coordinator: NSObject {
        var onBondTap: (VisualizerBond?) -> Void
        var bonds: [VisualizerBond] = []
        var lastPresetTick = -1
        var lastRadius = 0.0

        init(onBondTap: @escaping (VisualizerBond?) -> Void) {
            self.onBondTap = onBondTap
        }

        @objc func handleTap(_ gesture: PlatformGestureRecognizer) {
            guard let view = gesture.view as? SCNView else { return }
            let point = gesture.location(in: view)
            let hits = view.hitTest(point, options: [.searchMode: SCNHitTestSearchMode.all.rawValue])
            for hit in hits {
                var node: SCNNode? = hit.node
                while let n = node {
                    if let name = n.name, name.hasPrefix("bond-"),
                       let idx = Int(name.dropFirst(5)), idx < bonds.count {
                        onBondTap(bonds[idx])
                        return
                    }
                    node = n.parent
                }
            }
            onBondTap(nil)
        }
    }

    // MARK: Scene construction

    private func buildScene() -> SCNScene {
        let scene = SCNScene()
        scene.background.contents = PlatformColor.clear

        // Lights (web: ambient + point + spot).
        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = dark ? 550 : 420
        let ambientNode = SCNNode()
        ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)

        let key = SCNLight()
        key.type = .omni
        key.intensity = 900
        let keyNode = SCNNode()
        keyNode.light = key
        keyNode.position = SCNVector3(10, 10, 10)
        scene.rootNode.addChildNode(keyNode)

        let fill = SCNLight()
        fill.type = .omni
        fill.intensity = 400
        let fillNode = SCNNode()
        fillNode.light = fill
        fillNode.position = SCNVector3(-10, 10, 10)
        scene.rootNode.addChildNode(fillNode)

        // Camera: z-up like the web scene, framed on the actual structure.
        let content = contentBounds()
        let camera = SCNCamera()
        let p = preset.position(center: content.center, radius: content.radius)
        camera.zFar = max(1000, simd_length(p - content.center) * 4)
        camera.fieldOfView = 50
        let cameraNode = SCNNode()
        cameraNode.name = "camera"
        cameraNode.camera = camera
        cameraNode.position = SceneBuilder.v3(p)
        cameraNode.look(at: SceneBuilder.v3(content.center),
                        up: SCNVector3(0, 0, 1), localFront: SCNVector3(0, 0, -1))
        scene.rootNode.addChildNode(cameraNode)

        // Grid in the XY plane (web: gridHelper rotated into XY).
        scene.rootNode.addChildNode(gridNode(center: content.center, radius: content.radius))

        // Atoms + labels + spin arrows. Label sizes are in Angstrom, so they
        // scale with the cell to stay legible at the framing distance.
        let labelScale = max(1.0, content.radius / 9)
        let spinDirCart = lattice.cartesian(fromFractional: [0, 0, 1])
        for atom in atoms {
            let pos = lattice.cartesian(fromFractional: atom.pos)
            scene.rootNode.addChildNode(
                SceneBuilder.atomNode(at: pos, radius: 0.5,
                                      color: VizColor.atom(label: atom.label, dark: dark))
            )
            scene.rootNode.addChildNode(labelNode(atom.label, at: pos + SIMD3(0, 0, 0.9),
                                                  size: 0.35 * labelScale))
            scene.rootNode.addChildNode(
                SceneBuilder.arrowNode(origin: pos, direction: spinDirCart,
                                       length: 1.8, color: VizColor.spin)
            )
        }

        // Bonds.
        for (idx, bond) in bonds.enumerated() {
            guard bond.atomI < atoms.count, bond.atomJ < atoms.count else { continue }
            let fi = atoms[bond.atomI].pos
            var fj = atoms[bond.atomJ].pos
            for k in 0..<min(3, bond.offset.count) { fj[k] += Double(bond.offset[k]) }
            let start = lattice.cartesian(fromFractional: fi)
            let end = lattice.cartesian(fromFractional: fj)
            guard simd_length(end - start) > 1e-5 else { continue }

            let isSelected = selectedBond.map { sel in
                sel.atomI == bond.atomI && sel.atomJ == bond.atomJ && sel.offset == bond.offset
            } ?? false
            let color = isSelected ? VizColor.selected : VizColor.bond(type: bond.type)

            let bondGroup = SCNNode()
            bondGroup.name = "bond-\(idx)"
            let radius: CGFloat = isSelected ? 0.07 : (bond.type == "heisenberg" ? 0.05 : 0.035)
            if bond.type == "heisenberg" {
                bondGroup.addChildNode(SceneBuilder.cylinderNode(from: start, to: end,
                                                                 radius: radius, color: color))
            } else {
                // Dashed line for DM / anisotropic bonds, like the web view.
                for segment in dashSegments(from: start, to: end) {
                    bondGroup.addChildNode(SceneBuilder.cylinderNode(from: segment.0, to: segment.1,
                                                                     radius: radius, color: color))
                }
            }
            // Invisible fat cylinder for easier tapping (web's hit cylinder).
            let hit = SceneBuilder.cylinderNode(from: start, to: end, radius: 0.2,
                                                color: PlatformColor.red.withAlphaComponent(0.001))
            bondGroup.addChildNode(hit)
            scene.rootNode.addChildNode(bondGroup)

            // Bond label at midpoint (J1 / DM / Aniso).
            if let label = bond.label, !label.isEmpty {
                let mid = (start + end) / 2
                let offset: SIMD3<Double> = bond.type == "dm" ? SIMD3(0, 0, 0.4) : .zero
                scene.rootNode.addChildNode(labelNode(label, at: mid + offset,
                                                      color: color, size: 0.32 * labelScale))
            }

            // DM vector arrow at the midpoint.
            if bond.type == "dm", let dm = bond.dmVector, dm.count == 3,
               simd_length(SIMD3(dm[0], dm[1], dm[2])) > 1e-8 {
                let mid = (start + end) / 2
                scene.rootNode.addChildNode(
                    SceneBuilder.arrowNode(origin: mid, direction: SIMD3(dm[0], dm[1], dm[2]),
                                           length: 1.0, color: VizColor.dmVector)
                )
            }

            // Ghost atom at the far end for out-of-cell bonds.
            if bond.offset.contains(where: { $0 != 0 }) {
                scene.rootNode.addChildNode(ghostAtomNode(at: end))
                scene.rootNode.addChildNode(labelNode(atoms[bond.atomJ].label,
                                                      at: end + SIMD3(0, 0, 0.6),
                                                      color: VizColor.ghost, size: 0.28 * labelScale))
            }
        }

        return scene
    }

    private func dashSegments(from a: SIMD3<Double>, to b: SIMD3<Double>,
                              dash: Double = 0.2, gap: Double = 0.1) -> [(SIMD3<Double>, SIMD3<Double>)] {
        let d = b - a
        let length = simd_length(d)
        guard length > 1e-8 else { return [] }
        let dir = d / length
        var segments: [(SIMD3<Double>, SIMD3<Double>)] = []
        var t = 0.0
        while t < length {
            let end = min(t + dash, length)
            segments.append((a + dir * t, a + dir * end))
            t = end + gap
        }
        return segments
    }

    private func ghostAtomNode(at position: SIMD3<Double>) -> SCNNode {
        let sphere = SCNSphere(radius: 0.3)
        sphere.segmentCount = 20
        let material = SCNMaterial()
        material.diffuse.contents = VizColor.ghost
        material.transparency = 0.6
        sphere.materials = [material]
        let node = SCNNode(geometry: sphere)
        node.position = SceneBuilder.v3(position)
        return node
    }

    private func labelNode(_ text: String, at position: SIMD3<Double>,
                           color: PlatformColor = .white, size: CGFloat = 0.35) -> SCNNode {
        let scnText = SCNText(string: text, extrusionDepth: 0.02)
        scnText.font = .systemFont(ofSize: 1, weight: .semibold)
        scnText.flatness = 0.05
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.lightingModel = .constant
        scnText.materials = [material]

        let node = SCNNode(geometry: scnText)
        let (minB, maxB) = node.boundingBox
        node.pivot = SCNMatrix4MakeTranslation((minB.x + maxB.x) / 2, minB.y, 0)
        node.scale = SCNVector3(Float(size), Float(size), Float(size))
        node.position = SceneBuilder.v3(position)
        node.constraints = [SCNBillboardConstraint()]
        return node
    }

    /// Centre and bounding radius of everything drawn: atoms, bond endpoints
    /// (offset partners and their ghosts included) and the unit cell itself, so
    /// a one-atom cell is still framed by its box.
    private func contentBounds() -> (center: SIMD3<Double>, radius: Double) {
        var points: [SIMD3<Double>] = []
        for atom in atoms { points.append(lattice.cartesian(fromFractional: atom.pos)) }
        for bond in bonds where bond.atomJ < atoms.count {
            var fj = atoms[bond.atomJ].pos
            for k in 0..<min(3, bond.offset.count) { fj[k] += Double(bond.offset[k]) }
            points.append(lattice.cartesian(fromFractional: fj))
        }
        for corner in [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]] {
            points.append(lattice.cartesian(fromFractional: corner))
        }
        guard let first = points.first else { return (.zero, 6) }

        var lo = first, hi = first
        for p in points {
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        let center = (lo + hi) / 2
        // +1 A of margin for the atom spheres and their labels.
        let radius = max(simd_length(hi - lo) / 2 + 1.0, 3.0)
        return (center, radius)
    }

    private func gridNode(center: SIMD3<Double>, radius: Double) -> SCNNode {
        let node = SCNNode()
        let color = PlatformColor(white: dark ? 0.18 : 0.8, alpha: 0.7)
        // Cover the structure (was a fixed +/-10 A about the origin, i.e. a
        // patch smaller than a 20 A cell and off to one side of it), with the
        // spacing widened so the line count stays sane.
        let extent = max(5.0, (radius * 1.1).rounded(.up))
        let step = max(1.0, (extent / 10).rounded(.up))
        let lines = Int(extent / step)
        let cx = (center.x / step).rounded() * step
        let cy = (center.y / step).rounded() * step
        for i in -lines...lines {
            let d = Double(i) * step
            node.addChildNode(SceneBuilder.cylinderNode(
                from: SIMD3(cx - extent, cy + d, 0), to: SIMD3(cx + extent, cy + d, 0),
                radius: 0.006 * step, color: color))
            node.addChildNode(SceneBuilder.cylinderNode(
                from: SIMD3(cx + d, cy - extent, 0), to: SIMD3(cx + d, cy + extent, 0),
                radius: 0.006 * step, color: color))
        }
        return node
    }
}

#if os(macOS)
typealias PlatformGestureRecognizer = NSGestureRecognizer
#else
typealias PlatformGestureRecognizer = UIGestureRecognizer
#endif

// MARK: - Selected bond overlay (web's visualizer popover)

struct SelectedBondPanel: View {
    @EnvironmentObject var model: AppModel
    let bond: VisualizerBond

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Selected Bond")
                    .font(.callout.weight(.bold))
                    .foregroundStyle(.tint)
                Spacer()
                Button {
                    model.selectedBond = nil
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }

            HStack {
                Text("\(atomLabel(bond.atomI)) → \(atomLabel(bond.atomJ))")
                    .font(.caption.monospaced().weight(.semibold))
                Spacer()
                Text("Offset: [\(bond.offset.map(String.init).joined(separator: ","))]")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(6)
            .background(.background, in: RoundedRectangle(cornerRadius: 6))

            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    let matching = matchingBonds
                    if matching.isEmpty {
                        Text("No interactions")
                            .font(.caption.italic())
                            .foregroundStyle(.secondary)
                    }
                    ForEach(matching) { b in
                        VStack(alignment: .leading, spacing: 4) {
                            Label(b.type.replacingOccurrences(of: "_", with: " ").capitalized,
                                  systemImage: icon(for: b.type))
                                .font(.caption.weight(.bold))
                            if let matrix = ExchangeMatrix.symbolic(type: b.type, value: b.ruleValue ?? b.value) {
                                ExchangeTensorGrid(matrix: matrix)
                            } else {
                                Text((b.ruleValue ?? b.value)?.displayString ?? "—")
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(6)
                        .background(.background, in: RoundedRectangle(cornerRadius: 6))
                    }
                }
            }
            .frame(maxHeight: 230)

            Menu {
                Button("Heisenberg") { model.addRuleFromVisualizer(type: .heisenberg) }
                Button("DM Interaction") { model.addRuleFromVisualizer(type: .dm) }
                Button("Anisotropic Exchange") { model.addRuleFromVisualizer(type: .anisotropicExchange) }
                Button("Interaction Matrix") { model.addRuleFromVisualizer(type: .interactionMatrix) }
                Button("Kitaev") { model.addRuleFromVisualizer(type: .kitaev) }
            } label: {
                Label("Add Interaction", systemImage: "plus")
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(10)
        .frame(width: 260)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.tint.opacity(0.4)))
    }

    private var matchingBonds: [VisualizerBond] {
        (model.visualizerData?.bonds ?? []).filter {
            $0.atomI == bond.atomI && $0.atomJ == bond.atomJ && $0.offset == bond.offset
        }
    }

    private func atomLabel(_ idx: Int) -> String {
        model.visualizerData?.atoms.first { $0.idx == idx }?.label ?? String(idx)
    }

    private func icon(for type: String) -> String {
        if type == "heisenberg" { return "bolt" }
        if type.contains("dm") { return "wind" }
        if type.contains("anisotropic") { return "scope" }
        return "cube"
    }
}

/// Interactive minimized-structure view: atoms with spin arrows
/// (mag_structure.json from the runner). Positions are already cartesian.
struct SpinStructureSceneView: View {
    let structure: MagStructureResult

    var body: some View {
        SceneKitView(scene: buildScene())
    }

    private func buildScene() -> SCNScene {
        let positions = structure.atoms.map { SIMD3<Double>($0[safe: 0] ?? 0, $0[safe: 1] ?? 0, $0[safe: 2] ?? 0) }
        var center = SIMD3<Double>.zero
        for p in positions { center += p }
        if !positions.isEmpty { center /= Double(positions.count) }
        var radius = 3.0
        for p in positions { radius = max(radius, simd_length(p - center)) }

        let scene = SceneBuilder.standardScene(boundingRadius: radius, center: center)

        for (idx, pos) in positions.enumerated() {
            scene.rootNode.addChildNode(
                SceneBuilder.atomNode(at: pos, radius: 0.3, color: .darkGray)
            )
            if idx < structure.vectors.count {
                let v = structure.vectors[idx]
                let dir = SIMD3<Double>(v[safe: 0] ?? 0, v[safe: 1] ?? 0, v[safe: 2] ?? 0)
                scene.rootNode.addChildNode(
                    SceneBuilder.arrowNode(origin: pos, direction: dir,
                                           length: 1.5, color: .systemRed)
                )
            }
        }
        return scene
    }
}
