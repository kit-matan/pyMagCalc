import SceneKit
import SwiftUI

/// Interactive unit-cell view: atoms (fractional → cartesian), exchange bonds
/// colored per rule, and the cell wireframe. Native counterpart of the web
/// app's three.js Visualizer.
struct CrystalSceneView: View {
    let lattice: LatticeParameters
    let atoms: [VisualizerAtom]
    let bonds: [VisualizerBond]

    var body: some View {
        SceneKitView(scene: buildScene())
    }

    private func buildScene() -> SCNScene {
        let vectors = lattice.latticeVectors
        let cellCenter = (vectors[0] + vectors[1] + vectors[2]) / 2
        let radius = max(simd_length(vectors[0]), simd_length(vectors[1]), simd_length(vectors[2]))
        let scene = SceneBuilder.standardScene(boundingRadius: radius / 2 + 2, center: cellCenter)

        scene.rootNode.addChildNode(SceneBuilder.unitCellNode(latticeVectors: vectors))

        // Atoms — color by species (leading letters of the label).
        var speciesColors: [String: PlatformColor] = [:]
        for atom in atoms {
            let species = atom.label.prefix { $0.isLetter }
            let key = String(species)
            if speciesColors[key] == nil {
                speciesColors[key] = BondPalette.platformColor(for: key)
            }
            let pos = lattice.cartesian(fromFractional: atom.pos)
            scene.rootNode.addChildNode(
                SceneBuilder.atomNode(at: pos, radius: 0.28,
                                      color: speciesColors[key]!, label: atom.label)
            )
        }

        // Bonds — one cylinder per expanded interaction, colored by rule value.
        for bond in bonds {
            guard bond.atomI < atoms.count, bond.atomJ < atoms.count else { continue }
            let fi = atoms[bond.atomI].pos
            var fj = atoms[bond.atomJ].pos
            for k in 0..<min(3, bond.offset.count) {
                fj[k] += Double(bond.offset[k])
            }
            let a = lattice.cartesian(fromFractional: fi)
            let b = lattice.cartesian(fromFractional: fj)
            scene.rootNode.addChildNode(
                SceneBuilder.cylinderNode(from: a, to: b, radius: 0.045,
                                          color: BondPalette.platformColor(for: bond.valueKey))
            )
        }

        return scene
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
