import SceneKit
import SwiftUI

#if os(macOS)
typealias PlatformColor = NSColor
#else
typealias PlatformColor = UIColor
#endif

/// Stable, distinguishable colors for interaction rules, keyed by the rule's
/// display value ("J1", "[Dx, 0, 0]", …) — same idea as the web visualizer.
enum BondPalette {
    private static let palette: [Color] = [
        .orange, .cyan, .green, .pink, .purple, .yellow, .mint, .indigo, .red, .teal,
    ]

    static func color(for key: String) -> Color {
        var hash = 5381
        for byte in key.utf8 {
            hash = ((hash << 5) &+ hash) &+ Int(byte)
        }
        return palette[abs(hash) % palette.count]
    }

    static func platformColor(for key: String) -> PlatformColor {
        PlatformColor(color(for: key))
    }
}

enum SceneBuilder {
    static func v3(_ v: SIMD3<Double>) -> SCNVector3 {
        SCNVector3(Float(v.x), Float(v.y), Float(v.z))
    }

    /// Sphere node for an atom.
    static func atomNode(at position: SIMD3<Double>, radius: CGFloat,
                         color: PlatformColor, label: String? = nil) -> SCNNode {
        let sphere = SCNSphere(radius: radius)
        sphere.segmentCount = 24
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.specular.contents = PlatformColor.white
        sphere.materials = [material]
        let node = SCNNode(geometry: sphere)
        node.position = v3(position)
        if let label { node.name = label }
        return node
    }

    /// Cylinder between two points (bond / cell edge).
    static func cylinderNode(from a: SIMD3<Double>, to b: SIMD3<Double>,
                             radius: CGFloat, color: PlatformColor) -> SCNNode {
        let d = b - a
        let length = simd_length(d)
        guard length > 1e-8 else { return SCNNode() }

        let cylinder = SCNCylinder(radius: radius, height: CGFloat(length))
        cylinder.radialSegmentCount = 12
        let material = SCNMaterial()
        material.diffuse.contents = color
        cylinder.materials = [material]

        let node = SCNNode(geometry: cylinder)
        node.position = v3((a + b) / 2)
        node.orientRodAlong(direction: d)
        return node
    }

    /// Arrow (shaft + cone) for spin vectors, starting at `origin`.
    static func arrowNode(origin: SIMD3<Double>, direction: SIMD3<Double>,
                          length: Double, color: PlatformColor) -> SCNNode {
        let norm = simd_length(direction)
        guard norm > 1e-8 else { return SCNNode() }
        let dir = direction / norm
        let shaftLen = length * 0.7
        let headLen = length * 0.3

        let group = SCNNode()

        let shaft = SCNCylinder(radius: 0.05, height: CGFloat(shaftLen))
        shaft.radialSegmentCount = 12
        let shaftMat = SCNMaterial()
        shaftMat.diffuse.contents = color
        shaft.materials = [shaftMat]
        let shaftNode = SCNNode(geometry: shaft)
        shaftNode.position = v3(origin + dir * (shaftLen / 2))
        shaftNode.orientRodAlong(direction: dir)
        group.addChildNode(shaftNode)

        let head = SCNCone(topRadius: 0, bottomRadius: 0.14, height: CGFloat(headLen))
        head.radialSegmentCount = 12
        let headMat = SCNMaterial()
        headMat.diffuse.contents = color
        head.materials = [headMat]
        let headNode = SCNNode(geometry: head)
        headNode.position = v3(origin + dir * (shaftLen + headLen / 2))
        headNode.orientRodAlong(direction: dir)
        group.addChildNode(headNode)

        return group
    }

    /// Wireframe unit-cell box from lattice vectors.
    static func unitCellNode(latticeVectors: [SIMD3<Double>]) -> SCNNode {
        let node = SCNNode()
        let (va, vb, vc) = (latticeVectors[0], latticeVectors[1], latticeVectors[2])
        let corners: [SIMD3<Double>] = [
            .zero, va, vb, vc, va + vb, va + vc, vb + vc, va + vb + vc,
        ]
        let edges: [(Int, Int)] = [
            (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6),
            (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
        ]
        let color = PlatformColor.gray.withAlphaComponent(0.55)
        for (i, j) in edges {
            node.addChildNode(cylinderNode(from: corners[i], to: corners[j],
                                           radius: 0.018, color: color))
        }
        return node
    }

    /// Standard lighting + camera rig for a model of the given bounding radius.
    static func standardScene(boundingRadius: Double, center: SIMD3<Double>) -> SCNScene {
        let scene = SCNScene()
        scene.background.contents = PlatformColor.clear

        let camera = SCNCamera()
        camera.zFar = 1000
        camera.fieldOfView = 50
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        let dist = max(boundingRadius * 2.6, 6)
        cameraNode.position = v3(center + SIMD3(dist * 0.6, dist * 0.5, dist * 0.8))
        cameraNode.look(at: v3(center))
        scene.rootNode.addChildNode(cameraNode)

        let key = SCNLight()
        key.type = .omni
        key.intensity = 900
        let keyNode = SCNNode()
        keyNode.light = key
        keyNode.position = v3(center + SIMD3(10, 12, 14))
        scene.rootNode.addChildNode(keyNode)

        let fill = SCNLight()
        fill.type = .omni
        fill.intensity = 350
        let fillNode = SCNNode()
        fillNode.light = fill
        fillNode.position = v3(center + SIMD3(-10, -8, -12))
        scene.rootNode.addChildNode(fillNode)

        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = 420
        let ambientNode = SCNNode()
        ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)

        return scene
    }
}

extension SCNNode {
    /// Rotates a y-axis-aligned geometry (cylinder/cone) to point along `direction`.
    func orientRodAlong(direction: SIMD3<Double>) {
        let norm = simd_length(direction)
        guard norm > 1e-8 else { return }
        let dir = direction / norm
        let up = SIMD3<Double>(0, 1, 0)
        let dot = simd_dot(up, dir)
        if dot > 0.99999 { return }
        if dot < -0.99999 {
            rotation = SCNVector4(1, 0, 0, Float.pi)
            return
        }
        let axis = simd_normalize(simd_cross(up, dir))
        let angle = acos(min(max(dot, -1), 1))
        rotation = SCNVector4(Float(axis.x), Float(axis.y), Float(axis.z), Float(angle))
    }
}

/// SwiftUI wrapper around SCNView with orbit camera controls.
struct SceneKitView: PlatformViewRepresentable {
    let scene: SCNScene

    #if os(macOS)
    func makeNSView(context: Context) -> SCNView { makeView() }
    func updateNSView(_ view: SCNView, context: Context) { view.scene = scene }
    #else
    func makeUIView(context: Context) -> SCNView { makeView() }
    func updateUIView(_ view: SCNView, context: Context) { view.scene = scene }
    #endif

    private func makeView() -> SCNView {
        let view = SCNView()
        view.scene = scene
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = false
        view.antialiasingMode = .multisampling4X
        view.backgroundColor = .clear
        return view
    }
}

#if os(macOS)
protocol PlatformViewRepresentable: NSViewRepresentable {}
#else
protocol PlatformViewRepresentable: UIViewRepresentable {}
#endif
