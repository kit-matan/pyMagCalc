import React, { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, PerspectiveCamera, Line, Text, GizmoHelper, GizmoViewport } from '@react-three/drei'
import * as THREE from 'three'

function Atom({ position, color, label, isGhost = false }) {
    return (
        <group position={position}>
            <mesh>
                <sphereGeometry args={[isGhost ? 0.3 : 0.5, 32, 32]} />
                <meshStandardMaterial
                    color={color}
                    transparent={isGhost}
                    opacity={isGhost ? 0.6 : 1.0}
                    roughness={0.4}
                    metalness={0.2}
                    emissive={color}
                    emissiveIntensity={isGhost ? 0.05 : 0.1}
                />
            </mesh>
            {label && !isGhost && (
                <Text
                    position={[0, 0.7, 0]}
                    fontSize={0.35}
                    color="#ffffff"
                    anchorX="center"
                    anchorY="middle"
                    outlineWidth={0.03}
                    outlineColor="#000000"
                >
                    {label}
                </Text>
            )}
        </group>
    )
}

function SpinArrow({ position, direction, length = 1.8, color = '#10b981' }) {
    const dir = new THREE.Vector3(...direction).normalize()
    const pos = new THREE.Vector3(...position)

    return (
        <group position={pos}>
            <primitive
                object={new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), length, color, 0.5, 0.25)}
            />
        </group>
    )
}

function BondLine({ start, end, label, color, type, labelOffset = [0, 0, 0] }) {
    const mid = [
        (start[0] + end[0]) / 2 + labelOffset[0],
        (start[1] + end[1]) / 2 + labelOffset[1],
        (start[2] + end[2]) / 2 + labelOffset[2]
    ]
    return (
        <group>
            {type !== 'dm' && (
                <Line
                    points={[start, end]}
                    color={color}
                    lineWidth={type === 'heisenberg' ? 2 : 1}
                    dashed={type !== 'heisenberg'}
                />
            )}
            {type === 'dm' && (
                /* For DM, maybe draw a thinner line or no line if Heisenberg exists? 
                   But usually we want to see the connection. 
                   Let's draw a thin dashed line for DM to distinguish it. */
                <Line
                    points={[start, end]}
                    color={color}
                    lineWidth={1}
                    dashed={true}
                    dashSize={0.2}
                    gapSize={0.1}
                />
            )}
            {label && (
                <Text
                    position={mid}
                    fontSize={0.3}
                    color={color}
                    anchorX="center"
                    anchorY="middle"
                    outlineWidth={0.02}
                    outlineColor="#000000"
                >
                    {label}
                </Text>
            )}
        </group>
    )
}

function getLatticeMatrix(lattice) {
    const { a, b, c, alpha, beta, gamma } = lattice
    const d2r = Math.PI / 180
    const ca = Math.cos(alpha * d2r)
    const cb = Math.cos(beta * d2r)
    const cg = Math.cos(gamma * d2r)
    const sg = Math.sin(gamma * d2r)

    const v = Math.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)

    // Standard crystallography convention (a along x, b in xy plant)
    return [
        [a, b * cg, c * cb],
        [0, b * sg, c * (ca - cb * cg) / sg],
        [0, 0, (c * v) / sg]
    ]
}

function transformPos(pos, matrix) {
    const [u, v, w] = pos
    return [
        matrix[0][0] * u + matrix[0][1] * v + matrix[0][2] * w,
        matrix[1][0] * u + matrix[1][1] * v + matrix[1][2] * w,
        matrix[2][0] * u + matrix[2][1] * v + matrix[2][2] * w
    ]
}

export default function Visualizer({ atoms, lattice, isDark, dimensionality, zFilter, bonds = [] }) {
    const bgColor = isDark ? '#020617' : '#f8fafc'
    const gridMain = isDark ? '#1e293b' : '#cbd5e1'
    const gridSec = isDark ? '#0f172a' : '#f1f5f9'
    const spinColor = isDark ? '#10b981' : '#059669' // Brighter in dark, deeper in light

    const matrix = getLatticeMatrix(lattice)

    // Filter atoms if zFilter is active in 2D mode
    const filteredAtoms = (dimensionality === '2D' && zFilter)
        ? atoms.filter(a => Math.abs(a.pos[2]) < 0.01 || Math.abs(a.pos[2] - 1.0) < 0.01)
        : atoms

    return (
        <div style={{ width: '100%', height: '100%', borderRadius: '16px', overflow: 'hidden', background: bgColor }}>
            <Canvas shadows>
                <PerspectiveCamera makeDefault position={[12, 12, 10]} up={[0, 0, 1]} />
                <OrbitControls makeDefault />
                <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
                    <GizmoViewport axisColors={['#f87171', '#34d399', '#38bdf8']} labelColor="white" />
                </GizmoHelper>

                {isDark && <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />}

                <ambientLight intensity={isDark ? 0.7 : 0.4} />
                <pointLight position={[10, 10, 10]} intensity={isDark ? 1.5 : 2} castShadow />
                <spotLight position={[-10, 10, 10]} angle={0.15} penumbra={1} intensity={isDark ? 1.5 : 2} />

                <group>
                    {filteredAtoms.map((atom, idx) => {
                        const atomColor = atom.label.includes('Cu') ? '#f87171' : '#38bdf8'
                        const finalAtomColor = isDark ? atomColor : (atom.label.includes('Cu') ? '#ef4444' : '#0ea5e9')

                        const rawCartPos = transformPos(atom.pos, matrix)
                        const cartPos = [rawCartPos[0], rawCartPos[1], rawCartPos[2]]

                        const rawCartDir = transformPos(atom.magmom_classical || [0, 0, 1], matrix)
                        const cartDir = [rawCartDir[0], rawCartDir[1], rawCartDir[2]]

                        return (
                            <React.Fragment key={idx}>
                                <Atom
                                    position={cartPos}
                                    label={atom.label}
                                    color={finalAtomColor}
                                />
                                <SpinArrow
                                    position={cartPos}
                                    direction={cartDir}
                                    color={spinColor}
                                />
                            </React.Fragment>
                        )
                    })}

                    {bonds.map((bond, idx) => {
                        // Find start/end atoms by index
                        if (bond.atom_i >= atoms.length || bond.atom_j >= atoms.length) return null

                        const atomI = atoms[bond.atom_i]
                        const atomJ = atoms[bond.atom_j] // Base atom J

                        // Calculate wrapped position for J using offset
                        // bond.offset is usually integer offset [u, v, w]
                        // pos_j_actual = pos_j_uc + offset
                        const posI = atomI.pos
                        const posJ = atomJ.pos
                        const offset = bond.offset || [0, 0, 0]
                        const posJExpanded = [
                            posJ[0] + offset[0],
                            posJ[1] + offset[1],
                            posJ[2] + offset[2]
                        ]

                        const startRaw = transformPos(posI, matrix)
                        const start = [startRaw[0], startRaw[1], startRaw[2]]

                        const endRaw = transformPos(posJExpanded, matrix)
                        const end = [endRaw[0], endRaw[1], endRaw[2]]

                        // Colors
                        const bondColor = bond.type === 'dm' ? '#f472b6' : (bond.type === 'anisotropic' ? '#fbbf24' : '#94a3b8')

                        // DM Vector Logic
                        let dmArrow = null
                        if (bond.type === 'dm' && bond.dm_vector) {
                            // DM vector at midpoint
                            const mid = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2]
                            const dmVecRaw = transformPos(bond.dm_vector, matrix)
                            const dmVec = [dmVecRaw[0], dmVecRaw[1], dmVecRaw[2]]

                            dmArrow = (
                                <SpinArrow
                                    position={mid}
                                    direction={dmVec}
                                    length={1.0}
                                    color="#e879f9"
                                />
                            )
                        }

                        // Filter bonds if nodes are filtered? 
                        // If zFilter is on, only show bonds between visible atoms?
                        if (dimensionality === '2D' && zFilter) {
                            if (Math.abs(posI[2]) > 0.01 && Math.abs(posI[2] - 1.0) > 0.01) return null
                            // Allow bonds to other layers? Maybe restricts visualization to in-plane
                            if (Math.abs(posJExpanded[2]) > 0.01 && Math.abs(posJExpanded[2] - 1.0) > 0.01) return null
                        }

                        // Offset label for DM to avoid overlap with J
                        const labelOffset = bond.type === 'dm' ? [0, 0.4, 0] : [0, 0, 0]

                        // Determine if we need to render a ghost atom at 'end'
                        let ghostAtom = null
                        const isExternal = Math.abs(offset[0]) > 0.001 || Math.abs(offset[1]) > 0.001 || Math.abs(offset[2]) > 0.001

                        if (isExternal && bond.atom_j < atoms.length) {
                            const atomJ = atoms[bond.atom_j]
                            // Ghost color: desaturated and lighter/darker depending on theme
                            const ghostColor = isDark ? '#475569' : '#cbd5e1'

                            ghostAtom = (
                                <Atom
                                    key={`ghost-${idx}`}
                                    position={end}
                                    label={atomJ.label}
                                    color={ghostColor}
                                    isGhost={true}
                                />
                            )
                        }

                        return (
                            <React.Fragment key={`bond-${idx}`}>
                                <BondLine
                                    start={start}
                                    end={end}
                                    label={bond.label}
                                    color={bondColor}
                                    type={bond.type}
                                    labelOffset={labelOffset}
                                />
                                {dmArrow}
                                {ghostAtom}
                            </React.Fragment>
                        )
                    })}
                </group>

                {dimensionality === '2D' ? (
                    <gridHelper args={[20, 20, gridMain, gridSec]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -0.1]} />
                ) : (
                    <gridHelper args={[20, 20, gridMain, gridSec]} rotation={[Math.PI / 2, 0, 0]} />
                )}
            </Canvas>
        </div>
    )
}
