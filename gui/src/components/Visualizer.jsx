import React, { useRef, useState, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Stars, PerspectiveCamera, Line, Text, GizmoHelper, GizmoViewport } from '@react-three/drei'
import * as THREE from 'three'
import { ArrowUp, ArrowDown, Move, RotateCcw, Maximize, Box, Eye } from 'lucide-react'

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

function BondLine({ start, end, label, color, type, labelOffset = [0, 0, 0], onClick, isSelected }) {
    const [hovered, setHovered] = useState(false)

    // Derived state
    const mid = [
        (start[0] + end[0]) / 2 + labelOffset[0],
        (start[1] + end[1]) / 2 + labelOffset[1],
        (start[2] + end[2]) / 2 + labelOffset[2]
    ]

    // Calculate length and orientation for the hit cylinder
    const startVec = new THREE.Vector3(...start)
    const endVec = new THREE.Vector3(...end)
    const distance = startVec.distanceTo(endVec)
    const midVec = new THREE.Vector3().addVectors(startVec, endVec).multiplyScalar(0.5)

    // Cylinder lookAt logic is tricky because cylinder creates along Y by default.
    // We can use a group with lookAt.

    const displayColor = isSelected ? '#fbbf24' : (hovered ? '#ffffff' : color)
    const lineWidth = isSelected ? 4 : (hovered ? 3 : (type === 'heisenberg' ? 2 : 1))

    return (
        <group>
            {/* Hit Cylinder - Transparent but raycastable */}
            <group position={midVec} lookAt={endVec}>
                {/* Rotate cylinder to align with Z-axis lookAt? No, lookAt aligns +Z to target. 
                     Cylinder is along Y. We need to rotate X 90deg? 
                     Actually simpler: Make a mesh looking at endVec?
                 */}
                <mesh
                    quaternion={new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), endVec.clone().sub(startVec).normalize())}
                    onClick={(e) => { e.stopPropagation(); onClick && onClick() }}
                    onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer' }}
                    onPointerOut={(e) => { e.stopPropagation(); setHovered(false); document.body.style.cursor = 'default' }}
                >
                    <cylinderGeometry args={[0.2, 0.2, distance, 8]} />
                    <meshBasicMaterial transparent opacity={0.0} color="red" />
                </mesh>
            </group>


            {type !== 'dm' && (
                <Line
                    points={[start, end]}
                    color={displayColor}
                    lineWidth={lineWidth}
                    dashed={type !== 'heisenberg'}
                />
            )}
            {type === 'dm' && (
                <Line
                    points={[start, end]}
                    color={displayColor}
                    lineWidth={lineWidth}
                    dashed={true}
                    dashSize={0.2}
                    gapSize={0.1}
                />
            )}
            {label && (
                <Text
                    position={mid}
                    fontSize={hovered || isSelected ? 0.5 : 0.3}
                    color={displayColor}
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

// Internal component to handle programmatic camera moves
function CameraRig({ view }) {
    const { camera, controls } = useThree()

    useEffect(() => {
        if (!view) return;

        const dist = 15;
        const newPos = new THREE.Vector3();

        switch (view) {
            case 'TOP': // XY Plane
                newPos.set(0, 0, dist);
                break;
            case 'FRONT': // XZ Plane? Or similar. usually Front is -Y looking at XZ? 
                // In this coord system, z is up. Front could be y=-dist.
                newPos.set(0, -dist, 2);
                break;
            case 'SIDE': // YZ
                newPos.set(dist, 0, 2);
                break;
            case 'ISO':
            default:
                newPos.set(12, 12, 10);
                break;
        }

        // Animate? simpler to just set for now, or use small interpolation if we had a loop
        // OrbitControls makes 'set' tricky if we don't update target.
        // Let's just set position and reset target to 0,0,0
        camera.position.copy(newPos);
        camera.lookAt(0, 0, 0);

        if (controls) {
            controls.target.set(0, 0, 0);
            controls.update();
        }

    }, [view, camera, controls])

    return null;
}

export default function Visualizer({ atoms, lattice, isDark, dimensionality, zFilter, bonds = [], onBondClick, selectedBond }) {
    const bgColor = isDark ? '#020617' : '#f8fafc'
    const gridMain = isDark ? '#1e293b' : '#cbd5e1'
    const gridSec = isDark ? '#0f172a' : '#f1f5f9'
    const spinColor = isDark ? '#10b981' : '#059669'

    const matrix = getLatticeMatrix(lattice)

    const filteredAtoms = (dimensionality === '2D' && zFilter)
        ? atoms.filter(a => Math.abs(a.pos[2]) < 0.01 || Math.abs(a.pos[2] - 1.0) < 0.01)
        : atoms

    // Camera Actions
    const [viewMetric, setViewMetric] = useState(0); // Just a trigger
    const [desiredView, setDesiredView] = useState(null)

    const triggerView = (v) => {
        setDesiredView(v);
        // Toggle metric to ensure effect fires even if view name is same (resets)
        setViewMetric(p => p + 1);
    }

    return (
        <div style={{ width: '100%', height: '100%', borderRadius: '16px', overflow: 'hidden', background: bgColor, position: 'relative' }}>

            {/* Camera Toolbar */}
            <div className="absolute top-4 right-4 z-10 flex flex-col gap-2 pointer-events-none">
                <div className="flex gap-1 bg-surface/80 p-1 rounded-lg border border-color backdrop-blur shadow-sm pointer-events-auto">
                    <button className="icon-btn xs" title="Reset View" onClick={() => triggerView('ISO')}><RotateCcw size={14} /></button>
                    <div className="w-px bg-color mx-1"></div>
                    <button className="icon-btn xs font-xs font-bold w-8" title="Top View (XY)" onClick={() => triggerView('TOP')}>XY</button>
                    <button className="icon-btn xs font-xs font-bold w-8" title="Front View" onClick={() => triggerView('FRONT')}>XZ</button>
                    <button className="icon-btn xs font-xs font-bold w-8" title="Side View" onClick={() => triggerView('SIDE')}>YZ</button>
                </div>
                {/* Tips */}
                <div className="bg-surface/50 p-2 rounded-lg border border-color backdrop-blur shadow-sm text-xxs text-muted pointer-events-auto max-w-[150px]">
                    <div className="flex align-center gap-xs mb-1">
                        <Move size={12} /> <span>Right-click or 2-finger to pan</span>
                    </div>
                </div>
            </div>

            <Canvas shadows className="cursor-move">
                <PerspectiveCamera makeDefault position={[12, 12, 10]} up={[0, 0, 1]} />
                <OrbitControls
                    makeDefault
                    enableDamping={true}
                    dampingFactor={0.1}
                    rotateSpeed={0.5}
                    panSpeed={0.5}
                    zoomSpeed={0.8}
                />

                <CameraRig view={desiredView} key={viewMetric} />

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
                        if (bond.atom_i >= atoms.length || bond.atom_j >= atoms.length) return null

                        const atomI = atoms[bond.atom_i]
                        const atomJ = atoms[bond.atom_j]

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

                        // SAFETY: Skip zero-length bonds to prevent NaN crashes in Three.js
                        if (Math.abs(start[0] - end[0]) < 1e-5 &&
                            Math.abs(start[1] - end[1]) < 1e-5 &&
                            Math.abs(start[2] - end[2]) < 1e-5) {
                            return null
                        }

                        const bondColor = bond.type === 'dm' ? '#f472b6' : (bond.type === 'anisotropic' ? '#fbbf24' : '#94a3b8')

                        // Check selection
                        // Selection matches if: atom_i, atom_j, AND offset match?
                        // bond key in parent list could be enough, but let's check content if passed selectedBond object
                        let isSelected = false;
                        if (selectedBond) {
                            // Check if this bond matches selectedBond state
                            // Using precise match of i->j and offset
                            if (selectedBond.atom_i === bond.atom_i &&
                                selectedBond.atom_j === bond.atom_j &&
                                (selectedBond.offset || []).join(',') === (bond.offset || []).join(',')) {
                                isSelected = true;
                            }
                        }

                        // DM Vector Logic
                        let dmArrow = null
                        if (bond.type === 'dm' && bond.dm_vector) {
                            const mid = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2]
                            const dmVec = bond.dm_vector
                            dmArrow = (
                                <SpinArrow
                                    position={mid}
                                    direction={dmVec}
                                    length={1.0}
                                    color="#e879f9"
                                />
                            )
                        }

                        if (dimensionality === '2D' && zFilter) {
                            if (Math.abs(posI[2]) > 0.01 && Math.abs(posI[2] - 1.0) > 0.01) return null
                            if (Math.abs(posJExpanded[2]) > 0.01 && Math.abs(posJExpanded[2] - 1.0) > 0.01) return null
                        }

                        const labelOffset = bond.type === 'dm' ? [0, 0.4, 0] : [0, 0, 0]

                        let ghostAtom = null
                        const isExternal = Math.abs(offset[0]) > 0.001 || Math.abs(offset[1]) > 0.001 || Math.abs(offset[2]) > 0.001

                        if (isExternal && bond.atom_j < atoms.length) {
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
                                    isSelected={isSelected}
                                    onClick={() => onBondClick && onBondClick(bond)}
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
