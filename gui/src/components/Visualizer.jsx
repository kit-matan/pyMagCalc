import React, { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'

function Atom({ position, color = '#38bdf8', label }) {
    return (
        <mesh position={position}>
            <sphereGeometry args={[0.3, 32, 32]} />
            <meshStandardMaterial color={color} roughness={0.3} metalness={0.8} />
        </mesh>
    )
}

function SpinArrow({ position, direction, length = 1.2 }) {
    const dir = new THREE.Vector3(...direction).normalize()
    const pos = new THREE.Vector3(...position)

    return (
        <group position={pos}>
            <primitive
                object={new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), length, '#10b981', 0.2, 0.1)}
            />
        </group>
    )
}

export default function Visualizer({ atoms }) {
    return (
        <div style={{ width: '100%', height: '100%', borderRadius: '16px', overflow: 'hidden' }}>
            <Canvas shadows>
                <PerspectiveCamera makeDefault position={[10, 10, 10]} />
                <OrbitControls makeDefault />
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} castShadow />
                <spotLight position={[-10, 10, 10]} angle={0.15} penumbra={1} intensity={1} />

                <group>
                    {atoms.map((atom, idx) => (
                        <React.Fragment key={idx}>
                            <Atom
                                position={atom.pos}
                                label={atom.label}
                                color={atom.label.includes('Cu') ? '#f87171' : '#38bdf8'}
                            />
                            <SpinArrow
                                position={atom.pos}
                                direction={atom.magmom_classical || [0, 0, 1]}
                            />
                        </React.Fragment>
                    ))}
                </group>

                <gridHelper args={[20, 20, '#1e293b', '#0f172a']} />
            </Canvas>
        </div>
    )
}
