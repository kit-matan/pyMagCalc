import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import * as THREE from 'three';

const Atom = ({ position, color = "#333", size = 0.3 }) => {
    return (
        <mesh position={position}>
            <sphereGeometry args={[size, 32, 32]} />
            <meshStandardMaterial color={color} />
        </mesh>
    );
};

const SpinArrow = ({ position, vector, length = 1.0, color = "#ff0000" }) => {
    // vector is [sx, sy, sz]
    const origin = new THREE.Vector3(...position);
    const dir = new THREE.Vector3(...vector).normalize();

    // Create quaternion to rotate from default UP (0,1,0) to target direction
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    // Arrow cylinder length
    const shaftLen = length * 0.7;
    const headLen = length * 0.3;
    const shaftRadius = 0.05;
    const headRadius = 0.15;

    return (
        <group position={position} quaternion={quaternion}>
            {/* Shaft shifted up by half its length so it starts at origin */}
            <mesh position={[0, shaftLen / 2, 0]}>
                <cylinderGeometry args={[shaftRadius, shaftRadius, shaftLen, 12]} />
                <meshStandardMaterial color={color} />
            </mesh>
            {/* Head at the end of shaft */}
            <mesh position={[0, shaftLen + headLen / 2, 0]}>
                <coneGeometry args={[headRadius, headLen, 12]} />
                <meshStandardMaterial color={color} />
            </mesh>
        </group>
    );
};

const StructureScene = ({ data }) => {
    const { atoms, vectors, energy } = data;

    // Safety check
    if (!atoms || atoms.length === 0) return null;

    // Calculate center to center the camera
    const center = atoms.reduce(
        (acc, pos) => [acc[0] + pos[0], acc[1] + pos[1], acc[2] + pos[2]],
        [0, 0, 0]
    ).map(val => val / atoms.length);

    return (
        <>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <pointLight position={[-10, -10, -10]} intensity={0.5} />

            <group position={[-center[0], -center[1], -center[2]]}>
                {atoms.map((pos, idx) => (
                    <group key={`atom-${idx}`}>
                        <Atom position={pos} />
                        <SpinArrow position={pos} vector={vectors[idx]} length={1.5} />
                    </group>
                ))}

                {/* Draw unit cell box or axes helper if needed */}
                <gridHelper args={[20, 20, 0xdddddd, 0xeeeeee]} position={[0, -2, 0]} />
                <axesHelper args={[2]} />
            </group>

            <OrbitControls makeDefault />
        </>
    );
};

export default function MagneticStructureViewer({ data, style }) {
    if (!data) return null;

    return (
        <div style={{ width: '100%', height: '400px', position: 'relative', borderRadius: '8px', overflow: 'hidden', ...style }}>
            <Canvas camera={{ position: [5, 5, 8], fov: 50 }}>
                <StructureScene data={data} />
            </Canvas>
            <div style={{
                position: 'absolute',
                bottom: '10px',
                left: '10px',
                background: 'rgba(255,255,255,0.8)',
                padding: '8px',
                borderRadius: '4px',
                fontSize: '12px',
                pointerEvents: 'none'
            }}>
                <strong>Interactive 3D View</strong><br />
                Left Click: Rotate<br />
                Right Click: Pan<br />
                Scroll: Zoom
            </div>
        </div>
    );
}
