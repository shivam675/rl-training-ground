import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import RobotModel from './RobotModel.jsx'

/**
 * PyBullet is Z-up; three.js is Y-up. Rather than transform every streamed pose,
 * we put all simulation content under one group rotated -90° about X. Inside that
 * group we work in PyBullet's native frame (Z-up), so streamed positions/quats
 * apply verbatim, while the camera + OrbitControls stay in standard three Y-up.
 *
 * FOV 60 matches computeProjectionMatrixFOV(60, ...) in the Qt viewer.
 */
export default function Viewer({ scene, stateRef }) {
  return (
    <Canvas
      camera={{ fov: 60, near: 0.05, far: 200, position: [2.2, 1.6, 2.2] }}
      dpr={[1, 2]}
    >
      <color attach="background" args={['#202632']} />

      {/* Clean, geometry-accurate lighting (no hard shadows). */}
      <hemisphereLight args={['#ffffff', '#3a4150', 0.9]} />
      <ambientLight intensity={0.25} />
      <directionalLight position={[6, 10, 4]} intensity={1.15} />
      <directionalLight position={[-5, 4, -3]} intensity={0.35} />

      <group rotation={[-Math.PI / 2, 0, 0]}>
        <Ground />
        {scene?.bodies?.map((body) => (
          <RobotModel key={body.name} body={body} stateRef={stateRef} />
        ))}
      </group>

      <OrbitControls makeDefault target={[0, 0.3, 0]} enablePan />
    </Canvas>
  )
}

/** Flat ground at z=0 in PyBullet's frame (matches where plane.urdf sits). */
function Ground() {
  return (
    <group>
      {/* planeGeometry lies in XY with +Z normal — i.e. "up" in PyBullet's frame. */}
      <mesh receiveShadow>
        <planeGeometry args={[60, 60]} />
        <meshStandardMaterial color="#39414f" roughness={1} metalness={0} />
      </mesh>
      {/* gridHelper is XZ by default; rotate into the XY plane and lift slightly. */}
      <gridHelper
        args={[60, 60, '#5b6577', '#2b3240']}
        rotation={[Math.PI / 2, 0, 0]}
        position={[0, 0, 0.001]}
      />
    </group>
  )
}
