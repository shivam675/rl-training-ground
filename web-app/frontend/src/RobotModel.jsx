import { useEffect, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { LoadingManager, Mesh, MeshStandardMaterial } from 'three'
import URDFLoader from 'urdf-loader'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js'

/**
 * Loads the same URDF PyBullet simulates and drives it from the streamed state:
 * base pose sets the robot root, joint angles feed urdf-loader's forward
 * kinematics — so link transforms are computed identically to PyBullet.
 *
 * State is applied in useFrame (reading a ref) rather than via React state, so
 * the 60 Hz stream never causes a React re-render.
 */
export default function RobotModel({ body, stateRef }) {
  const [robot, setRobot] = useState(null)

  useEffect(() => {
    const manager = new LoadingManager()
    const loader = new URDFLoader(manager)

    // Resolve ROS `package://<pkg>/...` mesh URLs. The backend tells us where
    // each package's files are served (e.g. { x2_description: '/loaded/<token>' });
    // urdf-loader appends the remaining path.
    if (body.packages && Object.keys(body.packages).length) {
      loader.packages = body.packages
    }

    // URDF mesh filenames are resolved (relative or via packages) to a URL and
    // fetched from the backend. Pick a loader per file extension.
    loader.loadMeshCb = (path, mgr, done) => {
      if (path.startsWith('package://')) {
        // Unresolved package — skip rather than firing a doomed fetch.
        console.warn(`[${body.name}] unresolved package mesh: ${path}`)
        return done(null)
      }
      const ext = path.split('.').pop().toLowerCase()
      const onErr = (e) => {
        console.warn(`[${body.name}] mesh failed: ${path}`, e)
        done(null)
      }
      if (ext === 'stl') {
        new STLLoader(mgr).load(
          path,
          (geom) => {
            const mat = new MeshStandardMaterial({ color: 0xdadde2, metalness: 0.1, roughness: 0.7 })
            done(new Mesh(geom, mat))
          },
          undefined,
          onErr,
        )
      } else if (ext === 'obj') {
        new OBJLoader(mgr).load(path, (obj) => done(obj), undefined, onErr)
      } else if (ext === 'dae') {
        new ColladaLoader(mgr).load(path, (dae) => done(dae.scene), undefined, onErr)
      } else {
        console.warn(`[${body.name}] unsupported mesh type: ${path}`)
        done(null)
      }
    }

    let cancelled = false
    loader.load(
      body.urdf,
      (result) => {
        if (!cancelled) setRobot(result)
      },
      undefined,
      (err) => console.error(`[${body.name}] URDF load failed: ${body.urdf}`, err),
    )
    return () => {
      cancelled = true
    }
  }, [body.urdf, body.name, body.packages])

  const ref = useRef()
  useFrame(() => {
    const root = ref.current
    const st = stateRef.current
    if (!root || !st) return
    const bs = st.bodies[body.name]
    if (!bs) return

    root.position.set(bs.p[0], bs.p[1], bs.p[2])
    root.quaternion.set(bs.q[0], bs.q[1], bs.q[2], bs.q[3]) // PyBullet quat is [x,y,z,w]

    if (bs.j && root.joints) {
      for (const name in bs.j) {
        if (root.joints[name]) root.setJointValue(name, bs.j[name])
      }
    }
  })

  if (!robot) return null
  return <primitive ref={ref} object={robot} />
}
