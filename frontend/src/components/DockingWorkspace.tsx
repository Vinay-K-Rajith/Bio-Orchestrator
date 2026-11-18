import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { DockInteraction, DockPose, DockResponse, EgfrPdbId, Lead, TargetMetadata } from '@/types/api'
import { fixed } from '@/lib/format'
import { useTheme } from '@/lib/theme'
import { SectionTitle } from './primitives'

declare global {
  interface Window {
    $3Dmol?: any
  }
}

const REFERENCE_DRUGS: ReadonlyArray<{
  id: string
  name: string
  gen: string
  recommendedPdb: EgfrPdbId
  smiles: string
}> = [
  {
    id: 'CHEMBL553',
    name: 'Erlotinib (1st Gen WT)',
    gen: '1st Gen',
    recommendedPdb: '1M17',
    smiles: 'COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC',
  },
  {
    id: 'CHEMBL939',
    name: 'Gefitinib (1st Gen)',
    gen: '1st Gen',
    recommendedPdb: '2ITY',
    smiles: 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
  },
  {
    id: 'CHEMBL3353410',
    name: 'Osimertinib (3rd Gen Covalent)',
    gen: '3rd Gen',
    recommendedPdb: '4ZAU',
    smiles: 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1',
  },
  {
    id: 'CHEMBL3989924',
    name: 'EAI045 (4th Gen Allosteric)',
    gen: '4th Gen',
    recommendedPdb: '5D41',
    smiles: 'O=C(NC(c1ccccc1)c1nccs1)c1cc2ccccc2n1CC(=O)Nc1ccc(F)cc1O',
  },
]

const INTERACTION_COLORS: Record<DockInteraction['type'], { hex: string; badge: string; label: string }> = {
  hbond: { hex: '#facc15', badge: 'bg-warn-soft text-warn', label: 'H-Bond' },
  pi_stacking: { hex: '#c084fc', badge: 'bg-info-soft text-info', label: 'π-Stacking' },
  pi_cation: { hex: '#a855f7', badge: 'bg-info-soft text-info', label: 'Cation-π' },
  hydrophobic: { hex: '#38bdf8', badge: 'bg-accent-soft text-accent', label: 'Hydrophobic' },
  covalent_proximity: { hex: '#2dd4bf', badge: 'bg-good-soft text-good', label: 'Cys797 Proximity' },
  gatekeeper: { hex: '#94a3b8', badge: 'border border-rule bg-sunken text-ink-2', label: 'Gatekeeper' },
  gatekeeper_clash: { hex: '#f87171', badge: 'bg-crit-soft text-crit', label: 'Steric Clash' },
}

interface Props {
  targets: TargetMetadata[]
  selectedPdb: EgfrPdbId
  onSelectPdb: (pdb: EgfrPdbId) => void
  leads: Lead[]
  activeLead: Lead | null
  onSelectLead: (lead: Lead) => void
}

export default function DockingWorkspace({
  targets,
  selectedPdb,
  onSelectPdb,
  leads,
  activeLead,
  onSelectLead,
}: Props) {
  const { theme } = useTheme()
  const [engine, setEngine] = useState<'3dmol' | 'threejs'>('3dmol')
  const [pdbTextCache, setPdbTextCache] = useState<Partial<Record<EgfrPdbId, string>>>({})
  const [dockData, setDockData] = useState<DockResponse | null>(null)
  const [loadingDock, setLoadingDock] = useState(false)
  const [activeMode, setActiveMode] = useState<number>(1)

  // Visual toggles
  const [showSurface, setShowSurface] = useState(true)
  const [showCocrystal, setShowCocrystal] = useState(true)
  const [showInteractions, setShowInteractions] = useState(true)
  const [slabNear, setSlabNear] = useState<number>(-45)
  const [slabFar, setSlabFar] = useState<number>(45)
  const [slicePlane, setSlicePlane] = useState<number>(18)

  // Active ligand state (either the selected Lead from the funnel or a reference TKI)
  const initialLead = activeLead ?? leads[0] ?? null
  const [customMolecule, setCustomMolecule] = useState<{
    id: string
    name: string
    smiles: string
  }>(() =>
    initialLead
      ? {
          id: initialLead.molecule_id,
          name: initialLead.name
            ? `${initialLead.molecule_id} · ${initialLead.name}`
            : `Rank #${initialLead.rank} · ${initialLead.molecule_id}`,
          smiles: initialLead.smiles,
        }
      : {
          id: 'CHEMBL553',
          name: 'Erlotinib (1st Gen WT)',
          smiles: 'COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC',
        },
  )

  // Sync activeLead into customMolecule whenever a lead is clicked or screening finishes
  useEffect(() => {
    const target = activeLead ?? leads[0]
    if (target) {
      setCustomMolecule({
        id: target.molecule_id,
        name: target.name ? `${target.molecule_id} · ${target.name}` : `Rank #${target.rank} · ${target.molecule_id}`,
        smiles: target.smiles,
      })
    }
  }, [activeLead, leads])

  // Fetch PDB coordinates for selected target
  useEffect(() => {
    if (pdbTextCache[selectedPdb]) return
    let cancelled = false
    fetch(`/api/pdb/${selectedPdb}`)
      .then((r) => (r.ok ? r.text() : Promise.reject()))
      .then((txt) => {
        if (!cancelled) {
          setPdbTextCache((prev) => ({ ...prev, [selectedPdb]: txt }))
        }
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [selectedPdb, pdbTextCache])

  // Run 9-mode 3D docking whenever molecule or EGFR PDB target changes
  useEffect(() => {
    let cancelled = false
    setLoadingDock(true)
    fetch('/api/dock', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        smiles: customMolecule.smiles,
        pdb_id: selectedPdb,
        n_poses: 9,
        molecule_id: customMolecule.id,
      }),
    })
      .then((r) => (r.ok ? (r.json() as Promise<DockResponse>) : Promise.reject()))
      .then((res) => {
        if (!cancelled) {
          setDockData(res)
          setActiveMode(1)
          setLoadingDock(false)
        }
      })
      .catch(() => {
        if (!cancelled) setLoadingDock(false)
      })
    return () => {
      cancelled = true
    }
  }, [customMolecule.smiles, customMolecule.id, selectedPdb])

  const currentPose: DockPose | undefined = useMemo(
    () => dockData?.poses.find((p) => p.mode === activeMode) ?? dockData?.poses[0],
    [dockData, activeMode],
  )

  const targetMeta: TargetMetadata | undefined = useMemo(
    () => targets.find((t) => t.pdb_id === selectedPdb) ?? dockData?.target_info,
    [targets, selectedPdb, dockData],
  )

  // -------------------------------------------------------------------------
  // Engine A: 3Dmol.js Hardware-Accelerated Crystallographic Viewer
  // -------------------------------------------------------------------------
  const molContainerRef = useRef<HTMLDivElement | null>(null)
  const molViewerRef = useRef<any>(null)
  const pdbLoadedRef = useRef<string | null>(null)

  // Initialize / update protein model in 3Dmol.js when PDB or theme changes
  useEffect(() => {
    if (engine !== '3dmol') return
    const el = molContainerRef.current
    const $3Dmol = window.$3Dmol
    const pdbText = pdbTextCache[selectedPdb]
    if (!el || !$3Dmol || !pdbText) return

    const bgColor = theme === 'dark' ? '#0b0e1b' : '#111827'
    if (!molViewerRef.current) {
      molViewerRef.current = $3Dmol.createViewer(el, {
        backgroundColor: bgColor,
        antialias: true,
      })
    } else {
      molViewerRef.current.setBackgroundColor(bgColor)
    }

    const viewer = molViewerRef.current
    viewer.removeAllModels()
    viewer.removeAllSurfaces()
    viewer.removeAllShapes()
    viewer.removeAllLabels()

    // Model 0: EGFR Kinase Domain Crystal Structure
    viewer.addModel(pdbText, 'pdb')
    pdbLoadedRef.current = selectedPdb

    // Style EGFR Protein Ribbon
    viewer.setStyle(
      { model: 0, chain: 'A' },
      { cartoon: { color: 'spectrum', opacity: 0.88 } },
    )

    // Highlight key binding pocket residues (Thr790/Met790, Met793, Cys797, Lys745, Asp855)
    viewer.addStyle(
      { model: 0, resi: [718, 726, 743, 745, 766, 791, 792, 844, 854, 855, 856] },
      { stick: { colorscheme: 'whiteCarbon', radius: 0.14, opacity: 0.85 } },
    )
    viewer.addStyle(
      { model: 0, resi: [790, 793, 797] },
      { stick: { colorscheme: 'cyanCarbon', radius: 0.24 } },
    )

    // Optional Co-crystallized ligand overlay (AQ4 / IRE / YY3 / 57N)
    if (showCocrystal && targetMeta) {
      viewer.setStyle(
        { model: 0, resn: targetMeta.cocrystal_ligand_id },
        { stick: { colorscheme: 'magentaCarbon', radius: 0.12, opacity: 0.55 } },
      )
    } else if (targetMeta) {
      viewer.setStyle({ model: 0, resn: targetMeta.cocrystal_ligand_id }, {})
    }

    // Optional Active Pocket Surface Mesh
    if (showSurface) {
      viewer.addSurface(
        $3Dmol.SurfaceType.VDW,
        { opacity: 0.28, color: '#38bdf8' },
        { model: 0, resi: [718, 726, 743, 745, 790, 791, 792, 793, 796, 797, 844, 854, 855] },
      )
    }

    // Add residue labels for Gatekeeper 790, Hinge Met793, and Covalent Cys797
    const KeyLabels: Array<[number, string, string]> = [
      [790, targetMeta?.gatekeeper_residue.split(' ')[0] ?? 'Thr790', '#38bdf8'],
      [793, 'Met793 (Hinge)', '#facc15'],
      [797, 'Cys797 (Thiol)', '#2dd4bf'],
      [745, 'Lys745', '#a855f7'],
    ]
    for (const [resi, text, col] of KeyLabels) {
      viewer.addLabel(
        text,
        {
          fontSize: 11,
          fontColor: col,
          backgroundColor: '#0f172a',
          backgroundOpacity: 0.78,
          borderColor: col,
          borderThickness: 1,
          inFront: true,
        },
        { model: 0, resi: [resi], atom: 'CA' },
      )
    }

    viewer.zoomTo({ model: 0, resi: [790, 793, 797] })
    viewer.setSlab(slabNear, slabFar)
    viewer.render()
  }, [engine, pdbTextCache, selectedPdb, theme, showSurface, showCocrystal, targetMeta])

  // Update Docked Ligand Pose (Model 1) & Non-Covalent Interaction Cylinders without reloading protein
  useEffect(() => {
    if (engine !== '3dmol') return
    const viewer = molViewerRef.current
    if (!viewer || !currentPose || pdbLoadedRef.current !== selectedPdb) return

    // Remove previous ligand model if present (keep Model 0 = EGFR protein)
    const models = viewer.getModelList ? viewer.getModelList() : []
    if (models.length > 1) {
      for (let i = models.length - 1; i >= 1; i--) {
        viewer.removeModel(models[i])
      }
    }
    viewer.removeAllShapes()

    // Add current docked pose SDF as Model 1
    const ligModel = viewer.addModel(currentPose.sdf, 'sdf')
    ligModel.setStyle({}, {
      stick: { colorscheme: 'greenCarbon', radius: 0.24 },
      sphere: { colorscheme: 'greenCarbon', scale: 0.24 },
    })

    // Render non-covalent interactions as dashed 3D cylinders
    if (showInteractions) {
      for (const ix of currentPose.interactions) {
        const spec = INTERACTION_COLORS[ix.type]
        viewer.addCylinder({
          start: ix.start,
          end: ix.end,
          radius: ix.type === 'hbond' || ix.type === 'covalent_proximity' ? 0.09 : 0.06,
          color: spec.hex,
          dashed: true,
          fromCap: 1,
          toCap: 1,
        })
      }
    }

    viewer.setSlab(slabNear, slabFar)
    viewer.render()
  }, [engine, currentPose, selectedPdb, showInteractions, slabNear, slabFar])

  const focusResidues = useCallback((resiList: number[], near = -45, far = 45) => {
    setSlabNear(near)
    setSlabFar(far)
    if (molViewerRef.current && engine === '3dmol') {
      molViewerRef.current.zoomTo({ model: 0, resi: resiList }, 600)
      molViewerRef.current.setSlab(near, far)
      molViewerRef.current.render()
    }
  }, [engine])

  // -------------------------------------------------------------------------
  // Engine B: Custom Three.js InstancedMesh + Pocket Hull & Local Clipping
  // -------------------------------------------------------------------------
  const threeContainerRef = useRef<HTMLDivElement | null>(null)
  const clipPlaneRef = useRef<THREE.Plane>(new THREE.Plane(new THREE.Vector3(0, 0, -1), 18))

  useEffect(() => {
    clipPlaneRef.current.constant = slicePlane
  }, [slicePlane])

  useEffect(() => {
    if (engine !== 'threejs') return
    const container = threeContainerRef.current
    if (!container || !currentPose || !dockData) return

    const width = container.clientWidth || 800
    const height = container.clientHeight || 520

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(theme === 'dark' ? '#0b0e1b' : '#111827')

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000)
    camera.position.set(0, 4, 24)

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.localClippingEnabled = true
    container.replaceChildren(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.78))
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.25)
    dirLight1.position.set(14, 22, 18)
    scene.add(dirLight1)
    const dirLight2 = new THREE.DirectionalLight(0x60a5fa, 0.55)
    dirLight2.position.set(-16, -12, -14)
    scene.add(dirLight2)

    const cx = dockData.pocket_center.x
    const cy = dockData.pocket_center.y
    const cz = dockData.pocket_center.z
    const clipPlanes = [clipPlaneRef.current]

    // 1. Batch render Docked Ligand Atoms via THREE.InstancedMesh
    const ligAtoms = currentPose.atoms
    if (ligAtoms.length > 0) {
      const sphereGeo = new THREE.SphereGeometry(1, 20, 20)
      const atomMat = new THREE.MeshStandardMaterial({
        roughness: 0.25,
        metalness: 0.12,
        clippingPlanes: clipPlanes,
      })
      const instancedAtoms = new THREE.InstancedMesh(sphereGeo, atomMat, ligAtoms.length)
      const dummy = new THREE.Object3D()
      ligAtoms.forEach((a, i) => {
        dummy.position.set(a.x - cx, a.y - cy, a.z - cz)
        dummy.scale.setScalar(a.radius * 1.15)
        dummy.updateMatrix()
        instancedAtoms.setMatrixAt(i, dummy.matrix)
        instancedAtoms.setColorAt(i, new THREE.Color(a.color))
      })
      instancedAtoms.instanceMatrix.needsUpdate = true
      if (instancedAtoms.instanceColor) instancedAtoms.instanceColor.needsUpdate = true
      scene.add(instancedAtoms)
    }

    // 2. Batch render Docked Ligand Bonds via THREE.InstancedMesh
    const ligBonds = currentPose.bonds
    if (ligBonds.length > 0) {
      const cylGeo = new THREE.CylinderGeometry(0.11, 0.11, 1, 12)
      const bondMat = new THREE.MeshStandardMaterial({
        color: '#34d399',
        roughness: 0.35,
        metalness: 0.1,
        clippingPlanes: clipPlanes,
      })
      const instancedBonds = new THREE.InstancedMesh(cylGeo, bondMat, ligBonds.length)
      const dummy = new THREE.Object3D()
      const up = new THREE.Vector3(0, 1, 0)
      ligBonds.forEach((b, i) => {
        const p1 = new THREE.Vector3(b.start.x - cx, b.start.y - cy, b.start.z - cz)
        const p2 = new THREE.Vector3(b.end.x - cx, b.end.y - cy, b.end.z - cz)
        const dir = new THREE.Vector3().subVectors(p2, p1)
        const len = dir.length()
        dummy.position.copy(p1).addScaledVector(dir, 0.5)
        dummy.quaternion.setFromUnitVectors(up, dir.normalize())
        dummy.scale.set(b.order > 1 ? 1.35 : 1.0, len, b.order > 1 ? 1.35 : 1.0)
        dummy.updateMatrix()
        instancedBonds.setMatrixAt(i, dummy.matrix)
      })
      instancedBonds.instanceMatrix.needsUpdate = true
      scene.add(instancedBonds)
    }

    // 3. Batch render EGFR Key Pocket Residue Atoms via THREE.InstancedMesh
    const pocketPts = dockData.pocket_mesh_points
    if (pocketPts.length > 0) {
      const pGeo = new THREE.SphereGeometry(0.26, 12, 12)
      const pMat = new THREE.MeshStandardMaterial({
        roughness: 0.45,
        metalness: 0.05,
        transparent: true,
        opacity: 0.78,
        clippingPlanes: clipPlanes,
      })
      const instancedPocket = new THREE.InstancedMesh(pGeo, pMat, pocketPts.length)
      const dummy = new THREE.Object3D()
      pocketPts.forEach((pt, i) => {
        dummy.position.set(pt.x - cx, pt.y - cy, pt.z - cz)
        const isKey = pt.resi === 790 || pt.resi === 793 || pt.resi === 797
        dummy.scale.setScalar(isKey ? 1.35 : 0.85)
        dummy.updateMatrix()
        instancedPocket.setMatrixAt(i, dummy.matrix)
        instancedPocket.setColorAt(
          i,
          new THREE.Color(isKey && pt.elem === 'C' ? '#38bdf8' : pt.color),
        )
      })
      instancedPocket.instanceMatrix.needsUpdate = true
      if (instancedPocket.instanceColor) instancedPocket.instanceColor.needsUpdate = true
      scene.add(instancedPocket)
    }

    // 4. Binding Pocket Translucent Surface Mesh (with local clipping plane!)
    if (showSurface && pocketPts.length > 0) {
      const surfGeo = new THREE.SphereGeometry(1.85, 16, 16)
      const surfMat = new THREE.MeshStandardMaterial({
        color: '#38bdf8',
        transparent: true,
        opacity: 0.16,
        roughness: 0.2,
        metalness: 0.1,
        depthWrite: false,
        side: THREE.DoubleSide,
        clippingPlanes: clipPlanes,
      })
      const keyPts = pocketPts.filter((p) => p.atom === 'CA' || p.atom === 'CB' || p.resi === 790 || p.resi === 793 || p.resi === 797)
      const surfInst = new THREE.InstancedMesh(surfGeo, surfMat, keyPts.length)
      const dummy = new THREE.Object3D()
      keyPts.forEach((pt, i) => {
        dummy.position.set(pt.x - cx, pt.y - cy, pt.z - cz)
        dummy.updateMatrix()
        surfInst.setMatrixAt(i, dummy.matrix)
      })
      surfInst.instanceMatrix.needsUpdate = true
      scene.add(surfInst)
    }

    // 5. Non-Covalent Interaction Dashed Lines (THREE.LineDashedMaterial)
    if (showInteractions) {
      for (const ix of currentPose.interactions) {
        const spec = INTERACTION_COLORS[ix.type]
        const points = [
          new THREE.Vector3(ix.start.x - cx, ix.start.y - cy, ix.start.z - cz),
          new THREE.Vector3(ix.end.x - cx, ix.end.y - cy, ix.end.z - cz),
        ]
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points)
        const lineMat = new THREE.LineDashedMaterial({
          color: spec.hex,
          dashSize: 0.35,
          gapSize: 0.2,
          clippingPlanes: clipPlanes,
        })
        const line = new THREE.Line(lineGeo, lineMat)
        line.computeLineDistances()
        scene.add(line)
      }
    }

    let frameId = 0
    const animate = () => {
      frameId = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    const onResize = () => {
      if (!container) return
      const w = container.clientWidth || 800
      const h = container.clientHeight || 520
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      cancelAnimationFrame(frameId)
      controls.dispose()
      renderer.dispose()
    }
  }, [engine, currentPose, dockData, theme, showSurface, showInteractions])

  return (
    <section id="docking-workspace" className="surface overflow-hidden p-6 sm:p-7">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <SectionTitle
          icon={
            <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M10 2.2 17 6.2v7.6l-7 4-7-4V6.2l7-4Z" strokeLinejoin="round" />
              <path d="M10 10v7.8M10 10 3 6.2M10 10l7-3.8" strokeLinejoin="round" />
            </svg>
          }
          aside={
            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex rounded-lg border border-rule bg-sunken p-0.5">
                <button
                  type="button"
                  onClick={() => setEngine('3dmol')}
                  className={`rounded-md px-2.5 py-1 text-micro font-medium transition-colors ${
                    engine === '3dmol' ? 'bg-accent text-accent-on' : 'text-ink-2 hover:text-ink'
                  }`}
                >
                  3Dmol.js Ribbon & Surface
                </button>
                <button
                  type="button"
                  onClick={() => setEngine('threejs')}
                  className={`rounded-md px-2.5 py-1 text-micro font-medium transition-colors ${
                    engine === 'threejs' ? 'bg-accent text-accent-on' : 'text-ink-2 hover:text-ink'
                  }`}
                >
                  Three.js InstancedMesh + Clipping
                </button>
              </div>
            </div>
          }
        >
          3D EGFR Binding Pocket &amp; Conformer Docking Studio
        </SectionTitle>
      </div>

      {/* Step 1: EGFR Crystal Conformation Selector Table / Cards */}
      <div className="mt-2 grid gap-2.5 sm:grid-cols-2 lg:grid-cols-5">
        {(targets.length > 0 ? targets : []).map((t) => {
          const active = t.pdb_id === selectedPdb
          return (
            <button
              key={t.pdb_id}
              type="button"
              onClick={() => onSelectPdb(t.pdb_id)}
              className={`group flex flex-col justify-between rounded-panel border p-3 text-left transition-all ${
                active
                  ? 'border-accent bg-accent-soft/55 shadow-card'
                  : 'border-rule bg-sunken/60 hover:border-rule-strong hover:bg-panel'
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2">
                  <span className="numeral shrink-0 whitespace-nowrap rounded bg-accent px-1.5 py-0.5 text-micro font-semibold text-accent-on">
                    PDB {t.pdb_id}
                  </span>
                  <span className="truncate text-micro font-medium text-ink-3">{t.generation}</span>
                </div>
                <div className="mt-1.5 text-2xs font-medium leading-snug text-ink">
                  {t.conformation}
                </div>
              </div>
              <div className="mt-2 border-t border-rule/80 pt-1.5 text-micro leading-relaxed text-ink-3">
                Co-crystal: <span className="font-medium text-ink-2">{t.cocrystal_name}</span>
              </div>
            </button>
          )
        })}
      </div>

      {/* Step 2: Candidate / Reference Drug Selector Bar */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-panel border border-rule bg-sunken px-3.5 py-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="label mr-1">Docked Ligand</span>
          {leads.length > 0 &&
            leads.slice(0, 6).map((lead) => {
              const isSelected = customMolecule.id === lead.molecule_id
              return (
                <button
                  key={lead.molecule_id}
                  type="button"
                  onClick={() => onSelectLead(lead)}
                  className={`chip py-1 ${
                    isSelected ? 'border-accent bg-accent text-accent-on' : ''
                  }`}
                >
                  <span className="numeral font-medium">#{lead.rank}</span> {lead.molecule_id}
                  {lead.docking_score_kcal != null && (
                    <span className="numeral ml-1.5 opacity-80">
                      ({fixed(lead.docking_score_kcal, 1)} kcal/mol)
                    </span>
                  )}
                </button>
              )
            })}
          <span className="mx-1 text-ink-3">|</span>
          <span className="label mr-1">Clinical Controls</span>
          {REFERENCE_DRUGS.map((ref) => {
            const isSelected = customMolecule.id === ref.id
            return (
              <button
                key={ref.id}
                type="button"
                onClick={() => {
                  setCustomMolecule({ id: ref.id, name: ref.name, smiles: ref.smiles })
                }}
                className={`chip py-1 ${
                  isSelected ? 'border-accent bg-accent text-accent-on' : ''
                }`}
                title={`Dock ${ref.name} (recommended PDB: ${ref.recommendedPdb})`}
              >
                {ref.name}
              </button>
            )
          })}
        </div>

        <a
          href={`/api/docking/top_hit.sdf?pdb_id=${selectedPdb}&smiles=${encodeURIComponent(customMolecule.smiles)}`}
          download={`${customMolecule.id}_${selectedPdb}_mode1.sdf`}
          className="btn-quiet py-1 text-micro"
        >
          Download Pose (.SDF)
        </a>
      </div>

      {/* Step 3: Split Viewport (Left 3D WebGL Canvas + Right Pose & Interaction Inspector) */}
      <div className="mt-4 grid gap-5 lg:grid-cols-12">
        {/* 3D WebGL Canvas Column */}
        <div className="flex flex-col lg:col-span-8">
          <div className="relative h-[490px] w-full overflow-hidden rounded-panel border border-rule-strong bg-[#111827]">
            {engine === '3dmol' ? (
              <div ref={molContainerRef} className="h-full w-full" />
            ) : (
              <div ref={threeContainerRef} className="h-full w-full" />
            )}

            {/* Top-left floating HUD inside WebGL viewport */}
            <div className="pointer-events-none absolute left-3.5 top-3.5 z-10 flex flex-col gap-1.5">
              <div className="inline-flex items-center gap-2 rounded-md border border-white/15 bg-slate-950/80 px-3 py-1.5 text-xs text-slate-100 backdrop-blur-sm">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                <span className="font-medium">EGFR {selectedPdb}</span>
                <span className="text-slate-400">&middot;</span>
                <span className="numeral text-emerald-300">{customMolecule.name}</span>
                <span className="text-slate-400">&middot;</span>
                <span className="numeral text-amber-300">Mode {activeMode}/9</span>
              </div>
              {currentPose && (
                <div className="inline-flex flex-wrap items-center gap-2 rounded-md border border-white/10 bg-slate-950/75 px-3 py-1 text-micro text-slate-200 backdrop-blur-sm">
                  <span>
                    &Delta;G:{' '}
                    <strong className="numeral text-emerald-300">
                      {fixed(currentPose.delta_g_kcal, 2)} kcal/mol
                    </strong>
                  </span>
                  <span>&middot;</span>
                  <span>
                    Met793 Hinge:{' '}
                    <strong className={currentPose.hinge_hbond_met793 ? 'text-amber-300' : 'text-slate-400'}>
                      {currentPose.hinge_hbond_met793 ? 'H-Bonded' : 'No H-Bond'}
                    </strong>
                  </span>
                  {currentPose.cys797_distance_a != null && (
                    <>
                      <span>&middot;</span>
                      <span>
                        Cys797:SG:{' '}
                        <strong className="numeral text-teal-300">
                          {fixed(currentPose.cys797_distance_a, 2)} &Aring;
                        </strong>
                      </span>
                    </>
                  )}
                  {currentPose.gatekeeper_distance_a != null && (
                    <>
                      <span>&middot;</span>
                      <span>
                        Gatekeeper 790:{' '}
                        <strong
                          className={`numeral ${
                            currentPose.gatekeeper_clash ? 'text-rose-400' : 'text-sky-300'
                          }`}
                        >
                          {fixed(currentPose.gatekeeper_distance_a, 2)} &Aring;
                          {currentPose.gatekeeper_clash ? ' (CLASH)' : ''}
                        </strong>
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>

            {loadingDock && (
              <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/65 backdrop-blur-xs">
                <div className="rounded-panel border border-white/15 bg-slate-900 px-4 py-3 text-xs text-slate-100 shadow-lift">
                  Generating 9 ETKDGv3 conformers &amp; aligning in EGFR ({selectedPdb}) pocket&hellip;
                </div>
              </div>
            )}

            {/* Bottom Legend Bar inside Viewport */}
            <div className="pointer-events-none absolute bottom-3 left-3.5 right-3.5 z-10 flex flex-wrap items-center justify-between gap-2 rounded-md border border-white/10 bg-slate-950/75 px-3 py-1.5 text-micro text-slate-300 backdrop-blur-sm">
              <div className="flex flex-wrap items-center gap-3">
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" /> Docked Ligand
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-full bg-sky-400" /> Thr790 / Met793 / Cys797
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-full bg-fuchsia-400" /> Co-Crystal ({targetMeta?.cocrystal_ligand_id})
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-full bg-amber-300" /> H-Bond (&le;3.5 &Aring;)
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-full bg-purple-400" /> &pi;-Stacking
                </span>
              </div>
              <span className="text-slate-400">Drag to rotate &middot; Scroll to zoom</span>
            </div>
          </div>

          {/* Viewport Camera & Pocket Slice / Slab Controls */}
          <div className="mt-3 grid gap-3 rounded-panel border border-rule bg-sunken p-3 sm:grid-cols-2">
            <div>
              <div className="label mb-2">Camera Focus &amp; Layer Toggles</div>
              <div className="flex flex-wrap gap-1.5">
                <button
                  type="button"
                  onClick={() => focusResidues([790, 793, 797], -45, 45)}
                  className="btn-quiet px-2.5 py-1 text-micro"
                >
                  Active Cleft
                </button>
                <button
                  type="button"
                  onClick={() => focusResidues([791, 792, 793], -12, 12)}
                  className="btn-quiet px-2.5 py-1 text-micro"
                >
                  Met793 Hinge
                </button>
                <button
                  type="button"
                  onClick={() => focusResidues([790, 745, 766], -12, 12)}
                  className="btn-quiet px-2.5 py-1 text-micro"
                >
                  Gatekeeper 790
                </button>
                <button
                  type="button"
                  onClick={() => focusResidues([797, 796], -12, 12)}
                  className="btn-quiet px-2.5 py-1 text-micro"
                >
                  Cys797 Thiol
                </button>
                <label className="ml-1 inline-flex cursor-pointer items-center gap-1.5 text-micro text-ink-2">
                  <input
                    type="checkbox"
                    checked={showSurface}
                    onChange={(e) => setShowSurface(e.target.checked)}
                    className="accent-accent"
                  />
                  Pocket Surface
                </label>
                <label className="inline-flex cursor-pointer items-center gap-1.5 text-micro text-ink-2">
                  <input
                    type="checkbox"
                    checked={showCocrystal}
                    onChange={(e) => setShowCocrystal(e.target.checked)}
                    className="accent-accent"
                  />
                  Co-Crystal Ref
                </label>
                <label className="inline-flex cursor-pointer items-center gap-1.5 text-micro text-ink-2">
                  <input
                    type="checkbox"
                    checked={showInteractions}
                    onChange={(e) => setShowInteractions(e.target.checked)}
                    className="accent-accent"
                  />
                  H-Bonds &amp; Contacts
                </label>
              </div>
            </div>

            <div>
              {engine === '3dmol' ? (
                <>
                  <div className="flex items-center justify-between gap-2">
                    <span className="label">Pocket Slice / Slab Clipping (Near / Far)</span>
                    <span className="numeral shrink-0 whitespace-nowrap text-micro text-ink-2">
                      {slabNear} &Aring; to +{slabFar} &Aring;
                    </span>
                  </div>
                  <div className="mt-2 flex items-center gap-3">
                    <input
                      type="range"
                      min={-50}
                      max={-2}
                      step={1}
                      value={slabNear}
                      aria-label="Slab near clipping plane"
                      onChange={(e) => setSlabNear(Number(e.target.value))}
                      className="w-full accent-accent"
                    />
                    <input
                      type="range"
                      min={2}
                      max={50}
                      step={1}
                      value={slabFar}
                      aria-label="Slab far clipping plane"
                      onChange={(e) => setSlabFar(Number(e.target.value))}
                      className="w-full accent-accent"
                    />
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center justify-between">
                    <span className="label">Three.js Local Clipping Plane (Z-Slice)</span>
                    <span className="numeral text-micro text-ink-2">{slicePlane} &Aring;</span>
                  </div>
                  <div className="mt-2 flex items-center gap-3">
                    <input
                      type="range"
                      min={-10}
                      max={20}
                      step={0.5}
                      value={slicePlane}
                      aria-label="Three.js pocket clipping plane"
                      onChange={(e) => setSlicePlane(Number(e.target.value))}
                      className="w-full accent-accent"
                    />
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Pose Toggle (Modes 1-9), Energy Decomposition & Non-Covalent Interactions */}
        <div className="flex flex-col justify-between space-y-4 lg:col-span-4">
          {/* Pose Toggle (Modes 1 to 9) */}
          <div className="rounded-panel border border-rule bg-sunken p-4">
            <div className="flex items-baseline justify-between">
              <span className="label">Binding Pose Toggle (Modes 1&ndash;9)</span>
              <span className="numeral text-micro text-ink-3">Vina &Delta;G &middot; RMSD</span>
            </div>
            <div className="mt-2.5 grid grid-cols-3 gap-1.5">
              {(dockData?.poses ?? []).map((p) => {
                const selected = p.mode === activeMode
                return (
                  <button
                    key={p.mode}
                    type="button"
                    onClick={() => setActiveMode(p.mode)}
                    className={`flex flex-col rounded-md border px-2.5 py-1.5 text-left transition-all ${
                      selected
                        ? 'border-accent bg-accent text-accent-on shadow-sm'
                        : 'border-rule bg-panel text-ink hover:border-rule-strong'
                    }`}
                  >
                    <div className="flex items-center justify-between text-micro">
                      <span className="font-semibold">Mode {p.mode}</span>
                      <span className={`numeral ${selected ? 'text-accent-on/80' : 'text-ink-3'}`}>
                        {fixed(p.rmsd_ub, 1)}&Aring;
                      </span>
                    </div>
                    <div className="numeral mt-0.5 text-xs font-medium">
                      {fixed(p.delta_g_kcal, 2)} <span className="text-micro opacity-75">kcal</span>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Decomposed Binding Energy Terms */}
          {currentPose && (
            <div className="rounded-panel border border-rule bg-sunken p-4">
              <div className="flex items-baseline justify-between">
                <span className="label">Mode {currentPose.mode} Energy Breakdown</span>
                <span className="numeral text-sm font-semibold text-accent">
                  {fixed(currentPose.delta_g_kcal, 2)} kcal/mol
                </span>
              </div>
              <div className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-1.5 text-micro">
                <div className="flex justify-between">
                  <span className="text-ink-3">Gauss Steric:</span>
                  <span className="numeral text-ink">{fixed(currentPose.energy_terms.gauss_steric, 2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-3">Hydrophobic:</span>
                  <span className="numeral text-ink">{fixed(currentPose.energy_terms.hydrophobic, 2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-3">H-Bonding:</span>
                  <span className="numeral text-good">{fixed(currentPose.energy_terms.hbond, 2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-3">Steric Repulsion:</span>
                  <span
                    className={`numeral ${
                      currentPose.energy_terms.repulsion > 1.5 ? 'text-crit' : 'text-ink'
                    }`}
                  >
                    +{fixed(currentPose.energy_terms.repulsion, 2)}
                  </span>
                </div>
              </div>
              {targetMeta && (
                <p className="mt-2.5 border-t border-rule pt-2 text-micro leading-relaxed text-ink-3">
                  <strong className="font-medium text-ink-2">{targetMeta.pdb_id}:</strong>{' '}
                  {targetMeta.key_residues_summary}
                </p>
              )}
            </div>
          )}

          {/* Non-Covalent 3D Interactions List */}
          <div className="flex-1 rounded-panel border border-rule bg-sunken p-4">
            <div className="flex items-baseline justify-between">
              <span className="label">Non-Covalent Pocket Contacts</span>
              <span className="numeral text-micro text-ink-3">
                {currentPose?.interactions.length ?? 0} detected
              </span>
            </div>
            <ul className="mt-2.5 max-h-[195px] space-y-1.5 overflow-y-auto pr-1">
              {(currentPose?.interactions ?? []).map((ix, i) => {
                const spec = INTERACTION_COLORS[ix.type]
                return (
                  <li
                    key={`${ix.residue}-${ix.prot_atom}-${i}`}
                    onClick={() => focusResidues([ix.resi], -14, 14)}
                    className="flex cursor-pointer items-center justify-between gap-2 rounded-md border border-rule bg-panel px-2.5 py-1.5 text-micro transition-colors hover:border-accent/50"
                    title="Click to zoom 3D camera onto this residue contact"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      <span className={`badge shrink-0 ${spec.badge}`}>{spec.label}</span>
                      <span className="truncate font-medium text-ink">{ix.label}</span>
                    </div>
                    <span className="numeral shrink-0 font-medium text-ink-2">
                      {fixed(ix.distance, 2)} &Aring;
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
