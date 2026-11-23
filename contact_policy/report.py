import json
from pathlib import Path
import numpy as np
p=Path('contact_policy'); metrics=json.loads((p/'metrics.json').read_text());s=metrics['scores']
train=json.loads((p/'training.json').read_text());allrows=json.loads((p/'test-records.json').read_text())['rows']
r=[f"""# Camera and language control with real simulated grasp contacts

A MuJoCo robot learns which puck to pick up, which pad to place it on, and how to move and squeeze. Unlike the earlier latch benchmark, its pucks are independent free rigid bodies. The arm grips them between opposed friction fingers, lifts, carries, and releases on the floor. No attachment or object teleport exists in this branch.

## What was learned

- RGB: a seven-class chromatic prototype classifier supervised on MuJoCo segmentation for training only. Known camera intrinsics turn the learned pixel detections into planar coordinates. It caches reliable centers because targets are stationary until touch, not because the simulator provides their positions.
- Text: a learned bag-of-words lexical model from order-insensitive commands in two templates. It recombines known color words, not open-ended language.
- Action: a 96×96 tanh MLP clones expert examples and emits continuous delta X/Y/Z and finger aperture commands. Position targets are fixed inverse-kinematics and servo logic. Policy inputs are detected target/zone coordinates, wrist XYZ, two finger encoder readings and a short history of its own gripper commands. Ground-truth target positions, contacts, simulator state and teacher phases never reach the learned controller. Control is deterministic.

Trained from **{train['episodes']} physical-simulation expert demonstrations**, **{train['rows']:,} control samples**, with modest executed-action noise. The teacher successfully completed all training demos. The regression optimizer reached its fixed cap of {train['epochs']} epochs rather than its numerical stopping tolerance; this is a single development-validated checkpoint, not a claim of optimizer convergence. Training wall time {train['seconds']:.1f} s including data generation. Nominal development: 36/36 learned, 36/36 teacher.

## Frozen closed-loop evaluation

24 fresh layouts × 3 withheld color pairs with two withheld templates. Red→yellow, green→purple, blue→cyan are absent during training. Within each layout each condition uses the same geometry. Low-friction and altered-start cases also corrupt RGB with a fixed channel shift and Gaussian pixel noise. Corruption is bundled in those cases, so their effects are not identified separately. The wrist is displaced only at initialization; there is no hidden simulator-state correction during runs. Success includes 7 cm lift, bilateral force-bearing finger contact at altitude, carry, supported and stable release inside the pad tolerance, and no large bystander-object displacement.

| Condition | policy | requested goal | shown goal | capped |
|---|---|---:|---:|---:|"""]
for mode,ps in s.items():
 for name,x in ps.items():
  r.append(f"| {mode} | {name} | {x['success']}/{x['episodes']} | {x['shown_success']}/{x['episodes']} | {x['capped']} |")
nom=s['nominal']['full']
r.append(f"""
A wrong-command intervention changes the shown instruction without changing the scene or score target. Check shown-goal success to distinguish grounding from failure. Blank RGB is a direct inference intervention using the same learned weights, not a retrained no-image model. No unseen colors, unseen object shapes, continuous target movement, or real-world recovery claims. The test has 24 independent layouts, not 72 independent scenes per condition.

Median nominal learned trajectory: {nom['step_median']:.0f} commands, {nom['policy_median_ms']:.3f} ms per policy call, {nom['physics_median_ms']:.3f} ms per simulator step of 50 ms modeled time. The median call largely uses cached RGB positions; initial render measured separately at {nom['render_median_ms']:.2f} ms. The actual gripper control is real-valued; {nom['gripper_nonbinary_steps']} nominal actions used an interior command between .05 and .95. Wall times are machine-specific.

## Artifacts / reproduction

The NPZ checkpoint contains only plain numeric/string arrays, no executable pickle. Demonstration file stores features, reference actions, per-example weights and one sample RGB; the complete image streams can be regenerated with simulator code. Evaluation logs include all Cartesian/gripper commands, proprioception, target motion and contact counts. The visual demo comes from replaying the tested checkpoint. Earlier static, tracking, and physical expert modules remain byte-for-byte unchanged.

```
export PYTHONPATH=.:contact:contact_policy MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python contact_policy/learn.py
python contact_policy/eval.py dev
python contact_policy/eval.py test
python contact_policy/summarize.py
python contact_policy/report.py
python -m unittest -v contact_policy/test_contact_policy.py
python contact_policy/demo.py
```

Run from project root after installing requirements.txt. To verify the archived checkpoint rather than retrain, first run `sha256sum -c contact_policy/frozen.sha256`. This is a CPU-only contact-VLA benchmark, not a pretrained transformer or an RL result. No hardware run or deployment.

## Next rigorous milestone

Stress the essential shortcuts: hold out shape/color binding and camera calibration, use a learned fused encoder with image-space augmentation, and test recovery after incidental object motion or a failed grasp. Compare imitation to DAgger-style corrections or explicit RL with equal simulator and demonstration budgets only after fixing the test plan. The current scene family is intentionally simple and the gripper is idealized.
""")
(p/'README.md').write_text('\n'.join(r)+'\n')
