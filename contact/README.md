# Physical grasp and release feasibility

This new branch removes the old proximity latch. MuJoCo simulates five actuated DOFs: two arm joints, a lift axis, and two opposed sliding fingers. Three pucks are free-joint rigid bodies with gravity and randomized radius, height, contact friction and layout. The simulator has **zero mocap bodies and zero constraints attaching any puck to the arm**. Puck poses are initialized once per episode and never assigned during step logic. Fingers grip by opposed normal forces and friction.

The privileged-state expert uses initial target and pad poses to execute up, down, squeeze, lift, carry, lower, release and retreat trajectories. It is an expert for future demonstrations, **not a newly learned camera policy**. The paired control follows the identical pose targets but never squeezes. The success test requires sustained opposed contacts in the air, at least 7 cm lift and 9 cm carry, stable supported placement within 7.5 cm of the pad, fully open fingers with no remaining contact, and no significant displacement of the other pucks.

| Frozen held-out condition | Complete task | Lifted ≥7 cm | Sustained opposed contacts |
|---|---:|---:|---:|
| physical expert | 72/72 | 72/72 | 72/72 |
| same expert motion, fingers held open | 0/72 | 0/72 | 0/72 |

Those are three goals on each of 24 independent fresh layouts, not 72 separate layouts. Median expert lift **12.60 cm**; carry distance **52.85 cm**; final XY error **3.54 mm**. Opposed fingertip load co-occurred in at least 83 sampled lifted states for every expert episode, at 20 ms sample cadence. The open control reached effectively zero lift. Success isn't assurance on untested shapes, camera conditions, hardware, or real manipulation. Pads are visual scoring zones on a planar floor. The idealized arm links do not collide, and contact is confined to the gripper, pucks, and supporting plane.

Median expert+simulation time per episode approximately 66 ms on this CPU, excluding rendering, reset and compile. The simulated sequence spans approximately 6.46 s after reset. Top and oblique demo renders in physical-grasp.gif show actual simulator motion.

## Reproduce

From the project directory with dependencies installed as specified in the parent README:

```
export PYTHONPATH=.:contact MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum -c contact/frozen.sha256
python contact/evaluate.py
python contact/summarize.py
python -m unittest -v contact/test_physical.py
python contact/demo.py
```

Three tests pass: articulated model and no attachments; matched lift/contact/release vs open; repeatable trajectories from the same seed. Baseline and tracker input hashes also still match their original frozen runs. The evaluation writes the complete paired object trajectories and contact counts.

## Next checkpoint

Collect fresh RGB/action trajectories from this expert on train-only layouts, separately lock a future test namespace and randomize support/friction conditions. Train a camera/text policy controlling z and squeeze as well as XY. Use allowed wrist/finger encoders or valid simulated force sensing and verify actual physical lift/release in closed loop. Do not reuse the held-out expert QA scenes for model selection, and do not describe the existing latch-trained policy as a physical grasp policy.
