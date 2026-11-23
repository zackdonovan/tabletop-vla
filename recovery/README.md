# Can the arm recover after a target moves?

Frozen MLP and text/vision weights; compare first-location cache, two-frame RGB tracking, and state expert. The intervention swaps same-category objects or pads; it is a teleport, not a push/contact benchmark. Each row has 72 matched tasks across 24 new layouts. The both condition applies its pad jump after the policy first grasps, so that event is not forced when grasp never occurs.

| Intervention | expert | first-frame cache | tracker | tracker − cache, pp (paired scene 95% interval) |
|---|---:|---:|---:|---:|
| static | 72/72 | 72/72 | 72/72 | +0.0 [+0.0, +0.0] |
| puck | 72/72 | 0/72 | 72/72 | +100.0 [+100.0, +100.0] |
| pad | 72/72 | 0/72 | 72/72 | +100.0 [+100.0, +100.0] |
| both | 72/72 | 0/72 | 72/72 | +100.0 [+100.0, +100.0] |

Intervals resample whole layouts in this limited simulator distribution; degenerate intervals at extremes do not prove a population rate. See metrics.json for event exposure, failure counts and measured per-call compute. This is not learned tracking or RL: replacing cached image coordinates with a live RGB tracker is the intervention, and the learned controller stays frozen. No generalization claims for continuous moving targets, occlusion, real physical contact, or uncontrolled cameras.

Reproduce from the parent directory: PYTHONPATH=. python recovery/run.py; python recovery/summarize.py. Run recovery/test_tracking.py from the parent with PYTHONPATH=.:recovery.
