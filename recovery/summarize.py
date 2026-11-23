import json
from pathlib import Path
import numpy as np
p=Path("recovery"); rows=json.loads((p/"records.json").read_text());assert len(rows)==864
out={}; lines=["# Can the arm recover after a target moves?", "", "Frozen MLP and text/vision weights; compare first-location cache, two-frame RGB tracking, and state expert. The intervention swaps same-category objects or pads; it is a teleport, not a push/contact benchmark. Each row has 72 matched tasks across 24 new layouts. The both condition applies its pad jump after the policy first grasps, so that event is not forced when grasp never occurs.", "", "| Intervention | expert | first-frame cache | tracker | tracker − cache, pp (paired scene 95% interval) |", "|---|---:|---:|---:|---:|"]
for c in ['static','puck','pad','both']:
 rs={k:[r for r in rows if r['condition']==c and r['policy']==k] for k in ['expert','cache','track']}
 keys=lambda v:[(r['scene'],r['goal']) for r in v]
 assert all(len(v)==72 and keys(v)==keys(rs['cache']) for v in rs.values())
 rec={}
 for k,v in rs.items():
  pms=[y for x in v for y in x["policy_ms"]]
  rec[k]={'trials':len(v),'success':sum(x['success'] for x in v),'grasped':sum(x['grasped'] for x in v),'puck_jump_applied':sum('puck' in x['jumps'] for x in v),'pad_jump_applied':sum('pad' in x['jumps'] for x in v),'policy_p50_ms':float(np.median(pms)),'policy_p95_ms':float(np.quantile(pms,.95)),'failure':{t:sum(x['reason']==t for x in v) for t in ['wrong_puck','wrong_pad','timeout']}}
 delta=np.array([int(a['success'])-int(b['success']) for a,b in zip(rs['track'],rs['cache'])]).reshape(24,3).mean(1)
 idx=np.random.default_rng(888).integers(0,24,(5000,24)); m=float(delta.mean());lo,hi=np.quantile(delta[idx].mean(1),[.025,.975])
 rec['paired_delta']=[m,float(lo),float(hi)];out[c]=rec
 results=' | '.join(f"{rec[k]['success']}/72" for k in rs)
 lines.append(f"| {c} | {results} | {100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] |")
lines+=["", "Intervals resample whole layouts in this limited simulator distribution; degenerate intervals at extremes do not prove a population rate. See metrics.json for event exposure, failure counts and measured per-call compute. This is not learned tracking or RL: replacing cached image coordinates with a live RGB tracker is the intervention, and the learned controller stays frozen. No generalization claims for continuous moving targets, occlusion, real physical contact, or uncontrolled cameras.", "", "Reproduce from the parent directory: PYTHONPATH=. python recovery/run.py; python recovery/summarize.py. Run recovery/test_tracking.py from the parent with PYTHONPATH=.:recovery."]
(p/"metrics.json").write_text(json.dumps(out,indent=2));(p/"README.md").write_text('\n'.join(lines)+'\n')
