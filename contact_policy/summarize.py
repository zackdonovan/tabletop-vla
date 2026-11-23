import json
from pathlib import Path
import numpy as np
p=Path('contact_policy')
raw=json.loads((p/'test-records.json').read_text());rows=raw['rows'];assert len(rows)==576
out={}
for mode in ('nominal','lowfric','offset'):
 out[mode]={}
 for kind in ('teacher','full','shuffle','blank') if mode=='nominal' else ('teacher','full'):
  v=[r for r in rows if r['mode']==mode and r['policy']==kind];assert len(v)==72
  med=lambda name:float(np.median([x for r in v for x in r[name]])) if any(r[name] for r in v) else None
  out[mode][kind]={'episodes':len(v),'success':sum(r['success'] for r in v),'shown_success':sum(r['shown_success'] for r in v),'capped':sum(r['capped'] for r in v),
   'step_median':float(np.median([r['steps'] for r in v])),'policy_median_ms':med('calls_ms'),'render_median_ms':med('render_ms'),'physics_median_ms':med('physics_ms'),
   'gripper_nonbinary_steps':sum(.05<a[3]<.95 for r in v for a in r['actions']),
   'all_3_goals_success_scenes':sum(all(r['success'] for r in v if r['scene']==i) for i in range(24)),
   'passed':{k:sum(r['checks'][k] for r in v) for k in v[0]['checks']}}
  print(mode,kind,out[mode][kind])
(p/'metrics.json').write_text(json.dumps({'wall_seconds':raw['seconds'],'scores':out},indent=2))
