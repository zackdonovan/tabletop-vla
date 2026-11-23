from pathlib import Path
import json,numpy as np
p=Path("contact");data=json.loads((p/'records.json').read_text());rows=data['records']
key=lambda r:(r['scene'],r['object'],r['pad'])
a=[r for r in rows if r['squeeze']];b=[r for r in rows if not r['squeeze']]
assert len(a)==len(b)==72 and list(map(key,a))==list(map(key,b))
stats={}
for label,r in [('contact_expert',a),('open_finger_control',b)]:
    def get(k):return np.array([x[k] for x in r])
    s={"games":len(r),"successes":int(get('success').sum()),"layouts":len(set(x['scene'] for x in r)),
       "checks_passed":{k:sum(x['checks'][k] for x in r) for k in r[0]['checks']}}
    for k in ['lift_m','carry_m','dist_zone_m','other_object_displacement_m','contact_lift_samples','seconds']:
        v=get(k);s[k]={"min":float(v.min()),"median":float(np.median(v)),"max":float(v.max())}
    stats[label]=s
(p/'metrics.json').write_text(json.dumps({'total_wall_seconds':data['wall_seconds'],'scores':stats},indent=2))
for k,v in stats.items():print(k,v)
