from pathlib import Path
import json,time
from env import TEST,phrase,seed
from sim import Physical,config,place
import numpy as np

p=Path('contact')
def run():
    t=time.perf_counter()
    e=Physical()
    out=[]
    for i in range(24):
        cfg=config('confirm-v1',i)
        for j,(o,z) in enumerate(TEST):
            for squeeze in (True,False):
                e.reset(cfg)
                result=place(e,o,z,squeeze=squeeze)
                assert all(np.isfinite([result[k] for k in ('lift_m','carry_m','dist_zone_m','seconds')]))
                out.append(dict(scene=i,layout_seed=seed('contact-confirm-v1',i),physical_seed=seed('contact-physical-confirm-v1',i),object=o,pad=z,command=phrase(o,z,2+(i+j)%2),squeeze=squeeze,**result))
    (p/'records.json').write_text(json.dumps(dict(records=out,wall_seconds=time.perf_counter()-t,trace_columns=['stage','puck_x_m','puck_y_m','puck_z_m','finger_contact_count'])))
    for s in (True,False):
        rows=[r for r in out if r['squeeze']==s]
        print(s,sum(r['success'] for r in rows),len(rows), flush=True)
if __name__=='__main__':run()
