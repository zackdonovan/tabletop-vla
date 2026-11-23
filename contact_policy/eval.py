from pathlib import Path
import json,time,sys
import numpy as np
from roll import *
from learn import Learner
from env import TRAIN,TEST,phrase
from sim import Physical,config,SIZE
ROOT=Path('contact_policy')
def episode(e,p,cfg,i,goal,shown,mode,kind,t,frames=None):
    reset(e,cfg,mode,i)
    if p:p.fresh(phrase(*shown,t))
    o,z=goal; u,v=shown;m=Memory()
    beg=e.objects();xy=beg[o,:2];pad=e.m.geom_pos[e.pad[z],:2].copy()
    log=[];calls=[];render=[];phys=[];acts=[];tip=[]
    last=np.zeros((SIZE,SIZE,3),np.uint8)
    for j in range(CAP):
        s=sensors(e)
        if kind not in ('teacher','blank') and (frames is not None or j==0 or not np.isfinite(p.centers[list((p.goal[0],p.goal[1]+3))]).all()):
            ts=time.perf_counter();last=corrupt(e.rgb(),mode,i);render.append((time.perf_counter()-ts)*1e3)
        if frames is not None:frames.append(e.rgb())
        ts=time.perf_counter()
        a=teacher(s,m,xy,pad) if kind=='teacher' else p.act(last,s)
        if kind=='teacher':m.step(a[3])
        calls.append((time.perf_counter()-ts)*1e3)
        ts=time.perf_counter();move(e,a);phys.append((time.perf_counter()-ts)*1e3)
        coords=e.objects();log.append(np.c_[coords,[e.contact(k) for k in range(3)]])
        acts.append(a.tolist());tip.append(s.tolist())
        if (m if kind=='teacher' else p.memory).stop():break
    traces=np.asarray(log)
    expected=scorer(e,o,z,beg,traces[:,o]);presented=scorer(e,u,v,beg,traces[:,u])
    return {**expected,'shown_success':presented['success'],'steps':j+1,'capped':j+1==CAP,
            'calls_ms':calls,'render_ms':render,'physics_ms':phys,'trace':traces.tolist(),'actions':acts,'proprio':tip}
def run(split):
    assert split in ('dev','test')
    dev=split=='dev';t0=time.perf_counter()
    e=Physical(render=True);p=Learner();rows=[]
    for mode in ('nominal',) if dev else ('nominal','lowfric','offset'):
        for kind in ('teacher','full','shuffle','blank') if mode=='nominal' and not dev else ('teacher','full'):
            tests=6 if dev else 24
            for i in range(tests):
                cfg=config('vla-'+('dev' if dev else 'confirm-v1'),i)
                goals=TRAIN if dev else TEST
                for j,goal in enumerate(goals):
                    shown=goals[(j+1)%len(goals)] if kind=='shuffle' else goal
                    r=episode(e,p,cfg,i,goal,shown,mode,kind,(i+j)%2+(0 if dev else 2))
                    rows.append(dict(mode=mode,policy=kind,scene=i,goal=goal,shown=shown,**r))
            rr=[r for r in rows if r['mode']==mode and r['policy']==kind]
            print(mode,kind,sum(r['success'] for r in rr),len(rr),sum(r['capped'] for r in rr),flush=True)
    e.renderer.close()
    (ROOT/f'{split}-records.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t0,rows=rows)))
if __name__=='__main__':run(sys.argv[1])
