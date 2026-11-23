"""Frozen arm checkpoint challenged by reproducible target displacements."""
from pathlib import Path
import json,time
import numpy as np
from env import Table,scene,phrase,expert,TEST,RANGE,mujoco
from policy import Policy,detect,features,forward

class Track(Policy):
    def reset(self,command,none=False):
        super().reset(command,none)
        self.pending=np.full((6,2),np.nan)
    def act(self,rgb,proprio):
        self.refresh(detect(rgb,self.w["prototype"]),proprio)
        x=features(self.memory,proprio,self.obj,self.zone)
        if x is None:return np.zeros(3)
        output=forward(x,self.w)
        return np.r_[np.clip(output[:2],-1,1)*RANGE,np.clip(output[2],-1,1)]
    def refresh(self,found,proprio):
        for k in range(6):
            if proprio[2] and k<3:continue
            if not np.isfinite(found[k]).all():
                self.pending[k]=np.nan
                continue
            if not np.isfinite(self.memory[k]).all():
                self.memory[k]=found[k];continue
            if np.linalg.norm(found[k]-self.memory[k])>.08:
                if np.linalg.norm(found[k]-self.pending[k])<=.035:
                    self.memory[k]=found[k];self.pending[k]=np.nan
                else:self.pending[k]=found[k]
            else:self.pending[k]=np.nan

def swap(env,k,candidates):
    positions=env.positions()
    j=max((j for j in candidates if j!=k),key=lambda j:np.linalg.norm(positions[j]-positions[k]))
    env.d.mocap_pos[env.mids[k],:2]=positions[j];env.d.mocap_pos[env.mids[j],:2]=positions[k]
    mujoco.mj_forward(env.m,env.d)
    return [k,j,float(np.linalg.norm(positions[j]-positions[k]))]

def run():
    rows=[];env=Table();pols={"expert":None,"cache":Policy(),"track":Track()}
    for condition in ("static","puck","pad","both"):
        for kind,p in pols.items():
            for i in range(24):
                for j,(o,z) in enumerate(TEST):
                    env.reset(scene('motion-confirmation',i),o,z)
                    if p:p.reset(phrase(o,z,2+(i+j)%2))
                    jumps={}; times=[];grasped=False
                    while not env.done:
                        if condition in ("puck","both") and env.steps==1 and env.grasped is None:
                            jumps["puck"]=swap(env,o,range(3))
                        if condition in ("pad","both") and env.grasped is not None and "pad" not in jumps:
                            jumps["pad"]=swap(env,z+3,range(3,6))
                        rgb=env.rgb() if p else None
                        t=time.perf_counter();a=p.act(rgb,env.scalar()) if p else expert(env);times.append((time.perf_counter()-t)*1000)
                        env.action(a);grasped|=env.grasped==o
                    reason="none" if env.success else "wrong_puck" if env.wrong else "timeout" if env.released is None else "wrong_pad"
                    rows.append(dict(condition=condition,policy=kind,scene=i,goal=[o,z],success=bool(env.success),steps=env.steps,grasped=grasped,jumps=jumps,reason=reason,policy_ms=times))
            r=[x for x in rows if x["condition"]==condition and x["policy"]==kind]
            print(condition,kind,sum(r["success"] for r in r),len(r),flush=True)
    env.close();Path('recovery/records.json').write_text(json.dumps(rows))
if __name__=='__main__':run()
