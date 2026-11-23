import time
import numpy as np
from env import mujoco,ik,seed
from sim import Physical,BASE_Z,FINGER_CLOSE,LIFT_Z,PICK_Z,DT
LIMIT=np.array([.043,.043,.036]); HOLD=4; N=25; CAP=120
def pos(e):return e.wrist()
def sensors(e):return np.r_[pos(e),e.d.qpos[e.joints[-2:]]]
def move(e,a):
    a=np.asarray(a,float)
    if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError("bad action")
    targ=pos(e)+np.clip(a[:3],-1,1)*LIMIT
    targ=np.clip(targ,[.12,-.49,.028],[.91,.49,.185])
    if np.linalg.norm(targ[:2])>.965:targ[:2]*=.965/np.linalg.norm(targ[:2])
    g=np.clip(a[3],0,1)
    e.d.ctrl[:]=[*ik(targ[:2]),targ[2]-BASE_Z,FINGER_CLOSE*g,FINGER_CLOSE*g]
    mujoco.mj_step(e.m,e.d,nstep=N)
class Memory:
    def __init__(self): self.closed=0;self.released=0
    def step(self,g):
        # State captures own command history, not simulation grasp labels or object pose.
        if self.released:self.released+=1
        elif g>.65:self.closed+=1
        elif self.closed>=HOLD and g<.25:self.released=1
    def stop(self):return self.released>=12

def teacher(s,m,obj,pad):
    x=s[:3]
    if m.released: target=np.r_[pad,LIFT_Z];g=0.
    elif m.closed<HOLD:
        dx=np.linalg.norm(x[:2]-obj)
        if m.closed:
            target=np.r_[obj,PICK_Z];g=1.
        elif dx>.019 and x[2]>.085:target=np.r_[obj,LIFT_Z];g=0.
        elif dx>.032:target=np.r_[obj,x[2]];g=0.
        else:target=np.r_[obj,PICK_Z];g=float(x[2]<PICK_Z+.009 and dx<.017)
    else:
        dx=np.linalg.norm(x[:2]-pad)
        if x[2]<.132 and dx>.075:target=np.r_[x[:2],LIFT_Z];g=1.
        elif dx>.025 and x[2]>.105:target=np.r_[pad,LIFT_Z];g=1.
        elif dx>.033:target=np.r_[pad,x[2]];g=1.
        else:target=np.r_[pad,PICK_Z+.006];g=float(not (x[2]<PICK_Z+.017 and dx<.022))
    return np.r_[np.clip((target-x)/LIMIT,-1,1),g]
def signals(s,m,obj,pad):
    if not np.isfinite([*obj,*pad]).all():return None
    d0=(obj-s[:2])/.22;d1=(pad-s[:2])/.22
    return np.r_[d0,d1,np.linalg.norm(d0),np.linalg.norm(d1),s[2]/.16,s[3:]/.054,
                 min(m.closed,HOLD)/HOLD,bool(m.released)]
def scorer(e,obj,pad,beg,trace):
    end=e.objects()
    zone=e.m.geom_pos[e.pad[pad],:2]
    dh=trace[:,2]-beg[obj,2];high=dh>=.07
    carry=np.linalg.norm(trace[high,:2]-beg[obj,:2],axis=1)
    lift=max(0.,float(np.max(dh)));car=float(carry.max()) if len(carry) else 0.
    bct=int(np.sum(high&(trace[:,3]==2)))
    error=float(np.linalg.norm(end[obj,:2]-zone))
    disp=max(float(np.linalg.norm(end[j,:2]-beg[j,:2])) for j in range(3) if j!=obj)
    height=float(e.m.geom_size[e.og[obj],1]); v=float(np.linalg.norm(e.d.cvel[e.objs[obj]][3:]))
    checks=dict(lift=lift>=.07,bilateral=bct>=4,carry=car>=.09,zone=error<=.075,
                release=bool(e.is_open() and e.contact(obj)==0),support=bool(abs(end[obj,2]-height)<.009),rest=v<=.035,others=disp<=.04)
    return dict(success=all(checks.values()),checks=checks,lift_m=lift,carry_m=car,error_m=error,contact_samples=bct,max_other_m=disp)
def reset(e,cfg,mode,i):
    layout,tint,radii,heights,mu=cfg
    if mode=='lowfric':mu=np.maximum(.48,mu*.57)
    e.reset((layout,tint,radii,heights,mu))
    if mode=='offset':
        start=[.25, .15 if i%2==0 else -.15]
        e.d.qpos[e.joints[:3]]=[*ik(start),.136-BASE_Z]
        e.d.qvel[:]=0;e.d.ctrl[:3]=e.d.qpos[e.joints[:3]]
        mujoco.mj_forward(e.m,e.d)
def corrupt(rgb,kind,i):
    if kind not in ('lowfric','offset'):return rgb
    r=np.random.default_rng(seed("contact-vla-camera-"+kind,i))
    return np.clip(rgb.astype(float)*np.array([.89,1.09,.96])+r.normal(0,4,rgb.shape),0,255).astype(np.uint8)
