"""MuJoCo planar articulated tabletop with a transparent virtual grasp latch."""
from __future__ import annotations
import hashlib, os, re
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE","true")
import mujoco
import numpy as np

SIZE=80
EXTENT=1.1
REACH=(.51,.49)
RANGE=.054
GRASP=.066
DROP=.083
HORIZON=56
WORDS=("red","green","blue","yellow","purple","cyan")
PALETTE=((.9,.065,.06),(.06,.8,.13),(.06,.20,.94),(.94,.82,.05),(.77,.07,.84),(.04,.86,.84))
TRAIN=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))
TEST=((0,0),(1,1),(2,2))
TEMPLATES=("move the {obj} puck to the {zone} pad","pick up the {obj} puck and place it on the {zone} pad","put the puck that is {obj} on the pad that is {zone}","place on the {zone} pad the {obj} puck")
def phrase(obj,zone,t):
    return TEMPLATES[t].format(obj=WORDS[obj],zone=WORDS[zone+3])
def seed(split, i):
    return int.from_bytes(hashlib.sha256(f"vla-tabletop-v1|{split}|{i}".encode()).digest()[:8],"big")
def scene(split,i):
    r=np.random.default_rng(seed(split,i))
    # Slot identities and object/zone role freely permute; screen coordinates are not semantic.
    points=np.array([(x,y) for x in (.30,.54,.78) for y in (-.39,-.13,.13,.39)])
    points=points[np.linalg.norm(points,axis=1)<.92]
    chosen=r.choice(len(points),6,replace=False)
    return points[chosen]+r.uniform(-.045,.045,(6,2)),r.uniform(.84,1.12,3),r.uniform(.85,1.14)
def ik(xy):
    x,y=xy;l1,l2=REACH; c=np.clip((x*x+y*y-l1*l1-l2*l2)/(2*l1*l2),-.999,.999)
    b=np.arccos(c);return np.array([np.arctan2(y,x)-np.arctan2(l2*np.sin(b),l1+l2*np.cos(b)),b])
def xml():
    cols=[]
    for i,c in enumerate(PALETTE):
        items=' '.join(map(str,c))
        cols.append(f'<material name="m{i}" rgba="{items} 1" emission=".75" specular="0"/>')
    objs=''.join(f'<body name="item{i}" mocap="true" pos=".4 0 {.033 if i<3 else .008}"><geom name="g{i}" type="cylinder" size="{.049 if i<3 else .080} {.023 if i<3 else .006}" material="m{i}" contype="0" conaffinity="0"/></body>' for i in range(6))
    return f'''<mujoco><option gravity="0 0 0" timestep=".004" integrator="implicitfast"/><asset>{''.join(cols)}</asset>
<visual><global offwidth="{SIZE}" offheight="{SIZE}"/><headlight ambient=".42 .42 .42" diffuse=".55 .55 .55" specular="0 0 0"/></visual>
<worldbody><camera name="rgb" projection="orthographic" fovy="{EXTENT}" pos=".51 0 2" xyaxes="1 0 0 0 1 0"/>
<geom type="plane" size="2 2 .1" rgba=".13 .13 .13 1"/>
{objs}
<body pos="0 0 .115"><joint name="j0" type="hinge" axis="0 0 1" damping="2"/><geom type="capsule" fromto="0 0 0 .51 0 0" size=".025" rgba=".33 .34 .35 1" contype="0" conaffinity="0"/>
<body pos=".51 0 0"><joint name="j1" type="hinge" axis="0 0 1" damping="1"/><geom type="capsule" fromto="0 0 0 .49 0 0" size=".019" rgba=".43 .43 .44 1" contype="0" conaffinity="0"/>
<site name="wrist" pos=".49 0 0" size=".019" rgba=".94 .94 .94 1"/></body></body>
</worldbody><actuator><position joint="j0" kp="120" kv="12"/><position joint="j1" kp="100" kv="10"/></actuator></mujoco>'''
class Table:
    def __init__(self,render=True):
        self.m=mujoco.MjModel.from_xml_string(xml());self.d=mujoco.MjData(self.m)
        self.renderer=mujoco.Renderer(self.m,SIZE,SIZE) if render else None
        self.mids=[self.m.body(f"item{i}").mocapid[0] for i in range(6)]
        self.gs=[self.m.geom(f"g{i}").matid[0] for i in range(6)]
        self.wid=self.m.site("wrist").id
    def reset(self,layout,obj,zone):
        poses,channel,gain=layout
        self.target=(obj,zone);self.grasped=None;self.released=None;self.attempts=0
        self.wrong=False;self.done=False;self.success=False;self.steps=0;self._desired=np.array([.21,0.])
        mujoco.mj_resetData(self.m,self.d)
        self.d.qpos[:2]=ik(self._desired);self.d.ctrl[:2]=self.d.qpos[:2]
        self.d.mocap_pos[self.mids,:2]=poses
        self.d.mocap_pos[self.mids,2]=[.033]*3+[.008]*3
        for i,j in enumerate(self.gs):self.m.mat_rgba[j,:3]=np.clip(np.array(PALETTE[i])*channel*gain,0,1)
        mujoco.mj_forward(self.m,self.d)
    @property
    def wrist(self):return self.d.site_xpos[self.wid,:2].copy()
    def positions(self):return self.d.mocap_pos[self.mids,:2].copy()
    def rgb(self):
        self.renderer.update_scene(self.d,camera="rgb")
        return self.renderer.render().copy()
    def scalar(self):return np.r_[self.wrist,self.grasped is not None].astype(np.float32)
    def action(self,act):
        if self.done:raise RuntimeError("episode finished")
        a=np.asarray(act,float)
        if a.shape!=(3,) or not np.all(np.isfinite(a)):raise ValueError("expected finite dx,dy,grip")
        v=np.clip(a[:2],-RANGE,RANGE);self._desired=np.clip(self.wrist+v,[.14,-.51],[.90,.51])
        rad=np.linalg.norm(self._desired)
        if rad>.965:self._desired*=.965/rad
        self.d.ctrl[:2]=ik(self._desired);mujoco.mj_step(self.m,self.d,nstep=40)
        wrist=self.wrist
        if self.grasped is not None:
            self.d.mocap_pos[self.mids[self.grasped],:2]=wrist
            if a[2]<-.5:
                self.released=self.grasped;self.grasped=None;self.done=True
                self.success=self.released==self.target[0] and np.linalg.norm(wrist-self.positions()[self.target[1]+3])<=DROP
                self.d.mocap_pos[self.mids[self.released],2]=.033
        elif a[2]>.5:
            self.attempts+=1
            distances=np.linalg.norm(self.positions()[:3]-wrist,axis=1);j=int(np.argmin(distances))
            if distances[j]<=GRASP:
                self.grasped=j;self.wrong|=j!=self.target[0]
                self.d.mocap_pos[self.mids[j],2]=.074
        mujoco.mj_forward(self.m,self.d)
        self.steps+=1
        if self.steps>=HORIZON:self.done=True
    def close(self):
        if self.renderer is not None:self.renderer.close()
def expert(env):
    obj,zone=env.target;held=env.grasped is not None
    delta=env.positions()[zone+3 if held else obj]-env.wrist
    if np.linalg.norm(delta)<.035:return np.r_[delta,-1. if held else 1.]
    return np.r_[np.clip(delta,-RANGE,RANGE),0.]
