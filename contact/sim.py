"""Physical branch: free-joint pucks and friction-only bilateral grip."""
from __future__ import annotations
import time
import numpy as np
from env import mujoco, ik, PALETTE, scene, seed
SIZE=96
DT=.002
BASE_Z=.23
PICK_Z=.033
LIFT_Z=.160
FINGER_INIT=.069
FINGER_CLOSE=-.054
PAD_ERR=.075

def XML(radii=(.032,)*3,heights=(.025,)*3,frictions=(1.2,)*3):
    colors=''.join(f'<material name="mat{i}" rgba="{" ".join(map(str,c))} 1" emission=".4" specular="0"/>' for i,c in enumerate(PALETTE))
    objects=''.join(f'<body name="obj{i}" pos=".4 0 .027"><freejoint name="free{i}"/><geom name="obj_geom{i}" type="cylinder" size="{radii[i]} {heights[i]}" density="440" friction="{frictions[i]} .01 .001" material="mat{i}"/></body>' for i in range(3))
    pads=''.join(f'<geom name="pad{i}" pos=".5 0 .001" type="cylinder" size=".076 .001" contype="0" conaffinity="0" material="mat{i+3}"/>' for i in range(3))
    fingers=''.join(f'<body pos="0 {side*FINGER_INIT} 0"><joint name="finger{k}" axis="0 {side} 0" type="slide" damping="4" range="-.060 .005"/><geom name="contact{k}" type="box" size=".041 .011 .022" mass=".045" rgba=".94 .92 .87 1" friction=".65 .015 .001"/></body>' for k,side in enumerate((-1,1)))
    return f'''<mujoco model="physical tabletop"><compiler autolimits="true"/><option gravity="0 0 -9.81" timestep="{DT}" integrator="implicitfast" cone="elliptic" iterations="70"/>
<default><geom solref=".010 1" solimp=".94 .99 .002" condim="4"/></default><asset>{colors}</asset>
<visual><global offwidth="{SIZE}" offheight="{SIZE}"/><headlight ambient=".4 .4 .4" diffuse=".6 .6 .6" specular="0 0 0"/></visual>
<worldbody><camera name="top" projection="orthographic" fovy="1.1" pos=".51 0 2" xyaxes="1 0 0 0 1 0"/>
<geom name="ground" type="plane" size="2 2 .2" friction=".65 .01 .001" rgba=".18 .19 .2 1"/>
{objects}{pads}
<body pos="0 0 {BASE_Z}"><joint name="a0" type="hinge" axis="0 0 1" damping="4"/><geom type="capsule" fromto="0 0 0 .51 0 0" size=".024" mass=".9" rgba=".43 .47 .52 1" contype="0" conaffinity="0"/>
<body pos=".51 0 0"><joint name="a1" type="hinge" axis="0 0 1" damping="3"/><geom type="capsule" fromto="0 0 0 .49 0 0" size=".020" mass=".6" rgba=".57 .64 .69 1" contype="0" conaffinity="0"/>
<body pos=".49 0 0"><joint name="lift" type="slide" axis="0 0 1" range="-.207 .02" damping="10"/><geom type="cylinder" size=".025 .03" pos="0 0 .031" mass=".17" rgba=".73 .75 .78 1" contype="0" conaffinity="0"/>
<site name="tool" pos="0 0 0" rgba=".94 .94 .94 .4" size=".009"/>{fingers}</body></body></body>
</worldbody><actuator>
<position joint="a0" kp="450" kv="30"/><position joint="a1" kp="340" kv="24"/>
<position joint="lift" kp="1600" kv="40"/>
<position joint="finger0" kp="530" kv="13" forcerange="-11 11"/><position joint="finger1" kp="530" kv="13" forcerange="-11 11"/>
</actuator></mujoco>'''

def config(split,i):
    layout,tint,gain=scene('contact-'+split,i)
    r=np.random.default_rng(seed('contact-physical-'+split,i))
    return layout,tint*gain,r.uniform(.027,.037,3),r.uniform(.021,.027,3),r.uniform(.75,1.40,3)

class Physical:
    def __init__(self,render=False):
        self.render=render;self.renderer=None
    def _compile(self,radii,height,mu):
        if self.renderer is not None:self.renderer.close()
        self.m=mujoco.MjModel.from_xml_string(XML(radii,height,mu));self.d=mujoco.MjData(self.m)
        self.tool=self.m.site('tool').id
        self.objs=[self.m.body(f'obj{i}').id for i in range(3)]
        self.og=[self.m.geom(f'obj_geom{i}').id for i in range(3)]
        self.pad=[self.m.geom(f'pad{i}').id for i in range(3)]
        self.fingers=[self.m.geom(f'contact{i}').id for i in range(2)]
        self.joints=[int(self.m.joint(n).qposadr[0]) for n in ('a0','a1','lift','finger0','finger1')]
        self.free=[int(self.m.joint(f'free{i}').qposadr[0]) for i in range(3)]
        self.renderer=mujoco.Renderer(self.m,SIZE,SIZE) if self.render else None
    def reset(self,cfg):
        layout,tint,radii,height,mu=cfg
        self._compile(radii,height,mu)
        for i,a in enumerate(self.free):self.d.qpos[a:a+3]=[*layout[i],height[i]+.002]
        for j,k in enumerate(self.pad):self.m.geom_pos[k,:2]=layout[j+3]
        for i,clr in enumerate(PALETTE):self.m.mat_rgba[i,:3]=np.minimum(1,np.asarray(clr)*tint)
        self.d.qpos[self.joints]=[*ik((.22,0)),LIFT_Z-BASE_Z,0,0];self.d.ctrl[:]=self.d.qpos[self.joints]
        # These are the only object qpos writes; all later states are MuJoCo integration.
        mujoco.mj_forward(self.m,self.d);mujoco.mj_step(self.m,self.d,nstep=250)
    def wrist(self):return self.d.site_xpos[self.tool].copy()
    def objects(self):return self.d.xpos[self.objs].copy()
    def rgb(self):
        self.renderer.update_scene(self.d,camera='top')
        return self.renderer.render().copy()
    def drive(self,xyz,squeeze,n=10):
        self.d.ctrl[:]=[*ik(xyz[:2]),xyz[2]-BASE_Z,*([FINGER_CLOSE if squeeze else 0]*2)]
        mujoco.mj_step(self.m,self.d,nstep=n)
    def contact(self,obj):
        touch=set()
        for k in range(self.d.ncon):
            c=self.d.contact[k]
            if self.og[obj] not in (c.geom1,c.geom2):continue
            for f in self.fingers:
                if f in (c.geom1,c.geom2):
                    force=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,force)
                    if force[0]>.05:touch.add(f)
        return len(touch)
    def is_open(self):
        return bool(np.all(self.d.qpos[self.joints[-2:]]>-.015))

def place(e,obj,pad,squeeze=True,frames=None):
    """Expert interpolates pose commands; force and lift are exclusively physical."""
    beg=e.objects();p=e.m.geom_pos[e.pad[pad],:2].copy()
    xy=beg[obj,:2];start=e.wrist()
    stages=[('above',[*xy,LIFT_Z],False,40),('down',[*xy,PICK_Z],False,28),
            ('close',[*xy,PICK_Z],squeeze,28),('lift',[*xy,LIFT_Z],squeeze,36),
            ('carry',[*p,LIFT_Z],squeeze,62),('lower',[*p,PICK_Z+.006],squeeze,34),
            ('open',[*p,PICK_Z+.006],False,29),('retreat',[*p,LIFT_Z],False,36)]
    rows=[];t=time.perf_counter()
    for stage,target,closed,n in stages:
        target=np.asarray(target)
        for j in range(1,n+1):
            f=min(1.,j/max(1,n-10));u=f*f*(3-2*f)
            e.drive(start+(target-start)*u,closed)
            cur=e.objects()[obj]
            rows.append([stage,*cur,e.contact(obj)])
            if frames is not None and j%4==0:frames.append(e.rgb())
        start=target
    for _ in range(30):e.drive(start,False)
    final=e.objects();dist=float(np.linalg.norm(final[obj,:2]-p));height=float(e.m.geom_size[e.og[obj],1])
    lift_max=max(row[3] for row in rows)-beg[obj,2]
    contact_lift=int(sum(row[0] in ('lift','carry') and row[3]-beg[obj,2]>=.07 and row[4]==2 for row in rows))
    carry=[row[1:3] for row in rows if row[0]=='carry' and row[3]-beg[obj,2]>=.07]
    carry_dist=max((float(np.linalg.norm(np.array(x)-beg[obj,:2])) for x in carry),default=0.)
    disp=max(float(np.linalg.norm(final[i,:2]-beg[i,:2])) for i in range(3) if i!=obj)
    floor=abs(final[obj,2]-height)<.008
    speed=float(np.linalg.norm(e.d.cvel[e.objs[obj]][3:]))
    released=e.is_open() and e.contact(obj)==0
    pass_cond=dict(lift=bool(lift_max>=.07),bilateral=contact_lift>=8,carry=carry_dist>=.09,zone=dist<=PAD_ERR,floor=bool(floor),stable=speed<=.03,released=released,others=disp<=.04)
    return dict(success=all(pass_cond.values()),checks=pass_cond,lift_m=float(lift_max),carry_m=carry_dist,contact_lift_samples=contact_lift,dist_zone_m=dist,cylinder_speed_mps=speed,other_object_displacement_m=disp,seconds=time.perf_counter()-t,trace=[[a,*[float(j) for j in b]] for a,*b in rows])
