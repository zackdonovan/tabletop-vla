"""RGB and language conditioning plus controlled imitation on physical contacts."""
from pathlib import Path
import json,time
import numpy as np
from sklearn.neural_network import MLPRegressor
from env import phrase,TRAIN,seed,mujoco
from sim import Physical,config,SIZE
from roll import *
from policy import pixel_features,classify,tokens,text_vector
ROOT=Path('contact_policy')
def segment(e):
    e.renderer.enable_segmentation_rendering()
    try:e.renderer.update_scene(e.d,camera='top');return e.renderer.render().copy()
    finally:e.renderer.disable_segmentation_rendering()

def locations(rgb,proto):
    labels=classify(rgb,proto)
    p=np.full((6,2),np.nan)
    for i in range(6):
        y,x=np.where(labels==i)
        if len(x)>=(10 if i<3 else 50):
            p[i]=[.51+(np.median(x)+.5-SIZE/2)*1.1/SIZE,(SIZE/2-np.median(y)-.5)*1.1/SIZE]
    return p

def network(x,w):
    h=(x-w['mean'])/w['scale']
    for i in range(2):h=np.tanh(h@w[f'W{i}']+w[f'b{i}'])
    a=h@w['W2']+w['b2'];a[3]=(a[3]+1)/2
    return np.clip(a,[-1,-1,-1,0],[1,1,1,1])

class Learner:
    def __init__(self,path=ROOT/'model.npz'):
        with np.load(path,allow_pickle=False) as z:self.w=dict(z)
    def fresh(self,text):
        z=text_vector(text,self.w['vocab'])@self.w['text']
        self.goal=(int(z[:3].argmax()),int(z[3:].argmax()))
        self.centers=np.full((6,2),np.nan)
        self.memory=Memory()
    def act(self,rgb,s):
        if not np.isfinite(self.centers).all():
            new=locations(rgb,self.w['prototype']);missing=~np.isfinite(self.centers).all(axis=1)
            self.centers[missing]=new[missing]
        x=signals(s,self.memory,self.centers[self.goal[0]],self.centers[3+self.goal[1]])
        a=np.zeros(4) if x is None else network(x,self.w)
        self.memory.step(a[3]);return a

def fit(episodes=360):
    t=time.perf_counter();e=Physical(render=True);parts=[];rng=np.random.default_rng(seed('physical-vla-fit',0))
    for i in range(42):
        e.reset(config('vla-train',i)); pix=pixel_features(e.rgb())
        seg=segment(e).reshape(-1,2)
        masks=[(seg[:,0]==j)&(seg[:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) for j in e.og+e.pad]
        parts.append([pix[k] for k in [*masks,~np.any(masks,axis=0)]])
    proto=np.array([np.median(np.concatenate([u[j] for u in parts]),0) for j in range(7)],np.float32)
    rows=[(phrase(o,z,t),o,z) for o,z in TRAIN for t in (0,1)]
    vocab=np.array(sorted(set().union(*(tokens(s) for s,_,_ in rows))))
    X=np.array([text_vector(s,vocab) for s,_,_ in rows]);Y=np.zeros((len(rows),6))
    for i,(_,o,z) in enumerate(rows):Y[i,[o,z+3]]=1
    text=np.linalg.solve(X.T@X+.01*np.eye(X.shape[1]),X.T@Y)
    features=[];labels=[];weights=[];example=[];failures=0;invisible=0
    for i in range(episodes):
        o,z=TRAIN[i%len(TRAIN)];e.reset(config('vla-train',i));rgb=e.rgb();loc=locations(rgb,proto)
        beg=e.objects();obj=beg[o,:2].copy();pad=e.m.geom_pos[e.pad[z],:2].copy()
        m=Memory();tr=[]
        if not len(example):example.append(rgb.copy())
        prev=0.
        for j in range(CAP):
            s=sensors(e); a=teacher(s,m,obj,pad)
            x=signals(s,m,loc[o],loc[z+3])
            if x is not None:
                features.append(x);labels.append(np.r_[a[:3],a[3]*2-1]);weights.append(7. if a[3]!=prev or m.closed<HOLD and m.closed else 1.)
            else:invisible+=1
            # Modest demonstration perturbation, with unchanged reference label.
            ar=a.copy()
            if rng.random()<.19:ar[:3]+=rng.normal(0,[.12,.12,.08])
            move(e,ar);m.step(a[3]);prev=a[3];tr.append([*e.objects()[o],e.contact(o)])
            if m.stop():break
        failures+=not scorer(e,o,z,beg,np.array(tr))['success']
        if (i+1)%60==0:print(i+1,len(features),failures,invisible,flush=True)
    e.renderer.close()
    X=np.array(features);Y=np.array(labels);scale=np.maximum(.00001,X.std(0));mean=X.mean(0)
    np.savez_compressed(ROOT/'demonstrations.npz',features=X.astype(np.float32),labels=Y.astype(np.float32),weights=np.asarray(weights,np.float32),sample_rgb=example[0])
    mlp=MLPRegressor(hidden_layer_sizes=(96,96),activation='tanh',batch_size=256,
                    max_iter=220,learning_rate_init=.0015,random_state=13421,tol=2e-6)
    mlp.fit((X-mean)/scale,Y,sample_weight=np.asarray(weights))
    np.savez_compressed(ROOT/'model.npz',mean=mean,scale=scale,prototype=proto,vocab=vocab,text=text,**{f'W{i}':a for i,a in enumerate(mlp.coefs_)},**{f'b{i}':a for i,a in enumerate(mlp.intercepts_)})
    (ROOT/'training.json').write_text(json.dumps(dict(episodes=episodes,rows=len(X),teacher_failures=failures,unseen_features=invisible,loss=mlp.loss_,epochs=mlp.n_iter_,seconds=time.perf_counter()-t),indent=2))
if __name__=='__main__':fit()
