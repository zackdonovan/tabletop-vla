"""Learn pixel prototypes, bag-of-words grounding, and behavioral-cloning MLP."""
import json,time
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from env import *
from policy import *

def segment(t):
    t.renderer.enable_segmentation_rendering()
    try:t.renderer.update_scene(t.d,camera="rgb");return t.renderer.render().copy()
    finally:t.renderer.disable_segmentation_rendering()

def run():
    p=Path('artifacts');p.mkdir(exist_ok=True)
    t=time.perf_counter();rng=np.random.default_rng(seed('training-noise',0));env=Table()
    x=[]
    for i in range(48):
        env.reset(scene('train',i),0,1); rgb=env.rgb(); seg=segment(env)
        colors=pixel_features(rgb)
        mask=seg[:,:,0].reshape(-1)
        # Both geom ID and MuJoCo object category required; wrist site may reuse geom IDs.
        geom=seg[:,:,1].reshape(-1)==int(mujoco.mjtObj.mjOBJ_GEOM)
        ks=[(mask==env.m.geom(f'g{k}').id)&geom for k in range(6)]
        x.append([colors[a] for a in [*ks,~np.any(ks,axis=0)]])
    proto=np.array([np.median(np.concatenate([a[i] for a in x]),0) for i in range(7)],np.float32)
    instructions=[(phrase(o,z,t),o,z) for o,z in TRAIN for t in (0,1)]
    vocab=np.array(sorted(set().union(*(tokens(s) for s,_,_ in instructions))))
    v=np.array([text_vector(s,vocab) for s,_,_ in instructions])
    y=np.zeros((len(instructions),6))
    for i,(_,o,z) in enumerate(instructions):y[i,[o,z+3]]=1
    lang=np.linalg.solve(v.T@v+.01*np.eye(v.shape[1]),v.T@y)
    X,Y=[],[]; demos=0; lost=0; skipped=0
    for i in range(420):
        o,z=TRAIN[i%len(TRAIN)]
        env.reset(scene('train',i),o,z)
        memory=detect(env.rgb(),proto)
        while not env.done:
            propr=env.scalar(); a=expert(env)
            f=features(memory,propr,o,z)
            if f is not None:
                v=np.r_[a[:2]/RANGE,a[2]]
                # Reweight rare events with small pose jitter; use no future states.
                for j in range(5 if a[2] else 1):
                    d=f.copy()
                    if j:
                        d[:2]+=rng.normal(0,.012,2);d[2:4]=np.abs(d[:2])
                    X.append(d);Y.append(v)
            else:skipped+=1
            act=a.copy()
            if not a[2] and rng.random()<.18:act[:2]+=rng.normal(0,.045,2)
            env.action(act)
            if env.grasped is not None and env.grasped!=o:break
        demos+=1;lost+=not env.success
        if (i+1)%100==0:print(i+1,len(X),lost,flush=True)
    env.close()
    X,Y=np.asarray(X),np.asarray(Y);mean=X.mean(0);scale=np.maximum(X.std(0),1e-4)
    mlp=MLPRegressor(hidden_layer_sizes=(48,48),activation='tanh',max_iter=240,batch_size=256,random_state=4511,learning_rate_init=.003,tol=1e-6)
    mlp.fit((X-mean)/scale,Y)
    np.savez_compressed(p/'model.npz',vocab=vocab,language=lang,prototype=proto,mean=mean,scale=scale,**{f'w{i}':v for i,v in enumerate(mlp.coefs_)},**{f'b{i}':v for i,v in enumerate(mlp.intercepts_)})
    np.savez_compressed(p/'demonstrations.npz',features=X.astype(np.float32),targets=Y.astype(np.float32))
    (p/'train.json').write_text(json.dumps(dict(simulation='MuJoCo 4.0.1',episodes=demos,failures=lost,skipped_visual_labels=skipped,training_examples=len(X),loss=mlp.loss_,epochs=mlp.n_iter_,seconds=time.perf_counter()-t),indent=2))
if __name__=='__main__':run()
