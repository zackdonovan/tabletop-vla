"""Closed-loop frozen paired tests, with complete per-episode records."""
import argparse,json,time
from pathlib import Path
import numpy as np
from env import Table,scene,phrase,expert,TRAIN,TEST,WORDS
from policy import Policy

MODES=("expert","full","shuffle","no_language","fixed_vision")
def rollout(env,policy,layout,goal,shown,template,mode,fixed,frames=None):
    env.reset(layout,*goal);policy.reset(phrase(*shown,template),none=mode=="no_language")
    pt=[]; et=[]; reached=False; grasped=False
    while not env.done:
        if mode=="expert" or mode=="fixed_vision":
            rgb=env.rgb() if frames is not None else fixed
            rt=0.
        else:
            t=time.perf_counter();rgb=env.rgb();rt=time.perf_counter()-t
        if frames is not None:frames.append(rgb.copy())
        reached|=np.linalg.norm(env.wrist-env.positions()[goal[0]])<.066
        t=time.perf_counter();action=expert(env) if mode=="expert" else policy.act(rgb,env.scalar());pt.append(time.perf_counter()-t)
        t=time.perf_counter();env.action(action);et.append(rt+time.perf_counter()-t)
        grasped|=env.grasped==goal[0]
    loc=env.positions()
    requested=env.success
    supplied=env.released==shown[0] and np.linalg.norm(env.wrist-loc[shown[1]+3])<=.083
    failure="success" if requested else "wrong_puck" if env.wrong else "timeout" if env.released is None else "wrong_zone"
    return dict(success=bool(requested),supplied_success=bool(supplied),reached=bool(reached),correct_grasp=bool(grasped),failure=failure,steps=env.steps,released=env.released,policy_ms=[round(x*1e3,4) for x in pt],env_ms=[round(x*1e3,4) for x in et])
def run(dev=False):
    p=Path("artifacts");env=Table();model=Policy(p/"model.npz");env.reset(scene('train',0),*TRAIN[0]);fixed=env.rgb()
    rows=[]; t=time.perf_counter()
    domains=[('dev',8,TRAIN,(0,1))] if dev else [('seen',12,TRAIN,(0,1)),('test',24,TEST,(2,3))]
    for domain,n,goals,templates in domains:
        for mode in ("expert","full") if dev else MODES:
            for i in range(n):
                for j,goal in enumerate(goals):
                    shown=goals[(j+1)%len(goals)] if mode=="shuffle" else (0,0) if mode=="no_language" else goal
                    r=rollout(env,model,scene(domain,i),goal,shown,templates[(i+j)%2],mode,fixed)
                    rows.append(dict(domain=domain,scene=i,goal=goal,shown=shown,template=templates[(i+j)%2],mode=mode,**r))
            selected=[r for r in rows if r['domain']==domain and r['mode']==mode]
            print(domain,mode,sum(r['success'] for r in selected),len(selected),flush=True)
    env.close()
    (p/('dev-records.json' if dev else 'test-records.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-t,rows=rows)))
def summarize():
    p=Path("artifacts");raw=json.loads((p/"test-records.json").read_text()); rows=raw["rows"];out={}
    for domain in ("seen","test"):
        out[domain]={}
        for m in MODES:
            r=[x for x in rows if x["domain"]==domain and x["mode"]==m];vals=np.array([x["success"] for x in r]);sc=np.array([x["scene"] for x in r]);u=np.unique(sc)
            scores=np.array([vals[sc==i].mean() for i in u])
            rng=np.random.default_rng(75);ids=rng.integers(0,len(u),(2000,len(u)))
            med=lambda name:float(np.median([a for x in r for a in x[name]]))
            out[domain][m]=dict(episodes=len(r),success=int(vals.sum()),rate=float(vals.mean()),cluster_ci95=np.quantile(scores[ids].mean(1),[.025,.975]).tolist(),success_by_template={str(j):int(sum(x["success"] for x in r if x["template"]==j)) for j in set(x["template"] for x in r)},supplied_success=sum(x["supplied_success"] for x in r),grasped=sum(x["correct_grasp"] for x in r),reached=sum(x["reached"] for x in r),median_steps=float(np.median([x["steps"] for x in r])),policy_median_ms=med('policy_ms'),env_median_ms=med('env_ms'),failures={k:sum(x["failure"]==k for x in r) for k in ("wrong_puck","wrong_zone","timeout")})
        r=[x for x in rows if x["domain"]==domain and x["mode"]=="full"]
        out[domain]["paired_scenes_all_commands_success"]=sum(all(x["success"] for x in r if x["scene"]==i) for i in set(x["scene"] for x in r))
    (p/'metrics.json').write_text(json.dumps({'timing_note':'policy timing excludes RGB read, environment median includes render except expert/fixed image; simulated step=0.16s','test_wall_seconds':raw['seconds'],'scores':out},indent=2))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument('--dev',action='store_true');a.add_argument('--summary',action='store_true');args=a.parse_args()
    if args.summary:summarize()
    else:run(args.dev)
