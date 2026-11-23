"""A reproducible same-scene counterfactual visualization from actual rollouts."""
from PIL import Image,ImageDraw
from pathlib import Path
import numpy as np
from env import Table,scene,phrase,TEST
from policy import Policy
from evaluate import rollout

e=Table();p=Policy();layout=scene("test",0);streams=[]
for goal in TEST:
 f=[];result=rollout(e,p,layout,goal,goal,2,"full",None,f)
 # Include the final release state instead of stopping at pre-release.
 f.append(e.rgb());streams.append((phrase(*goal,2).replace("put the puck that is ","").replace(" on the pad that is "," → "),f,result))
out=[];n=max(len(f) for _,f,_ in streams)
for i in range(n+10):
 frame=Image.new('RGB',(768,310),'#101722');d=ImageDraw.Draw(frame);d.text((12,10),'SAME CAMERA SCENE, THREE DIFFERENT INSTRUCTIONS',fill='#d9e7fc')
 for j,(title,frames,result) in enumerate(streams):
  x=8+j*256;frame.paste(Image.fromarray(frames[min(i,len(frames)-1)]).resize((244,244),Image.Resampling.NEAREST),(x,36))
  text=title.replace('→','to')
  status=('  |  placed' if result['success'] else '  |  '+result['failure']) if i>=len(frames)-1 else ''
  d.text((x,289),text+status,fill='#d9e7fc')
 out.append(frame)
out[0].save("artifacts/same-scene.gif",save_all=True,append_images=out[1:],duration=120,loop=0)
e.close()
