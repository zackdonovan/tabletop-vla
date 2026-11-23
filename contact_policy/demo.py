from PIL import Image,ImageDraw
from eval import episode
from learn import Learner
from sim import Physical,config
from env import TEST,phrase

sim=Physical(render=True);p=Learner();out=[]
for goal in TEST:
 frames=[];r=episode(sim,p,config('vla-confirm-v1',0),0,goal,goal,'nominal','full',2,frames)
 frames.append(sim.rgb());out.append((phrase(*goal,2).replace('put the puck that is ','').replace(' on the pad that is ',' to '),frames[::2],r['success']))
n=max(len(f) for _,f,_ in out)
movie=[]
for j in range(n+10):
 image=Image.new('RGB',(768,310),'#101722');d=ImageDraw.Draw(image)
 d.text((12,10),'LEARNED CONTINUOUS CONTROL WITH PHYSICAL GRASP CONTACT',fill='#d9e7fc')
 for i,(title,frames,passed) in enumerate(out):
  x=8+i*256;image.paste(Image.fromarray(frames[min(j,len(frames)-1)]).resize((244,244)),(x,36))
  final=(' | placed' if passed else ' | failed') if j>=len(frames)-1 else ''
  d.text((x,290),title+final,fill='#d9e7fc')
 movie.append(image)
movie[0].save('contact_policy/learned-contact.gif',save_all=True,append_images=movie[1:],duration=100,loop=0)
sim.renderer.close()
