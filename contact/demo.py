from PIL import Image,ImageDraw
from sim import Physical,config,place,mujoco

e=Physical(render=True);e.reset(config('confirm-v1',0));raw=e.rgb
free=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(e.m,free)
free.lookat[:]=[.44,0,.06];free.distance=1.5;free.azimuth=144;free.elevation=-36
def combined():
    top=Image.fromarray(raw()).resize((288,288))
    e.renderer.update_scene(e.d,camera=free)
    side=Image.fromarray(e.renderer.render().copy()).resize((288,288))
    frame=Image.new('RGB',(602,334),'#101722');draw=ImageDraw.Draw(frame)
    draw.text((12,10),'MuJoCo: independent free-body puck + opposed friction fingers',fill='#edf2fd')
    frame.paste(top,(8,31));frame.paste(side,(306,31))
    draw.text((10,320),'top camera',fill='#edf2fd');draw.text((306,320),'oblique camera',fill='#edf2fd');return frame
# Capture the exact physical episode with top and side render; render has no dynamics writes.
e.rgb=combined
frames=[];r=place(e,0,0,frames=frames);frames.append(combined())
frames[0].save('contact/physical-grasp.gif',save_all=True,append_images=frames[1:],loop=0,duration=80)
e.renderer.close()
print('success',r['success'])
