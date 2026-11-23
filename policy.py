"""Portable NumPy inference. RGB-only scene geometry, text, and proprioception."""
import re
from pathlib import Path
import numpy as np
from env import SIZE,EXTENT,RANGE

def tokens(text):return set(re.findall(r"[a-z]+",text.lower()))
def text_vector(text,vocab):return np.r_[1.,[t in tokens(text) for t in vocab]]
def pixel_features(rgb):
    x=rgb.reshape(-1,3).astype(np.float32)/255.
    return x/np.maximum(x.sum(1,keepdims=True),.08)
def classify(rgb,proto):
    x=pixel_features(rgb);return np.argmin((x*x).sum(1,keepdims=True)-2*(x@proto.T)+(proto*proto).sum(1),1).reshape(rgb.shape[:2])
def detect(rgb,proto):
    labels=classify(rgb,proto)
    out=np.full((6,2),np.nan)
    for j in range(6):
        y,x=np.where(labels==j)
        if len(x)>=(20 if j<3 else 43):
            out[j]=[.51+(np.median(x)+.5-SIZE/2)*EXTENT/SIZE,(SIZE/2-np.median(y)-.5)*EXTENT/SIZE]
    return out
def update(memory,detected):
    known=np.isfinite(memory).all(axis=1); ok=np.isfinite(detected).all(axis=1)
    memory[~known&ok]=detected[~known&ok]
    return memory
def features(memory,proprio,obj,zone):
    wrist=proprio[:2];holding=proprio[2]
    target=memory[zone+3 if holding else obj]
    if not np.isfinite(target).all():return None
    delta=(target-wrist)/.2
    return np.r_[delta,np.abs(delta),holding]
def forward(x,w):
    h=(x-w["mean"])/w["scale"]
    for i in range(2):h=np.tanh(h@w[f"w{i}"]+w[f"b{i}"])
    return h@w["w2"]+w["b2"]
class Policy:
    def __init__(self,path="artifacts/model.npz"):
        with np.load(path,allow_pickle=False) as w:self.w=dict(w)
        self.reset("")
    def reset(self,command,none=False):
        s=text_vector(command,self.w["vocab"])@self.w["language"]
        self.obj,self.zone=(0,0) if none else (int(np.argmax(s[:3])),int(np.argmax(s[3:])))
        self.memory=np.full((6,2),np.nan)
    def act(self,rgb,proprio):
        if not np.isfinite(self.memory).all():update(self.memory,detect(rgb,self.w["prototype"]))
        x=features(self.memory,proprio,self.obj,self.zone)
        if x is None:return np.array([0.,0.,0.])
        out=forward(x,self.w)
        return np.r_[np.clip(out[:2],-1,1)*RANGE,np.clip(out[2],-1,1)]
