import numpy as np, json
np.set_printoptions(precision=4, suppress=True)
d = {}
W,H,fps=1920,1080,30
d['rate']=W*H*3*fps/1e6; d['hour_GB']=W*H*3*fps*3600/1e9; d['frame_MB']=W*H*3/1e6
# LK
Ix=np.array([4,4,3,0,1,2,5,1,3]);Iy=np.array([0,1,3,4,5,2,1,3,0]);It=np.array([-8,-9,-9,-4,-7,-6,-11,-6,-5])
A=np.c_[Ix,Iy].astype(float);b=-It.astype(float)
ATA=A.T@A;ATb=A.T@b;uv=np.linalg.solve(ATA,ATb);res=A@uv-b
ev=np.linalg.eigvalsh(ATA)
d.update(ATA=ATA.tolist(),ATb=ATb.tolist(),det=float(np.linalg.det(ATA)),uv=uv.tolist(),res=res.tolist(),rss=float(res@res),ev=ev.tolist(),cond=float(ev[1]/ev[0]),
 SIx2=int((Ix**2).sum()),SIxIy=int((Ix*Iy).sum()),SIy2=int((Iy**2).sum()),SIxIt=int((Ix*It).sum()),SIyIt=int((Iy*It).sum()))
# vertical edge
A2=np.c_[np.full(9,5.),np.zeros(9)];ATA2=A2.T@A2;d['ATA2']=ATA2.tolist();d['det2']=float(np.linalg.det(ATA2));d['u2']=10/5
# SAD
ref=np.array([[10,20],[30,40]]);S=np.array([[8,15,22,19],[12,11,21,17],[30,25,12,22],[28,34,31,41]])
sad={}
for dy in(-1,0,1):
  for dx in(-1,0,1):
    r,c=1+dy,1+dx;blk=S[r:r+2,c:c+2];sad[(dx,dy)]=int(abs(blk-ref).sum())
d['sad']={f'{k[0]},{k[1]}':v for k,v in sad.items()};best=min(sad,key=sad.get);d['best']=best
d['blocks']={f'{k[0]},{k[1]}':S[1+k[1]:3+k[1],1+k[0]:3+k[0]].tolist() for k in sad}
# costs
blocks=(W//16)*(H//16)  # 120*67 (1080/16=67.5)
blocks_exact=120*67.5
full_cand=33*33;full_ops=blocks*full_cand*256; full_ops_exact=blocks_exact*full_cand*256
four=blocks*33*256;four_exact=blocks_exact*33*256
d.update(nblocks=blocks,full_cand=full_cand,full_ops=full_ops,full_ops_exact=full_ops_exact,four_ops=four,four_exact=four_exact,
 full_ms=full_ops/5e9*1e3,four_ms=four/5e9*1e3,budget_ops=5e9*0.033,full_exact_ms=full_ops_exact/5e9*1e3,four_exact_ms=four_exact/5e9*1e3)
# stereo
f=1200;B=1.0;cx,cy=960,540
st={}
for n,(l,r) in {'Q1':((700,400),(640,400)),'Q2':((900,500),(850,503)),'Q3':((500,300),(560,300))}.items():
  dd=l[0]-r[0];st[n]=dict(dr=r[1]-l[1],d=dd)
d['stereo']=st
dd=60;Z=f*B/dd;X=(700-cx)*Z/f;Y=(400-cy)*Z/f
d.update(Z=Z,X=X,Y=Y,dZ_plus=f*B/(dd-1)-Z,dZ_minus=Z-f*B/(dd+1),dZ_approx=Z**2/(f*B),Zmax_C3=None)
# max Z with dZ<=0.5: Z^2/(fB)<=0.5
d['Zmax']=float(np.sqrt(0.5*f*B))
# speed
u_full=2*8  # level-3 flow? computed from LK u
u3=uv[0];d['u_full']=float(u3*8);d['v_full']=float(uv[1]*8)
for lab,u in(('lk',u3*8),('sad',float(best[0]))):
  v=u*Z*fps/f;d['spd_'+lab]=dict(u=float(u),ms=float(v),kmh=float(v*3.6))
d['speed_limit_px']=25/3.6*f/(Z*fps)
# bg
Bg,I,al,th=100,180,.05,30
d['bg']=dict(diff=abs(I-Bg),fg=abs(I-Bg)>th,new=(1-al)*Bg+al*I,new05=(1-.5)*Bg+.5*I)
# frames to absorb for alpha: |I-B_t|<=30 → (0.95)^n*80<=30
import math
d['n05']=math.ceil(math.log(30/80)/math.log(.95));d['n5']=math.ceil(math.log(30/80)/math.log(.5))
json.dump(d,open('co5/calc.json','w'),indent=1,default=str)
print(json.dumps(d,indent=1,default=str))
