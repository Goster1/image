import numpy as np
from scipy.optimize import least_squares
from common import *
d=load_json(CACHE+'/proto_plumb.json'); pcs=[np.array(p) for p in d['pieces']]
F0=1300.0
def resid(x, spec, pcs, per=False):
    p=dict(zip(spec,x)); K=K_from(F0,F0,p.get('cx',W/2),p.get('cy',H/2))
    dist=[p.get('k1',0),p.get('k2',0),p.get('p1',0),p.get('p2',0),p.get('k3',0)]
    out=[];pp=[]
    for Q in pcs:
        n=undistort_points(Q[::3],K,dist,iters=20)*F0
        c=n.mean(0); u,s,vt=np.linalg.svd(n-c,full_matrices=False)
        r=(n-c)@vt[1]; t=(n-c)@vt[0]; r=r*(np.hypot(*(Q[-1]-Q[0]))/max(np.ptp(t),1e-9))
        out.append(r/np.sqrt(len(r))); pp.append(np.sqrt(np.mean(r**2)))
    return pp if per else np.concatenate(out)
init={'k1':-0.2,'k2':0.0,'k3':0.0,'p1':0.0,'p2':0.0,'cx':W/2,'cy':H/2}
lo={'k1':-2,'k2':-5,'k3':-10,'p1':-.2,'p2':-.2,'cx':-2000,'cy':-2000}
hi={'k1':2,'k2':5,'k3':10,'p1':.2,'p2':.2,'cx':4000,'cy':3000}
sc={'k1':.01,'k2':.01,'k3':.01,'p1':.001,'p2':.001,'cx':10,'cy':10}
for spec in [['k1','cx','cy'],['k1','k2','cx','cy'],['k1','k2'],['k1','k2','k3','cx','cy'],['k1','k2','p1','p2'],['k1','k2','p1','p2','cx','cy']]:
    x0=[init[s] for s in spec]
    r=least_squares(resid,x0,args=(spec,pcs),loss='soft_l1',f_scale=0.3,x_scale=[sc[s] for s in spec],bounds=([lo[s] for s in spec],[hi[s] for s in spec]))
    J=r.jac; C=np.linalg.pinv(J.T@J)*(2*r.cost/(len(r.fun)-len(r.x)))
    per=np.array(resid(r.x,spec,pcs,True))
    print(spec, ' '.join(f"{s}={v:.4f}±{e:.4f}" for s,v,e in zip(spec,r.x,np.sqrt(np.diag(C)))), 'median rms',round(np.median(per),3),'mean',round(per.mean(),3))
