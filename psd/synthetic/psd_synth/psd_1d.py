import os, math, warnings, itertools
import numpy as np
import pandas as pd
import yaml

from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.special import lambertw
from numpy.fft import fft, ifft, fftfreq

FULL_RESOLUTION = False
nx = 2048 if FULL_RESOLUTION else 512
L_um = 2500.0
dx_um = L_um / nx

dt = 0.05
n_frames = 30
times = np.arange(n_frames)*dt
FRAME_LIST = [5,10,15,20,25,29]
FRAME_T = 25
SNAP_FRAME = 28

K_MAX = 0.08
K_MIN_SNAPSHOT = 0.005
K_MAX_SNAPSHOT = 0.15
R2_MIN = 0.80

# In 1D, GRID_N means number of non-overlapping segments.
GRID_N = 8
GRID_LIST = [2,4,6,8,10,12,16]

# Known fixed SH rate coefficient (1/time). This makes dimensions explicit.
KAPPA_SH = 1.0

RUN_PER_MODEL_FIGURES = False
SAVE_FIGURES = False
FIG_DIR = "master2_1d_clean_figures"


def configure(path=None,**overrides):
    g=globals()
    cfg={}
    if path is not None:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update(overrides)
    for k,v in cfg.items():
        g[k]=v
    g["nx"]=2048 if g["FULL_RESOLUTION"] else 512
    g["dx_um"]=g["L_um"]/g["nx"]
    g["times"]=np.arange(g["n_frames"])*g["dt"]
    g["x"]=np.linspace(0,g["L_um"],g["nx"],endpoint=False)
    if g["SAVE_FIGURES"]:
        os.makedirs(g["FIG_DIR"],exist_ok=True)
    return dict(cfg)

ELEMENTARY = {
    "Diff": {"params":{"D":750.0}},
    "Adv": {"params":{"c":120.0}},
    "Hyper": {"params":{"nu":2.0e5}},
    "SH": {"params":{"ell":22.0}},
    "FracDiff": {"params":{"Df":350.0,"alpha":1.5}},
    "Disp": {"params":{"beta3":8.0e4}},
    "Sixth": {"params":{"mu6":4.0e7}},
    "Loss": {"params":{"eta":0.20}},
    "Nonlocal": {"params":{"chi":0.80,"sigma":45.0}},
}
ELEMENTARY_NAMES = list(ELEMENTARY)

MODEL_LIBRARY = {}

# 9 original elementary single-operator models
for op in ELEMENTARY_NAMES:
    true = {"R":0.65, **ELEMENTARY[op]["params"]}
    if op == "SH":
        true["R"] = 1.25
    MODEL_LIBRARY["R+"+op] = {
        "kind":"elementary",
        "ops":(op,),
        "true":true,
    }

# 36 original pairwise combinations
for op1,op2 in itertools.combinations(ELEMENTARY_NAMES,2):
    true = {"R":0.65}
    true.update(ELEMENTARY[op1]["params"])
    true.update(ELEMENTARY[op2]["params"])
    if "SH" in (op1,op2):
        true["R"] = 1.25
    MODEL_LIBRARY[f"R+{op1}+{op2}"] = {
        "kind":"pairwise",
        "ops":(op1,op2),
        "true":true,
    }

# Missing 3-way and 4-way combinations of the four core operators.
CORE = ("Diff","Adv","Hyper","SH")
for n_ops in (3,4):
    for ops in itertools.combinations(CORE,n_ops):
        true = {"R":1.25 if "SH" in ops else 0.65}
        if "Diff" in ops:
            true["D"] = 300.0 if "SH" in ops else 500.0
        if "Adv" in ops:
            true["c"] = 120.0
        if "Hyper" in ops:
            true["nu"] = 1.0e5
        if "SH" in ops:
            true["ell"] = 22.0
        MODEL_LIBRARY["R+"+"+".join(ops)] = {
            "kind":"core_mixed",
            "ops":ops,
            "true":true,
        }

# Original named models
MODEL_LIBRARY["R+KSlin"] = {
    "kind":"KS",
    "ops":("KSlin",),
    "true":{"R":0.65,"a2":450.0,"a4":1.5e5},
}
MODEL_LIBRARY["CHlin"] = {
    "kind":"CH",
    "ops":("CHlin",),
    "true":{"A2":450.0,"A4":1.5e5},
}
MODEL_LIBRARY["R+Mixed"] = {
    "kind":"mixed",
    "ops":("Diff","Adv","Disp","Hyper"),
    "true":{"R":0.65,"D":500.0,"c":120.0,"beta3":8.0e4,"nu":1.0e5},
}

MODEL_VARIANTS = list(MODEL_LIBRARY)
assert len(MODEL_VARIANTS) == 53

SPECIAL_MODELS = {
    "R+Telegraph":{
        "equation":"u_tt + tau^{-1}u_t = cT^2 u_xx + R u",
        "true":{"R":0.50,"tau":0.80,"cT":90.0},
    },
    "R+DelayDiff":{
        "equation":"u_t = R u(t-tau) + D u_xx",
        "true":{"R":0.50,"D":500.0,"tau":0.50},
    },
}

x = np.linspace(0,L_um,nx,endpoint=False)

def periodic_distance(a,b,L=None):
    if L is None: L=L_um
    d=np.abs(a-b)
    return np.minimum(d,L-d)

def gaussian_periodic(center,sigma,amplitude=1.0):
    d=periodic_distance(x,center)
    return amplitude*np.exp(-0.5*(d/sigma)**2)

def make_initial_conditions():
    rng=np.random.default_rng(42)
    random_field=rng.normal(0,1,nx)
    k=2*np.pi*fftfreq(nx,d=dx_um)
    filt=np.exp(-(np.abs(k)/0.08)**4)
    random_field=np.real(ifft(fft(random_field)*filt))
    random_field-=random_field.min()
    random_field/=random_field.max()+1e-30

    one_cluster=gaussian_periodic(L_um/2,180)
    one_cluster/=one_cluster.max()

    many=np.zeros(nx)
    centers=np.linspace(150,L_um-150,9)
    for c0 in centers:
        many+=gaussian_periodic(c0,55)
    many/=many.max()

    mixed=many.copy()
    for c0 in [260,730,1320,1980,2280]:
        mixed+=0.45*gaussian_periodic(c0,20)
    mixed/=mixed.max()

    return {
        "Random smooth":random_field,
        "One central cluster":one_cluster,
        "Many clusters":many,
        "Clusters + singles":mixed,
    }


def model_parameters(tag):
    return MODEL_LIBRARY[tag]["true"].copy()

def operator_symbol(k,op,p):
    kk=np.asarray(k)
    if op=="Diff": return -p["D"]*kk**2
    if op=="Adv": return -1j*p["c"]*kk
    if op=="Hyper": return -p["nu"]*kk**4
    if op=="SH": return -KAPPA_SH*(1-p["ell"]**2*kk**2)**2
    if op=="FracDiff": return -p["Df"]*np.abs(kk)**p["alpha"]
    if op=="Disp": return 1j*p["beta3"]*kk**3
    if op=="Sixth": return -p["mu6"]*kk**6
    if op=="Loss": return -p["eta"]+0*kk
    if op=="Nonlocal": return p["chi"]*(np.exp(-0.5*p["sigma"]**2*kk**2)-1.0)
    raise ValueError(op)

def complex_symbol(k,tag,p,include_reaction=True):
    spec=MODEL_LIBRARY[tag]
    k=np.asarray(k)
    lam=np.zeros_like(k,dtype=complex)

    if spec["kind"]=="CH":
        return p["A2"]*k**2-p["A4"]*k**4

    if include_reaction:
        lam+=p.get("R",0.0)

    if spec["kind"]=="KS":
        return lam+p["a2"]*k**2-p["a4"]*k**4

    for op in spec["ops"]:
        lam+=operator_symbol(k,op,p)
    return lam

def real_symbol(k,tag,p):
    return np.real(complex_symbol(np.asarray(k),tag,p,True))

def invisible_names(tag):
    out=[]
    for op in MODEL_LIBRARY[tag]["ops"]:
        if op=="Adv": out.append("c")
        if op=="Disp": out.append("beta3")
    return list(dict.fromkeys(out))

def primitive_confounded(tag):
    ops=set(MODEL_LIBRARY[tag]["ops"])
    return "SH" in ops and ("Diff" in ops or "Hyper" in ops)

def effective_poly_truth(tag):
    p=MODEL_LIBRARY[tag]["true"]
    ops=set(MODEL_LIBRARY[tag]["ops"])
    c0=p.get("R",0.0); c2=c4=c6=0.0
    if "Loss" in ops: c0-=p["eta"]
    if "Diff" in ops: c2-=p["D"]
    if "Hyper" in ops: c4-=p["nu"]
    if "Sixth" in ops: c6-=p["mu6"]
    if "SH" in ops:
        c0-=KAPPA_SH
        c2+=2*KAPPA_SH*p["ell"]**2
        c4-=KAPPA_SH*p["ell"]**4
    return {"c0":c0,"c2":c2,"c4":c4,"c6":c6}

def identifiable_names(tag):
    spec=MODEL_LIBRARY[tag]
    kind=spec["kind"]
    ops=spec["ops"]

    if kind=="CH": return ["A2","A4"]
    if kind=="KS": return ["R","a2","a4"]

    if primitive_confounded(tag):
        names=["c0","c2","c4"]
        if "Sixth" in ops: names.append("c6")
        return names

    names=["R"]
    for op in ops:
        if op=="Diff": names.append("D")
        elif op=="Hyper": names.append("nu")
        elif op=="SH": names.append("ell")
        elif op=="FracDiff": names += ["Df","alpha"]
        elif op=="Sixth": names.append("mu6")
        elif op=="Loss":
            if "R" in names: names.remove("R")
            names.append("R_eff")
        elif op=="Nonlocal": names += ["chi","sigma"]
    return list(dict.fromkeys(names))

def true_identifiable_value(tag,name):
    p=MODEL_LIBRARY[tag]["true"]
    if name in p: return p[name]
    if name=="R_eff": return p["R"]-p["eta"]
    if name in ("c0","c2","c4","c6"): return effective_poly_truth(tag)[name]
    return np.nan

def simulate_linear(u0,tag,p):
    u=np.asarray(u0,float).copy()
    k=2*np.pi*fftfreq(nx,d=dx_um)
    growth=np.exp(complex_symbol(k,tag,p,True)*dt)
    stack=[]
    for _ in range(n_frames):
        stack.append(u.copy())
        u=np.real(ifft(fft(u)*growth))
    return np.asarray(stack)
def psd_1d(u,dx=None):
    if dx is None: dx=dx_um
    z=np.asarray(u,float)-np.mean(u)
    F=fft(z)
    P=np.abs(F)**2/len(z)
    k=2*np.pi*fftfreq(len(z),d=dx)
    pos=k>0
    return k[pos],P[pos]

def normalize_1d(k,P):
    v=np.isfinite(k)&np.isfinite(P)&(P>0)&(k>0)
    if v.sum()<3:
        return np.full_like(P,np.nan,dtype=float),np.nan
    norm=np.trapezoid(P[v],k[v])
    return P/(norm+1e-30),norm

def fit_real_symbol(k,y,tag):
    k=np.asarray(k,float); y=np.asarray(y,float)
    names=identifiable_names(tag)
    true=MODEL_LIBRARY[tag]["true"]

    if len(names)==1 and names[0] in ("R","R_eff"):
        val=float(np.mean(y))
        pred=np.full_like(y,val)
        return {names[0]:val},np.nan,pred

    if any(n in names for n in ("c0","c2","c4","c6")):
        degree=3 if "c6" in names else 2
        cf=np.polyfit(k**2,y,degree)
        pred=np.polyval(cf,k**2)
        a=cf[::-1]
        fit={"c0":a[0],"c2":a[1],"c4":a[2]}
        if degree==3: fit["c6"]=a[3]
        ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
        return fit,(1-ssr/sst if sst>0 else np.nan),pred

    p0=[]; lo=[]; hi=[]
    for nm in names:
        tv=true_identifiable_value(tag,nm)
        if nm in ("R","R_eff","a2","A2"):
            p0.append(tv*0.9 if tv!=0 else 0.01)
            lo.append(-20*abs(tv)-10); hi.append(20*abs(tv)+10)
        elif nm=="alpha":
            p0.append(tv*0.9); lo.append(0.2); hi.append(6.0)
        elif nm in ("ell","sigma"):
            p0.append(tv*0.9); lo.append(0.1); hi.append(500.0)
        else:
            p0.append(tv*0.9); lo.append(0.0); hi.append(max(1.0,20*abs(tv)))

    def fitted_curve(k,*theta):
        p=true.copy()
        for nm,val in zip(names,theta):
            if nm=="R_eff":
                p["R"]=float(val)+p["eta"]
            else:
                p[nm]=float(val)
        return real_symbol(k,tag,p)

    popt,_=curve_fit(
        fitted_curve,k,y,p0=np.asarray(p0),
        bounds=(np.asarray(lo),np.asarray(hi)),maxfev=300000
    )
    pred=fitted_curve(k,*popt)
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    R2=1-ssr/sst if sst>0 else np.nan
    return dict(zip(names,popt)),R2,pred

def relative_error(est,true):
    return 100*(est-true)/(abs(true)+1e-30)
INVISIBLE_OPS={"Adv","Disp"}

def identifiable_combinations_S3(tag):
    spec=MODEL_LIBRARY[tag]; kind=spec["kind"]
    ops=tuple(o for o in spec["ops"] if o not in INVISIBLE_OPS)
    if kind=="CH": return ["A2/A4"]
    if kind=="KS": return ["a2/R","a4/R"]
    if ops==("SH",): return ["ell","R","2*dt (requires known fixed kappa_SH)"]
    if "SH" in ops: return ["net polynomial coefficient ratios; primitive SH/Diff/Hyper separation not claimed"]
    c0="R-eta" if "Loss" in ops else "R"
    out=[]
    if "Diff" in ops: out.append(f"D/({c0})")
    if "Hyper" in ops: out.append(f"nu/({c0})")
    if "Sixth" in ops: out.append(f"mu6/({c0})")
    if "FracDiff" in ops: out += ["alpha",f"Df/({c0})"]
    if "Nonlocal" in ops: out += ["sigma",f"chi/({c0})"]
    return out if out else ["nothing: constant-only real symbol"]

def recover_S3(k,y,tag):
    k=np.asarray(k,float); y=np.asarray(y,float)
    spec=MODEL_LIBRARY[tag]; kind=spec["kind"]
    ops=tuple(o for o in spec["ops"] if o not in INVISIBLE_OPS)

    if kind=="CH":
        cf=np.polyfit(k**2,y,2); pred=np.polyval(cf,k**2)
        b4,b2,b0=cf; rec={}
        if abs(b4)>1e-30: rec["A2/A4"]=-b2/b4
        return rec,pred

    if kind=="KS":
        cf=np.polyfit(k**2,y,2); pred=np.polyval(cf,k**2)
        b4,b2,b0=cf; rec={}
        if abs(b0)>1e-30:
            rec["a2/R"]=b2/b0; rec["a4/R"]=-b4/b0
        return rec,pred

    if ops==("SH",):
        cf=np.polyfit(k**2,y,2); pred=np.polyval(cf,k**2)
        c0,c2,c4=cf[::-1]; rec={}
        if abs(c2)>1e-30:
            ell2=-2*c4/c2
            if ell2>0:
                ell=np.sqrt(ell2)
                s=c2/(2*KAPPA_SH*ell2)
                if abs(s)>1e-30:
                    rec["ell"]=ell
                    rec["2*dt"]=s
                    rec["R"]=c0/s+KAPPA_SH
        return rec,pred

    if "SH" in ops:
        degree=3 if "Sixth" in ops else 2
        cf=np.polyfit(k**2,y,degree); pred=np.polyval(cf,k**2)
        a=cf[::-1]; c0=a[0]; rec={}
        if abs(c0)>1e-30:
            if len(a)>1: rec["net_k2/c0"]=a[1]/c0
            if len(a)>2: rec["net_k4/c0"]=a[2]/c0
            if len(a)>3: rec["net_k6/c0"]=a[3]/c0
        rec["note"]="SH shares spectral powers with other operators; primitive separation not claimed."
        return rec,pred

    has_frac="FracDiff" in ops
    has_nonlocal="Nonlocal" in ops
    has_diff="Diff" in ops
    has_hyper="Hyper" in ops
    has_sixth="Sixth" in ops

    if has_frac or has_nonlocal:
        def model(k,*th):
            it=iter(th); A=next(it)
            out=np.full_like(k,A,dtype=float)
            if has_diff: out+=next(it)*k**2
            if has_hyper: out+=next(it)*k**4
            if has_sixth: out+=next(it)*k**6
            if has_frac:
                B=next(it); al=next(it); out+=B*np.abs(k)**al
            if has_nonlocal:
                C=next(it); sg=next(it); out+=C*(np.exp(-0.5*sg**2*k**2)-1.0)
            return out

        p0=[float(np.mean(y))]; lo=[-np.inf]; hi=[np.inf]
        if has_diff: p0+=[-100]; lo+=[-np.inf]; hi+=[np.inf]
        if has_hyper: p0+=[-1e5]; lo+=[-np.inf]; hi+=[np.inf]
        if has_sixth: p0+=[-1e7]; lo+=[-np.inf]; hi+=[np.inf]
        if has_frac: p0+=[-100,1.5]; lo+=[-np.inf,0.2]; hi+=[np.inf,6]
        if has_nonlocal: p0+=[-1,45]; lo+=[-np.inf,0.1]; hi+=[np.inf,500]

        try:
            popt,_=curve_fit(model,k,y,p0=p0,bounds=(lo,hi),maxfev=300000)
            pred=model(k,*popt)
        except Exception:
            return {"note":"non-polynomial fit failed"},np.full_like(y,np.mean(y))

        it=iter(popt); A=next(it); rec={}
        if has_diff:
            B=next(it)
            if abs(A)>1e-30: rec["D/c0"]=-B/A
        if has_hyper:
            B=next(it)
            if abs(A)>1e-30: rec["nu/c0"]=-B/A
        if has_sixth:
            B=next(it)
            if abs(A)>1e-30: rec["mu6/c0"]=-B/A
        if has_frac:
            B=next(it); al=next(it); rec["alpha"]=al
            if abs(A)>1e-30: rec["Df/c0"]=-B/A
        if has_nonlocal:
            C=next(it); sg=next(it); rec["sigma"]=sg
            if abs(A)>1e-30: rec["chi/c0"]=C/A
        return rec,pred

    has_loss="Loss" in ops
    degree=0
    if has_diff: degree=max(degree,1)
    if has_hyper: degree=max(degree,2)
    if has_sixth: degree=max(degree,3)

    if degree==0:
        return {"note":"constant-only real symbol; time scale not identifiable"},np.full_like(y,np.mean(y))

    cf=np.polyfit(k**2,y,degree); pred=np.polyval(cf,k**2)
    a=cf[::-1]; c0=a[0]
    c0lab="R-eta" if has_loss else "R"
    rec={}
    if abs(c0)>1e-30:
        if has_diff: rec[f"D/({c0lab})"]=-a[1]/c0
        if has_hyper: rec[f"nu/({c0lab})"]=-a[2]/c0
        if has_sixth: rec[f"mu6/({c0lab})"]=-a[3]/c0
    return rec,pred
def scenario1(stack,tag,kmax=None):
    if kmax is None: kmax=K_MAX
    kref,_=psd_1d(stack[0])
    P=np.asarray([psd_1d(fr)[1] for fr in stack])
    lam=np.full(len(kref),np.nan)
    tr2=np.full(len(kref),np.nan)

    for j in range(len(kref)):
        q=np.isfinite(P[:,j])&(P[:,j]>0)
        if q.sum()<6: continue
        lr=linregress(times[q],np.log(P[q,j]))
        lam[j]=lr.slope/2
        tr2[j]=lr.rvalue**2

    v=np.isfinite(lam)&(kref>0)&(kref<kmax)&(tr2>0.5)
    if v.sum()<5: return None
    fit,R2,pred=fit_real_symbol(kref[v],lam[v],tag)
    return {"k":kref[v],"lambda":lam[v],"pred":pred,"fit":fit,"R2":R2,"temporal_R2":tr2[v]}

def scenario2(stack,tag,frame_t=None,kmax=None):
    if frame_t is None: frame_t=FRAME_T
    if kmax is None: kmax=K_MAX
    k,P0=psd_1d(stack[0]); _,Pt=psd_1d(stack[frame_t])
    t=times[frame_t]
    v=np.isfinite(P0)&np.isfinite(Pt)&(P0>0)&(Pt>0)&(k>0)&(k<kmax)
    kv=k[v]; lam=np.log(Pt[v]/P0[v])/(2*t)
    fit,R2,pred=fit_real_symbol(kv,lam,tag)
    return {"k":kv,"lambda":lam,"pred":pred,"fit":fit,"R2":R2}

def scenario3(stack,tag,frame_t=None,kmax=None):
    if frame_t is None: frame_t=FRAME_T
    if kmax is None: kmax=K_MAX
    k,P0=psd_1d(stack[0]); _,Pt=psd_1d(stack[frame_t])
    v=np.isfinite(P0)&np.isfinite(Pt)&(P0>0)&(Pt>0)&(k>0)&(k<kmax)
    kv=k[v]; y=np.log(Pt[v]/P0[v])
    rec,pred=recover_S3(kv,y,tag)
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    return {"k":kv,"y":y,"pred":pred,"recovered":rec,"R2":1-ssr/sst if sst>0 else np.nan}

def snapshot_degree(tag):
    spec=MODEL_LIBRARY[tag]; kind=spec["kind"]; ops=spec["ops"]
    if kind in ("KS","CH"): return 2
    if "Sixth" in ops: return 3
    if "Hyper" in ops or "SH" in ops: return 2
    if "Diff" in ops: return 1
    if "FracDiff" in ops or "Nonlocal" in ops: return 3
    return 0

def scenario4(stack,tag,frame=None,kmin=None,kmax=None):
    if frame is None: frame=SNAP_FRAME
    if kmin is None: kmin=K_MIN_SNAPSHOT
    if kmax is None: kmax=K_MAX_SNAPSHOT
    k,P=psd_1d(stack[frame]); Pn,_=normalize_1d(k,P)
    v=np.isfinite(Pn)&(Pn>0)&(k>=kmin)&(k<=kmax)
    kv=k[v]; y=np.log(Pn[v]+1e-30)
    if len(kv)<5: return None
    deg=snapshot_degree(tag)
    if deg==0:
        coef=np.array([np.mean(y)]); pred=np.full_like(y,coef[0])
    else:
        coef=np.polyfit(kv**2,y,deg); pred=np.polyval(coef,kv**2)
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    return {"k":kv,"logP":y,"coef":coef,"pred":pred,
            "R2":1-ssr/sst if sst>0 else np.nan,
            "status":"single-snapshot spectral-shape diagnostic only"}

def fit_segment_shape(segment,tag):
    segment=np.asarray(segment,float)
    if len(segment)<16 or np.std(segment)<1e-10: return None
    k,P=psd_1d(segment,dx=dx_um); Pn,_=normalize_1d(k,P)
    v=np.isfinite(Pn)&(Pn>0)&(k>=K_MIN_SNAPSHOT)&(k<=K_MAX_SNAPSHOT)
    kv=k[v]; y=np.log(Pn[v]+1e-30)
    if len(kv)<5: return None
    deg=snapshot_degree(tag)
    if deg==0:
        coef=np.array([np.mean(y)]); pred=np.full_like(y,coef[0])
    else:
        coef=np.polyfit(kv**2,y,deg); pred=np.polyval(coef,kv**2)
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    R2=1-ssr/sst if sst>0 else np.nan
    metric=coef[-2] if len(coef)>=2 else coef[-1]
    return {"metric":float(metric),"R2":float(R2)}

def scenario5(stack,tag,frame=None,grid=None,r2_min=None):
    if frame is None: frame=SNAP_FRAME
    if grid is None: grid=GRID_N
    if r2_min is None: r2_min=R2_MIN
    u=stack[frame]; n=len(u)//grid
    rows=[]
    for j in range(grid):
        rr=fit_segment_shape(u[j*n:(j+1)*n],tag)
        if rr is not None: rows.append({"segment":j,**rr})
    if not rows: return None
    df=pd.DataFrame(rows)
    good=df[np.isfinite(df["R2"])&(df["R2"]>=r2_min)].copy()
    return {
        "df_segments":df,
        "median_metric":float(np.nanmedian(good["metric"])) if len(good) else np.nan,
        "mean_metric":float(np.nanmean(good["metric"])) if len(good) else np.nan,
        "median_R2":float(np.nanmedian(good["R2"])) if len(good) else np.nan,
        "mean_R2":float(np.nanmean(good["R2"])) if len(good) else np.nan,
        "n_good":int(len(good)),
        "n_total":int(len(df)),
        "grid":grid,
    }
def fit_segment_RD(segment,R_known,elapsed_time):
    segment=np.asarray(segment,float)
    if len(segment)<16 or np.std(segment)<1e-10: return None
    k,P=psd_1d(segment,dx=dx_um)
    v=np.isfinite(P)&(P>0)&(k>K_MIN_SNAPSHOT)&(k<K_MAX_SNAPSHOT)
    kf=k[v]; Pf=P[v]
    if len(kf)<5: return None
    X=np.column_stack([np.ones(len(kf)),kf**2])
    y=np.log10(Pf+1e-30)
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    A,B=beta; pred=X@beta
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    R2=1-ssr/sst if sst>0 else np.nan
    D_hat=-B*np.log(10)/(2*elapsed_time)
    lodr=np.log10(D_hat/R_known) if D_hat>0 and R_known>0 else np.nan
    return {"D_hat":float(D_hat),"log10_DR":lodr,"R2":float(R2)}

def fit_tiled_RD_1d(img,R_known,elapsed_time,grid_n,r2_min=None):
    if r2_min is None: r2_min=R2_MIN
    n=len(img)//grid_n; rows=[]
    for j in range(grid_n):
        rr=fit_segment_RD(img[j*n:(j+1)*n],R_known,elapsed_time)
        if rr is not None: rows.append({"segment":j,**rr})
    if not rows: return None
    df=pd.DataFrame(rows)
    good=df[np.isfinite(df["log10_DR"])&np.isfinite(df["R2"])&(df["R2"]>=r2_min)].copy()
    if not len(good):
        return {"median":np.nan,"mean":np.nan,"std":np.nan,"n_good":0,"n_total":len(df),"df_segments":df}
    vals=good["log10_DR"].values
    return {"median":float(np.median(vals)),"mean":float(np.mean(vals)),
            "std":float(np.std(vals)),"n_good":len(good),"n_total":len(df),"df_segments":df}
def fmt_dict(d):
    if not isinstance(d,dict): return ""
    out=[]
    for k,v in d.items():
        if isinstance(v,str): out.append(f"{k}: {v}")
        else:
            try: out.append(f"{k}={float(v):.5g}")
            except Exception: out.append(f"{k}={v}")
    return "; ".join(out)

