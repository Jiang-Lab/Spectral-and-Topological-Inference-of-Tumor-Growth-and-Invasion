import os, math, itertools, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from scipy.optimize import curve_fit
from scipy.stats import linregress
from numpy.fft import fftn, ifftn, fftfreq

FULL_RESOLUTION=False
n=64 if FULL_RESOLUTION else 32

L_um=2500.0
dx_um=L_um/n

dt=0.05
n_frames=30
times=np.arange(n_frames)*dt

FRAME_LIST=[5,10,15,20,25,29]
FRAME_T=25
SNAP_FRAME=28

K_MAX=0.08
K_MIN_SNAPSHOT=0.005
K_MAX_SNAPSHOT=0.15

R2_MIN=0.80
GRID_N=2
GRID_LIST=[2,3,4]

KAPPA_SH=1.0

RUN_PER_MODEL_FIGURES=False
SAVE_FIGURES=False
FIG_DIR="master_3d_figures"


def configure(path=None,**overrides):
    g=globals()
    cfg={}
    if path is not None:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update(overrides)
    for k,v in cfg.items():
        g[k]=v
    g["n"]=64 if g["FULL_RESOLUTION"] else 32
    g["dx_um"]=g["L_um"]/g["n"]
    g["times"]=np.arange(g["n_frames"])*g["dt"]
    g["coords"]=np.linspace(0,g["L_um"],g["n"],endpoint=False)
    g["X"],g["Y"],g["Z"]=np.meshgrid(g["coords"],g["coords"],g["coords"],indexing="ij")
    g["k1"]=2*np.pi*fftfreq(g["n"],d=g["dx_um"])
    g["KX"],g["KY"],g["KZ"]=np.meshgrid(g["k1"],g["k1"],g["k1"],indexing="ij")
    g["K2"]=g["KX"]**2+g["KY"]**2+g["KZ"]**2
    g["KMAG"]=np.sqrt(g["K2"])
    if g["SAVE_FIGURES"]:
        os.makedirs(g["FIG_DIR"],exist_ok=True)
    return dict(cfg)

ELEMENTARY={
    "Diff":{"params":{"D":750.0}},
    "Adv":{"params":{"c":120.0}},
    "Hyper":{"params":{"nu":2.0e5}},
    "SH":{"params":{"ell":22.0}},
    "FracDiff":{"params":{"Df":350.0,"alpha":1.5}},
    "Disp":{"params":{"beta3":8.0e4}},
    "Sixth":{"params":{"mu6":4.0e7}},
    "Loss":{"params":{"eta":0.20}},
    "Nonlocal":{"params":{"chi":0.80,"sigma":45.0}},
}
ELEMENTARY_NAMES=list(ELEMENTARY)

MODEL_LIBRARY={}

for opname in ELEMENTARY_NAMES:
    tag="R+"+opname
    true={"R":1.25 if opname=="SH" else 0.65,**ELEMENTARY[opname]["params"]}
    MODEL_LIBRARY[tag]={"kind":"elementary","ops":(opname,),"true":true}

for op1,op2 in itertools.combinations(ELEMENTARY_NAMES,2):
    tag=f"R+{op1}+{op2}"
    true={"R":1.25 if "SH" in (op1,op2) else 0.65}
    true.update(ELEMENTARY[op1]["params"])
    true.update(ELEMENTARY[op2]["params"])
    MODEL_LIBRARY[tag]={"kind":"pairwise","ops":(op1,op2),"true":true}

CORE=("Diff","Adv","Hyper","SH")
for n_ops in (3,4):
    for ops in itertools.combinations(CORE,n_ops):
        tag="R+"+"+".join(ops)
        true={"R":1.25 if "SH" in ops else 0.65}
        if "Diff" in ops: true["D"]=300.0 if "SH" in ops else 500.0
        if "Adv" in ops: true["c"]=120.0
        if "Hyper" in ops: true["nu"]=1.0e5
        if "SH" in ops: true["ell"]=22.0
        MODEL_LIBRARY[tag]={"kind":"core_mixed","ops":ops,"true":true}

MODEL_LIBRARY["R+KSlin"]={
    "kind":"KS","ops":("KSlin",),
    "true":{"R":0.65,"a2":450.0,"a4":1.5e5}
}
MODEL_LIBRARY["CHlin"]={
    "kind":"CH","ops":("CHlin",),
    "true":{"A2":450.0,"A4":1.5e5}
}
MODEL_LIBRARY["R+Mixed"]={
    "kind":"mixed","ops":("Diff","Adv","Disp","Hyper"),
    "true":{"R":0.65,"D":500.0,"c":120.0,"beta3":8.0e4,"nu":1.0e5}
}

MODEL_VARIANTS=list(MODEL_LIBRARY)
assert len(MODEL_VARIANTS)==53

SPECIAL_MODELS={
    "R+Telegraph":{
        "equation":"u_tt + tau^{-1}u_t = cT^2 Laplacian(u) + R u",
        "true":{"R":0.50,"tau":0.80,"cT":90.0}
    },
    "R+DelayDiff":{
        "equation":"u_t = R u(x,t-tau) + D Laplacian(u)",
        "true":{"R":0.50,"D":500.0,"tau":0.50}
    }
}

coords=np.linspace(0,L_um,n,endpoint=False)
X,Y,Z=np.meshgrid(coords,coords,coords,indexing="ij")

def pdist(a,b,L=None):
    if L is None: L=L_um
    d=np.abs(a-b)
    return np.minimum(d,L-d)

def gaussian3_periodic(cx,cy,cz,sigma,amplitude=1.0):
    dx=pdist(X,cx); dy=pdist(Y,cy); dz=pdist(Z,cz)
    r2=dx**2+dy**2+dz**2
    return amplitude*np.exp(-0.5*r2/sigma**2)

def normalize_field(u):
    u=np.asarray(u,float)
    u=u-u.min()
    return u/(u.max()+1e-30)

def make_initial_conditions_3d():
    rng=np.random.default_rng(42)
    z=rng.normal(0,1,(n,n,n))
    kk=2*np.pi*fftfreq(n,d=dx_um)
    kx,ky,kz=np.meshgrid(kk,kk,kk,indexing="ij")
    km=np.sqrt(kx**2+ky**2+kz**2)
    filt=np.exp(-(km/0.05)**4)
    random_smooth=normalize_field(np.real(ifftn(fftn(z)*filt)))

    one=normalize_field(gaussian3_periodic(L_um/2,L_um/2,L_um/2,220))

    many=np.zeros((n,n,n),float)
    centers=[
        (420,430,450),(850,700,1250),(1250,1250,1250),
        (1750,850,650),(2050,1850,1700),(700,1850,2000),
        (1750,1750,450),(450,1350,1750)
    ]
    for c0 in centers:
        many+=gaussian3_periodic(*c0,95)
    many=normalize_field(many)

    mixed=many.copy()
    singles=[
        (250,1150,350),(1100,300,1950),(1500,2100,1200),
        (2200,900,2200),(600,2200,850)
    ]
    for c0 in singles:
        mixed+=0.45*gaussian3_periodic(*c0,40)
    mixed=normalize_field(mixed)

    return {
        "Random smooth":random_smooth,
        "One central cluster":one,
        "Many clusters":many,
        "Clusters + singles":mixed
    }


k1=2*np.pi*fftfreq(n,d=dx_um)
KX,KY,KZ=np.meshgrid(k1,k1,k1,indexing="ij")
K2=KX**2+KY**2+KZ**2
KMAG=np.sqrt(K2)

def model_parameters(tag):
    return MODEL_LIBRARY[tag]["true"].copy()

def operator_symbol_3d(op,p):
    if op=="Diff": return -p["D"]*K2
    if op=="Adv": return -1j*p["c"]*KX
    if op=="Hyper": return -p["nu"]*K2**2
    if op=="SH": return -KAPPA_SH*(1-p["ell"]**2*K2)**2
    if op=="FracDiff": return -p["Df"]*KMAG**p["alpha"]
    if op=="Disp": return 1j*p["beta3"]*KX**3
    if op=="Sixth": return -p["mu6"]*K2**3
    if op=="Loss": return -p["eta"]+0*K2
    if op=="Nonlocal": return p["chi"]*(np.exp(-0.5*p["sigma"]**2*K2)-1)
    raise ValueError(op)

def complex_symbol_3d(tag,p):
    spec=MODEL_LIBRARY[tag]
    if spec["kind"]=="CH":
        return p["A2"]*K2-p["A4"]*K2**2

    lam=np.zeros_like(K2,dtype=complex)+p.get("R",0.0)

    if spec["kind"]=="KS":
        return lam+p["a2"]*K2-p["a4"]*K2**2

    for op in spec["ops"]:
        lam+=operator_symbol_3d(op,p)
    return lam

def real_symbol_radial(k,tag,p):
    k=np.asarray(k,float)
    spec=MODEL_LIBRARY[tag]

    if spec["kind"]=="CH":
        return p["A2"]*k**2-p["A4"]*k**4

    lam=np.zeros_like(k)+p.get("R",0.0)

    if spec["kind"]=="KS":
        return lam+p["a2"]*k**2-p["a4"]*k**4

    for op in spec["ops"]:
        if op=="Diff": lam-=p["D"]*k**2
        elif op=="Hyper": lam-=p["nu"]*k**4
        elif op=="SH": lam-=KAPPA_SH*(1-p["ell"]**2*k**2)**2
        elif op=="FracDiff": lam-=p["Df"]*np.abs(k)**p["alpha"]
        elif op=="Sixth": lam-=p["mu6"]*k**6
        elif op=="Loss": lam-=p["eta"]
        elif op=="Nonlocal": lam+=p["chi"]*(np.exp(-0.5*p["sigma"]**2*k**2)-1)
        # Adv and Disp are purely imaginary.
    return lam

def simulate_linear_3d(u0,tag,p):
    lam=complex_symbol_3d(tag,p)
    growth=np.exp(lam*dt)
    u=np.asarray(u0,float).copy()
    stack=[]
    for _ in range(n_frames):
        stack.append(u.copy())
        u=np.real(ifftn(fftn(u)*growth))
    return np.asarray(stack)

def radial_psd_3d(u,dx=None):
    if dx is None: dx=dx_um
    # Spherical-shell average of the 3D Fourier power.
    u=np.asarray(u,float)
    m=u.shape[0]
    z=u-np.mean(u)
    F=fftn(z)
    P=np.abs(F)**2/z.size

    kk=2*np.pi*fftfreq(m,d=dx)
    kx,ky,kz=np.meshgrid(kk,kk,kk,indexing="ij")
    km=np.sqrt(kx**2+ky**2+kz**2)

    dk=2*np.pi/(m*dx)
    bins=np.floor(km.ravel()/dk+0.5).astype(int)

    psum=np.bincount(bins,weights=P.ravel())
    count=np.bincount(bins)
    ksum=np.bincount(bins,weights=km.ravel())

    good=count>0
    kval=ksum[good]/count[good]
    pval=psum[good]/count[good]

    keep=kval>0
    return kval[keep],pval[keep]

def normalize_radial_psd(k,P):
    use=np.isfinite(k)&np.isfinite(P)&(P>0)&(k>0)
    if use.sum()<3:
        return np.full_like(P,np.nan,float),np.nan
    norm=np.trapezoid(P[use],k[use])
    return P/(norm+1e-30),norm

EVOLUTION_FRAMES=[0,5,10,15,20,29]

PLOT_EVOLUTION_FOR_ALL_MODELS=False

EVOLUTION_MODELS=[
    "R+Diff",
    "R+Adv",
    "R+Hyper",
    "R+SH",
    "R+FracDiff",
    "R+KSlin",
    "CHlin"
]

def normalize_snapshot(u):
    u=np.asarray(u,float)
    return (u-np.nanmin(u))/(np.nanmax(u)-np.nanmin(u)+1e-30)

def plot_pde_evolution_3d(
    u0,
    tag,
    ic_name,
    frames=None,
    show_absolute=True,
    show_normalized=True,
    show_psd=True
):
    if frames is None: frames=EVOLUTION_FRAMES
    p=model_parameters(tag)
    stack=simulate_linear_3d(u0,tag,p)
    mid=stack.shape[-1]//2

    if show_absolute:
        selected=[stack[f,:,:,mid] for f in frames]
        vmin=min(np.nanmin(a) for a in selected)
        vmax=max(np.nanmax(a) for a in selected)

        fig,axs=plt.subplots(
            1,len(frames),
            figsize=(3.0*len(frames),3.2),
            constrained_layout=True
        )
        axs=np.atleast_1d(axs)

        for ax,fid in zip(axs,frames):
            im=ax.imshow(
                stack[fid,:,:,mid].T,
                origin="lower",
                extent=[0,L_um,0,L_um],
                aspect="equal",
                vmin=vmin,
                vmax=vmax
            )
            ax.set_title(f"t={times[fid]:.2f}")
            ax.set_xlabel("x (um)")
        axs[0].set_ylabel("y (um)")

        fig.colorbar(im,ax=axs.tolist(),shrink=.78,label="u")
        fig.suptitle(
            f"{tag} | {ic_name}\nAbsolute PDE evolution",
            fontweight="bold"
        )
        plt.show()

    if show_normalized:
        fig,axs=plt.subplots(
            1,len(frames),
            figsize=(3.0*len(frames),3.2),
            constrained_layout=True
        )
        axs=np.atleast_1d(axs)

        for ax,fid in zip(axs,frames):
            us=normalize_snapshot(stack[fid])
            im=ax.imshow(
                us[:,:,mid].T,
                origin="lower",
                extent=[0,L_um,0,L_um],
                aspect="equal",
                vmin=0,
                vmax=1
            )
            ax.set_title(f"t={times[fid]:.2f}")
            ax.set_xlabel("x (um)")
        axs[0].set_ylabel("y (um)")

        fig.colorbar(im,ax=axs.tolist(),shrink=.78,label="Normalized field")
        fig.suptitle(
            f"{tag} | {ic_name}\nNormalized morphology",
            fontweight="bold"
        )
        plt.show()

    if show_psd:
        fig,ax=plt.subplots(figsize=(8,5))
        for fid in frames:
            k,P=radial_psd_3d(stack[fid])
            use=np.isfinite(k)&np.isfinite(P)&(P>0)&(k>0)&(k<=K_MAX_SNAPSHOT)
            if np.any(use):
                ax.semilogy(
                    k[use],P[use],
                    linewidth=1.6,
                    label=f"t={times[fid]:.2f}"
                )

        ax.set_xlabel("Radial wavenumber k")
        ax.set_ylabel("3D radial PSD")
        ax.set_title(f"{tag} | {ic_name} — PSD evolution")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8,ncol=2)
        plt.tight_layout()
        plt.show()

    return stack

def plot_initial_vs_final_3d(u0,tag,ic_name,final_frame=29):
    p=model_parameters(tag)
    stack=simulate_linear_3d(u0,tag,p)
    mid=stack.shape[-1]//2

    arr0=stack[0,:,:,mid]
    arr1=stack[final_frame,:,:,mid]

    vmin=min(arr0.min(),arr1.min())
    vmax=max(arr0.max(),arr1.max())

    fig,axs=plt.subplots(1,2,figsize=(8,3.8),constrained_layout=True)

    im=axs[0].imshow(
        arr0.T,origin="lower",
        extent=[0,L_um,0,L_um],
        aspect="equal",
        vmin=vmin,vmax=vmax
    )
    axs[0].set_title("Initial")
    axs[0].set_xlabel("x (um)")
    axs[0].set_ylabel("y (um)")

    axs[1].imshow(
        arr1.T,origin="lower",
        extent=[0,L_um,0,L_um],
        aspect="equal",
        vmin=vmin,vmax=vmax
    )
    axs[1].set_title(f"After PDE, t={times[final_frame]:.2f}")
    axs[1].set_xlabel("x (um)")

    fig.colorbar(im,ax=axs.tolist(),shrink=.8,label="u")
    fig.suptitle(f"{tag} | {ic_name}: initial vs evolved",fontweight="bold")
    plt.show()

    return stack

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
    p=MODEL_LIBRARY[tag]["true"]; ops=set(MODEL_LIBRARY[tag]["ops"])
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
    spec=MODEL_LIBRARY[tag]; ops=spec["ops"]
    if spec["kind"]=="CH": return ["A2","A4"]
    if spec["kind"]=="KS": return ["R","a2","a4"]
    if primitive_confounded(tag):
        names=["c0","c2","c4"]
        if "Sixth" in ops: names.append("c6")
        return names
    names=["R"]
    for op in ops:
        if op=="Diff": names.append("D")
        elif op=="Hyper": names.append("nu")
        elif op=="SH": names.append("ell")
        elif op=="FracDiff": names+=["Df","alpha"]
        elif op=="Sixth": names.append("mu6")
        elif op=="Loss":
            if "R" in names: names.remove("R")
            names.append("R_eff")
        elif op=="Nonlocal": names+=["chi","sigma"]
    return list(dict.fromkeys(names))

def true_identifiable_value(tag,name):
    p=MODEL_LIBRARY[tag]["true"]
    if name in p: return p[name]
    if name=="R_eff": return p["R"]-p["eta"]
    if name in ("c0","c2","c4","c6"): return effective_poly_truth(tag)[name]
    return np.nan

def relative_error(est,true):
    return 100*(est-true)/(abs(true)+1e-30)

def fit_real_symbol(k,y,tag):
    k=np.asarray(k,float); y=np.asarray(y,float)
    names=identifiable_names(tag); true=MODEL_LIBRARY[tag]["true"]

    if len(names)==1 and names[0] in ("R","R_eff"):
        val=float(np.mean(y))
        return {names[0]:val},np.nan,np.full_like(y,val)

    if any(nm in names for nm in ("c0","c2","c4","c6")):
        degree=3 if "c6" in names else 2
        cf=np.polyfit(k**2,y,degree)
        pred=np.polyval(cf,k**2)
        asc=cf[::-1]
        fit={"c0":asc[0],"c2":asc[1],"c4":asc[2]}
        if degree==3: fit["c6"]=asc[3]
        ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
        return fit,(1-ssr/sst if sst>0 else np.nan),pred

    def unpack(theta):
        p=true.copy()
        for nm,val in zip(names,theta):
            if nm=="R_eff": p["R"]=float(val)+p["eta"]
            else: p[nm]=float(val)
        return p

    p0=[]; lo=[]; hi=[]
    for nm in names:
        tv=true_identifiable_value(tag,nm)
        if nm in ("R","R_eff","a2","A2"):
            p0.append(tv*.9 if tv!=0 else .01); lo.append(-20*abs(tv)-10); hi.append(20*abs(tv)+10)
        elif nm=="alpha":
            p0.append(tv*.9); lo.append(.2); hi.append(6)
        elif nm in ("ell","sigma"):
            p0.append(tv*.9); lo.append(.1); hi.append(500)
        else:
            p0.append(tv*.9); lo.append(0); hi.append(max(1,20*abs(tv)))

    def f(k,*theta):
        return real_symbol_radial(k,tag,unpack(theta))

    popt,_=curve_fit(f,k,y,p0=p0,bounds=(lo,hi),maxfev=300000)
    pred=f(k,*popt)
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    R2=1-ssr/sst if sst>0 else np.nan
    return dict(zip(names,popt)),R2,pred

def recover_S3(k,y,tag):
    k=np.asarray(k,float); y=np.asarray(y,float)
    spec=MODEL_LIBRARY[tag]
    ops=tuple(o for o in spec["ops"] if o not in ("Adv","Disp"))

    if spec["kind"]=="CH":
        cf=np.polyfit(k**2,y,2); pred=np.polyval(cf,k**2)
        b4,b2,b0=cf
        return ({"A2/A4":-b2/b4} if abs(b4)>1e-30 else {}),pred

    if spec["kind"]=="KS":
        cf=np.polyfit(k**2,y,2); pred=np.polyval(cf,k**2)
        b4,b2,b0=cf; rec={}
        if abs(b0)>1e-30:
            rec["a2/R"]=b2/b0
            rec["a4/R"]=-b4/b0
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
                    rec={"ell":ell,"2*dt":s,"R":c0/s+KAPPA_SH}
        return rec,pred

    if "SH" in ops:
        degree=3 if "Sixth" in ops else 2
        cf=np.polyfit(k**2,y,degree); pred=np.polyval(cf,k**2)
        asc=cf[::-1]; c0=asc[0]
        rec={"note":"effective polynomial ratios only"}
        if abs(c0)>1e-30:
            if len(asc)>1: rec["net_k2/c0"]=asc[1]/c0
            if len(asc)>2: rec["net_k4/c0"]=asc[2]/c0
            if len(asc)>3: rec["net_k6/c0"]=asc[3]/c0
        return rec,pred

    has_frac="FracDiff" in ops
    has_nonlocal="Nonlocal" in ops
    has_diff="Diff" in ops
    has_hyper="Hyper" in ops
    has_sixth="Sixth" in ops
    has_loss="Loss" in ops

    if has_frac or has_nonlocal:
        def model(k,*th):
            it=iter(th)
            A=next(it)
            out=np.full_like(k,A,float)
            if has_diff: out+=next(it)*k**2
            if has_hyper: out+=next(it)*k**4
            if has_sixth: out+=next(it)*k**6
            if has_frac:
                B=next(it); alpha=next(it)
                out+=B*np.abs(k)**alpha
            if has_nonlocal:
                C=next(it); sigma=next(it)
                out+=C*(np.exp(-.5*sigma**2*k**2)-1)
            return out

        p0=[np.mean(y)]; lo=[-np.inf]; hi=[np.inf]
        if has_diff: p0+= [-100]; lo+= [-np.inf]; hi+=[np.inf]
        if has_hyper: p0+= [-1e5]; lo+= [-np.inf]; hi+=[np.inf]
        if has_sixth: p0+= [-1e7]; lo+= [-np.inf]; hi+=[np.inf]
        if has_frac: p0+= [-100,1.5]; lo+= [-np.inf,.2]; hi+=[np.inf,6]
        if has_nonlocal: p0+= [-1,45]; lo+= [-np.inf,.1]; hi+=[np.inf,500]

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
            B=next(it); alpha=next(it)
            rec["alpha"]=alpha
            if abs(A)>1e-30: rec["Df/c0"]=-B/A
        if has_nonlocal:
            C=next(it); sigma=next(it)
            rec["sigma"]=sigma
            if abs(A)>1e-30: rec["chi/c0"]=C/A
        return rec,pred

    degree=0
    if has_diff: degree=max(degree,1)
    if has_hyper: degree=max(degree,2)
    if has_sixth: degree=max(degree,3)

    if degree==0:
        return {"note":"constant-only real symbol; unknown-time scale unidentifiable"},np.full_like(y,np.mean(y))

    cf=np.polyfit(k**2,y,degree)
    pred=np.polyval(cf,k**2)
    asc=cf[::-1]
    c0=asc[0]
    label="R-eta" if has_loss else "R"
    rec={}
    if abs(c0)>1e-30:
        if has_diff: rec[f"D/({label})"]=-asc[1]/c0
        if has_hyper: rec[f"nu/({label})"]=-asc[2]/c0
        if has_sixth: rec[f"mu6/({label})"]=-asc[3]/c0
    return rec,pred

def scenario1(stack,tag,kmax=None):
    if kmax is None: kmax=K_MAX
    kref,_=radial_psd_3d(stack[0])
    P=np.asarray([radial_psd_3d(fr)[1] for fr in stack])

    lam=np.full(len(kref),np.nan)
    temporal_r2=np.full(len(kref),np.nan)

    for j in range(len(kref)):
        valid=np.isfinite(P[:,j])&(P[:,j]>0)
        if valid.sum()<6: continue
        lr=linregress(times[valid],np.log(P[valid,j]))
        lam[j]=lr.slope/2
        temporal_r2[j]=lr.rvalue**2

    use=np.isfinite(lam)&(kref>0)&(kref<kmax)&(temporal_r2>0.5)
    if use.sum()<5: return None

    fit,R2,pred=fit_real_symbol(kref[use],lam[use],tag)
    return {"k":kref[use],"lambda":lam[use],"pred":pred,"fit":fit,"R2":R2,
            "temporal_R2":temporal_r2[use]}

def scenario2(stack,tag,frame_t=None,kmax=None):
    if frame_t is None: frame_t=FRAME_T
    if kmax is None: kmax=K_MAX
    k,P0=radial_psd_3d(stack[0])
    _,Pt=radial_psd_3d(stack[frame_t])
    elapsed=times[frame_t]
    use=np.isfinite(P0)&np.isfinite(Pt)&(P0>0)&(Pt>0)&(k>0)&(k<kmax)
    kv=k[use]
    lam=np.log(Pt[use]/P0[use])/(2*elapsed)
    fit,R2,pred=fit_real_symbol(kv,lam,tag)
    return {"k":kv,"lambda":lam,"pred":pred,"fit":fit,"R2":R2}

def scenario3(stack,tag,frame_t=None,kmax=None):
    if frame_t is None: frame_t=FRAME_T
    if kmax is None: kmax=K_MAX
    k,P0=radial_psd_3d(stack[0])
    _,Pt=radial_psd_3d(stack[frame_t])
    use=np.isfinite(P0)&np.isfinite(Pt)&(P0>0)&(Pt>0)&(k>0)&(k<kmax)
    kv=k[use]
    y=np.log(Pt[use]/P0[use])
    rec,pred=recover_S3(kv,y,tag)
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    R2=1-ssr/sst if sst>0 else np.nan
    return {"k":kv,"y":y,"pred":pred,"recovered":rec,"R2":R2}

def snapshot_degree(tag):
    spec=MODEL_LIBRARY[tag]; ops=spec["ops"]
    if spec["kind"] in ("KS","CH"): return 2
    if "Sixth" in ops: return 3
    if "Hyper" in ops or "SH" in ops: return 2
    if "Diff" in ops: return 1
    if "FracDiff" in ops or "Nonlocal" in ops: return 3
    return 0

def scenario4(stack,tag,frame=None,kmin=None,kmax=None):
    if frame is None: frame=SNAP_FRAME
    if kmin is None: kmin=K_MIN_SNAPSHOT
    if kmax is None: kmax=K_MAX_SNAPSHOT
    k,P=radial_psd_3d(stack[frame])
    Pn,_=normalize_radial_psd(k,P)
    use=np.isfinite(Pn)&(Pn>0)&(k>=kmin)&(k<=kmax)
    kv=k[use]; y=np.log(Pn[use]+1e-30)
    if len(kv)<5: return None

    degree=snapshot_degree(tag)
    if degree==0:
        coef=np.array([np.mean(y)])
        pred=np.full_like(y,coef[0])
    else:
        coef=np.polyfit(kv**2,y,degree)
        pred=np.polyval(coef,kv**2)

    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    R2=1-ssr/sst if sst>0 else np.nan
    return {"k":kv,"logP":y,"coef":coef,"pred":pred,"R2":R2,
            "status":"single-snapshot radial spectral-shape diagnostic only"}

def fit_cube_shape(cube,tag):
    if cube.size<64 or np.std(cube)<1e-10: return None
    k,P=radial_psd_3d(cube,dx=dx_um)
    Pn,_=normalize_radial_psd(k,P)
    use=np.isfinite(Pn)&(Pn>0)&(k>=K_MIN_SNAPSHOT)&(k<=K_MAX_SNAPSHOT)
    kv=k[use]; y=np.log(Pn[use]+1e-30)
    if len(kv)<4: return None

    degree=snapshot_degree(tag)
    if degree==0:
        coef=np.array([np.mean(y)])
        pred=np.full_like(y,coef[0])
    else:
        degree=min(degree,max(1,len(kv)-2))
        coef=np.polyfit(kv**2,y,degree)
        pred=np.polyval(coef,kv**2)

    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    R2=1-ssr/sst if sst>0 else np.nan
    metric=coef[-2] if len(coef)>=2 else coef[-1]
    return {"metric":float(metric),"R2":float(R2)}

def scenario5(stack,tag,frame=None,grid=None,r2_min=None):
    if frame is None: frame=SNAP_FRAME
    if grid is None: grid=GRID_N
    if r2_min is None: r2_min=R2_MIN
    vol=stack[frame]
    m=vol.shape[0]//grid
    rows=[]

    for i in range(grid):
        for j in range(grid):
            for q in range(grid):
                cube=vol[i*m:(i+1)*m,j*m:(j+1)*m,q*m:(q+1)*m]
                rr=fit_cube_shape(cube,tag)
                if rr is not None:
                    rows.append({"i":i,"j":j,"k":q,**rr})

    if not rows: return None

    df=pd.DataFrame(rows)
    good=df[np.isfinite(df["R2"])&(df["R2"]>=r2_min)]

    return {
        "df_tiles":df,
        "median_metric":float(np.nanmedian(good["metric"])) if len(good) else np.nan,
        "mean_metric":float(np.nanmean(good["metric"])) if len(good) else np.nan,
        "median_R2":float(np.nanmedian(good["R2"])) if len(good) else np.nan,
        "mean_R2":float(np.nanmean(good["R2"])) if len(good) else np.nan,
        "n_good":int(len(good)),
        "n_total":int(len(df)),
        "grid":grid
    }

def fit_cube_RD(cube,R_known,elapsed_time):
    if cube.size<64 or np.std(cube)<1e-10: return None
    k,P=radial_psd_3d(cube,dx=dx_um)
    use=np.isfinite(P)&(P>0)&(k>K_MIN_SNAPSHOT)&(k<K_MAX_SNAPSHOT)
    kf=k[use]; Pf=P[use]
    if len(kf)<4: return None

    x2=kf**2
    y=np.log10(Pf+1e-30)
    Xmat=np.column_stack([np.ones_like(x2),x2])
    beta,*_=np.linalg.lstsq(Xmat,y,rcond=None)
    pred=Xmat@beta
    A,B=beta

    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    R2=1-ssr/sst if sst>0 else np.nan

    D_hat=-B*np.log(10)/(2*elapsed_time)
    log10_DR=np.log10(D_hat/R_known) if D_hat>0 and R_known>0 else np.nan
    return {"D_hat":float(D_hat),"log10_DR":float(log10_DR) if np.isfinite(log10_DR) else np.nan,
            "R2":float(R2)}

def fit_tiled_RD_3d(vol,R_known,elapsed_time,grid_n,r2_min=None):
    if r2_min is None: r2_min=R2_MIN
    m=vol.shape[0]//grid_n
    rows=[]
    for i in range(grid_n):
        for j in range(grid_n):
            for q in range(grid_n):
                cube=vol[i*m:(i+1)*m,j*m:(j+1)*m,q*m:(q+1)*m]
                rr=fit_cube_RD(cube,R_known,elapsed_time)
                if rr is not None:
                    rows.append({"i":i,"j":j,"k":q,**rr})

    if not rows: return None
    df=pd.DataFrame(rows)
    good=df[np.isfinite(df["log10_DR"])&np.isfinite(df["R2"])&(df["R2"]>=r2_min)]
    return {
        "median":float(np.median(good["log10_DR"])) if len(good) else np.nan,
        "mean":float(np.mean(good["log10_DR"])) if len(good) else np.nan,
        "std":float(np.std(good["log10_DR"])) if len(good) else np.nan,
        "n_good":int(len(good)),
        "n_total":int(len(df)),
        "df_tiles":df
    }


def fmt_dict(d):
    if not isinstance(d,dict): return ""
    out=[]
    for k,v in d.items():
        if isinstance(v,str):
            out.append(f"{k}: {v}")
        else:
            try: out.append(f"{k}={float(v):.5g}")
            except Exception: out.append(f"{k}={v}")
    return "; ".join(out)

