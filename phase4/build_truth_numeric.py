"""
build_truth_numeric.py  —  PHASE 5 truth WITHOUT external downloads
====================================================================
Truth = high-fidelity numerical propagation (two-body + J2 + J3 + J4 + drag),
which is MORE accurate than SGP4 for LEO -> residual = SGP4 - truth is the real
SGP4 modelling error and IS learnable. Uses your real TLEs for the SGP4 baseline
and your real dataset for input features/states. No CDDIS/IGS/OEM files needed.

Emits the SAME tensors phase5_TRAINER_rtn.py consumes:
  X_*.pt  y_*.pt(truth)  baseline_*.pt(SGP4)  age_*.pt  residual_rtn_*.pt
  feature_scaler.pkl  truth_meta.json

Run from the phase4/ folder (where idsmass_data/ lives):
  python build_truth_numeric.py
Then:  python phase5_TRAINER_rtn.py
"""
import os, json, math, time, joblib, numpy as np, pandas as pd, torch
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import StandardScaler
from sgp4.api import Satrec, jday

CFG = {
    "DATASET": "idsmass_data/processed/historical_trajectory_dataset_v2.csv",
    "TLE_CSV": "idsmass_data/raw/spacetrack/historical_tles.csv",
    "SAVE_DIR": ".",
    "SEQ": 30, "HORIZON_DAYS": 1.0, "DT": 120.0,
    "MAX_PER_SAT": 1500, "TRAIN_FRACTION": 0.8, "USE_DRAG": True, "SEED": 42,
}
MU=398600.4418; RE=6378.137
J2=1.08262668e-3; J3=-2.53265648e-6; J4=-1.61962159e-6
OMEGA_E=7.2921159e-5  # rad/s
FEAT=["x_km","y_km","z_km","vx_kms","vy_kms","vz_kms","semi_major_axis","eccentricity","inclination",
"raan","arg_perigee","true_anomaly","specific_energy","angular_momentum_mag","bstar","f107","kp",
"sin_doy","cos_doy","sin_tod","cos_tod","mass_kg","cross_section_m2"]
# Vallado exponential atmosphere (base_alt km, rho0 kg/m^3, scale_height km)
ATM=np.array([[350,7.014e-12,53.298],[400,2.803e-12,58.515],[450,1.184e-12,60.828],
[500,5.215e-13,63.822],[600,1.137e-13,71.835],[700,3.070e-14,88.667],
[800,1.136e-14,124.64],[900,5.759e-15,181.05],[1000,3.561e-15,268.00]])
def density(alt_km):
    alt=np.clip(alt_km,ATM[0,0],ATM[-1,0]); idx=np.searchsorted(ATM[:,0],alt,side="right")-1
    idx=np.clip(idx,0,len(ATM)-1); b=ATM[idx]
    return b[:,1]*np.exp(-(alt-b[:,0])/b[:,2])  # kg/m^3

# ---- forces (vectorized, FD gradient of zonal potential for J2/J3/J4) ----
def Vz(r):
    rn=np.linalg.norm(r,axis=1); s=r[:,2]/rn
    p=(J2*(RE/rn)**2*0.5*(3*s**2-1)+J3*(RE/rn)**3*0.5*(5*s**3-3*s)
       +J4*(RE/rn)**4*(35*s**4-30*s**2+3)/8)
    return MU/rn*(1-p)
def grav(r,eps=1e-3):
    a=np.empty_like(r)
    for k in range(3):
        rp=r.copy(); rp[:,k]+=eps; rm=r.copy(); rm[:,k]-=eps; a[:,k]=(Vz(rp)-Vz(rm))/(2*eps)
    return a
def drag(r,v,B):  # B = Cd*A/m [m^2/kg], per-window
    rn=np.linalg.norm(r,axis=1); alt=rn-RE; rho=density(alt)            # kg/m^3
    vr=v.copy(); vr[:,0]+=OMEGA_E*r[:,1]; vr[:,1]-=OMEGA_E*r[:,0]        # v - omega x r (km/s)
    vr_ms=vr*1e3; sp=np.linalg.norm(vr_ms,axis=1,keepdims=True)          # m/s
    a_ms=-0.5*(B[:,None]*rho[:,None])*sp*vr_ms                           # m/s^2
    return a_ms*1e-3                                                     # km/s^2
def deriv(st,B,use_drag):
    r,v=st[:,:3],st[:,3:]; a=grav(r)
    if use_drag: a=a+drag(r,v,B)
    return np.concatenate([v,a],axis=1)
def rk4(st,B,steps,dt,use_drag):
    for _ in range(steps):
        k1=deriv(st,B,use_drag); k2=deriv(st+0.5*dt*k1,B,use_drag)
        k3=deriv(st+0.5*dt*k2,B,use_drag); k4=deriv(st+dt*k3,B,use_drag)
        st=st+dt/6*(k1+2*k2+2*k3+k4)
    return st
def to_rtn(vec,r,v):
    R=r/np.linalg.norm(r); W=np.cross(r,v); W=W/np.linalg.norm(W); S=np.cross(W,R)
    return np.array([vec@R,vec@S,vec@W],np.float32)

def tle_epoch(t1):
    yy=int(t1[18:20]); yr=2000+yy if yy<57 else 1900+yy
    return datetime(yr,1,1,tzinfo=timezone.utc)+timedelta(days=float(t1[20:32])-1)

def main():
    c=CFG; np.random.seed(c["SEED"]); os.makedirs(c["SAVE_DIR"],exist_ok=True)
    df=pd.read_csv(c["DATASET"]); df["timestamp"]=pd.to_datetime(df["timestamp"],format="mixed",utc=True)
    df=df.dropna(subset=FEAT)
    tle=pd.read_csv(c["TLE_CSV"]); tle["epoch"]=tle["tle1"].apply(tle_epoch)
    tidx={}
    for nid,g in tle.groupby("norad_id"):
        g=g.sort_values("epoch").reset_index(drop=True)
        tidx[int(nid)]={"ep":g["epoch"].values.astype("datetime64[ns]"),
                        "t1":g["tle1"].tolist(),"t2":g["tle2"].tolist()}
    seq=c["SEQ"]; hor=np.timedelta64(int(c["HORIZON_DAYS"]*86400),"s"); Cd=2.2

    Xs,IC,Bal,Bcoef,Tt,sat,age=[],[],[],[],[],[],[]
    for nid,sub in df.groupby("norad_id"):
        nid=int(nid)
        if nid not in tidx: continue
        sub=sub.sort_values("timestamp").reset_index(drop=True)
        F=sub[FEAT].values.astype(np.float32)
        S=sub[["x_km","y_km","z_km","vx_kms","vy_kms","vz_kms"]].values.astype(np.float64)
        T=sub["timestamp"].values.astype("datetime64[ns]")
        mass=sub["mass_kg"].values; area=sub["cross_section_m2"].values
        last=len(sub)-seq
        if last<=0: continue
        idxs=np.arange(last)
        if len(idxs)>c["MAX_PER_SAT"]: idxs=idxs[np.linspace(0,len(idxs)-1,c["MAX_PER_SAT"]).astype(int)]
        te=tidx[nid]
        for i in idxs:
            j=i+seq-1; in_t=T[j]; tgt_t=in_t+hor
            k=max(0,int(np.searchsorted(te["ep"],in_t,side="right"))-1)
            ts=pd.Timestamp(tgt_t).to_pydatetime()
            satrec=Satrec.twoline2rv(te["t1"][k],te["t2"][k])
            jd,fr=jday(ts.year,ts.month,ts.day,ts.hour,ts.minute,ts.second+ts.microsecond/1e6)
            err,r,v=satrec.sgp4(jd,fr)
            if err!=0: continue
            Xs.append(F[i:i+seq]); IC.append(S[j]); Bal.append([*r,*v])
            Bcoef.append(Cd*max(area[j],1.0)/max(mass[j],1.0))  # m^2/kg
            sat.append(nid); age.append(float((tgt_t-te["ep"][k])/np.timedelta64(1,"D")))
    Xs=np.array(Xs,np.float32); IC=np.array(IC,np.float64); Bal=np.array(Bal,np.float32)
    B=np.array(Bcoef,np.float64); sat=np.array(sat); age=np.array(age,np.float32)
    print(f"windows: {len(IC)}  ({df.norad_id.nunique()} sats)")

    steps=int(round(c["HORIZON_DAYS"]*86400/c["DT"])); t0=time.time()
    truth=rk4(IC.copy(),B,steps,c["DT"],c["USE_DRAG"]).astype(np.float32)
    print(f"high-fidelity propagation {len(IC)}x{steps} steps in {time.time()-t0:.1f}s")

    err=np.linalg.norm(Bal[:,:3]-truth[:,:3],axis=1)
    print(f"SGP4 vs high-fidelity truth: median={np.median(err):.2f} mean={err.mean():.2f} "
          f"p90={np.percentile(err,90):.2f} rmse={np.sqrt((err**2).mean()):.2f} km")

    res_rtn=np.array([to_rtn((Bal[i]-truth[i])[:3],Bal[i,:3],Bal[i,3:]) for i in range(len(IC))],np.float32)
    tr=np.zeros(len(IC),bool)
    for s in np.unique(sat):
        m=np.where(sat==s)[0]; tr[m[:int(c["TRAIN_FRACTION"]*len(m))]]=True
    va=~tr
    sc=StandardScaler().fit(Xs[tr].reshape(-1,23)); joblib.dump(sc,f"{c['SAVE_DIR']}/feature_scaler.pkl")
    def nrm(a):
        s=a.shape; return sc.transform(a.reshape(-1,23)).reshape(s).astype(np.float32)
    T_=lambda a: torch.tensor(np.array(a,np.float32))
    sv=lambda a,f: torch.save(T_(a),f"{c['SAVE_DIR']}/{f}")
    sv(nrm(Xs[tr]),"X_train.pt"); sv(nrm(Xs[va]),"X_val.pt")
    sv(truth[tr],"y_train.pt");   sv(truth[va],"y_val.pt")
    sv(Bal[tr],"baseline_train.pt"); sv(Bal[va],"baseline_val.pt")
    sv(res_rtn[tr],"residual_rtn_train.pt"); sv(res_rtn[va],"residual_rtn_val.pt")
    sv(age[tr],"age_train.pt"); sv(age[va],"age_val.pt")
    meta={"config":c,"n_train":int(tr.sum()),"n_val":int(va.sum()),
          "sgp4_vs_truth_km":{"median":float(np.median(err)),"mean":float(err.mean()),
                              "p90":float(np.percentile(err,90)),"rmse":float(np.sqrt((err**2).mean()))}}
    json.dump(meta,open(f"{c['SAVE_DIR']}/truth_meta.json","w"),indent=2,default=str)
    print(f"train {int(tr.sum())}  val {int(va.sum())}  -> tensors saved. Now run phase5_TRAINER_rtn.py")

if __name__=="__main__":
    main()