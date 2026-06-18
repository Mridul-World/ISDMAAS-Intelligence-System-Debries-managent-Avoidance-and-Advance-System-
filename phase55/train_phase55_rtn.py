"""
================================================================================
PHASE 5.5 — RTN RESIDUAL TRAINER ON REAL PRECISE-ORBIT TRUTH
================================================================================
Identical architecture, losses, optimizer and reconstruction to the FROZEN
phase5_TRAINER_rtn.py. Only paths/report names differ. The single change in
the world is the data: y/baseline/residual now come from REAL ESA POD
ephemerides instead of RK4-of-J2J3J4 synthetic truth.

target convention (unchanged):
    residual_rtn = RTN_baseline( baseline_pos - truth_pos )
    => truth_pos = baseline_pos - RTN_to_ECI(residual_rtn, baseline_r, baseline_v)
================================================================================
"""
import os, json, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import warnings; warnings.filterwarnings("ignore")

CFG = {"TENSOR_DIR": ".", "MODEL_DIR": "model", "REPORT_DIR": "reports",
       "D_MODEL":64,"NHEAD":4,"NUM_LAYERS":2,"DIM_FF":128,"DROPOUT":0.1,
       "EPOCHS":40,"BATCH":512,"LR":4e-4,"WD":1e-4,"PATIENCE":8,
       "USE_PHYSICS":True,"W_E":0.05,"W_M":0.05,"W_A":0.02,"W_T":0.01,"SEED":42}
MU=398600.4418; RE=6378.137

class PosEnc(nn.Module):
    def __init__(s,d,ml=512,dr=0.1):
        super().__init__(); s.dr=nn.Dropout(dr); pe=torch.zeros(ml,d)
        pos=torch.arange(ml).unsqueeze(1).float(); dv=torch.exp(torch.arange(0,d,2).float()*(-math.log(1e4)/d))
        pe[:,0::2]=torch.sin(pos*dv); pe[:,1::2]=torch.cos(pos*dv); s.register_buffer("pe",pe.unsqueeze(0))
    def forward(s,x): return s.dr(x+s.pe[:,:x.size(1)])
class RTNTransformer(nn.Module):
    def __init__(s,nf,d,nh,nl,ff,dr):
        super().__init__(); s.ip=nn.Sequential(nn.Linear(nf,d),nn.LayerNorm(d),nn.GELU()); s.pe=PosEnc(d,dr=dr)
        e=nn.TransformerEncoderLayer(d,nh,ff,dr,batch_first=True,norm_first=True,activation="gelu")
        s.enc=nn.TransformerEncoder(e,nl,enable_nested_tensor=False)
        s.hd=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(dr),nn.Linear(d//2,3))  # 3 = RTN
        [nn.init.xavier_uniform_(p) for p in s.parameters() if p.dim()>1]
    def forward(s,x): return s.hd(s.enc(s.pe(s.ip(x)))[:,-1,:])

def rtn_basis(state):
    r,v=state[:,:3],state[:,3:]
    R=r/torch.norm(r,dim=-1,keepdim=True)
    W=torch.cross(r,v,dim=-1); W=W/torch.norm(W,dim=-1,keepdim=True)
    S=torch.cross(W,R,dim=-1)
    return R,S,W
def rtn_to_eci(rtn,ref_state):
    R,S,W=rtn_basis(ref_state)
    return rtn[:,0:1]*R + rtn[:,1:2]*S + rtn[:,2:3]*W

def inv(st):
    r,v=st[:,:3],st[:,3:]; rn=torch.norm(r,dim=-1).clamp_min(1.); E=0.5*(v*v).sum(-1)-MU/rn
    h=torch.cross(r,v,dim=-1); a=-MU/(2*torch.clamp(E,max=-1e-3)); T=2*math.pi*torch.sqrt((a.abs()**3)/MU)
    return E,h,a,T
def physics(pred,true,c):
    Ep,hp,ap,Tp=inv(pred); Et,ht,at,Tt=inv(true)
    return (c["W_E"]*(Ep-Et).abs().mean()/100 + c["W_M"]*torch.norm(hp-ht,dim=-1).mean()/1e4
            + c["W_A"]*(ap-at).abs().mean()/1000 + c["W_T"]*(Tp-Tt).abs().mean()/1000)
def panel(p,t):
    e=np.linalg.norm(p[:,:3]-t[:,:3],axis=1); k=e<=np.percentile(e,99)
    return dict(rmse=float(np.sqrt((e**2).mean())),median=float(np.median(e)),
                p90=float(np.percentile(e,90)),p99=float(np.percentile(e,99)),
                trimmed_rmse=float(np.sqrt((e[k]**2).mean())))

def main():
    import sys
    c=dict(CFG)
    pure = "--pure-ml" in sys.argv
    if pure:
        c["USE_PHYSICS"]=False
        print(">>> ABLATION: pure-ML (no physics loss)")
    tag = "rtn_pure_ml_phase55" if pure else "rtn_physics_ml_phase55"
    torch.manual_seed(c["SEED"]); np.random.seed(c["SEED"])
    os.makedirs(c["MODEL_DIR"],exist_ok=True); os.makedirs(c["REPORT_DIR"],exist_ok=True)
    td=c["TENSOR_DIR"]; ld=lambda f: torch.load(os.path.join(td,f),weights_only=False).float()
    Xtr,btr,ytr,rtr=ld("X_train.pt"),ld("baseline_train.pt"),ld("y_train.pt"),ld("residual_rtn_train.pt")
    Xva,bva,yva,rva=ld("X_val.pt"),ld("baseline_val.pt"),ld("y_val.pt"),ld("residual_rtn_val.pt")
    assert len(Xva)==len(yva)==len(bva)==len(rva), "ALIGNMENT MISMATCH — rebuild with export_phase55_tensors.py"
    rm=rtr.mean(0,keepdim=True); rs=rtr.std(0,keepdim=True).clamp_min(1e-6); rtr_n=(rtr-rm)/rs

    base=panel(bva.numpy(),yva.numpy())
    print(f"SGP4 baseline vs REAL truth (val): rmse={base['rmse']:.3f} median={base['median']:.3f} p90={base['p90']:.3f} km")

    dev="cpu"; net=RTNTransformer(Xtr.shape[2],c["D_MODEL"],c["NHEAD"],c["NUM_LAYERS"],c["DIM_FF"],c["DROPOUT"]).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=c["LR"],weight_decay=c["WD"])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c["EPOCHS"]); lf=nn.SmoothL1Loss()
    loader=DataLoader(TensorDataset(Xtr,rtr_n,btr,ytr),batch_size=c["BATCH"],shuffle=True)
    rm_d,rs_d=rm.to(dev),rs.to(dev); best={"rmse":1e9}; bad=0

    # persist residual normalization for validate/infer
    torch.save({"rm":rm,"rs":rs},os.path.join(c["MODEL_DIR"],"residual_norm_phase55.pt"))

    def evaluate():
        net.eval()
        with torch.no_grad():
            outs=[net(Xva[j:j+1024])*rs_d+rm_d for j in range(0,len(Xva),1024)]
            pred_rtn=torch.cat(outs,0)
            final=bva.clone(); final[:,:3]=bva[:,:3]-rtn_to_eci(pred_rtn,bva)   # deployable: baseline frame
        return panel(final.numpy(),yva.numpy())

    for ep in range(1,c["EPOCHS"]+1):
        net.train()
        for xb,rb,bb,yb in loader:
            opt.zero_grad(); out=net(xb); loss=lf(out,rb)
            if c["USE_PHYSICS"]:
                rec=bb.clone(); rec[:,:3]=bb[:,:3]-rtn_to_eci(out*rs_d+rm_d,bb)
                loss=loss+physics(rec,yb,c)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        sch.step(); m=evaluate()
        imp=m["rmse"]<best["rmse"]-1e-6; fl=""
        if imp: best={**m,"epoch":ep}; bad=0; fl="  <-best"; torch.save(net.state_dict(),os.path.join(c["MODEL_DIR"],f"{tag}.pt"))
        else: bad+=1
        print(f"ep{ep:3d} rmse{m['rmse']:8.3f} median{m['median']:7.3f} p90{m['p90']:8.3f}{fl}")
        if bad>=c["PATIENCE"]: print("early stop"); break

    rep={"truth_source":"REAL ESA POD precise ephemeris (AUX_POEORB)",
         "baseline_sgp4":base,"model":best,"variant":tag,
         "improvement_pct":{"rmse":100*(1-best["rmse"]/base["rmse"]),"median":100*(1-best["median"]/base["median"])}}
    json.dump(rep,open(os.path.join(c["REPORT_DIR"],f"{tag}_validation.json"),"w"),indent=2,default=str)
    print("\n"+"="*56)
    print(f"SGP4       rmse={base['rmse']:.3f}  median={base['median']:.3f} km   [REAL truth]")
    print(f"{'pure-ML    ' if pure else 'physics+ML'} rmse={best['rmse']:.3f}  median={best['median']:.3f} km   [REAL truth]")
    print(f"reduction  {rep['improvement_pct']['rmse']:.1f}% RMSE  {rep['improvement_pct']['median']:.1f}% median")
    print(f"Saved: model/{tag}.pt, reports/{tag}_validation.json")

if __name__=="__main__":
    main()
