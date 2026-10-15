import numpy as np, sys
from config import Design
from scenarios import ScenarioCfg, sample
from sim import simulate, metrics
from controllers import SafeSwitch
from train import load_risk
from experiments import scenarios_main, scenarios_stress, make_cfg, TEST_SEED0
D=Design(); R=load_risk()
S={**scenarios_main(),**scenarios_stress()}
variants={'base':{}, 'gross':dict(gross_emax=1.0), 'gross+rec':dict(gross_emax=1.0,recovery=True), 'auth2':dict(auth_period=2), 'auth2+gross+rec':dict(auth_period=2,gross_emax=1.0,recovery=True)}
for scen in ['E1 nominal','E2 bias 1.0','E2 mult (HD)','E5 bias+net','E10 long bias 6 speed†','E2 ramp 0.2/s']:
  for vn,kw in variants.items():
    res=[]
    for s in range(8):
      cfg=make_cfg(scen,S[scen],s); sc=sample(cfg,TEST_SEED0+s,xmax=D.xmax)
      if 'auth_period' in kw: sc['auth_period']=kw['auth_period']
      res.append(metrics(D,simulate(D,SafeSwitch(D,'threshold',risk=R,**kw),sc)))
    print(f"{scen:24s}{vn:16s}",' '.join(f"{k}={np.nanmean([r[k] for r in res]):.3f}" for k in ['rmse','viol_any','contain','infeas','alarm_frac','n_alarm','budget_min_s','budget_sound']),flush=True)
