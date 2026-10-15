import numpy as np, sys
from config import Design
from scenarios import ScenarioCfg, sample
from sim import simulate, metrics
from controllers import SafeSwitch
from train import load_risk
D=Design(); R=load_risk()
for nm,cfg in [('short6',ScenarioCfg('s',attack='bias',attack_mag=6.0,attack_channels=(1,),on_mean=0.1,min_on=0.1,off_mean=1.0)),
               ('short6c0',ScenarioCfg('s',attack='bias',attack_mag=6.0,attack_channels=(0,),on_mean=0.1,min_on=0.1,off_mean=1.0)),
               ('long6',ScenarioCfg('s',attack='bias',attack_mag=6.0,attack_channels=(1,)))]:
  for f in [True,False]:
    res=[metrics(D,simulate(D,SafeSwitch(D,'always_N',risk=R,use_filter=f),sample(cfg,40000+s))) for s in range(5)]
    print(nm,'filter' if f else 'nofilt',' '.join(f"{k}={np.nanmean([r[k] for r in res]):.3f}" for k in ['rmse','max_dev','viol_any','infeas','contain','interv']),flush=True)
