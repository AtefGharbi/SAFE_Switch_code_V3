import numpy as np, sys
from config import Design
from scenarios import ScenarioCfg, sample
from sim import simulate, metrics
from controllers import *
from train import load_risk, load_policy
D=Design(); R=load_risk()
pols=[load_policy('full',s)[0] for s in [int(a) for a in sys.argv[1].split(',')]]
cfgs=[ScenarioCfg('E1'),ScenarioCfg('mult',attack='mult'),ScenarioCfg('bias1',attack='bias',attack_mag=1.0,attack_channels=(1,)),
      ScenarioCfg('comb',attack='bias',attack_mag=1.0,attack_channels=(1,),p_loss=0.2,burst=True,dmax=3,pa_loss=0.1)]
for cfg in cfgs:
  ms={}
  for nm,mk in [('thr',lambda:SafeSwitch(D,'threshold',risk=R))]+[(f'ppo{i}',(lambda p=p:SafeSwitch(D,'ppo',policy=p,risk=R))) for i,p in enumerate(pols)]:
    res=[metrics(D,simulate(D,mk(),sample(cfg,30000+s))) for s in range(5)]
    print(cfg.name,nm,' '.join(f"{k}={np.nanmean([r[k] for r in res]):.3f}" for k in ['rmse','cost','viol_any','frac_R','switches','interv']),flush=True)
