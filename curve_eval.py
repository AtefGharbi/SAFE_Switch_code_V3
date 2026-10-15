"""Learning convergence on a fixed validation set (checkpoint after every PPO update).

Retrains an additional seed (seed 10) with exactly the recipe of train.py and evaluates the
greedy policy after behaviour cloning and after each update on 8 fixed validation episodes
(disjoint from training and test seeds). The risk-threshold supervisor and PPO without
warm start (seed 10) are evaluated on the same episodes for reference.
"""
import json, os, sys, copy
import numpy as np
from config import Design
from scenarios import sample
from sim import simulate, metrics
from controllers import SafeSwitch
from train import load_risk, make_train_sc, reward, bc_data, pretrain, OBS_DIM
from ppo import PPOPolicy, gae

OUT = os.path.join(os.path.dirname(__file__), "results")


def val_cost(D, risk, pol=None, sup="ppo", seeds=range(900_000, 900_008)):
    J = []
    for s in seeds:
        cfg, sc = make_train_sc(D, s)
        c = SafeSwitch(D, sup, policy=pol, risk=risk)
        out = simulate(D, c, sc, reward_fn=reward)
        J.append(-out["RW"].sum())
    return float(np.mean(J)), [float(j) for j in J]


def run(warm=True, seed=10, iters=30, eps=6):
    D = Design(); risk = load_risk()
    pol = PPOPolicy(OBS_DIM, seed=seed, lr=1e-4 if warm else 3e-4)
    if warm:
        O, A, G = bc_data(D, risk)
        pretrain(pol, O, A, G)
    rng = np.random.default_rng(10_000 * (seed + 1))
    hist = [dict(it=0, val=val_cost(D, risk, pol)[0])]
    for it in range(iters):
        Ob, Ac, LP, ADV, RET = [], [], [], [], []
        for e in range(eps):
            s = int(rng.integers(0, 2 ** 31 - 1))
            cfg, sc = make_train_sc(D, s)
            ctrl = SafeSwitch(D, "ppo", policy=pol, risk=risk, deterministic=False)
            out = simulate(D, ctrl, sc, reward_fn=reward)
            tr = ctrl.trace
            rew = out["RW"][:len(tr)]
            adv, ret = gae(rew, [t[3] for t in tr], gamma=0.95, lam=0.9)
            Ob += [t[0] for t in tr]; Ac += [t[1] for t in tr]; LP += [t[2] for t in tr]
            ADV.append(adv); RET.append(ret)
        pol.update(np.array(Ob), np.array(Ac), np.array(LP), np.concatenate(ADV), np.concatenate(RET),
                   ent_coef=0.003 if warm else 0.01)
        hist.append(dict(it=it + 1, val=val_cost(D, risk, pol)[0]))
        print(warm, hist[-1], flush=True)
    return hist


if __name__ == "__main__":
    D = Design(); risk = load_risk()
    res = {"threshold": val_cost(D, risk, sup="threshold")[0],
           "always_R": val_cost(D, risk, sup="always_R")[0],
           "always_N": val_cost(D, risk, sup="always_N")[0]}
    res["warm"] = run(True)
    res["cold"] = run(False)
    json.dump(res, open(os.path.join(OUT, "curve_eval.json"), "w"), indent=1)
