"""E12 supervisor variants: 'mlp' (= train.py recipe), 'gru' (recurrent PPO), 'cvar' (mean-CVaR
weighted MLP-PPO). All: behaviour-cloning warm start from the risk-threshold rule, same reward,
same training distribution, same number of updates."""
import os, sys, time, pickle
import numpy as np
from config import Design
from controllers import SafeSwitch
from sim import simulate
from train import load_risk, make_train_sc, reward, bc_data, pretrain, OBS_DIM, OUT
from ppo import PPOPolicy, gae
from gru_ppo import GRUPolicy

ALPHA = 1 / 3   # CVaR tail fraction per batch


def train(kind, seed, iters=30, eps=6):
    D = Design(); risk = load_risk()
    O, A, G = bc_data(D, risk)
    if kind == "gru":
        pol = GRUPolicy(OBS_DIM, seed=seed, lr=1e-4)
        T = 600
        epi = [(O[i:i + T], A[i:i + T].astype(int), G[i:i + T]) for i in range(0, len(O), T)]
        pol.bc(epi, epochs=8)
    else:
        pol = PPOPolicy(OBS_DIM, seed=seed, lr=1e-4)
        pretrain(pol, O, A, G)
    rng = np.random.default_rng(10_000 * (seed + 1) + (7 if kind != "mlp" else 0))
    curve = []
    t0 = time.time()
    for it in range(iters):
        E = []
        for e in range(eps):
            s = int(rng.integers(0, 2 ** 31 - 1))
            cfg, sc = make_train_sc(D, s)
            ctrl = SafeSwitch(D, "ppo", policy=pol, risk=risk, deterministic=False)
            out = simulate(D, ctrl, sc, reward_fn=reward)
            tr = ctrl.trace
            rew = out["RW"][:len(tr)]
            adv, ret = gae(rew, [t[3] for t in tr], gamma=0.95, lam=0.9)
            ep = dict(obs=np.array([t[0] for t in tr]), act=np.array([t[1] for t in tr]),
                      logp=np.array([t[2] for t in tr]), adv=adv, ret=ret, R=float(rew.sum()))
            if kind == "gru":
                ep["h0s"] = np.array(pol.hist)
            E.append(ep)
        Rs = np.array([e["R"] for e in E])
        w_ep = np.ones(len(E))
        if kind == "cvar":
            k = max(1, int(np.ceil(ALPHA * len(E))))
            w_ep[np.argsort(Rs)[:k]] = 1.0 / ALPHA   # emphasise the worst-return episodes
        if kind == "gru":
            st = pol.update(E, weights=w_ep)
        else:
            sw = np.concatenate([np.full(len(e["act"]), w) for e, w in zip(E, w_ep)])
            st = pol.update(np.concatenate([e["obs"] for e in E]), np.concatenate([e["act"] for e in E]),
                            np.concatenate([e["logp"] for e in E]), np.concatenate([e["adv"] for e in E]),
                            np.concatenate([e["ret"] for e in E]), ent_coef=0.003, sample_w=sw)
        curve.append(dict(it=it, ret=float(Rs.mean()), t=time.time() - t0))
        print(f"[{kind} {seed}] it {it} ret {Rs.mean():.1f} t {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, f"ppo_{kind}_seed{seed}.pkl"), "wb") as f:
        pickle.dump(dict(state=pol.state(), curve=curve, kind=kind), f)


def load(kind, seed):
    if kind in ("mlp", "full"):
        from train import load_policy
        return load_policy("full", seed)[0]
    with open(os.path.join(OUT, f"ppo_{kind}_seed{seed}.pkl"), "rb") as f:
        d = pickle.load(f)
    pol = GRUPolicy(OBS_DIM, seed=seed) if kind == "gru" else PPOPolicy(OBS_DIM, seed=seed)
    pol.load(d["state"])
    return pol


if __name__ == "__main__":
    kind = sys.argv[1]
    for s in [int(x) for x in sys.argv[2].split(",")]:
        if kind == "mlp":
            from train import train_ppo
            train_ppo(Design(), load_risk(), s, iters=30, tag="full")
        else:
            train(kind, s)
