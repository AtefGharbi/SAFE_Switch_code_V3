"""Offline training: (1) calibrated attack-risk model, (2) PPO supervisor (multi-seed)."""
import json, os, sys, time, pickle
import numpy as np
from config import Design
from scenarios import ScenarioCfg, sample
from sim import simulate, metrics
from controllers import SafeSwitch, RiskModel
from ppo import PPOPolicy, gae

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
OBS_DIM = 12


def train_cfg(rng, T=600, profile=None):
    """Training scenario distribution (disjoint seeds from evaluation)."""
    prof = dict(profile or {})
    T = prof.pop("T", T)
    att = rng.choice(["none", "mult", "bias", "ramp", "stealth"], p=[0.2, 0.2, 0.3, 0.15, 0.15])
    nch = rng.choice([1, 2], p=[0.6, 0.4]) if att != "mult" else 2
    ch = tuple(sorted(rng.choice([0, 1], size=nch, replace=False).tolist()))
    mag = {"none": 0.0, "mult": 0.0, "bias": rng.uniform(0.1, 1.5), "ramp": rng.uniform(0.05, 0.6),
           "stealth": rng.uniform(0.01, 0.08)}[att]
    lam = (rng.uniform(-2.0, -0.3), rng.uniform(-1.0, 0.5)) if att == "mult" else (-1.5, -0.5)
    return ScenarioCfg(name="train", T=T, attack=att, attack_mag=mag, attack_channels=ch, Lam=lam,
                       p_loss=rng.uniform(0, 0.35), burst=bool(rng.random() < 0.5),
                       dmax=int(rng.integers(0, 4)), pa_loss=rng.uniform(0, 0.2),
                       rel_unc=0.03, x0_random=True, **prof)


def reward(x, u, info, D):
    viol = float(np.any(np.abs(x) > D.xmax))
    r = -(float(x @ x) + 0.01 * float(u @ u)) - 0.2 * info["switch"] \
        - 0.1 * (info["us"] - info["uc"]) ** 2 - 5.0 * viol
    return max(r, -20.0)


def make_train_sc(D, s):
    from bench_designs import PROFILES
    cfg = train_cfg(np.random.default_rng(s), profile=PROFILES[D.bench])
    sc = sample(cfg, s, xmax=D.xmax)
    r0 = np.full(D.plant.n, D.vbar + D.abar)
    f0 = D.cert.f(sc["x0"], r0)
    if f0 > 0.9 * D.ell:  # initial information state inside the certified set
        sc["x0"] = sc["x0"] * (0.9 * D.ell - D.cert.p @ r0) / D.cert.normP(sc["x0"])
    return cfg, sc


# ------------------------------------------------------------------ risk model
def train_risk(D, n_train=160, n_test=60):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss

    def collect(seeds):
        Phi, Y = [], []
        for s in seeds:
            rng = np.random.default_rng(s)
            cfg, sc = make_train_sc(D, s)
            ctrl = SafeSwitch(D, supervisor="random")
            ctrl.rng = np.random.default_rng(s + 7)
            phis = []
            orig = ctrl.step

            def st(k, p, u, _o=orig, _l=phis):
                r = _o(k, p, u)
                _l.append(r[2]["phi"])
                return r
            ctrl.step = st
            simulate(D, ctrl, sc)
            Phi.append(np.array(phis))
            Y.append(sc["attack_on"][:len(phis)].astype(int))
        return np.vstack(Phi), np.concatenate(Y)

    Xtr, ytr = collect(range(1000, 1000 + n_train))
    Xte, yte = collect(range(5000, 5000 + n_test))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit((Xtr - mu) / sd, ytr)
    p = lr.predict_proba((Xte - mu) / sd)[:, 1]
    # expected calibration error (10 bins)
    bins = np.linspace(0, 1, 11)
    ece, rel = 0.0, []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - yte[m].mean())
            rel.append((float(p[m].mean()), float(yte[m].mean()), int(m.sum())))
    pred = p > 0.5
    res = dict(theta=[float(lr.intercept_[0])] + lr.coef_[0].tolist(), mu=mu.tolist(), sd=sd.tolist(),
               features=RiskModel.FEATURES,
               auc=float(roc_auc_score(yte, p)), brier=float(brier_score_loss(yte, p)), ece=float(ece),
               tpr=float(pred[yte == 1].mean()), fpr=float(pred[yte == 0].mean()),
               base_rate=float(yte.mean()), n_train=int(len(ytr)), n_test=int(len(yte)), reliability=rel)
    return res


def load_risk(path=os.path.join(OUT, "risk_model.json")):
    r = json.load(open(path))
    return RiskModel(np.array(r["theta"]), np.array(r["mu"]), np.array(r["sd"]))


# ------------------------------------------------------------------ PPO
def bc_data(D, risk, n_eps=48, gamma=0.95):
    """Demonstrations of the risk-threshold supervisor (used to warm-start PPO)."""
    path = os.path.join(OUT, "bc_data.npz" if D.bench == "dcmotor" else f"bc_data_{D.bench}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["O"], d["A"], d["G"]
    O, A, G = [], [], []
    for e in range(n_eps):
        s = 777_000 + e
        cfg, sc = make_train_sc(D, s)
        ctrl = SafeSwitch(D, "threshold", risk=risk)
        obs = []
        orig = ctrl.step

        def st(k, p, u, _o=orig, _l=obs):
            r = _o(k, p, u)
            _l.append((r[2]["obs"], r[2]["mode"]))
            return r
        ctrl.step = st
        out = simulate(D, ctrl, sc, reward_fn=reward)
        rew = out["RW"][:len(obs)]
        g, run = np.zeros(len(rew)), 0.0
        for t in reversed(range(len(rew))):
            run = rew[t] + gamma * run
            g[t] = run
        O += [o for o, _ in obs]; A += [a for _, a in obs]; G.append(g)
    O, A, G = np.array(O), np.array(A), np.concatenate(G)
    np.savez_compressed(path, O=O, A=A, G=G)
    return O, A, G


def pretrain(pol, O, A, G, epochs=30, mb=1024):
    """Behaviour cloning of the actor (cross-entropy) and Monte-Carlo fit of the critic."""
    from ppo import _softmax
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(len(O))
        for s in range(0, len(O), mb):
            j = idx[s:s + mb]
            logits, acts = pol.actor.forward(O[j])
            p = _softmax(logits)
            onehot = np.zeros_like(p); onehot[np.arange(len(j)), A[j]] = 1
            gW, gb = pol.actor.backward(acts, (p - onehot) / len(j))
            pol.opt_a.step(pol.actor.params(), gW + gb)
            v, cacts = pol.critic.forward(O[j])
            gW, gb = pol.critic.backward(cacts, (v[:, 0] - G[j])[:, None] / len(j))
            pol.opt_c.step(pol.critic.params(), gW + gb)
    logits, _ = pol.actor.forward(O)
    return float(np.mean(np.argmax(logits, 1) == A))


def train_ppo(D, risk, seed, iters=40, eps_per_iter=6, risk_in_obs=True, use_filter=True, tag="full"):
    pol = PPOPolicy(OBS_DIM, seed=seed, lr=1e-4)
    O, A, G = bc_data(D, risk)
    if not risk_in_obs:
        O = O.copy(); O[:, [5, 6, 11]] = 0.0
    bc_acc = pretrain(pol, O, A, G)
    print(f"[{tag} seed {seed}] behaviour-cloning accuracy {bc_acc:.3f}", flush=True)
    rng = np.random.default_rng(10_000 * (seed + 1))
    curve = []
    t0 = time.time()
    for it in range(iters):
        O, A, LP, ADV, RET, eprets = [], [], [], [], [], []
        for e in range(eps_per_iter):
            s = int(rng.integers(0, 2 ** 31 - 1))
            cfg, sc = make_train_sc(D, s)
            ctrl = SafeSwitch(D, "ppo", policy=pol, risk=risk, deterministic=False,
                              risk_in_obs=risk_in_obs, use_filter=use_filter)
            out = simulate(D, ctrl, sc, reward_fn=reward)
            tr = ctrl.trace
            n = len(tr)
            rew = out["RW"][:n]
            val = [t[3] for t in tr]
            adv, ret = gae(rew, val, gamma=0.95, lam=0.9)
            O += [t[0] for t in tr]; A += [t[1] for t in tr]; LP += [t[2] for t in tr]
            ADV.append(adv); RET.append(ret); eprets.append(float(rew.sum()))
        st = pol.update(np.array(O), np.array(A), np.array(LP), np.concatenate(ADV), np.concatenate(RET), ent_coef=0.003)
        curve.append(dict(it=it, ret=float(np.mean(eprets)), ent=float(st[1]), vloss=float(st[2]),
                          frac_R=float(np.mean(A)), t=time.time() - t0))
        print(f"[{tag} seed {seed}] it {it} ret {np.mean(eprets):.2f} ent {st[1]:.3f} fracR {np.mean(A):.2f}",
              flush=True)
    with open(os.path.join(OUT, f"ppo_{tag}_seed{seed}.pkl"), "wb") as f:
        pickle.dump(dict(state=pol.state(), curve=curve, bc_acc=bc_acc), f)
    return pol, curve


def load_policy(tag, seed):
    with open(os.path.join(OUT, f"ppo_{tag}_seed{seed}.pkl"), "rb") as f:
        d = pickle.load(f)
    pol = PPOPolicy(OBS_DIM, seed=seed)
    pol.load(d["state"])
    return pol, d["curve"]


if __name__ == "__main__":
    what = sys.argv[1]
    D = Design()
    if what == "risk":
        res = train_risk(D)
        json.dump(res, open(os.path.join(OUT, "risk_model.json"), "w"), indent=1)
        print({k: v for k, v in res.items() if k not in ("reliability", "mu", "sd")})
    elif what == "ppo":
        tag = sys.argv[2]
        seeds = [int(s) for s in sys.argv[3].split(",")]
        iters = int(sys.argv[4]) if len(sys.argv) > 4 else 40
        risk = load_risk()
        kw = dict(full={}, norisk=dict(risk_in_obs=False), nofilter=dict(use_filter=False))[tag]
        for s in seeds:
            train_ppo(D, risk, s, iters=iters, tag=tag, **kw)
