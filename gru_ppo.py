"""Recurrent (GRU) PPO actor in NumPy with truncated BPTT; MLP critic.

Actor:  o_t -> GRU(H) -> h_t -> linear -> 2 logits.
Critic: o_t -> 64 -> 64 -> 1 (feed-forward, from ppo.MLP).
"""
import numpy as np
from ppo import MLP, Adam, _softmax


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class GRUActor:
    def __init__(self, nin, H, nout, rng):
        s = 1.0 / np.sqrt(H)
        self.p = {
            "Wz": rng.uniform(-s, s, (nin, H)), "Uz": rng.uniform(-s, s, (H, H)), "bz": np.zeros(H),
            "Wr": rng.uniform(-s, s, (nin, H)), "Ur": rng.uniform(-s, s, (H, H)), "br": np.zeros(H),
            "Wn": rng.uniform(-s, s, (nin, H)), "Un": rng.uniform(-s, s, (H, H)), "bn": np.zeros(H),
            "bnh": np.zeros(H),
            "Wo": rng.normal(0, 0.01, (H, nout)), "bo": np.zeros(nout),
        }
        self.H = H

    def step(self, x, h):
        p = self.p
        z = _sig(x @ p["Wz"] + h @ p["Uz"] + p["bz"])
        r = _sig(x @ p["Wr"] + h @ p["Ur"] + p["br"])
        hn = h @ p["Un"] + p["bnh"]
        n = np.tanh(x @ p["Wn"] + r * hn + p["bn"])
        hp = (1 - z) * n + z * h
        return hp, (x, h, z, r, n, hn)

    def forward_seq(self, X, h0):
        """X: (B, T, nin), h0: (B, H) -> logits (B, T, nout), caches."""
        B, T, _ = X.shape
        h = h0
        Hs, caches = [], []
        for t in range(T):
            h, cache = self.step(X[:, t], h)
            Hs.append(h); caches.append(cache)
        Hs = np.stack(Hs, 1)
        logits = Hs @ self.p["Wo"] + self.p["bo"]
        return logits, Hs, caches

    def backward_seq(self, dlogits, Hs, caches):
        p = self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        B, T, _ = dlogits.shape
        g["Wo"] = np.einsum("bth,bto->ho", Hs, dlogits)
        g["bo"] = dlogits.sum((0, 1))
        dH = dlogits @ p["Wo"].T
        dh_next = np.zeros((B, self.H))
        for t in reversed(range(T)):
            x, h, z, r, n, hn = caches[t]
            dh = dH[:, t] + dh_next
            dz = dh * (h - n)
            dn = dh * (1 - z)
            dhp = dh * z
            dan = dn * (1 - n ** 2)
            g["Wn"] += x.T @ dan; g["bn"] += dan.sum(0)
            dr = dan * hn
            dhn = dan * r
            g["Un"] += h.T @ dhn; g["bnh"] += dhn.sum(0)
            dhp += dhn @ p["Un"].T
            daz = dz * z * (1 - z)
            g["Wz"] += x.T @ daz; g["Uz"] += h.T @ daz; g["bz"] += daz.sum(0)
            dhp += daz @ p["Uz"].T
            dar = dr * r * (1 - r)
            g["Wr"] += x.T @ dar; g["Ur"] += h.T @ dar; g["br"] += dar.sum(0)
            dhp += dar @ p["Ur"].T
            dh_next = dhp
        return g


class GRUPolicy:
    """Drop-in replacement for ppo.PPOPolicy with a recurrent actor."""
    recurrent = True

    def __init__(self, obs_dim, n_act=2, H=32, seed=0, lr=3e-4, chunk=32):
        self.rng = np.random.default_rng(seed)
        self.actor = GRUActor(obs_dim, H, n_act, self.rng)
        self.critic = MLP([obs_dim, 64, 64, 1], self.rng, out_scale=1.0)
        self.keys = list(self.actor.p.keys())
        self.opt_a = Adam([self.actor.p[k] for k in self.keys], lr)
        self.opt_c = Adam(self.critic.params(), lr * 3)
        self.H, self.chunk = H, chunk
        self.reset()

    def reset(self):
        self.h = np.zeros((1, self.H))
        self.hist = []          # hidden state BEFORE each step (for BPTT chunk starts)

    def act(self, obs, deterministic=False):
        self.hist.append(self.h[0].copy())
        self.h, _ = self.actor.step(obs[None], self.h)
        logits = self.h @ self.actor.p["Wo"] + self.actor.p["bo"]
        p = _softmax(logits)[0]
        a = int(np.argmax(p)) if deterministic else int(self.rng.random() < p[1])
        v, _ = self.critic.forward(obs[None])
        return a, float(np.log(p[a] + 1e-12)), float(v[0, 0])

    def update(self, episodes, clip=0.2, epochs=6, ent_coef=0.003, max_norm=0.5, weights=None):
        """episodes: list of dicts with obs (T,d), act, logp, adv, ret, h0s (T,H)."""
        # build chunks
        X, A, LP, ADV, RET, H0, M, W = [], [], [], [], [], [], [], []
        allA = np.concatenate([e["adv"] for e in episodes])
        mu, sd = allA.mean(), allA.std() + 1e-8
        for ei, e in enumerate(episodes):
            T = len(e["act"])
            w_ep = 1.0 if weights is None else weights[ei]
            for s in range(0, T, self.chunk):
                t1 = min(T, s + self.chunk)
                L = t1 - s
                pad = self.chunk - L
                def P(x, fill=0.0):
                    x = np.asarray(x[s:t1])
                    if pad:
                        x = np.concatenate([x, np.full((pad,) + x.shape[1:], fill)], 0)
                    return x
                X.append(P(e["obs"])); A.append(P(e["act"]).astype(int)); LP.append(P(e["logp"]))
                ADV.append(P((e["adv"] - mu) / sd)); RET.append(P(e["ret"])); H0.append(e["h0s"][s])
                M.append(np.r_[np.ones(L), np.zeros(pad)]); W.append(w_ep)
        X, A, LP, ADV, RET = map(np.array, (X, A, LP, ADV, RET))
        H0, M, W = np.array(H0), np.array(M), np.array(W)
        nC = len(X)
        stats = []
        for _ in range(epochs):
            idx = self.rng.permutation(nC)
            for s in range(0, nC, 16):
                j = idx[s:s + 16]
                logits, Hs, caches = self.actor.forward_seq(X[j], H0[j])
                p = _softmax(logits)
                Bj, Tj = A[j].shape
                bi, ti = np.meshgrid(np.arange(Bj), np.arange(Tj), indexing="ij")
                pa = p[bi, ti, A[j]]
                logp = np.log(pa + 1e-12)
                ratio = np.exp(logp - LP[j])
                Aw = ADV[j] * W[j][:, None]
                un, cl = ratio * Aw, np.clip(ratio, 1 - clip, 1 + clip) * Aw
                use = un <= cl
                Nm = M[j].sum()
                dlogp = -np.where(use, un, 0.0) * M[j] / Nm
                onehot = np.zeros_like(p); onehot[bi, ti, A[j]] = 1
                dlog = dlogp[..., None] * (onehot - p)
                Hent = -(p * np.log(p + 1e-12)).sum(-1)
                dHe = -p * (np.log(p + 1e-12) + Hent[..., None])
                dlog += -ent_coef * dHe * M[j][..., None] / Nm
                g = self.actor.backward_seq(dlog, Hs, caches)
                gl = [g[k] for k in self.keys]
                gn = np.sqrt(sum((x ** 2).sum() for x in gl))
                if gn > max_norm:
                    gl = [x * max_norm / gn for x in gl]
                self.opt_a.step([self.actor.p[k] for k in self.keys], gl)
                o = X[j].reshape(-1, X.shape[-1]); mm = M[j].reshape(-1)
                v, cacts = self.critic.forward(o)
                dv = ((v[:, 0] - RET[j].reshape(-1)) * mm)[:, None] / mm.sum()
                gW, gb = self.critic.backward(cacts, dv)
                gc = gW + gb
                gn = np.sqrt(sum((x ** 2).sum() for x in gc))
                if gn > max_norm:
                    gc = [x * max_norm / gn for x in gc]
                self.opt_c.step(self.critic.params(), gc)
                stats.append((float((np.minimum(un, cl) * M[j]).sum() / Nm), float((Hent * M[j]).sum() / Nm)))
        return np.mean(stats, 0)

    def bc(self, episodes, epochs=20):
        """Behaviour cloning on demonstration episodes (list of (obs (T,d), act (T,), ret (T,)))."""
        for _ in range(epochs):
            for (O, Aa, G) in episodes:
                T = len(Aa)
                h0 = np.zeros((1, self.H))
                for s in range(0, T, self.chunk):
                    t1 = min(T, s + self.chunk)
                    X = O[None, s:t1]
                    logits, Hs, caches = self.actor.forward_seq(X, h0)
                    h0 = Hs[:, -1].copy()
                    p = _softmax(logits)
                    onehot = np.zeros_like(p); onehot[0, np.arange(t1 - s), Aa[s:t1]] = 1
                    g = self.actor.backward_seq((p - onehot) / (t1 - s), Hs, caches)
                    self.opt_a.step([self.actor.p[k] for k in self.keys], [g[k] for k in self.keys])
                    v, cacts = self.critic.forward(O[s:t1])
                    gW, gb = self.critic.backward(cacts, (v[:, 0] - G[s:t1])[:, None] / (t1 - s))
                    self.opt_c.step(self.critic.params(), gW + gb)

    def _h_at(self, O, s):
        h = np.zeros((1, self.H))
        for t in range(s):
            h, _ = self.actor.step(O[None, t], h)
        return h

    def state(self):
        return dict(actor={k: v.copy() for k, v in self.actor.p.items()}, critic=self.critic.get())

    def load(self, st):
        for k, v in st["actor"].items():
            self.actor.p[k] = v.copy()
        self.critic.set(st["critic"])


def gradcheck(seed=0):
    rng = np.random.default_rng(seed)
    a = GRUActor(5, 6, 2, rng)
    X = rng.normal(size=(3, 7, 5)); h0 = rng.normal(size=(3, 6)) * 0.1
    T = rng.normal(size=(3, 7, 2))
    logits, Hs, caches = a.forward_seq(X, h0)
    g = a.backward_seq(logits - T, Hs, caches)
    errs = []
    for k in ["Wz", "Un", "Ur", "bnh", "Wo"]:
        P = a.p[k]; idx = tuple(rng.integers(0, s) for s in P.shape)
        eps = 1e-6
        P[idx] += eps; lp = 0.5 * ((a.forward_seq(X, h0)[0] - T) ** 2).sum()
        P[idx] -= 2 * eps; lm = 0.5 * ((a.forward_seq(X, h0)[0] - T) ** 2).sum()
        P[idx] += eps
        errs.append(abs((lp - lm) / (2 * eps) - g[k][idx]))
    return max(errs)


if __name__ == "__main__":
    print("GRU gradcheck max error", gradcheck())
