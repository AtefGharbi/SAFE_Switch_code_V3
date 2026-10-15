"""Minimal, dependency-free PPO (clipped objective, GAE) with NumPy MLPs.

Actor : obs -> 64 tanh -> 64 tanh -> 2 logits (nominal / robust mode)
Critic: obs -> 64 tanh -> 64 tanh -> 1
"""
import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / a) if i < len(sizes) - 2 else out_scale / np.sqrt(a)
            self.W.append(rng.normal(0, scale, (a, b)))
            self.b.append(np.zeros(b))

    def forward(self, x):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        return h, acts

    def backward(self, acts, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1 - acts[i] ** 2)
        return gW, gb

    def params(self):
        return self.W + self.b

    def get(self):
        return [p.copy() for p in self.params()]

    def set(self, ps):
        n = len(self.W)
        self.W = [p.copy() for p in ps[:n]]
        self.b = [p.copy() for p in ps[n:]]


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class PPOPolicy:
    def __init__(self, obs_dim, n_act=2, hidden=64, seed=0, lr=3e-4):
        self.rng = np.random.default_rng(seed)
        self.actor = MLP([obs_dim, hidden, hidden, n_act], self.rng, out_scale=0.01)
        self.critic = MLP([obs_dim, hidden, hidden, 1], self.rng, out_scale=1.0)
        self.opt_a = Adam(self.actor.params(), lr)
        self.opt_c = Adam(self.critic.params(), lr * 3)
        self.obs_dim = obs_dim

    def act(self, obs, deterministic=False):
        logits, _ = self.actor.forward(obs[None])
        p = _softmax(logits)[0]
        a = int(np.argmax(p)) if deterministic else int(self.rng.random() < p[1])
        v, _ = self.critic.forward(obs[None])
        return a, float(np.log(p[a] + 1e-12)), float(v[0, 0])

    def update(self, obs, act, logp_old, adv, ret, clip=0.2, epochs=6, mb=512, ent_coef=0.01,
               max_norm=0.5, sample_w=None):
        N = len(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if sample_w is not None:
            adv = adv * sample_w
        stats = []
        for _ in range(epochs):
            idx = self.rng.permutation(N)
            for s in range(0, N, mb):
                j = idx[s:s + mb]
                o, a, lo, A, R = obs[j], act[j], logp_old[j], adv[j], ret[j]
                logits, acts = self.actor.forward(o)
                p = _softmax(logits)
                logp = np.log(p[np.arange(len(j)), a] + 1e-12)
                ratio = np.exp(logp - lo)
                unclipped = ratio * A
                clipped = np.clip(ratio, 1 - clip, 1 + clip) * A
                use_unclipped = unclipped <= clipped
                # d(-min)/dlogp
                dlogp = -np.where(use_unclipped, unclipped, 0.0) / len(j)
                onehot = np.zeros_like(p)
                onehot[np.arange(len(j)), a] = 1
                dlogits = dlogp[:, None] * (onehot - p)
                # entropy bonus gradient: dH/dz_k = -p_k (log p_k + H)
                H = -(p * np.log(p + 1e-12)).sum(1)
                dH = -p * (np.log(p + 1e-12) + H[:, None])
                dlogits += -ent_coef * dH / len(j)
                gW, gb = self.actor.backward(acts, dlogits)
                g = gW + gb
                gn = np.sqrt(sum((x ** 2).sum() for x in g))
                if gn > max_norm:
                    g = [x * max_norm / gn for x in g]
                self.opt_a.step(self.actor.params(), g)
                v, cacts = self.critic.forward(o)
                dv = (v[:, 0] - R)[:, None] / len(j)
                gW, gb = self.critic.backward(cacts, dv)
                g = gW + gb
                gn = np.sqrt(sum((x ** 2).sum() for x in g))
                if gn > max_norm:
                    g = [x * max_norm / gn for x in g]
                self.opt_c.step(self.critic.params(), g)
                stats.append((float(np.mean(np.minimum(unclipped, clipped))), float(np.mean(H)),
                              float(np.mean((v[:, 0] - R) ** 2))))
        return np.mean(stats, 0)

    def state(self):
        return dict(actor=self.actor.get(), critic=self.critic.get())

    def load(self, st):
        self.actor.set(st["actor"])
        self.critic.set(st["critic"])


def gae(rew, val, gamma=0.98, lam=0.95):
    T = len(rew)
    adv = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(val)


def gradcheck(seed=0):
    """Finite-difference check of the MLP backward pass."""
    rng = np.random.default_rng(seed)
    net = MLP([5, 7, 7, 2], rng)
    x = rng.normal(size=(4, 5))
    t = rng.normal(size=(4, 2))
    out, acts = net.forward(x)
    gW, gb = net.backward(acts, out - t)
    eps = 1e-6
    W = net.W[0]
    i, j = 1, 2
    W[i, j] += eps
    lp = 0.5 * ((net.forward(x)[0] - t) ** 2).sum()
    W[i, j] -= 2 * eps
    lm = 0.5 * ((net.forward(x)[0] - t) ** 2).sum()
    W[i, j] += eps
    return abs((lp - lm) / (2 * eps) - gW[0][i, j])


if __name__ == "__main__":
    print("gradcheck error", gradcheck())
