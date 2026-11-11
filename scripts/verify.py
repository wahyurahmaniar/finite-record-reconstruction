"""verify.py: numerical checks and values reported in the manuscript.

Usage: python scripts/verify.py [section ...]      (all sections by default)

  two     Lemma 11 and Theorem 12: the ellipse bound on random measure-and-prepare
          protocols and its affine certificate, uniqueness of the minimizer, strictness
          against guess-and-prepare, the small-overlap expansions, the maximum of V(c),
          the accessible fidelity of the pair, and the two-state Haar baseline
  sdp     Table 2: primal and dual semidefinite programs over channels with a PPT Choi
          operator, and the support function of the achievable set
  perrun  Proposition 13: the per-run loss on random protocols and mixed pairs
  trine   Sect. 5: trine ensemble (unrestricted optimum, accessible fidelity, the
          fidelity-optimal protocols, record-size search, visible values, Haar baselines)
  sic     Sect. 5: tetrahedral ensemble (unrestricted optimum, accessible fidelity,
          record-size search, visible values)
  misc    Remark 14 (Schmidt-basis protocol) and the mixed example after Corollary 8

Every check prints a PASS or FAIL line; sections two, sdp, trine and sic write
data/results_<section>.json.
"""
import json
import os
import sys

import numpy as np
import cvxpy as cp
from scipy.optimize import brentq, minimize, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)
rng = np.random.default_rng(20261111)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
PAULI = (SX, SY, SZ)
OK = True


def check(name, cond, detail=""):
    global OK
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""), flush=True)
    OK = OK and bool(cond)


def rho_of(n):
    return 0.5 * (I2 + n[0] * SX + n[1] * SY + n[2] * SZ)


def bloch(r):
    return np.real([np.trace(r @ P) for P in PAULI])


def trd(a, b):
    return 0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum()


# ----------------------------------------------------------------- two states
def S_of(c):
    return np.sqrt(1.0 - c * c)


def g_fun(phi, c):
    return (1 - c * c) * np.cos(phi) - c * np.sin(phi) + c * c * np.sin(phi) * np.cos(phi)


def phistar(c):
    return brentq(lambda p: g_fun(p, c), 0.0, np.pi / 2, xtol=1e-15)


def V(c):
    S = S_of(c)
    p = phistar(c)
    return 0.5 * np.sqrt(S * S * (1 - np.sin(p)) ** 2 + (c - np.cos(p)) ** 2)


def gp(c):
    S = S_of(c)
    return 0.5 * (1 - S) * S


def pair(c):
    S = S_of(c)
    return rho_of((-S, 0, c)), rho_of((S, 0, c))


# ----------------------------------------------------------------- SDP tools
def choi_var():
    J = cp.Variable((4, 4), hermitian=True)
    cons = [J >> 0, cp.partial_transpose(J, [2, 2], 1) >> 0,
            cp.partial_trace(J, [2, 2], 1) == I2]
    return J, cons


def apply(J, r):
    return cp.partial_trace(cp.kron(r.T, I2) @ J, [2, 2], 0)


def solve(prob):
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12,
               max_iter=500)
    ex = prob.solver_stats.extra_stats
    dual = getattr(ex, "obj_val_dual", None) if ex is not None else None
    return prob.value, dual


def scan_sdp(states, priors):
    J, cons = choi_var()
    obj = 0
    for p, r in zip(priors, states):
        out = apply(J, r)
        P = cp.Variable((2, 2), hermitian=True)
        N = cp.Variable((2, 2), hermitian=True)
        cons += [P >> 0, N >> 0, r - out == P - N]
        obj += p * 0.5 * cp.real(cp.trace(P + N))
    return solve(cp.Problem(cp.Minimize(obj), cons))


def scan_dual(states, priors):
    """Dual SDP: max sum_i p_i Tr(Z_i rho_i) - Tr(Y), 0 <= Z_i <= I,
    Y x I - sum_i p_i rho_i^T x Z_i = A + B^Gamma, A, B >= 0."""
    Y = cp.Variable((2, 2), hermitian=True)
    A = cp.Variable((4, 4), hermitian=True)
    B = cp.Variable((4, 4), hermitian=True)
    Z = [cp.Variable((2, 2), hermitian=True) for _ in states]
    cons = [A >> 0, B >> 0] + [z >> 0 for z in Z] + [I2 - z >> 0 for z in Z]
    H = sum(p * cp.kron(r.T, z) for p, r, z in zip(priors, states, Z))
    cons.append(cp.kron(Y, I2) - H == A + cp.partial_transpose(B, [2, 2], 1))
    obj = sum(p * cp.real(cp.trace(z @ r)) for p, r, z in zip(priors, states, Z)) - cp.real(cp.trace(Y))
    return solve(cp.Problem(cp.Maximize(obj), cons))[0]


def fid_sdp(states, priors):
    J, cons = choi_var()
    obj = sum(p * cp.real(cp.trace(r @ apply(J, r))) for p, r in zip(priors, states))
    return solve(cp.Problem(cp.Maximize(obj), cons))


# ----------------------------------------------------------------- MP_d search
def povm_from(x, d):
    A = x[: 8 * d].reshape(d, 2, 2, 2)
    E = [(A[m, 0] + 1j * A[m, 1]).conj().T @ (A[m, 0] + 1j * A[m, 1]) for m in range(d)]
    G = sum(E)
    w, U = np.linalg.eigh(G)
    if w.min() < 1e-10:
        return None
    Gi = U @ np.diag(w ** -0.5) @ U.conj().T
    return [Gi @ e @ Gi for e in E]


def states_from(x, d):
    B = x[8 * d:].reshape(d, 2, 2, 2)
    out = []
    for m in range(d):
        Y = B[m, 0] + 1j * B[m, 1]
        t = Y.conj().T @ Y
        out.append(t / np.trace(t).real)
    return out


def loss(Q, T, states, priors):
    return sum(p * trd(r, sum(np.trace(q @ r).real * t for q, t in zip(Q, T)))
               for p, r in zip(priors, states))


def powell(states, priors, d, starts=20, seed=1):
    g = np.random.default_rng(seed)
    best = (10.0, None)
    for _ in range(starts):
        x0 = g.normal(size=16 * d)

        def f(x):
            Q = povm_from(x, d)
            return 10.0 if Q is None else loss(Q, states_from(x, d), states, priors)
        r = minimize(f, x0, method="Powell",
                     options={"xtol": 1e-10, "ftol": 1e-13, "maxiter": 20000, "maxfev": 80000})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    x = best[1]
    return best[0], povm_from(x, d), states_from(x, d)


def alt_sdp(states, priors, Q, T, iters=60):
    """Alternating minimization: each step is an SDP (the loss is biconvex)."""
    d = len(Q)
    val = loss(Q, T, states, priors)
    for _ in range(iters):
        # preparations for fixed measurement
        Tv = [cp.Variable((2, 2), hermitian=True) for _ in range(d)]
        cons = [t >> 0 for t in Tv] + [cp.real(cp.trace(t)) == 1 for t in Tv]
        obj = 0
        for p, r in zip(priors, states):
            out = sum(float(np.trace(q @ r).real) * t for q, t in zip(Q, Tv))
            P = cp.Variable((2, 2), hermitian=True)
            N = cp.Variable((2, 2), hermitian=True)
            cons += [P >> 0, N >> 0, r - out == P - N]
            obj += p * 0.5 * cp.real(cp.trace(P + N))
        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL)
        T = [t.value for t in Tv]
        # measurement for fixed preparations
        Qv = [cp.Variable((2, 2), hermitian=True) for _ in range(d)]
        cons = [q >> 0 for q in Qv] + [sum(Qv) == I2]
        obj = 0
        for p, r in zip(priors, states):
            out = sum(cp.real(cp.trace(q @ r)) * t for q, t in zip(Qv, T))
            P = cp.Variable((2, 2), hermitian=True)
            N = cp.Variable((2, 2), hermitian=True)
            cons += [P >> 0, N >> 0, r - out == P - N]
            obj += p * 0.5 * cp.real(cp.trace(P + N))
        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.CLARABEL)
        Q = [q.value for q in Qv]
        new = loss(Q, T, states, priors)
        if val - new < 1e-12:
            val = min(val, new)
            break
        val = new
    return val, Q, T


def scan_d(states, priors, d, starts=20):
    v0, Q, T = powell(states, priors, d, starts=starts)
    v1, Q, T = alt_sdp(states, priors, Q, T)
    return min(v0, v1), v0, v1


def hull_dist(n, P):
    """Euclidean distance from n to the convex hull of the rows of P (at most four points)."""
    import itertools
    best = np.inf
    k = len(P)
    for r in range(1, k + 1):
        for sub in itertools.combinations(range(k), r):
            Q = P[list(sub)]
            if r == 1:
                best = min(best, np.linalg.norm(n - Q[0]))
                continue
            A = (Q[1:] - Q[0]).T                      # 3 x (r-1)
            lam, *_ = np.linalg.lstsq(A, n - Q[0], rcond=None)
            w = np.concatenate([[1 - lam.sum()], lam])
            if np.all(w >= -1e-12):
                best = min(best, np.linalg.norm(n - Q[0] - A @ lam))
    return best


def visible_d(nvecs, priors, d, starts=12, seed=3):
    """Convex-hull form: min over d Bloch points of the mean half-distance to their hull."""
    g = np.random.default_rng(seed)
    nvecs = np.array(nvecs)

    def f(x):
        P = x.reshape(d, 3)
        nrm = np.linalg.norm(P, axis=1)
        P = P / np.maximum(1.0, nrm)[:, None]
        return sum(p * 0.5 * hull_dist(n, P) for p, n in zip(priors, nvecs))
    best = 10.0
    for _ in range(starts):
        r = minimize(f, g.normal(size=3 * d) * 0.6, method="Powell",
                     options={"xtol": 1e-9, "ftol": 1e-11, "maxfev": 4000})
        best = min(best, r.fun)
    return best


# ----------------------------------------------------------------- sections
def sec_two():
    global rng
    rng = np.random.default_rng(20261111)
    res = {}
    # ellipse lemma on random measure-and-prepare protocols, any number of outcomes
    worst = -1.0
    for trial in range(20000):
        c = rng.uniform(0.02, 0.98)
        S = S_of(c)
        r0, r1 = pair(c)
        d = int(rng.integers(1, 7))
        x = rng.normal(size=16 * d)
        Q = povm_from(x, d)
        if Q is None:
            continue
        if rng.uniform() < 0.5:   # half of the trials with mixed preparations
            T = [rho_of(v / max(1.0, np.linalg.norm(v)) * rng.uniform())
                 for v in rng.normal(size=(d, 3))]
        else:
            T = states_from(x, d)
        o0 = sum(np.trace(q @ r0).real * t for q, t in zip(Q, T))
        o1 = sum(np.trace(q @ r1).real * t for q, t in zip(Q, T))
        b0, b1 = bloch(o0), bloch(o1)
        xi, zeta = (b1[0] - b0[0]) / 2, (b1[2] + b0[2]) / 2
        worst = max(worst, (xi / S) ** 2 + zeta ** 2)
    check("ellipse lemma on 20000 random protocols", worst <= 1 + 1e-12, f"max={worst:.12f}")
    res["ellipse_random_max"] = worst
    # dual certificate g(beta) <= 1 + c sin^2(alpha) beta_z on the unit ball
    viol = -1.0
    for _ in range(200000):
        c = rng.uniform(0, 1)
        a = rng.uniform(0, 2 * np.pi)
        b = rng.normal(size=3)
        b *= rng.uniform() ** (1 / 3) / np.linalg.norm(b)
        gval = np.sqrt(np.cos(a) ** 2 * b[0] ** 2 + np.sin(a) ** 2 * (1 + c * b[2]) ** 2)
        viol = max(viol, gval - (1 + c * np.sin(a) ** 2 * b[2]))
    check("affine certificate on the unit ball", viol <= 1e-12, f"max violation={viol:.2e}")
    # minimizer uniqueness: g' <= -S^2 sin(phi) < 0 on (0, pi/2)
    worst = -1e9
    for c in np.linspace(0.01, 0.99, 99):
        ph = np.linspace(1e-4, np.pi / 2 - 1e-4, 2001)
        gpr = -(1 - c * c) * np.sin(ph) - c * np.cos(ph) + c * c * np.cos(2 * ph)
        worst = max(worst, np.max(gpr + (1 - c * c) * np.sin(ph)))
    check("g' <= -S^2 sin(phi) on (0, pi/2)", worst <= 1e-14, f"max(g'+S^2 sin)={worst:.2e}")
    # global minimizer equals the root, by dense grid
    errs = []
    for c in np.linspace(0.02, 0.98, 49):
        S = S_of(c)
        ph = np.linspace(0, np.pi, 400001)
        f = S * S * (1 - np.sin(ph)) ** 2 + (c - np.cos(ph)) ** 2
        errs.append(abs(0.5 * np.sqrt(f.min()) - V(c)))
    check("V(c) equals the grid minimum over [0, pi]", max(errs) < 1e-8, f"max err={max(errs):.1e}")
    # strictness against guess-and-prepare and g at arccos c
    cs = np.linspace(0.01, 0.99, 99)
    check("V(c) < guess-and-prepare on (0,1)", all(V(c) < gp(c) for c in cs))
    check("g(arccos c) = c S^2 (1 - S)",
          max(abs(g_fun(np.arccos(c), c) - c * S_of(c) ** 2 * (1 - S_of(c))) for c in cs) < 1e-14)
    # expansions
    for c in (0.02, 0.04, 0.08):
        e1 = (V(c) - (c * c / 4 - 3 * c ** 4 / 16)) / c ** 6
        e2 = (gp(c) - V(c)) / c ** 4
        check(f"expansions at c={c}", abs(e1) < 1.0 and abs(e2 - 0.125) < 0.02,
              f"(V - c^2/4 + 3c^4/16)/c^6={e1:.4f}, (gp - V)/c^4={e2:.5f}")
    cstar = minimize_scalar(lambda c: -V(c), bounds=(0.5, 0.99), method="bounded",
                            options={"xatol": 1e-12}).x
    res.update(cstar=cstar, Vstar=V(cstar), gp_at_cstar=gp(cstar),
               gain_at_cstar=1 - V(cstar) / gp(cstar), gp_peak_c=np.sqrt(3) / 2,
               gp_peak=gp(np.sqrt(3) / 2), phistar_0861=phistar(0.861))
    check("maximizer of V", abs(cstar - 0.8615) < 5e-5, f"c*={cstar:.6f}, V={V(cstar):.6f}")
    check("guess-and-prepare peak", abs(gp(np.sqrt(3) / 2) - 0.125) < 1e-15)
    print(f"relative gain at c*: {100 * res['gain_at_cstar']:.2f}%")
    # accessible fidelity from the ellipse
    for c in (0.4, 0.8):
        S = S_of(c)
        th = np.linspace(0, 2 * np.pi, 200001)
        fmax = 0.5 * (1 + np.max(S * S * np.cos(th) + c * np.sin(th)))
        check(f"accessible fidelity from the ellipse at c={c}",
              abs(fmax - 0.5 * (1 + np.sqrt(S ** 4 + c * c))) < 1e-9)
    # two-state Haar: fixed-basis average equals the randomized channel
    for c in (0.3, 0.6, 0.9):
        S = S_of(c)
        u = rng.normal(size=(1_000_000, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        pc = 0.5 + S * np.abs(u[:, 0]) / 2
        fixed = np.mean((1 - pc) * S)
        rand = (1 - pc.mean()) * S
        check(f"two-state Haar at c={c}", abs(fixed - (S / 2 - S * S / 4)) < 2e-3
              and abs(fixed - rand) < 1e-12, f"fixed={fixed:.5f} rand={rand:.5f}")
    json.dump(res, open(os.path.join(DATA, "results_two.json"), "w"), indent=1)


def sec_sdp():
    res = {"rows": []}
    cs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8615, 0.9, 0.95, 0.99]
    for c in cs:
        r0, r1 = pair(c)
        val, _ = scan_sdp([r0, r1], [0.5, 0.5])
        dual = scan_dual([r0, r1], [0.5, 0.5])
        row = dict(c=c, V=V(c), sdp=val, dual=dual, gap=val - dual)
        res["rows"].append(row)
        print(f"c={c:<7} V={V(c):.12f} sdp={val:.12f} dual={dual:.12f} gap={val - dual:.1e} "
              f"|sdp-V|={abs(val - V(c)):.1e}", flush=True)
    check("SDP over EB channels reproduces V(c)",
          max(abs(r["sdp"] - r["V"]) for r in res["rows"]) < 1e-8)
    # support function of the achievable set: max e.(xi/S, zeta) = 1
    worst = 0.0
    for c in (0.3, 0.7):
        S = S_of(c)
        r0, r1 = pair(c)
        for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
            J, cons = choi_var()
            o0, o1 = apply(J, r0), apply(J, r1)
            xi = (cp.real(cp.trace(SX @ o1)) - cp.real(cp.trace(SX @ o0))) / 2
            ze = (cp.real(cp.trace(SZ @ o1)) + cp.real(cp.trace(SZ @ o0))) / 2
            val, _ = solve(cp.Problem(cp.Maximize(np.cos(a) * xi / S + np.sin(a) * ze), cons))
            worst = max(worst, abs(val - 1))
    check("support function of the achievable set equals one", worst < 1e-8, f"max dev={worst:.1e}")
    json.dump(res, open(os.path.join(DATA, "results_sdp.json"), "w"), indent=1)


def sec_perrun():
    global rng
    rng = np.random.default_rng(20261112)
    worst = 1e9
    for _ in range(4000):
        a, b = rng.normal(size=3), rng.normal(size=3)
        a *= rng.uniform() / np.linalg.norm(a)
        b *= rng.uniform() / np.linalg.norm(b)
        r0, r1 = rho_of(a), rho_of(b)
        p = rng.uniform(0.05, 0.95)
        pg = 0.5 * (1 + np.abs(np.linalg.eigvalsh(p * r0 - (1 - p) * r1)).sum())
        d = int(rng.integers(1, 6))
        x = rng.normal(size=16 * d)
        Q = povm_from(x, d)
        if Q is None:
            continue
        T = states_from(x, d)
        run = sum(pi * sum(np.trace(q @ r).real * trd(r, t) for q, t in zip(Q, T))
                  for pi, r in ((p, r0), (1 - p, r1)))
        worst = min(worst, run - (1 - pg) * trd(r0, r1))
    check("per-run loss >= (1 - Pguess) D on random protocols and mixed pairs", worst >= -1e-12,
          f"min slack={worst:.2e}")
    # equality for guess-and-prepare
    c = 0.7
    r0, r1 = pair(c)
    w, U = np.linalg.eigh(0.5 * r0 - 0.5 * r1)
    Qp = U[:, w > 0] @ U[:, w > 0].conj().T
    Q = [Qp, I2 - Qp]
    run = sum(0.5 * sum(np.trace(q @ r).real * trd(r, t) for q, t in zip(Q, (r0, r1)))
              for r in (r0, r1))
    pg = 0.5 * (1 + np.abs(w).sum())
    check("guess-and-prepare attains the per-run bound", abs(run - (1 - pg) * trd(r0, r1)) < 1e-12)


def trine_states():
    nv = [np.array([np.cos(2 * np.pi * j / 3), np.sin(2 * np.pi * j / 3), 0.0]) for j in range(3)]
    return nv, [rho_of(n) for n in nv]


def sic_states():
    nv = [np.array(v) / np.sqrt(3) for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    return nv, [rho_of(n) for n in nv]


def haar(nv, seed, samples=4_000_000, chunk=500_000):
    """Haar-random projective measurement, then preparation of the likelier target:
    distortion averaged over fixed-basis protocols, and of the randomized channel."""
    g = np.random.default_rng(seed)
    nv = np.array(nv)
    K = len(nv)
    fixed = 0.0
    racc = np.zeros((K, 3))
    done = 0
    while done < samples:
        u = g.normal(size=(chunk, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        proj = u @ nv.T                         # chunk x K
        gplus, gminus = proj.argmax(1), proj.argmin(1)
        for j in range(K):
            pp = (1 + proj[:, j]) / 2
            r = pp[:, None] * nv[gplus] + (1 - pp)[:, None] * nv[gminus]
            fixed += np.sum(0.5 * np.linalg.norm(nv[j] - r, axis=1)) / K
            racc[j] += r.sum(0)
        done += chunk
    fixed /= samples
    rbar = racc / samples
    rand = np.mean([0.5 * np.linalg.norm(nv[j] - rbar[j]) for j in range(K)])
    return fixed, rand


def sec_trine():
    res = {}
    nv, st = trine_states()
    pr = [1 / 3] * 3
    sd, _ = scan_sdp(st, pr)
    res["sdp"], res["sdp_dual"] = sd, scan_dual(st, pr)
    fa, _ = fid_sdp(st, pr)
    check("trine: unrestricted optimum 1/4", abs(sd - 0.25) < 1e-8, f"{sd:.10f} dual {res['sdp_dual']:.10f}")
    check("trine: accessible fidelity 3/4", abs(fa - 0.75) < 1e-8, f"{fa:.10f}")
    # two-outcome fidelity-optimal family: axis u in the plane, prepare +/-u
    Ds = []
    for th in np.linspace(0, np.pi / 3, 6001):
        u = np.array([np.cos(th), np.sin(th), 0])
        F = D = 0.0
        for n in nv:
            a = n @ u
            F += ((1 + a) ** 2 / 4 + (1 - a) ** 2 / 4) / 3
            D += 0.5 * np.linalg.norm(n - a * u) / 3
        Ds.append(D)
        assert abs(F - 0.75) < 1e-12
    res.update(fid2_min=min(Ds), fid2_max=max(Ds))
    check("trine: two-outcome fidelity-optimal family", abs(min(Ds) - 1 / (2 * np.sqrt(3))) < 1e-9
          and abs(max(Ds) - 1 / 3) < 1e-9, f"[{min(Ds):.4f}, {max(Ds):.4f}]")
    # pretty-good measurement then prepare: F = 3/4, D = 1/4
    Q = [2 / 3 * r for r in st]
    out = [sum(np.trace(q @ r).real * t for q, t in zip(Q, st)) for r in st]
    F = np.mean([np.trace(r @ o).real for r, o in zip(st, out)])
    D = np.mean([trd(r, o) for r, o in zip(st, out)])
    check("trine: PGM fidelity 3/4 and distortion 1/4", abs(F - .75) < 1e-12 and abs(D - .25) < 1e-12)
    vals = {}
    for d in (1, 2, 3, 4):
        best, v0, v1 = scan_d(st, pr, d, starts=12)
        vals[d] = best
        print(f"trine scan d={d}: powell={v0:.6f} alt-SDP={v1:.6f}", flush=True)
    res["scan"] = vals
    check("trine: scan values 0.500 0.274 0.250 0.250",
          [round(vals[d], 3) for d in (1, 2, 3, 4)] == [0.5, 0.274, 0.25, 0.25])
    vis = {d: visible_d(nv, pr, d) for d in (1, 2, 3)}
    res["vis"] = vis
    print("trine visible:", {k: round(v, 6) for k, v in vis.items()})
    fixed, rand = haar(nv, seed=3)
    res.update(haar_fixed=fixed, haar_rand=rand)
    check("trine Haar: fixed-basis average 0.354, randomized channel 0.338",
          abs(fixed - 0.354) < 1.5e-3 and abs(rand - 0.338) < 1.5e-3, f"{fixed:.4f}, {rand:.4f}")
    json.dump(res, open(os.path.join(DATA, "results_trine.json"), "w"), indent=1)


def sec_sic():
    res = {}
    nv, st = sic_states()
    pr = [0.25] * 4
    sd, _ = scan_sdp(st, pr)
    res["sdp"], res["sdp_dual"] = sd, scan_dual(st, pr)
    fa, _ = fid_sdp(st, pr)
    check("SIC: unrestricted optimum 1/3", abs(sd - 1 / 3) < 1e-8, f"{sd:.10f} dual {res['sdp_dual']:.10f}")
    check("SIC: accessible fidelity 2/3", abs(fa - 2 / 3) < 1e-8, f"{fa:.10f}")
    vals = {}
    for d in (1, 2, 3, 4, 5):
        best, v0, v1 = scan_d(st, pr, d, starts=16)
        vals[d] = best
        print(f"SIC scan d={d}: powell={v0:.6f} alt-SDP={v1:.6f}", flush=True)
    res["scan"] = vals
    vis = {d: visible_d(nv, pr, d) for d in (1, 2, 3, 4)}
    res["vis"] = vis
    print("SIC visible:", {k: round(v, 6) for k, v in vis.items()})
    # two-outcome fidelity-optimal protocols: projective measurement along any axis u,
    # preparations +u and -u; vertex axes and edge-midpoint axes are the extremes
    g = np.random.default_rng(4)
    U = g.normal(size=(400_000, 3))
    U /= np.linalg.norm(U, axis=1)[:, None]
    U = np.vstack([U, nv, np.eye(3)])
    X = U @ np.array(nv).T
    F = np.mean((1 + X ** 2) / 2, axis=1)
    D = np.mean(0.5 * np.sqrt(np.clip(1 - X ** 2, 0, None)), axis=1)
    check("SIC: every two-outcome axis protocol has fidelity 2/3", np.max(np.abs(F - 2 / 3)) < 1e-12)
    res.update(fid2_min=D.min(), fid2_max=D.max())
    check("SIC: two-outcome range [1/(2 sqrt 2), 1/sqrt 6], vertex axis to edge-midpoint axis",
          abs(D.min() - 1 / (2 * np.sqrt(2))) < 1e-12 and abs(D.max() - 1 / np.sqrt(6)) < 1e-12,
          f"[{D.min():.6f}, {D.max():.6f}]")
    check("SIC: the vertex-axis protocol attains the d = 2 optimum and the visible value",
          abs(vals[2] - D.min()) < 1e-9 and abs(vis[2] - D.min()) < 1e-9,
          f"scan {vals[2]:.12f}, visible {vis[2]:.12f}")
    fixed, rand = haar(nv, seed=4)
    res.update(haar_fixed=fixed, haar_rand=rand)
    check("SIC Haar: fixed-basis average 0.411, randomized channel 0.376",
          abs(fixed - 0.411) < 1.5e-3 and abs(rand - 0.376) < 1.5e-3, f"{fixed:.4f}, {rand:.4f}")
    json.dump(res, open(os.path.join(DATA, "results_sic.json"), "w"), indent=1)


def sec_misc():
    # Schmidt-basis measure-and-reprepare attains 1 - 1/n
    for n in (2, 3, 5):
        om = np.zeros(n * n)
        for k in range(n):
            om[k * n + k] = 1 / np.sqrt(n)
        Om = np.outer(om, om)
        sig = np.zeros((n * n, n * n))
        for k in range(n):
            sig[k * n + k, k * n + k] = 1 / n
        check(f"EB bound tight at n={n}", abs(trd(Om, sig) - (1 - 1 / n)) < 1e-12)
    # mixed example of the Koashi-Imoto criterion
    r0, r1 = np.diag([0.7, 0.3]).astype(complex), np.diag([0.3, 0.7]).astype(complex)
    P0 = np.diag([1, 0]).astype(complex)
    P1 = I2 - P0
    for r in (r0, r1):
        out = np.trace(P0 @ r).real * P0 + np.trace(P1 @ r).real * P1
        check("block measurement rebuilds the mixed example", np.allclose(out, r))


if __name__ == "__main__":
    secs = sys.argv[1:] or ["two", "sdp", "perrun", "trine", "sic", "misc"]
    for s in secs:
        print(f"== {s}", flush=True)
        globals()["sec_" + s]()
    print("ALL PASS" if OK else "SOME CHECKS FAILED")
