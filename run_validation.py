#!/usr/bin/env python3
"""
run_validation.py — Falsification suite for FDFA
=================================================

Tests the predictions of the FDFA theory note (fdfa_theory_note.tex):

  P1 (Exponent universality). On fGn with Hurst H, every admissible
     functional Phi yields slope ~ H. Differences are prefactors only.

  P2 (Heavy-tail robustness). On iid symmetric alpha-stable noise
     (alpha < 2) the true exponent is 1/alpha. Variance-based DFA
     estimates are erratic across realizations (infinite variance of
     the summands); quantile/median/Huber functionals concentrate.

  P3 (Robustness-sensitivity trade-off). On a binomial multiplicative
     cascade, robust functionals attenuate the right side (q > 0) of
     h(q) less faithfully than RMS: extreme events ARE the multifractal
     signal, so robustness is not free.

Usage:
    python run_validation.py            # full run (~ a few minutes)
    python run_validation.py --quick    # reduced sizes, sanity check

Outputs go to ./results/ : PNG figures + CSV tables + summary.txt.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fdfa import (FUNCTIONALS, fdfa, mfdfa_hq, fdfa_multi, mfdfa_hq_multi,
                  default_scales)
from generators import (fgn, alpha_stable, binomial_cascade,
                        cascade_hq_theory, mrw, mrw_hq_theory, arfima,
                        tempered_stable)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ----------------------------------------------------------------------
# Figure style: scienceplots [science, nature] with LaTeX if available
# ----------------------------------------------------------------------

def _setup_style() -> str:
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        return "default (scienceplots not installed)"
    try:
        plt.style.use(["science", "nature"])
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$F_\Phi(s)$")
        fig.canvas.draw()          # probe: raises if usetex broken
        plt.close(fig)
        return "science+nature (usetex)"
    except Exception:
        plt.close("all")
        plt.style.use(["science", "nature", "no-latex"])
        return "science+nature (mathtext fallback)"


STYLE = _setup_style()

# readable font sizes on top of the nature style (sized for our figures)
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def add_grid(ax, axis="both"):
    """Subtle grid: full for line/scatter/log plots, y-only for bars."""
    ax.grid(True, axis=axis, which="major", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)

#: standard-notation display names for functionals (math mode)
PHI_TEX = {
    "rms":    r"$\mathrm{RMS}$",
    "l1":     r"$L^{1}$",
    "lp0.5":  r"$L^{1/2}$",
    "mad":    r"$\mathrm{MAD}$",
    "q0.75":  r"$Q_{0.75}$",
    "q0.9":   r"$Q_{0.90}$",
    "iqr":    r"$\mathrm{IQR}$",
    "huber":  r"$\mathrm{Huber}$",
    "tstd10": r"$\sigma_{\mathrm{tr}}$",
    "linf":   r"$L^{\infty}$",
    "entpow": r"$e^{\hat{h}_{\mathrm{ent}}}$",
    "rms-cl": r"$\mathrm{RMS},\,\mathcal{A}_{2}$",
}


def disp(name: str) -> str:
    return PHI_TEX.get(name, name)

# Functionals used in each experiment (subset keeps figures readable)
FGN_PHIS = ["rms", "l1", "lp0.5", "mad", "q0.75", "q0.9", "iqr",
            "huber", "tstd10", "linf", "entpow"]
STABLE_PHIS = ["rms", "l1", "mad", "q0.75", "huber", "tstd10"]
CASCADE_PHIS = ["rms", "mad", "q0.75", "huber"]

COLORS = plt.cm.tab10.colors + plt.cm.Dark2.colors


def _write_csv(path: str, header: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------
# Experiment 1 — fGn: exponent universality (P1)
# ----------------------------------------------------------------------

def experiment_fgn(n: int, n_real: int, order: int, rng, log) -> list:
    log("\n=== Experiment 1: fGn, prediction P1 (slope ~ H for all Phi) ===")
    H_grid = [0.3, 0.5, 0.7]
    rows = []
    est = {(H, name): [] for H in H_grid for name in FGN_PHIS}

    specs = [(name, FUNCTIONALS[name], "qmean", 2.0) for name in FGN_PHIS]
    for H in H_grid:
        for r in range(n_real):
            x = fgn(n, H, rng)
            res = fdfa_multi(x, specs, order=order)
            for name in FGN_PHIS:
                est[(H, name)].append(res[name].slope)

    log(f"{'Phi':>8} | " + " | ".join(f"H={H:.1f}" for H in H_grid))
    log("-" * 60)
    for name in FGN_PHIS:
        cells = []
        for H in H_grid:
            v = np.array(est[(H, name)])
            cells.append(f"{v.mean():.3f}+-{v.std():.3f}")
            rows.append(["fgn", name, H, v.mean(), v.std(), n, n_real])
        log(f"{name:>8} | " + " | ".join(cells))

    # Figure 1: example log-log curves at H = 0.7
    x = fgn(n, 0.7, rng)
    res_all = fdfa_multi(x, specs, order=order)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(FGN_PHIS):
        res = res_all[name]
        ax.loglog(res.scales, res.F, "o-", ms=3, lw=1,
                  color=COLORS[i % len(COLORS)],
                  label=disp(name) + rf" ($\hat{{H}}={res.slope:.3f}$)")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$F_{\Phi}(s)$")
    ax.set_title(r"fGn, $H=0.7$: exponent universality across $\Phi$")
    ax.legend(fontsize=10, ncol=2)
    add_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig1_fgn_loglog.png"), dpi=150)
    plt.close(fig)

    # Figure 2: estimated exponent vs H per functional
    fig, ax = plt.subplots(figsize=(7, 5))
    xpos = np.arange(len(FGN_PHIS))
    width = 0.25
    for j, H in enumerate(H_grid):
        means = [np.mean(est[(H, nm)]) for nm in FGN_PHIS]
        stds = [np.std(est[(H, nm)]) for nm in FGN_PHIS]
        ax.bar(xpos + (j - 1) * width, means, width, yerr=stds,
               label=rf"$H={H}$", capsize=2)
        ax.axhline(H, color="k", lw=0.5, ls="--")
    ax.set_xticks(xpos)
    ax.set_xticklabels([disp(nm) for nm in FGN_PHIS],
                       rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(r"$\hat{H}$")
    ax.set_title(r"P1: exponent universality on fGn")
    ax.legend()
    add_grid(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig2_fgn_universality.png"), dpi=150)
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------
# Experiment 2 — alpha-stable: robustness under infinite variance (P2)
# ----------------------------------------------------------------------

def experiment_stable(n: int, n_real: int, order: int, rng, log) -> list:
    log("\n=== Experiment 2: iid alpha-stable, prediction P2 (true exp = 1/alpha) ===")
    a_grid = [1.2, 1.5, 1.8]
    rows = []

    # 'rms-cl' = fully classical DFA: RMS inner + q=2 moment outer.
    # All others use a median outer layer (fully robust pipeline).
    variants = [("rms-cl", "rms", "qmean")] + \
               [(nm, nm, "median") for nm in STABLE_PHIS]
    specs = [(lab, FUNCTIONALS[name], agg, 2.0)
             for lab, name, agg in variants]
    est = {(a, lab): [] for a in a_grid for lab, _, _ in variants}

    for a in a_grid:
        for r in range(n_real):
            x = alpha_stable(n, a, rng=rng)
            res = fdfa_multi(x, specs, order=order)
            for lab, _, _ in variants:
                est[(a, lab)].append(res[lab].slope)

    labels = [lab for lab, _, _ in variants]
    log(f"{'Phi':>8} | " + " | ".join(
        f"a={a:.1f} (1/a={1/a:.3f})" for a in a_grid))
    log("-" * 78)
    for lab in labels:
        cells = []
        for a in a_grid:
            v = np.array(est[(a, lab)])
            cells.append(f"{v.mean():.3f}+-{v.std():.3f}")
            rows.append(["stable", lab, a, v.mean(), v.std(), n, n_real])
        log(f"{lab:>8} | " + " | ".join(cells))

    # Figure 3: spread of estimates, classical DFA vs robust functionals
    fig, axes = plt.subplots(1, len(a_grid), figsize=(11, 4), sharey=True)
    for k, a in enumerate(a_grid):
        ax = axes[k]
        data = [est[(a, lab)] for lab in labels]
        ax.boxplot(data, tick_labels=[disp(lab) for lab in labels])
        ax.axhline(1.0 / a, color="r", ls="--", lw=1,
                   label=rf"$1/\alpha={1/a:.3f}$")
        ax.set_title(rf"$\alpha={a}$")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        if k == 0:
            ax.set_ylabel(r"$\hat{H}$")
        ax.legend(fontsize=10)
        add_grid(ax, "y")
    fig.suptitle(r"P2: exponent estimation under infinite variance")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_stable_robustness.png"), dpi=150)
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------
# Experiment 3 — cascade: robustness vs multifractal sensitivity (P3)
# ----------------------------------------------------------------------

def experiment_cascade(J: int, n_real: int, order: int, rng, log) -> list:
    log("\n=== Experiment 3: binomial cascade, prediction P3 (trade-off) ===")
    m0 = 0.6
    qs = np.linspace(-4, 4, 17)
    hq_th = cascade_hq_theory(qs, m0)
    rows = []
    hq_acc = {name: [] for name in CASCADE_PHIS}

    for r in range(n_real):
        x = binomial_cascade(J, m0, rng)
        x = x - x.mean()
        res = mfdfa_hq_multi(x, qs, {nm: FUNCTIONALS[nm]
                                     for nm in CASCADE_PHIS}, order=order)
        for name in CASCADE_PHIS:
            hq_acc[name].append(res[name].hq)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(qs, hq_th, "k-", lw=2, label=r"theory (binomial cascade)")
    for i, name in enumerate(CASCADE_PHIS):
        hq_m = np.mean(hq_acc[name], axis=0)
        hq_s = np.std(hq_acc[name], axis=0)
        ax.errorbar(qs, hq_m, yerr=hq_s, fmt="o-", ms=3, lw=1, capsize=2,
                    color=COLORS[i % len(COLORS)], label=disp(name))
        for q, hm, hs in zip(qs, hq_m, hq_s):
            rows.append(["cascade", name, q, hm, hs, 2 ** J, n_real])
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$h(q)$")
    ax.set_title(r"P3: genuine multifractality, robust vs classical $\Phi$")
    ax.legend(fontsize=11)
    add_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig4_cascade_hq.png"), dpi=150)
    plt.close(fig)

    # log multifractal width Delta h = h(-4) - h(4)
    log(f"{'Phi':>8} | width Dh = h(-4) - h(4)   (theory: {hq_th[0]-hq_th[-1]:.3f})")
    log("-" * 50)
    for name in CASCADE_PHIS:
        hq_m = np.mean(hq_acc[name], axis=0)
        log(f"{name:>8} | {hq_m[0] - hq_m[-1]:.3f}")
    return rows


# ----------------------------------------------------------------------
# Experiment 4 — spurious multifractality of iid stable noise (P4)
# ----------------------------------------------------------------------

def experiment_spurious(n: int, n_real: int, order: int, rng, log) -> list:
    log("\n=== Experiment 4: iid alpha-stable h(q), prediction P4 "
        "(robust Phi suppresses spurious width) ===")
    a = 1.5
    qs = np.linspace(-4, 4, 17)
    # bifractal theory for iid stable noise: h(q) = 1/a (q <= a), 1/q (q > a)
    hq_th = np.full_like(qs, 1.0 / a)
    hq_th[qs > a] = 1.0 / qs[qs > a]
    rows = []
    phis = ["rms", "mad", "q0.75", "huber"]
    hq_acc = {name: [] for name in phis}

    for r in range(n_real):
        x = alpha_stable(n, a, rng=rng)
        res = mfdfa_hq_multi(x, qs, {nm: FUNCTIONALS[nm] for nm in phis},
                             order=order)
        for name in phis:
            hq_acc[name].append(res[name].hq)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(qs, hq_th, "k--", lw=1.5, label=r"bifractal theory")
    ax.axhline(1.0 / a, color="gray", lw=0.8, ls=":",
               label=rf"$1/\alpha={1/a:.3f}$")
    for i, name in enumerate(phis):
        hq_m = np.mean(hq_acc[name], axis=0)
        hq_s = np.std(hq_acc[name], axis=0)
        ax.errorbar(qs, hq_m, yerr=hq_s, fmt="o-", ms=3, lw=1, capsize=2,
                    color=COLORS[i % len(COLORS)], label=disp(name))
        for q, hm, hs in zip(qs, hq_m, hq_s):
            rows.append(["spurious", name, q, hm, hs, n, n_real])
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$h(q)$")
    ax.set_title(rf"P4: distributional multifractality, iid stable $\alpha={a}$")
    ax.legend(fontsize=11)
    add_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig5_spurious_hq.png"), dpi=150)
    plt.close(fig)

    log(f"{'Phi':>8} | width Dh = h(-4) - h(4)   "
        f"(genuine MF width would be 0)")
    log("-" * 50)
    for name in phis:
        hq_m = np.mean(hq_acc[name], axis=0)
        log(f"{name:>8} | {hq_m[0] - hq_m[-1]:.3f}")
    return rows


# ----------------------------------------------------------------------
# Experiment 5 — the (M3) discriminator: genuine vs spurious MF (P5)
# ----------------------------------------------------------------------

def experiment_discriminator(n: int, J: int, n_real: int, order: int,
                             rng, log) -> list:
    """Quantile-exponent profile theta(tau).

    For fixed tau, fit  log Q_tau({F(v,s)}_v)  vs  log s  -> theta(tau).

    Prediction P5: windows of a heavy-tailed *monofractal* signal are
    exchangeable with a common scale factor, so every fixed cross-window
    quantile scales identically: theta(tau) ~ const (flat), even though
    the classical q-moment width Dh is large (spurious MF). Genuinely
    multifractal signals have scale-structured window heterogeneity:
    theta(tau) varies with tau AND Dh is large. Thus
        Dtheta = theta(0.9) - theta(0.1)  ~ 0  &  Dh >> 0  -> spurious
        Dtheta >> 0                       &  Dh >> 0  -> genuine
    """
    log("\n=== Experiment 5: (M3) discriminator, prediction P5 ===")
    taus = np.linspace(0.1, 0.9, 9)
    qs = np.linspace(-4, 4, 17)
    rows = []

    def make_signals():
        return [
            (r"fGn $H=0.7$ (mono)", fgn(n, 0.7, rng)),
            (r"cascade $m_0=0.6$ (genuine)",
             binomial_cascade(J, 0.6, rng) - 1.0),
            (r"stable $\alpha=1.5$ (spurious)",
             alpha_stable(n, 1.5, rng=rng)),
        ]

    names = [nm for nm, _ in make_signals()]
    theta_acc = {nm: [] for nm in names}
    dh_acc = {nm: [] for nm in names}

    for r in range(n_real):
        for nm, x in make_signals():
            # restrict to scales with >= 64 windows: high quantiles of the
            # cross-window ensemble need a populated ensemble
            sc = default_scales(x.size, s_max=max(64, x.size // 32))
            res = fdfa(x, FUNCTIONALS["rms"], order=order, scales=sc,
                       keep_windows=True)
            ls = np.log(np.array(sorted(res.Fv.keys()), dtype=float))
            theta = np.empty(taus.size)
            for i, tau in enumerate(taus):
                lq = np.array([np.log(np.quantile(res.Fv[s], tau))
                               for s in sorted(res.Fv.keys())])
                theta[i] = np.polyfit(ls, lq, 1)[0]
            theta_acc[nm].append(theta)
            hq = mfdfa_hq(x, qs, FUNCTIONALS["rms"], order=order).hq
            dh_acc[nm].append(hq[0] - hq[-1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    log(f"{'signal':>32} | Dtheta = th(.9)-th(.1) | Dh = h(-4)-h(4) | verdict")
    log("-" * 92)
    dtheta_bar, dh_bar = [], []
    for i, nm in enumerate(names):
        th_m = np.mean(theta_acc[nm], axis=0)
        th_s = np.std(theta_acc[nm], axis=0)
        axes[0].errorbar(taus, th_m, yerr=th_s, fmt="o-", ms=3, lw=1,
                         capsize=2, color=COLORS[i], label=nm)
        dth = th_m[-1] - th_m[0]
        dh = float(np.mean(dh_acc[nm]))
        dtheta_bar.append(dth)
        dh_bar.append(dh)
        verdict = ("genuine MF" if (dh > 0.15 and abs(dth) > 0.08)
                   else "spurious MF" if dh > 0.15 else "monofractal")
        log(f"{nm:>32} | {dth:23.3f} | {dh:15.3f} | {verdict}")
        rows.append(["discriminator", nm, "dtheta", dth,
                     float(np.std([t[-1] - t[0] for t in theta_acc[nm]])),
                     n, n_real])
        rows.append(["discriminator", nm, "dh", dh,
                     float(np.std(dh_acc[nm])), n, n_real])
    axes[0].set_xlabel(r"$\tau$")
    axes[0].set_ylabel(r"$\theta(\tau)$")
    axes[0].set_title(r"quantile-exponent profile (outer layer)")
    axes[0].legend(fontsize=10)
    add_grid(axes[0])

    xb = np.arange(len(names))
    axes[1].bar(xb - 0.18, dh_bar, 0.36, label=r"$\Delta h$ ($q$-moments)")
    axes[1].bar(xb + 0.18, dtheta_bar, 0.36,
                label=r"$\Delta\theta$ (quantiles)")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels([r"fGn", r"cascade", r"stable"], fontsize=11)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title(r"P5: genuine vs spurious discriminator")
    axes[1].legend(fontsize=11)
    add_grid(axes[1], "y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig6_discriminator.png"), dpi=150)
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------
# Experiment 6 — MRW: genuine MF with Gaussian tails (P6)
# ----------------------------------------------------------------------

def experiment_mrw(n: int, n_real: int, order: int, rng, log) -> list:
    """The most demanding test: log-normal MRW has *genuine*
    multifractality but Gaussian marginals -> no heavy-tail artifacts.

    P6: (a) h(q) is recovered by RMS and robust Phi alike (M2);
        (b) the discriminator says 'genuine': |Dtheta| >> 0.
    """
    log("\n=== Experiment 6: MRW (lognormal), prediction P6 ===")
    lam2 = 0.05
    qs = np.linspace(-4, 4, 17)
    hq_th = mrw_hq_theory(qs, lam2)
    phis = ["rms", "mad", "q0.75", "huber"]
    taus = np.linspace(0.1, 0.9, 9)
    rows = []
    hq_acc = {name: [] for name in phis}
    theta_acc = []

    for r in range(n_real):
        x = mrw(n, lam2, rng=rng)
        res = mfdfa_hq_multi(x, qs, {nm: FUNCTIONALS[nm] for nm in phis},
                             order=order)
        for name in phis:
            hq_acc[name].append(res[name].hq)
        # discriminator on the same signal
        sc = default_scales(n, s_max=max(64, n // 32))
        rf_ = fdfa(x, FUNCTIONALS["rms"], order=order, scales=sc,
                   keep_windows=True)
        ls = np.log(np.array(sorted(rf_.Fv.keys()), dtype=float))
        theta = [np.polyfit(ls, [np.log(np.quantile(rf_.Fv[s], tau))
                                 for s in sorted(rf_.Fv.keys())], 1)[0]
                 for tau in taus]
        theta_acc.append(theta)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(qs, hq_th, "k-", lw=2,
            label=rf"theory: $h(q)=1/2+\lambda^2-\frac{{\lambda^2}}{{2}}q$, $\lambda^2={lam2}$")
    for i, name in enumerate(phis):
        hq_m = np.mean(hq_acc[name], axis=0)
        hq_s = np.std(hq_acc[name], axis=0)
        ax.errorbar(qs, hq_m, yerr=hq_s, fmt="o-", ms=3, lw=1, capsize=2,
                    color=COLORS[i % len(COLORS)], label=disp(name))
        for q, hm, hs in zip(qs, hq_m, hq_s):
            rows.append(["mrw", name, q, hm, hs, n, n_real])
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$h(q)$")
    ax.set_title(r"P6: MRW --- genuine MF, Gaussian tails")
    ax.legend(fontsize=10)
    add_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig7_mrw_hq.png"), dpi=150)
    plt.close(fig)

    th_m = np.mean(theta_acc, axis=0)
    dth = th_m[-1] - th_m[0]
    dh_th = hq_th[0] - hq_th[-1]
    log(f"{'Phi':>8} | width Dh = h(-4)-h(4)   (theory: {dh_th:.3f})")
    log("-" * 50)
    for name in phis:
        hq_m = np.mean(hq_acc[name], axis=0)
        log(f"{name:>8} | {hq_m[0] - hq_m[-1]:.3f}")
    log(f"discriminator on MRW: Dtheta = {dth:.3f} "
        f"(genuine expected: |Dtheta| >> 0)")
    rows.append(["mrw", "rms", "dtheta", dth,
                 float(np.std([t[-1] - t[0] for t in theta_acc])),
                 n, n_real])
    return rows


# ----------------------------------------------------------------------
# Experiment 7 — ARFIMA: long memory, different construction (P7)
# ----------------------------------------------------------------------

def experiment_arfima(n: int, n_real: int, order: int, rng, log) -> list:
    log("\n=== Experiment 7: ARFIMA(0,d,0), prediction P7 (slope = d+1/2) ===")
    d_grid = [-0.2, 0.2, 0.4]
    phis = ["rms", "l1", "mad", "q0.75", "huber", "tstd10"]
    specs = [(nm, FUNCTIONALS[nm], "qmean", 2.0) for nm in phis]
    rows = []
    est = {(d, nm): [] for d in d_grid for nm in phis}

    for d in d_grid:
        for r in range(n_real):
            x = arfima(n, d, rng)
            res = fdfa_multi(x, specs, order=order)
            for nm in phis:
                est[(d, nm)].append(res[nm].slope)

    log(f"{'Phi':>8} | " + " | ".join(
        f"d={d:+.1f} (H={d+0.5:.1f})" for d in d_grid))
    log("-" * 66)
    for nm in phis:
        cells = []
        for d in d_grid:
            v = np.array(est[(d, nm)])
            cells.append(f"{v.mean():.3f}+-{v.std():.3f}")
            rows.append(["arfima", nm, d, v.mean(), v.std(), n, n_real])
        log(f"{nm:>8} | " + " | ".join(cells))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xpos = np.arange(len(phis))
    width = 0.25
    for j, d in enumerate(d_grid):
        means = [np.mean(est[(d, nm)]) for nm in phis]
        stds = [np.std(est[(d, nm)]) for nm in phis]
        ax.bar(xpos + (j - 1) * width, means, width, yerr=stds,
               label=rf"$d={d:+.1f}$", capsize=2)
        ax.axhline(d + 0.5, color="k", lw=0.5, ls="--")
    ax.set_xticks(xpos)
    ax.set_xticklabels([disp(nm) for nm in phis], rotation=45,
                       ha="right", fontsize=11)
    ax.set_ylabel(r"$\hat{H}$")
    ax.set_title(r"P7: ARFIMA$(0,d,0)$, $H=d+1/2$")
    ax.legend(fontsize=11)
    add_grid(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig8_arfima.png"), dpi=150)
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------
# Experiment 8 — agreement with the shuffle-surrogate test (P8)
# ----------------------------------------------------------------------

def experiment_surrogates(n: int, J: int, n_real: int, order: int,
                          rng, log) -> list:
    """Community-standard check: shuffling destroys correlations but
    keeps the marginal distribution. Genuine MF -> Dh collapses after
    shuffling; distributional MF -> Dh survives shuffling.

    P8: the (Dh, Dtheta) discriminator agrees with the shuffle verdict
    on every signal, at a fraction of the cost.
    """
    log("\n=== Experiment 8: shuffle surrogates vs discriminator (P8) ===")
    qs = np.linspace(-4, 4, 17)
    taus = np.linspace(0.1, 0.9, 9)
    n_shuffle = 5
    rows = []

    def signals():
        return [
            ("fGn H=0.7", fgn(n, 0.7, rng)),
            ("cascade m0=0.6", binomial_cascade(J, 0.6, rng) - 1.0),
            ("stable a=1.5", alpha_stable(n, 1.5, rng=rng)),
            ("MRW lam2=0.05", mrw(n, 0.05, rng=rng)),
            ("tempered stable", tempered_stable(n, 1.5, ell=10.0, rng=rng)),
        ]

    names = [nm for nm, _ in signals()]
    acc = {nm: {"dh": [], "dhs": [], "dth": []} for nm in names}

    for r in range(n_real):
        for nm, x in signals():
            hq = mfdfa_hq(x, qs, FUNCTIONALS["rms"], order=order).hq
            acc[nm]["dh"].append(hq[0] - hq[-1])
            dhs = []
            for _ in range(n_shuffle):
                xs = rng.permutation(x)
                hqs = mfdfa_hq(xs, qs, FUNCTIONALS["rms"], order=order).hq
                dhs.append(hqs[0] - hqs[-1])
            acc[nm]["dhs"].append(np.mean(dhs))
            sc = default_scales(x.size, s_max=max(64, x.size // 32))
            rf_ = fdfa(x, FUNCTIONALS["rms"], order=order, scales=sc,
                       keep_windows=True)
            ls = np.log(np.array(sorted(rf_.Fv.keys()), dtype=float))
            th = [np.polyfit(ls, [np.log(np.quantile(rf_.Fv[s], tau))
                                  for s in sorted(rf_.Fv.keys())], 1)[0]
                  for tau in taus]
            acc[nm]["dth"].append(th[-1] - th[0])

    log(f"{'signal':>18} | {'Dh':>6} | {'Dh_shuf':>7} | {'Dtheta':>7} | "
        f"{'shuffle verdict':>16} | {'theta verdict':>14} | agree?")
    log("-" * 92)
    dh_b, dhs_b, dth_b = [], [], []
    for nm in names:
        dh = float(np.mean(acc[nm]["dh"]))
        dhs = float(np.mean(acc[nm]["dhs"]))
        dth = float(np.mean(acc[nm]["dth"]))
        dh_b.append(dh); dhs_b.append(dhs); dth_b.append(dth)
        # shuffle verdict: genuine iff correlations carry the width
        v_shuf = ("genuine" if (dh > 0.15 and dh - dhs > 0.5 * dh)
                  else "spurious" if dh > 0.15 else "mono")
        v_th = ("genuine" if (dh > 0.15 and abs(dth) > 0.08)
                else "spurious" if dh > 0.15 else "mono")
        agree = "YES" if v_shuf == v_th else "NO"
        log(f"{nm:>18} | {dh:6.3f} | {dhs:7.3f} | {dth:7.3f} | "
            f"{v_shuf:>16} | {v_th:>14} | {agree}")
        rows += [["surrogate", nm, "dh", dh, float(np.std(acc[nm]['dh'])), n, n_real],
                 ["surrogate", nm, "dh_shuffled", dhs, float(np.std(acc[nm]['dhs'])), n, n_real],
                 ["surrogate", nm, "dtheta", dth, float(np.std(acc[nm]['dth'])), n, n_real]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xb = np.arange(len(names))
    ax.bar(xb - 0.25, dh_b, 0.25, label=r"$\Delta h$ (original)")
    ax.bar(xb, dhs_b, 0.25, label=r"$\Delta h$ (shuffled)")
    ax.bar(xb + 0.25, np.abs(dth_b), 0.25, label=r"$|\Delta\theta|$")
    ax.set_xticks(xb)
    ax.set_xticklabels([r"fGn", r"cascade", r"stable", r"MRW",
                        r"tempered"], fontsize=11)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(r"P8: discriminator vs shuffle surrogates")
    ax.legend(fontsize=11)
    add_grid(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig9_surrogates.png"), dpi=150)
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------
# Experiment 9 — finite-size and detrending-order study (P9)
# ----------------------------------------------------------------------

def experiment_finitesize(n_real: int, order_max: int, rng, log) -> list:
    log("\n=== Experiment 9: RMSE vs n and detrending order m (P9) ===")
    n_grid = [2 ** k for k in range(10, 15)]
    orders = list(range(1, order_max + 1))
    phis = ["rms", "q0.75", "huber"]
    rows = []

    cases = [("fGn H=0.7", 0.7,
              lambda n: fgn(n, 0.7, rng), "qmean"),
             ("stable a=1.5", 1.0 / 1.5,
              lambda n: alpha_stable(n, 1.5, rng=rng), "median")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ic, (cname, target, gen, agg) in enumerate(cases):
        ax = axes[ic]
        # errs[(nm, m, n)] accumulated over realizations; one signal per
        # (n, r) shared across all (phi, m) pipelines.
        errs = {(nm, m, n): [] for nm in phis for m in orders
                for n in n_grid}
        specs = [(nm, FUNCTIONALS[nm], agg, 2.0) for nm in phis]
        for n in n_grid:
            for r in range(n_real):
                x = gen(n)
                for m in orders:
                    res = fdfa_multi(x, specs, order=m)
                    for nm in phis:
                        errs[(nm, m, n)].append(res[nm].slope - target)
        for i, nm in enumerate(phis):
            for m in orders:
                rmse = []
                for n in n_grid:
                    e = np.array(errs[(nm, m, n)])
                    rmse.append(float(np.sqrt(np.mean(e ** 2))))
                    rows.append(["finitesize", nm, f"{cname}|m={m}|n={n}",
                                 rmse[-1], float(np.std(e)), n, n_real])
                ax.loglog(n_grid, rmse, marker="o", ms=3, lw=1,
                          ls=["-", "--", ":"][(m - 1) % 3],
                          color=COLORS[i % len(COLORS)],
                          label=(disp(nm) + rf", $m={m}$"))
        # n^{-1/2} guide
        ref = np.array(n_grid, dtype=float)
        ax.loglog(ref, rmse[0] * (ref / ref[0]) ** -0.5, "k:", lw=0.8,
                  label=r"$\propto n^{-1/2}$")
        ax.set_xlabel(r"$n$")
        if ic == 0:
            ax.set_ylabel(r"RMSE$(\hat{H})$")
        ax.set_title(cname.replace("a=1.5", r"$\alpha=1.5$")
                          .replace("H=0.7", r"$H=0.7$"))
        ax.legend(fontsize=9, ncol=2)
        add_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig10_finitesize.png"), dpi=150)
    plt.close(fig)

    log("RMSE curves written to fig10_finitesize.png "
        "(see results.csv, experiment='finitesize')")
    return rows


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="small sizes for a fast sanity check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order", type=int, default=1,
                    help="polynomial detrending order m")
    ap.add_argument("--only", type=int, default=0, choices=range(10),
                    help="run a single experiment (1-9); 0 = all")
    ap.add_argument("--nreal", type=int, default=0,
                    help="override number of realizations (0 = defaults)")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.quick:
        n, n_real, J, n_real_c = 2 ** 12, 4, 12, 3
    else:
        n, n_real, J, n_real_c = 2 ** 14, 20, 15, 10
    if args.nreal > 0:
        n_real = n_real_c = args.nreal

    lines = []

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.time()
    log(f"FDFA validation | n={n}, realizations={n_real}, order={args.order}, "
        f"seed={args.seed}{' (QUICK)' if args.quick else ''} | style: {STYLE}")

    experiments = {
        1: lambda: experiment_fgn(n, n_real, args.order, rng, log),
        2: lambda: experiment_stable(n, n_real, args.order, rng, log),
        3: lambda: experiment_cascade(J, n_real_c, args.order, rng, log),
        4: lambda: experiment_spurious(n, n_real_c, args.order, rng, log),
        5: lambda: experiment_discriminator(n, J, n_real_c, args.order,
                                            rng, log),
        6: lambda: experiment_mrw(n, n_real_c, args.order, rng, log),
        7: lambda: experiment_arfima(n, n_real_c, args.order, rng, log),
        8: lambda: experiment_surrogates(n, J, max(3, n_real_c // 2),
                                         args.order, rng, log),
        9: lambda: experiment_finitesize(n_real_c, 3, rng, log),
    }
    run_ids = [args.only] if args.only else list(experiments)

    all_rows = []
    for i in run_ids:
        all_rows += experiments[i]()

    suffix = f"_exp{args.only}" if args.only else ""
    _write_csv(os.path.join(RESULTS, f"results{suffix}.csv"),
               ["experiment", "phi", "param", "mean", "std",
                "n", "n_realizations"], all_rows)

    log(f"\nDone in {time.time() - t0:.1f} s. "
        f"Figures and results{suffix}.csv written to {RESULTS}/")
    with open(os.path.join(RESULTS, f"summary{suffix}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
