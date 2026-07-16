#!/usr/bin/env python3
"""
run_application.py — FDFA on real heliophysical series (OMNI2)
===============================================================

Series (from real_data/omni2_*.lst.txt, hourly 1976-2026):

  * sunspot R   : daily (hourly file repeats the daily value), 1976-2026
  * Dst index   : hourly, 1976-2026 (complete)
  * scalar B    : hourly, 1996-2026 (fill fraction < 1%)
  * dB          : first differences of B

Reference results to compare against:
  - Movahed et al. 2006 (JSTAT): sunspot MFDFA; three crossovers from
    the 11-yr cycle; after Fourier detrending h(2) = 0.12 +- 0.01;
    multifractality "almost entirely due to long-range correlations".
  - Macek & Wawrzaszek 2011 (PSS): multifractal structure of IMF
    fluctuations at small and large scales.

For each series this script reports:
  1. classical MFDFA h(q) and width Dh = h(-4) - h(4),
  2. Dh after shuffling (destroys correlations, keeps distribution),
  3. the FDFA discriminator Dtheta (quantile-exponent profile),
  4. classical H (RMS, A_2) vs robust H (Huber inner, median outer),
     both with m = 2 (prescription from the P9 study),
  5. a genuine / spurious / mixed verdict, cross-checked two ways.

Scale ranges are restricted below the known periodicities
(11-yr cycle for R; annual/solar-cycle scales for Dst and B).
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fdfa import FUNCTIONALS, fdfa, mfdfa_hq, default_scales
from run_validation import disp, STYLE, COLORS, RESULTS, add_grid

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "real_data", "omni2_a7kZN7QddQ.lst.txt")

QS = np.linspace(-4, 4, 17)
TAUS = np.linspace(0.1, 0.9, 9)
N_SHUFFLE = 5
ORDER = 2          # FDFA-2, per the finite-size study (P9)


def load_series():
    d = np.loadtxt(DATA)
    yr, hour = d[:, 0], d[:, 2]
    B, R, Dst = d[:, 3], d[:, 5], d[:, 6]

    # sunspot: one value per day (hour 0), full period
    day = hour == 0
    r = R[day]
    r = r[r < 999]

    dst = Dst[np.abs(Dst) < 9999]

    mB = (yr >= 1996) & (B < 999)
    b = B[mB]
    missB = 1.0 - b.size / np.sum(yr >= 1996)
    db = np.diff(b)

    return [
        # (name, series, fit_smin, fit_smax, note)
        # R: s_min=64 d — 13% exact zeros at solar minima blow up the
        #    negative-q moments below the typical zero-run scale;
        #    s_max=1024 d stays below the 11-yr-cycle crossover.
        (r"sunspot $R$ (daily)", r, 64, 1024, f"n={r.size}"),
        (r"$D_{st}$ (hourly)", dst, 16, 4096, f"n={dst.size}"),
        (r"$B$ (hourly, 1996+)", b, 16, 4096,
         f"n={b.size}, miss={missB*100:.1f}%"),
        (r"$\delta B$ (hourly)", db, 16, 4096, f"n={db.size}"),
    ]


def theta_profile(x, s_min, s_max, order=ORDER):
    sc = default_scales(x.size, s_min=s_min,
                        s_max=min(s_max, max(64, x.size // 32)))
    res = fdfa(x, FUNCTIONALS["rms"], order=order, scales=sc,
               keep_windows=True)
    ls = np.log(np.array(sorted(res.Fv.keys()), dtype=float))
    th = np.array([np.polyfit(
        ls, [np.log(np.quantile(res.Fv[s], tau))
             for s in sorted(res.Fv.keys())], 1)[0] for tau in TAUS])
    return th


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(42)
    t0 = time.time()
    lines, rows = [], []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log(f"FDFA application to OMNI2 | order m={ORDER} | style: {STYLE}")
    series = load_series()

    fig_hq, axes_hq = plt.subplots(2, 2, figsize=(9, 7))
    fig_th, axes_th = plt.subplots(1, 2, figsize=(10, 4.2))

    log(f"\n{'series':>22} | {'h(2)':>5} | {'H_rob':>5} | {'Dh':>6} | "
        f"{'Dh_shuf':>7} | {'Dtheta':>7} | {'shuffle':>8} | {'theta':>8}")
    log("-" * 96)

    names, dh_b, dhs_b, dth_b = [], [], [], []
    for i, (name, x, s_min, s_max, note) in enumerate(series):
        x = np.asarray(x, dtype=float)
        scales = default_scales(x.size, s_min=s_min, s_max=s_max)

        # 1) classical h(q)
        hq = mfdfa_hq(x, QS, FUNCTIONALS["rms"], order=ORDER,
                      scales=scales).hq
        dh = hq[0] - hq[-1]
        h2 = hq[np.argmin(np.abs(QS - 2.0))]

        # 2) shuffled
        dhs = []
        hqs_mean = np.zeros_like(hq)
        for _ in range(N_SHUFFLE):
            xs = rng.permutation(x)
            hqs = mfdfa_hq(xs, QS, FUNCTIONALS["rms"], order=ORDER,
                           scales=scales).hq
            hqs_mean += hqs / N_SHUFFLE
            dhs.append(hqs[0] - hqs[-1])
        dhs = float(np.mean(dhs))

        # 3) discriminator
        th = theta_profile(x, s_min, s_max)
        dth = float(th[-1] - th[0])

        # 4) classical vs robust H
        H_cl = fdfa(x, FUNCTIONALS["rms"], order=ORDER, scales=scales,
                    aggregate="qmean", q=2.0).slope
        H_rob = fdfa(x, FUNCTIONALS["huber"], order=ORDER, scales=scales,
                     aggregate="median").slope

        v_shuf = ("genuine" if (dh > 0.15 and dh - dhs > 0.5 * dh)
                  else "spurious" if dh > 0.15 else "mono")
        v_th = ("genuine" if (dh > 0.15 and abs(dth) > 0.08)
                else "spurious" if dh > 0.15 else "mono")

        plain = (name.replace("$", "").replace("\\", "")
                     .replace("{", "").replace("}", ""))
        log(f"{plain:>22} | {h2:5.3f} | {H_rob:5.3f} | {dh:6.3f} | "
            f"{dhs:7.3f} | {dth:7.3f} | {v_shuf:>8} | {v_th:>8}   ({note})")

        names.append(name)
        dh_b.append(dh); dhs_b.append(dhs); dth_b.append(abs(dth))
        rows += [["application", plain, "h2", h2, 0.0, x.size, 1],
                 ["application", plain, "H_classic", H_cl, 0.0, x.size, 1],
                 ["application", plain, "H_robust", H_rob, 0.0, x.size, 1],
                 ["application", plain, "dh", dh, 0.0, x.size, 1],
                 ["application", plain, "dh_shuffled", dhs, 0.0, x.size, 1],
                 ["application", plain, "dtheta", dth, 0.0, x.size, 1]]

        # panels
        ax = axes_hq.ravel()[i]
        ax.plot(QS, hq, "o-", ms=3, lw=1, color=COLORS[0],
                label=r"original")
        ax.plot(QS, hqs_mean, "s--", ms=3, lw=1, color=COLORS[1],
                label=r"shuffled")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$h(q)$")
        ax.legend(fontsize=10)
        add_grid(ax)

        axes_th[0].plot(TAUS, th, "o-", ms=3, lw=1,
                        color=COLORS[i % len(COLORS)], label=name)

    axes_th[0].set_xlabel(r"$\tau$")
    axes_th[0].set_ylabel(r"$\theta(\tau)$")
    axes_th[0].set_title(r"quantile-exponent profiles")
    axes_th[0].legend(fontsize=10)
    add_grid(axes_th[0])

    xb = np.arange(len(names))
    axes_th[1].bar(xb - 0.25, dh_b, 0.25, label=r"$\Delta h$")
    axes_th[1].bar(xb, dhs_b, 0.25, label=r"$\Delta h$ (shuffled)")
    axes_th[1].bar(xb + 0.25, dth_b, 0.25, label=r"$|\Delta\theta|$")
    axes_th[1].set_xticks(xb)
    axes_th[1].set_xticklabels([r"$R$", r"$D_{st}$", r"$B$",
                                r"$\delta B$"], fontsize=11)
    axes_th[1].set_title(r"discriminator summary")
    axes_th[1].legend(fontsize=11)
    add_grid(axes_th[1], "y")

    fig_hq.tight_layout()
    fig_hq.savefig(os.path.join(RESULTS, "fig11_application_hq.png"),
                   dpi=150)
    fig_th.tight_layout()
    fig_th.savefig(os.path.join(RESULTS, "fig12_application_disc.png"),
                   dpi=150)
    plt.close("all")

    with open(os.path.join(RESULTS, "results_app.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "series", "quantity", "value", "std",
                    "n", "n_realizations"])
        w.writerows(rows)
    with open(os.path.join(RESULTS, "summary_app.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"\nDone in {time.time() - t0:.1f} s. Figures fig11, fig12 and "
        f"results_app.csv in {RESULTS}/")


if __name__ == "__main__":
    main()
