"""Figure 1 of the paper from s10_norm_dependence.py output.
usage: python make_figure1.py ../data/r1/norm_dependence.csv out_dir/"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3, 'savefig.bbox': 'tight'})
df = pd.read_csv(sys.argv[1]); OUT = sys.argv[2]; df = df[df.eps > 0]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.scatter(df.l2_norm, df.cpop, s=16, alpha=0.35, color='#888', linewidths=0)
g = df.groupby('eps')[['l2_norm', 'cpop']].mean()
ax.plot(g.l2_norm, g.cpop, 'o-', color='#1A1A2E', lw=2, ms=8, label='per-budget mean')
rb = df[df.l2_norm > 1.0]; sl, b = np.polyfit(np.log10(rb.l2_norm), np.log10(rb.cpop), 1)
xf = np.logspace(0, np.log10(rb.l2_norm.max()), 50)
ax.plot(xf, 10 ** (sl * np.log10(xf) + b), '-', color='#E15759', lw=2.5, label=f'log-log fit for norm > 1: slope $={sl:.2f}$')
pl = df[df.l2_norm < 0.5].cpop.mean()
ax.axhline(pl, color='#59A14F', ls='--', lw=2, alpha=.7, label=f'mean for norm < 0.5 $\\approx {pl:.3f}$')
ax.axvline(1.0, color='gray', ls=':', alpha=.6)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\|\eta\|_2$  (L2 perturbation norm, [0,1] scale)'); ax.set_ylabel('CPOP')
ax.legend(loc='lower left', fontsize=9.5); fig.savefig(OUT + 'fig_two_regime.pdf'); plt.close(); print(f'slope {sl:.3f}, plateau {pl:.4f}')
