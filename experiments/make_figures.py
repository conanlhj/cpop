"""Figures 2 and 3 of the paper from the analysis output (a1_analysis.py -> a1_tables.json).
fig_tpr_fpr.pdf  : R2 TPR vs target FPR (PGD eps=4, mean over 8 classifiers)
fig_optransfer.pdf: R1 realized FPR at a 15% target over the 56 ordered classifier pairs
usage: python make_figures.py ../data/r1/analysis/a1_tables.json out_dir/"""
import sys, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
J = json.load(open(sys.argv[1])); OUT = sys.argv[2]
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3, 'savefig.bbox': 'tight'})
GRID = [1, 2, 5, 10, 15, 20]
SERIES = [('cpop', 'CPOP', '#1f77b4', '-', 'o'), ('score_norm', 'Score norm', '#17becf', '-', 's'), ('raw_var', 'Raw variation', '#9467bd', '-', 'v'),
          ('maha1', 'Mahalanobis (single)', '#d62728', '--', 'D'), ('maha_cc', 'Mahalanobis (class)', '#ff7f0e', '--', 'd'),
          ('kde_cc', 'KDE (class)', '#8c564b', '--', 'P'), ('fs_bit2', 'FS 2-bit', '#2ca02c', '--', '^'), ('lid', 'LID', '#7f7f7f', '--', 'x')]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for k, lab, c, ls, mk in SERIES:
    y = J['t12'][k]; ax.plot(GRID, y, ls=ls, marker=mk, color=c, lw=2.2 if k == 'cpop' else 1.4, ms=5, label=lab)
for x, y in zip(GRID[2:], J['t12']['cpop'][2:]): ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(9, -12), ha='left', fontsize=8.5, color='#1f77b4')
ax.axvline(15, ls=':', color='gray', alpha=.6); ax.set_xticks(GRID); ax.set_ylim(0, 1.02)
ax.set_xlabel('target false positive rate (%)'); ax.set_ylabel('true positive rate'); ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
fig.savefig(OUT + 'fig_tpr_fpr.pdf'); fig.savefig(OUT + 'fig_tpr_fpr.png', dpi=80) if len(sys.argv) > 3 else None; plt.close()
DETS = [('cpop', 'CPOP'), ('score_norm', 'Score\nnorm'), ('raw_var', 'Raw\nvar.'), ('fs_common', 'FS\ncommon'), ('fs_bit2', 'FS\n2-bit'), ('lid', 'LID'),
        ('maha1', 'Maha\n(single)'), ('maha_cc', 'Maha\n(class)'), ('kd1', 'KDE\n(global)'), ('kde_cc', 'KDE\n(class)')]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
bp = ax.boxplot([J['r1_fpr15'][k] for k, _ in DETS], tick_labels=[l for _, l in DETS], showmeans=True, patch_artist=True, whis=(0, 100))
for i, p in enumerate(bp['boxes']): p.set_facecolor('#1f77b4' if i < 3 else '#bbbbbb'); p.set_alpha(.6)
ax.axhline(0.15, ls='--', color='green', label='target FPR = 15%'); ax.set_ylabel('realized FPR on $P_{\\mathrm{test}}$ (R1)')
ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=9, loc='upper left'); ax.tick_params(axis='x', labelsize=8.5)
fig.savefig(OUT + 'fig_optransfer.pdf'); fig.savefig(OUT + 'fig_optransfer.png', dpi=80) if len(sys.argv) > 3 else None; plt.close(); print('ok')
