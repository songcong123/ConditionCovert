# -*- coding: utf-8 -*-
"""元旦距离分布的正确看法: CCDF(log-log) + 分位数对比 + KS 检验 (替代不易读的小提琴图)"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats

W = np.load(r'D:/AI/条件转化/.flow_agg_winter.npz', allow_pickle=True)
dist = W['dist'].astype(np.float64); dist_day = W['dist_day'].astype(int)
dates = list(W['dates']); did = {d: i for i, d in enumerate(dates)}
dec_dry = [did[d] for d in dates if d[:7] == '2024-12' and W['klass'][did[d]] == 'dry']

def day_d(idx, lo=0.1, hi=60):
    s = dist[np.isin(dist_day, idx)]
    return s[(s > lo) & (s < hi)]

d_dec = day_d(dec_dry)                    # 12月干日(基线)
d31 = day_d([did['2024-12-31']])          # 跨年夜
d01 = day_d([did['2025-01-01']])          # 元旦
print(f'n: 12月干日 {len(d_dec):,} | 12/31 {len(d31):,} | 1/1 {len(d01):,}')

qs = [10, 25, 50, 75, 90]
rows = {}
for nm, d in [('12月干日', d_dec), ('12/31 跨年夜', d31), ('1/1 元旦', d01)]:
    rows[nm] = dict(mean=float(np.mean(d)), med=float(np.median(d)),
                    **{f'P{q}': float(np.percentile(d, q)) for q in qs})
    print(f"{nm}: mean={rows[nm]['mean']:.2f}  P10={rows[nm]['P10']:.2f}  P25={rows[nm]['P25']:.2f}  "
          f"P50={rows[nm]['P50']:.2f}  P75={rows[nm]['P75']:.2f}  P90={rows[nm]['P90']:.2f}")

print('\nKS 检验(D 统计量, n 巨大时 p 值必显著, 看 D 的量级):')
for a, b, nm in [(d31, d_dec, '12/31 vs 基线'), (d01, d_dec, '1/1 vs 基线'), (d01, d31, '1/1 vs 12/31')]:
    D = stats.ks_2samp(a, b).statistic
    print(f'  {nm}: D={D:.4f}')

# 长尾(≥8mi, 机场+跨区)份额
for nm, d in [('12月干日', d_dec), ('12/31', d31), ('1/1', d01)]:
    print(f'  ≥8mi 份额 {nm}: {(d>=8).mean()*100:.2f}% | ≥12mi: {(d>=12).mean()*100:.2f}%')

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5))
fig.suptitle('元旦距离分布的正确看法：CCDF（中位右移）＋ 分位数对比（尾部变肥）', fontsize=13.5, fontweight='bold', y=1.0)

ax = axes[0]
grid = np.geomspace(0.15, 40, 300)
for nm, d, cc, ls in [('12月干日', d_dec, '#70AD47', '--'), ('12/31 跨年夜', d31, '#BFBFBF', '-'), ('1/1 元旦', d01, '#ED7D31', '-')]:
    ccdf = 1 - np.searchsorted(np.sort(d), grid, 'right') / len(d)
    ax.plot(grid, ccdf, lw=2.2 if nm != '12月干日' else 1.8, color=cc, ls=ls, label=f'{nm} (中位 {np.median(d):.2f}mi)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.axvline(2.70, color='#BFBFBF', ls=':', lw=1.2); ax.axvline(3.20, color='#ED7D31', ls=':', lw=1.2)
ax.annotate('中位 2.70', (2.70, 0.5), xytext=(4.2, 0.55), fontsize=9.5, color='#7F7F7F',
            arrowprops=dict(arrowstyle='->', color='#7F7F7F'))
ax.annotate('中位 3.20\n(右移 +18%)', (3.20, 0.5), xytext=(1.2, 0.28), fontsize=9.5, color='#ED7D31',
            arrowprops=dict(arrowstyle='->', color='#ED7D31'))
ax.set_xlabel('单程里程 (mi, log)'); ax.set_ylabel('P(距离 > x)')
ax.set_title('① CCDF：曲线整体右移 = 元旦行程变长', fontsize=11.5, loc='left')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9.5, loc='lower left')

ax = axes[1]
x = np.arange(5); wd = 0.26
series = [('12月干日', '#70AD47'), ('12/31 跨年夜', '#BFBFBF'), ('1/1 元旦', '#ED7D31')]
for k, (nm, cc) in enumerate(series):
    vals = [rows[nm][f'P{q}'] for q in qs]
    ax.bar(x + (k - 1) * wd, vals, wd, color=cc, label=nm, alpha=0.9)
    for x_, v_ in zip(x + (k - 1) * wd, vals):
        ax.annotate(f'{v_:.1f}', (x_, v_), ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([f'P{q}' for q in qs])
ax.set_ylabel('里程 (mi)'); ax.set_title('② 分位数对比：低端几乎不变，中位及以上右移', fontsize=11.5, loc='left')
ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=9.5, loc='upper left')

plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/元旦跨年分析/元旦_距离CCDF与分位数.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 元旦_距离CCDF与分位数.png')
