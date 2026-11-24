# -*- coding: utf-8 -*-
"""validate_53b: 距离双轨结论 · 散点版(数据/协议同 validate_53)
   逐日散点(131雨日) + 公式曲线 + RF曲线, 训练边界15mm分区
   输出: 距离外推公式胜_散点版.png
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#17375E'; DGREEN = '#2E7D32'; ORANGE = '#ED7D31'; MUTED = '#666666'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates'])
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
wd = np.array([((np.datetime64(d, 'D').view('int64') + 3) % 7 < 5).astype(int) for d in dates])

dry_i = np.where(kl == 'dry')[0]
BAS = {}
for g in [1, 0]:
    dvg = dist[np.isin(dist_day, [i for i in dry_i if wd[i] == g])]
    BAS[g] = float(np.median(dvg[(dvg > 0.3) & (dvg < 60)]))

Rs, Ys = [], []
for i in np.where(kl != 'dry')[0]:
    dv = dist[dist_day == i]; dv = dv[(dv > 0.3) & (dv < 60)]
    if len(dv) < 500 or dp[i] < 0.5: continue
    Rs.append(float(dp[i])); Ys.append((np.median(dv) / BAS[wd[i]] - 1) * 100)
Rs = np.array(Rs); Ys = np.array(Ys)

RC = 5.4
m_tr = Rs < 15
th = float(np.sum(Ys[m_tr] * np.log1p(Rs[m_tr] / RC)) / np.sum(np.log1p(Rs[m_tr] / RC) ** 2))
formula = lambda R: th * np.log1p(np.asarray(R, float) / RC)
rf = RandomForestRegressor(400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(np.log1p(Rs[m_tr]).reshape(-1, 1), Ys[m_tr])

fig, ax = plt.subplots(figsize=(13, 7.6))
ax.axvspan(0.4, 15, color='#EAF1F9', zorder=0)
ax.axvspan(15, 190, color='#FDF1E8', zorder=0)
ax.axvline(15, color='#C00000', ls='--', lw=2, zorder=2)

ax.scatter(Rs[m_tr], Ys[m_tr], s=30, color='#7FA6CC', alpha=0.75, edgecolor='white', linewidth=0.4, zorder=3, label=None)
ax.scatter(Rs[~m_tr], Ys[~m_tr], s=34, color='#E8A47C', alpha=0.85, edgecolor='white', linewidth=0.4, zorder=3)

RR = np.linspace(0.5, 175, 500)
ax.plot(RR, formula(RR), lw=3, color=DGREEN, zorder=5)
ax.plot(RR, rf.predict(np.log1p(RR).reshape(-1, 1)), lw=2.6, color=ORANGE, ls='--', zorder=4)

# 箱均值点(实心方块叠在散点层上)
for lo, hi, cal in [(0.5, 5, 1), (5, 15, 1), (15, 30, 0), (30, 60, 0), (60, 200, 0)]:
    m = (Rs >= lo) & (Rs < hi)
    ax.plot([Rs[m].mean()], [Ys[m].mean()], 's', ms=12, zorder=6,
            markerfacecolor=NAVY, markeredgecolor='white', markeredgewidth=1.4)

ax.text(4.0, 2.3, '训练时见过\n(小中雨 <15mm/日)', ha='center', fontsize=20, fontweight='bold', color='#1F4E79')
ax.text(70, 3.6, '没见过(大雨盲测)', ha='center', fontsize=20, fontweight='bold', color='#843C0C')
ax.text(16.5, -11.6, '训练边界 15mm', fontsize=16, color='#C00000', fontweight='bold')

ax.text(6.5, -2.0, '公式', fontsize=22, fontweight='bold', color=DGREEN, rotation=-14)
ax.text(45, -4.3, 'RF：躺平', fontsize=21, fontweight='bold', color=ORANGE)
ax.text(1.1, -7.3, '方块 = 分箱均值\n(散点噪声大,\n看方块走势)', fontsize=15, color=MUTED)

ax.set_xscale('log')
ax.set_xlim(0.4, 190)
ax.set_ylim(-12.5, 5.0)
ax.set_xticks([1, 5, 15, 50, 150])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('日降水 (mm)', fontsize=19)
ax.set_ylabel('中位距离效应 (%)', fontsize=19)
ax.tick_params(labelsize=17)
ax.grid(alpha=0.25, which='both')
fig.text(0.5, 0.012,
         f'每个点 = 一个雨日(当日行程中位距离 ÷ 同日型晴天基线 − 1), n={len(Rs)}天; 蓝点=训练域, 橙点=盲测域\n公式 {th:+.2f}·ln(1+R/{RC})(RC先验), RF只用<15mm训练 | 盲测域: 公式贴方块(差0.2~0.9pp), RF躺平(差1.4~5.3pp)',
         ha='center', va='bottom', fontsize=13.5, color='#375623', fontweight='bold')
plt.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig(r'D:/AI/条件转化/距离外推公式胜_散点版.png', dpi=200, facecolor='white')
print('saved 距离外推公式胜_散点版.png')
