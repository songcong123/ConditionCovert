# -*- coding: utf-8 -*-
"""validate_52b: 距离中位效应拟合 · 简版三柱图(结论一眼可见)
   数据/口径同 validate_52: 实测箱效应 vs 公式(三点标定) vs RF(<30mm训练)
   输出: 距离中位拟合_简版.png
"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; DGREEN = '#375623'; ORANGE = '#ED7D31'; MUTED = '#666666'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates']); n = len(dates)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
wd = np.array([((np.datetime64(d, 'D').view('int64') + 3) % 7 < 5).astype(int) for d in dates])

dry_i = np.where(kl == 'dry')[0]
dd_wd = dist[np.isin(dist_day, [i for i in dry_i if wd[i] == 1])]; dd_wd = dd_wd[(dd_wd > 0.3) & (dd_wd < 60)]
dd_we = dist[np.isin(dist_day, [i for i in dry_i if wd[i] == 0])]; dd_we = dd_we[(dd_we > 0.3) & (dd_we < 60)]
base = {1: float(np.median(dd_wd)), 0: float(np.median(dd_we))}

Rs, Ys = [], []
for i in np.where(kl != 'dry')[0]:
    dv = dist[dist_day == i]; dv = dv[(dv > 0.3) & (dv < 60)]
    if len(dv) < 500 or dp[i] < 0.5: continue
    Rs.append(float(dp[i])); Ys.append(float(np.median(dv) / base[wd[i]] - 1))
Rs = np.array(Rs); Ys = np.array(Ys)

BINS = [(0.5, 5, '0.5-5\n(标定)'), (5, 15, '5-15\n(标定)'), (15, 30, '15-30'), (30, 60, '30-60\n(盲测)'), (60, 1e9, '≥60\n(盲测)')]
obs, fm, rf_v, ns, rbar = [], [], [], [], []
m_tr = Rs < 30
rf = RandomForestRegressor(400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(np.log1p(Rs[m_tr]).reshape(-1, 1), Ys[m_tr])
bx3 = [(np.mean(Rs[(Rs >= lo) & (Rs < hi)]), np.mean(Ys[(Rs >= lo) & (Rs < hi)])) for lo, hi, _ in BINS[:3]]
popt, _ = curve_fit(lambda R, th, RC: th * np.log1p(R / RC),
                    [b[0] for b in bx3], [b[1] for b in bx3], p0=[-0.02, 3.0],
                    bounds=([-0.2, 0.05], [-1e-4, 100]), maxfev=20000)
th3, RC3 = popt
print(f'三点标定: {th3:+.5f}·ln(1+R/{RC3:.2f})')

for lo, hi, nm in BINS:
    m = (Rs >= lo) & (Rs < hi)
    obs.append(float(Ys[m].mean()))
    fm.append(float(th3 * np.log1p(np.mean(Rs[m]) / RC3)))
    rf_v.append(float(rf.predict(np.log1p(np.array([np.mean(Rs[m])])).reshape(-1, 1))[0]))
    ns.append(int(m.sum())); rbar.append(float(Rs[m].mean()))

fig, ax = plt.subplots(figsize=(11.5, 6.8))
x = np.arange(len(BINS)); bw = 0.26
ax.axvspan(-0.5, 2.5, color='#EAF1F9', zorder=0)
ax.axvspan(2.5, 4.5, color='#FDF1E8', zorder=0)
ax.text(1.0, 0.4, '公式标定域', ha='center', fontsize=11, color=NAVY, fontweight='bold')
ax.text(3.5, 0.4, '盲测域(没见过)', ha='center', fontsize=11, color='#843C0C', fontweight='bold')

b1 = ax.bar(x - bw, np.array(obs) * 100, bw, color=NAVY, label='实测', zorder=3)
b2 = ax.bar(x, np.array(fm) * 100, bw, color='#70AD47', label='公式(三点标定)', zorder=3)
b3 = ax.bar(x + bw, np.array(rf_v) * 100, bw, color=ORANGE, label='RF(<30mm 训练)', zorder=3)
for bars, vals in [(b1, obs), (b2, fm), (b3, rf_v)]:
    for r_, v in zip(bars, vals):
        ax.annotate(f'{v*100:+.1f}', (r_.get_x() + r_.get_width() / 2, v * 100), xytext=(0, 4),
                    textcoords='offset points', ha='center', fontsize=9.5)
for i_ in range(5):
    ax.annotate(f'n={ns[i_]}', (x[i_], -10.3), ha='center', fontsize=8.5, color=MUTED)

ax.annotate('公式低估 2.6pp\n(形状偏浅)', xy=(x[3], fm[3] * 100), xytext=(x[3] + 0.28, fm[3] * 100 + 0.7),
            fontsize=10.5, color=DGREEN, fontweight='bold')
ax.annotate('RF 躺平值恰好命中\n(实测已饱和: -7.8→-8.4 几乎不变)', xy=(x[3] + bw, rf_v[3] * 100),
            xytext=(2.72, -5.9), fontsize=10.5, color='#843C0C', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#843C0C', lw=1.6))

ax.set_xticks(x); ax.set_xticklabels([b[2] for b in BINS], fontsize=10.5)
ax.set_ylabel('中位距离效应 (%)', fontsize=12)
ax.set_xlabel('日降水 (mm)', fontsize=12)
ax.set_ylim(-10.5, 0.8)
ax.legend(fontsize=11, loc='lower left')
ax.grid(alpha=0.3, axis='y')
fig.text(0.5, 0.005,
         '口径: 效应=当日中位距离÷同日型晴天基线−1(131个雨日, 分工作日/周末)。结论: 雨越大典型一趟越短(晴天2.95mi→大雨-8%≈2.71mi), 30mm后效应饱和——饱和的量, RF躺平恰好够; 还在涨的量(如出发量), RF必崩',
         ha='center', fontsize=9.5, color=DGREEN, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(r'D:/AI/条件转化/距离中位拟合_简版.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved 距离中位拟合_简版.png')
