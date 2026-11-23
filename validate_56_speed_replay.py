# -*- coding: utf-8 -*-
"""validate_56: '模型重放训练分布'换数据复验 —— 速度版(627万行程, 逐时520点)
   与出发量/距离不同: 这是连续特征、log空间、样本量最大的量。
   RF(<4mm/h 训练) vs 公式 δv=-0.0221·ln(1+r/0.2), 盲测域 r≥4。
   输出: 速度_重放失真_其他数据.png
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY_D = '#17375E'; NAVY = '#1F4E79'; DGREEN = '#2E7D32'; ORANGE = '#ED7D31'; MUTED = '#666666'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); n = len(dates)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)
OK = (spdV > 1) & (spdV < 60)
mu0 = np.full((7, 24), np.nan)
for k in range(7):
    sel = np.where((kl == 'dry') & (dow == k))[0]
    for h in range(24):
        m = np.isin(spdD, sel) & (spdH == h) & OK
        if m.sum() > 50: mu0[k, h] = np.median(np.log(spdV[m]))

keys = spdD.astype(np.int64) * 100 + spdH
order = np.argsort(keys, kind='stable'); v_s = spdV[order]; k_s = keys[order]
uniq, st = np.unique(k_s, return_index=True); en = np.append(st[1:], len(k_s))
bk = dict(zip(uniq.tolist(), zip(st.tolist(), en.tolist())))

pts_r, pts_dv = [], []
for i in np.where(kl != 'dry')[0]:
    for h in range(24):
        r = inten[i, h]
        if r < 0.3: continue
        b0 = mu0[dow[i], h]
        if np.isnan(b0): continue
        kk = i * 100 + h
        if kk not in bk: continue
        s0, e0 = bk[kk]
        vs = v_s[s0:e0]; vs = vs[(vs > 1) & (vs < 60)]
        if len(vs) < 30: continue
        pts_r.append(r); pts_dv.append((np.median(np.log(vs)) - b0) * 100)
pts_r = np.array(pts_r); pts_dv = np.array(pts_dv)
print(f'逐时观测点: {len(pts_r)} 个')

m_tr = pts_r < 4
rf = RandomForestRegressor(400, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(np.log1p(pts_r[m_tr]).reshape(-1, 1), pts_dv[m_tr])
fm = lambda r: -0.0221 * np.log1p(np.asarray(r, float) / 0.2) * 100

print(f'\n{"箱":8s} {"n":>4s} {"实测%":>8s} {"RF%":>8s} {"公式%":>8s} {"RF差":>6s} {"公式差":>6s}')
for lo, hi, nm, dom in [(0.3, 1, '0.3-1', '内插'), (1, 4, '1-4', '内插'),
                        (4, 12, '4-12', '盲测'), (12, 1e9, '≥12', '盲测')]:
    m = (pts_r >= lo) & (pts_r < hi)
    o = pts_dv[m].mean(); rp = rf.predict(np.log1p(pts_r[m]).reshape(-1, 1)).mean(); fp = fm(pts_r[m]).mean()
    print(f'{nm+dom:8s} {int(m.sum()):4d} {o:+8.2f} {rp:+8.2f} {fp:+8.2f} {abs(rp-o):6.2f} {abs(fp-o):6.2f}')

# ============ 图(同距离散点版风格) ============
fig, ax = plt.subplots(figsize=(13, 7.6))
ax.axvspan(0.25, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 32, color='#FDF1E8', zorder=0)
ax.axvline(4, color='#C00000', ls='--', lw=2, zorder=2)

ax.scatter(pts_r[m_tr], pts_dv[m_tr], s=30, color='#7FA6CC', alpha=0.7, edgecolor='white', linewidth=0.4, zorder=3)
ax.scatter(pts_r[~m_tr], pts_dv[~m_tr], s=34, color='#E8A47C', alpha=0.85, edgecolor='white', linewidth=0.4, zorder=3)

RR = np.linspace(0.3, 30, 500)
ax.plot(RR, fm(RR), lw=3, color=DGREEN, zorder=5)
ax.plot(RR, rf.predict(np.log1p(RR).reshape(-1, 1)), lw=2.6, color=ORANGE, ls='--', zorder=4)

for lo, hi in [(0.3, 1), (1, 4), (4, 12), (12, 1e9)]:
    m = (pts_r >= lo) & (pts_r < hi)
    ax.plot([pts_r[m].mean()], [pts_dv[m].mean()], 's', ms=13, zorder=6,
            markerfacecolor=NAVY, markeredgecolor='white', markeredgewidth=1.4)

ax.text(1.5, 1.6, '训练时见过\n(雨强<4mm/h)', ha='center', fontsize=15, fontweight='bold', color=NAVY)
ax.text(14, 1.6, '没见过(大雨盲测)', ha='center', fontsize=15, fontweight='bold', color='#843C0C')
ax.text(4.5, -12.6, '训练边界 4mm/h', fontsize=12, color='#C00000', fontweight='bold')
ax.text(0.55, -6.2, '公式', fontsize=17, fontweight='bold', color=DGREEN, rotation=-18)
ax.text(13.5, -3.6, 'RF：躺平', fontsize=15, fontweight='bold', color=ORANGE)
ax.text(0.42, -11.3, '方块 = 分箱均值', fontsize=11.5, color=MUTED)

ax.set_xscale('log')
ax.set_xlim(0.25, 32)
ax.set_ylim(-13.5, 3)
ax.set_xticks([0.3, 1, 4, 12, 30])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('小时雨强 (mm/h)', fontsize=15)
ax.set_ylabel('中位速度效应 (%)', fontsize=15)
ax.tick_params(labelsize=13)
ax.grid(alpha=0.25, which='both')
fig.text(0.5, 0.005,
         f'换数据复验: 速度(NYC 627万行程, 逐时{n}个观测点, 每点=当小时全部行程中位log速度−同钟点干日基线)。RF只用<4mm/h训练。\n'
         '盲测域(≥4): 实测减速持续加深到-9.1%, RF躺平在-4.5%(差4.6pp), 公式沿饱和曲线跟涨到-10.5%(差1.4pp) —— 与出发量、距离同一规律',
         ha='center', fontsize=11.5, color='#375623', fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(r'D:/AI/条件转化/速度_重放失真_其他数据.png', dpi=200, bbox_inches='tight', facecolor='white')
print('\nsaved 速度_重放失真_其他数据.png')
