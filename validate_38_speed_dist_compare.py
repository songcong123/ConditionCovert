# -*- coding: utf-8 -*-
"""三量直白对比(速度+距离; 出发量沿用37结论): 内插/外推 × RF/公式 谁更接近实际
   速度: 小时级中位减速(%, 相对同星期钟点干日基线); RF 用 r<4 训练
   距离: 日级中位里程偏移(%, 相对同年同日型干日基线); RF 用 日降水<30 训练
   出发量: validate_37 的结果(硬编码, 已验证)
输出: 三量_内插外推对比.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; GREY = '#404040'; MUTED = '#666666'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); n = len(dates)
hc = A['hour_cnt'].astype(float)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
wd = (dow + 3) % 7
yr = np.array([int(str(d)[:4]) for d in dates])
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)

# ================= 速度: 小时级中位减速 =================
spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)
okv = (spdV > 1) & (spdV < 60)
mu0_h = np.zeros((7, 24))
for k in range(7):
    for h in range(24):
        m = okv & np.isin(spdD, np.where((kl == 'dry') & (dow == k))[0]) & (spdH == h)
        mu0_h[k, h] = np.median(np.log(spdV[m])) if m.sum() > 50 else np.nan

samples = []   # (r, hour中位残差, lag, peak, we, 权重)
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        r = inten[i, h]
        if r < 0.3 or np.isnan(mu0_h[dow[i], h]): continue
        m = okv & (spdD == i) & (spdH == h)
        if m.sum() < 30: continue
        res = float(np.median(np.log(spdV[m])) - mu0_h[dow[i], h])
        lag = inten[i, h - 1] if h >= 1 else 0
        samples.append((r, res, np.log1p(lag),
                        1 if (7 <= h <= 9 or 17 <= h <= 19) else 0,
                        1 if wd[i] >= 5 else 0, h, lam[dow[i], h]))
SMP = np.array(samples)
r_s, y_s = SMP[:, 0], SMP[:, 1]
X_s = SMP[:, [0, 2, 3, 4, 5]]; X_s[:, 0] = np.log1p(r_s)
rf_s = RandomForestRegressor(500, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
m_tr = r_s < 4
rf_s.fit(X_s[m_tr], y_s[m_tr])
pred_s = rf_s.predict(X_s)
th_v, RC_v = -0.0221, 0.20
f_v = lambda r: th_v * np.log1p(np.asarray(r, float) / RC_v)

BINS = [(0.3, 1, '0.3-1'), (1, 4, '1-4'), (4, 8, '4-8'), (8, 16, '8-16'), (16, 140, '≥16')]
v_act, v_rf, v_f = [], [], []
for lo, hi, nm in BINS:
    m = (r_s >= lo) & (r_s < hi)
    w = SMP[m, 6]
    v_act.append((np.exp(np.average(y_s[m], weights=w)) - 1) * 100)
    v_rf.append((np.exp(np.average(pred_s[m], weights=w)) - 1) * 100)
    v_f.append((np.exp(np.average(f_v(r_s[m]), weights=w)) - 1) * 100)
print('速度: 箱 实际 RF 公式')
for (l, h, nm), a, b, c in zip(BINS, v_act, v_rf, v_f):
    print(f'  {nm:6s} {a:6.1f} {b:6.1f} {c:6.1f}')

# ================= 距离: 日级中位里程偏移 =================
lmd = np.full(n, np.nan)
for i in range(n):
    s = dist[dist_day == i]; s = s[(s > 0.1) & (s < 80)]
    if s.size >= 500: lmd[i] = np.median(np.log(s))
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
base_d = {}
for yi in [2024, 2025]:
    for w in [0, 1]:
        sel = (kl == 'dry') & (yr == yi) & (wd == w)
        v = lmd[sel]; v = v[~np.isnan(v)]
        if len(v) >= 5: base_d[(yi, w)] = float(np.median(v))
dR, dY = [], []
for i in range(n):
    if kl[i] == 'dry' or np.isnan(lmd[i]): continue
    b = base_d.get((int(yr[i]), wd[i]))
    if b is None: continue
    dR.append(dp[i]); dY.append(lmd[i] - b)
dR, dY = np.array(dR), np.array(dY)
dW = np.ones_like(dR)
rf_d = RandomForestRegressor(500, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
m_trd = dR < 30
rf_d.fit(np.log1p(dR[m_trd]).reshape(-1, 1), dY[m_trd])
pred_d = rf_d.predict(np.log1p(dR).reshape(-1, 1))
f_d = lambda r: -0.0171 * np.log1p(np.asarray(r, float) / 5.4)     # validate_25 μ1 曲线

BINS_D = [(0.5, 5, '0.5-5'), (5, 15, '5-15'), (15, 30, '15-30'), (30, 60, '30-60'), (60, 200, '≥60')]
d_act, d_rf, d_f = [], [], []
for lo, hi, nm in BINS_D:
    m = (dR >= lo) & (dR < hi)
    d_act.append((np.exp(np.mean(dY[m])) - 1) * 100)
    d_rf.append((np.exp(np.mean(pred_d[m])) - 1) * 100)
    d_f.append((np.exp(f_d(np.mean(dR[m]))) - 1) * 100)
print('距离(日级): 箱 实际 RF 公式')
for (l, h, nm), a, b, c in zip(BINS_D, d_act, d_rf, d_f):
    print(f'  {nm:6s} {a:6.1f} {b:6.1f} {c:6.1f}')

# ================= 出发量(validate_37 结果) =================
v_act0 = [9.0, 12.8, 17.3, 28.2, 24.8]; v_rf0 = [9.3, 12.5, 14.3, 15.3, 13.5]; v_f0 = [8.2, 12.6, 17.6, 21.0, 26.2]

# ================= 图 =================
fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
fig.suptitle('三个量的内插/外推对比：RF 和公式谁更接近实际值（都只用训练域内的数据训练）',
             fontsize=14.5, fontweight='bold', y=1.03)

def panel(ax, act, rf_v, f_v_, bins, split, ylab, title, ylim):
    nb = len(bins)
    yl0 = (min(0, min(act), min(f_v_)) * 1.5, max(max(act), max(f_v_)) * 1.35)
    ax.add_patch(plt.Rectangle((-0.5, yl0[0]), split + 0.5, yl0[1] - yl0[0], color='#E2F0D9', alpha=0.5, zorder=0))
    ax.add_patch(plt.Rectangle((split - 0.5, yl0[0]), nb - split, yl0[1] - yl0[0], color='#FDEADA', alpha=0.55, zorder=0))
    x = np.arange(nb); bw = 0.26
    ax.bar(x - bw, act, bw, color=GREY, label='实际值', zorder=3)
    ax.bar(x, rf_v, bw, color=RED, label='RF', zorder=3)
    ax.bar(x + bw, f_v_, bw, color=NAVY, label='公式', zorder=3)
    for k in range(nb):
        dr, df_ = abs(rf_v[k] - act[k]), abs(f_v_[k] - act[k])
        win = 'RF✓' if dr < df_ else '公式✓'
        wc = RED if dr < df_ else NAVY
        ax.annotate(f'{dr:.1f}|{df_:.1f} {win}', (x[k], yl0[1] * 0.86), ha='center', fontsize=8.5, color=wc, fontweight='bold')
    ax.axhline(0, color='#999', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([b[2] for b in bins], fontsize=9.5)
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=11.5, loc='left', color=NAVY)
    top = max(max(act), max(f_v_))
    bot = min(0, min(act), min(f_v_))
    ax.set_ylim(bot * 1.55 if bot < 0 else 0, top * 1.38 if top > 0 else 0.5)
    ax.grid(alpha=0.3, axis='y')

panel(axes[0], v_act0, v_rf0, v_f0, BINS, 2, '出发量效应 (%)',
      '① 出发量（小时级，r<4 训练）', None)
panel(axes[1], v_act, v_rf, v_f, BINS, 2, '速度变化 (%)',
      '② 速度（小时级减速，r<4 训练）', None)
panel(axes[2], d_act, d_rf, d_f, BINS_D, 3, '中位里程变化 (%)',
      '③ 距离（日级，日降水<30 训练）', None)

# 域标注
for ax, txt_l, txt_r in [(axes[0], '内插', '外推'), (axes[1], '内插', '外推'), (axes[2], '内插', '外推')]:
    yl = ax.get_ylim()
    ax.text(-0.0, yl[1] * 0.97, '◀ 绿底=内插域', fontsize=9, color=DGREEN, va='top')
    ax.text(1.0, yl[1] * 0.97, '橙底=外推域 ▶', fontsize=9, color='#843C0C', va='top', ha='right', transform=ax.transAxes)

fig.text(0.5, 0.005,
         '每箱上方数字 = |RF−实际| |公式−实际| + 赢家。速度为减速(负=变慢); 距离为日级中位里程偏移(负=变短); 出发量数据来自 validate_37',
         ha='center', fontsize=9.5, color=MUTED)
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/三量_内插外推对比.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 三量_内插外推对比.png')
