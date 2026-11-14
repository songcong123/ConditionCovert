# -*- coding: utf-8 -*-
"""速度分布的"正态形式"展示(参考 validate_06_heavy.py 同口径):
   速度模型 v ~ LN(μ(t), σ₀): 在 log 速度轴上是严格正态 —— 雨天只把 μ 左移(减速), σ 不变。
   三面板:
     ① log 速度轴: 基线钟形 vs 雨天钟形(同 σ、μ 平移) + 真实样本直方图叠加验证
     ② 线性速度轴: 同两条曲线的右偏形态(中位箭头平移)
     ③ δv(r) 剂量曲线: 平移量随雨强(两点标定的公式)
数据: .flow_spd.npz(6.27M 样本) + validate_06 同款基线(干周X逐时中位/σ₀)
输出: 速度分布_正态形式.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; GREY = '#7F7F7F'; MUTED = '#666666'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); n = len(dates)
klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])   # 0=周日(ISD口径同06)
dry_i = np.where(kl == 'dry')[0]
he_i = np.where(kl == 'heavy')[0]
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)

# ---- 基线(早峰 8 时, 干周三池化; 与 validate_06 同思路) ----
H_DEMO = 8
wed = np.where((kl == 'dry') & (dow == 3))[0]                    # 干周三
m_base = np.isin(spdD, wed) & (spdH == H_DEMO) & (spdV > 1) & (spdV < 60)
v_base = spdV[m_base]
mu0 = float(np.median(np.log(v_base)))                           # log 空间中位 = LN 的 μ
sig0 = float(np.std(np.log(v_base)))
print(f'干周三 {H_DEMO}时: n={len(v_base):,} | μ₀={mu0:.3f}(中位 {np.exp(mu0):.1f} mph) | σ₀={sig0:.3f}')

# ---- 雨天(强雨日中 8 时实际有雨的样本池化) ----
he8 = [i for i in he_i if inten[i, H_DEMO] >= 4]
m_rain = np.isin(spdD, he8) & (spdH == H_DEMO) & (spdV > 1) & (spdV < 60)
v_rain = spdV[m_rain]
mu_r = float(np.median(np.log(v_rain)))
print(f'强雨日({len(he8)}天) {H_DEMO}时 r≥4mm/h: n={len(v_rain):,} | μ={mu_r:.3f}(中位 {np.exp(mu_r):.1f} mph)')
r_med = float(np.median(inten[he8, H_DEMO]))

# ---- validate_06 两点标定公式(同款) ----
th_v, RC_v = -0.0221, 0.20                                        # δ_v 带符号(负=减速)
dv = lambda r: th_v * np.log1p(r / RC_v)

# ================= 图 =================
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
fig.suptitle(f'速度分布的正态形式：log 轴上 v~N(μ, σ₀)，雨天只平移 μ、σ 不变（早高峰 {H_DEMO} 时真实样本）',
             fontsize=13.5, fontweight='bold', y=1.03)

# ---- ① log 速度轴: 严格正态钟形 ----
ax = axes[0]
gx = np.linspace(mu0 - 4 * sig0, mu0 + 4 * sig0, 400)
ax.hist(np.log(v_base), bins=40, density=True, alpha=0.30, color='#8FAADC', label=f'晴天样本(n={len(v_base):,})')
ax.hist(np.log(v_rain), bins=40, density=True, alpha=0.30, color='#F4B6B6', label=f'强雨样本(n={len(v_rain):,})')
ax.plot(gx, stats.norm.pdf(gx, mu0, sig0), lw=2.6, color=NAVY, label=f'晴天正态 $N(\mu_0={mu0:.2f},\ \sigma_0={sig0:.2f})$')
ax.plot(gx, stats.norm.pdf(gx, mu_r, sig0), lw=2.6, color=RED, label=f'雨天正态 $N(\mu_r={mu_r:.2f},\ \sigma_0)$ ← $\mu$ 左移, 同宽')
ax.axvline(mu0, color=NAVY, ls='--', lw=1.4); ax.axvline(mu_r, color=RED, ls='--', lw=1.4)
ax.set_xlabel('log 速度 (ln mph)'); ax.set_ylabel('密度')
ax.set_title('① log 速度轴：严格正态，形状不变', fontsize=11.5, loc='left', color=NAVY)
ax.legend(fontsize=8.5, loc='upper left'); ax.grid(alpha=0.3)

# ---- ② 线性速度轴: 右偏形态 ----
ax = axes[1]
vx = np.linspace(2, 40, 400)
ax.hist(v_base, bins=45, density=True, alpha=0.28, color='#8FAADC', label='晴天样本')
ax.hist(v_rain, bins=45, density=True, alpha=0.28, color='#F4B6B6', label='强雨样本')
ax.plot(vx, stats.lognorm.pdf(vx, sig0, scale=np.exp(mu0)), lw=2.6, color=NAVY, label='晴天 LN')
ax.plot(vx, stats.lognorm.pdf(vx, sig0, scale=np.exp(mu_r)), lw=2.6, color=RED, label='雨天 LN(中位左移)')
ax.axvline(np.exp(mu0), color=NAVY, ls='--', lw=1.4); ax.axvline(np.exp(mu_r), color=RED, ls='--', lw=1.4)
ax.annotate('', xy=(np.exp(mu_r), 0.105), xytext=(np.exp(mu0), 0.105),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.8))
ax.text((np.exp(mu0) + np.exp(mu_r)) / 2, 0.109, f'慢 {abs(np.exp(mu_r)/np.exp(mu0)-1)*100:.0f}%',
        ha='center', fontsize=10, color=RED, fontweight='bold')
ax.set_xlabel('速度 (mph)'); ax.set_ylabel('密度')
ax.set_title('② 线性轴：同样的两条曲线（右偏形态）', fontsize=11.5, loc='left', color=NAVY)
ax.legend(fontsize=8.5, loc='upper right'); ax.grid(alpha=0.3)

# ---- ③ δv(r) 剂量曲线 ----
ax = axes[2]
# ---- 数据准备(分箱统计 + 逐时观测) ----
rainy = np.where(kl != 'dry')[0]
dow_map = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
mu0_how = np.zeros((7, 24))
for k7 in range(7):
    sel = np.where((kl == 'dry') & (dow_map == k7))[0]
    for h in range(24):
        mm = np.isin(spdD, sel) & (spdH == h) & (spdV > 1) & (spdV < 60)
        mu0_how[k7, h] = np.median(np.log(spdV[mm])) if mm.sum() > 50 else np.nan
boxes = [(0.3, 1.0, '0.3-1'), (1.0, 4.0, '1-4'), (4.0, 12.0, '4-12'), (12.0, 1e9, '>=12')]
pt_r, pt_dv = [], []
for lo, hi, nm in boxes:
    bm = np.zeros((n, 24), bool)
    bm[rainy] = (inten[rainy] >= lo) & (inten[rainy] < hi)
    msk = bm[spdD, spdH] & (spdV > 1) & (spdV < 60)
    base_h = mu0_how[dow_map[spdD], spdH]
    ok = msk & ~np.isnan(base_h)
    if ok.sum() > 200:
        pt_r.append(float(inten[bm].mean())); pt_dv.append(float(np.median(np.log(spdV[ok]) - base_h[ok])))
pt_r = np.array(pt_r); pt_dv = np.array(pt_dv)
# 层1: 逐时真实观测散点(每个雨时一个点: 当小时雨强, 当小时中位log残差) —— 未做任何聚合
sc_r, sc_dv = [], []
for i2 in rainy:
    for h in range(24):
        r_h = inten[i2, h]
        if r_h < 0.3: continue
        b0 = mu0_how[dow_map[i2], h]
        if np.isnan(b0): continue
        mm2 = (spdD == i2) & (spdH == h) & (spdV > 1) & (spdV < 60)
        if mm2.sum() < 30: continue
        sc_r.append(r_h); sc_dv.append(float(np.median(np.log(spdV[mm2])) - b0))
sc_r = np.array(sc_r); sc_dv = np.array(sc_dv)
print(f'逐时观测点: {len(sc_r)} 个')
ax.scatter(sc_r, sc_dv, s=14, alpha=0.35, color='#B0B0B0', zorder=3, label=f'逐时真实观测(n={len(sc_r)})')
# 层2: 公式曲线(只用 0.3-1 / 1-4 两箱标定)
rr = np.linspace(0, 30, 200)
ax.plot(rr, dv(rr), lw=2.6, color=RED, zorder=4, label='公式(仅低两箱标定)')
# 层3: 分箱中位 —— 标定箱(空心) vs 盲测箱(实心)
for k3, (lo, hi, nm) in enumerate(boxes):
    cal = k3 < 2      # 前两箱参与标定
    ax.plot([pt_r[k3]], [pt_dv[k3]], 's', ms=13, zorder=6,
            markerfacecolor='none' if cal else NAVY, markeredgecolor=GREY if cal else NAVY, markeredgewidth=2)
    ax.annotate(f'{pt_dv[k3]:+.3f}', (pt_r[k3], pt_dv[k3]), textcoords='offset points',
                xytext=(9, 4 if cal else -14), fontsize=9, color=GREY if cal else NAVY)
ax.plot([], [], 's', ms=13, markerfacecolor='none', markeredgecolor=GREY, markeredgewidth=2, label='标定箱中位(在曲线上是必然)')
ax.plot([], [], 's', ms=13, markerfacecolor=NAVY, markeredgecolor=NAVY, label='盲测箱中位(未参与拟合)')
ax.set_xlabel('雨强 r (mm/h)'); ax.set_ylabel('μ 的平移量 δv(r)')
ax.set_xscale('log')
ax.set_title('③ 平移量公式 $\delta_v$=-0.0221·ln(1+r/0.2) + 真实观测', fontsize=11.5, loc='left', color=NAVY)
ax.text(0.97, 0.05, '曲线仅穿过灰空心点(其标定箱); 蓝实心点(≥4mm/h)为盲测', transform=ax.transAxes,
        fontsize=8.5, color=MUTED, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF2CC', ec='#D6B656', alpha=0.9))
ax.grid(alpha=0.3, which='both')
ax.grid(alpha=0.3)

fig.text(0.5, 0.002,
         '口径同 validate_06：$\sigma_0$ 跨雨强恒定(锁死)、雨天只动 $\mu$ —— "方差不变，改均值"的严格含义是 log 空间正态的 μ 平移',
         ha='center', fontsize=10.5, color=DGREEN, fontweight='bold')
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/速度分布_正态形式.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 速度分布_正态形式.png')
