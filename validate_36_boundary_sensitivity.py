# -*- coding: utf-8 -*-
"""训练域边界敏感性可视化: 边界 4/8/16/30/42 × 两种估计器
   主图: 各边界拟合的公式曲线 + 实测分箱(体积加权, 误差棒=IQR) + 极端雨时散点
   副图: 各边界在盲测箱的偏差热力对比
输出: 训练域边界敏感性.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; GREY = '#7F7F7F'; MUTED = '#666666'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates'])
hc = A['hour_cnt'].astype(float)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)
inten = np.zeros((len(dates), 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

RS, ES, WS = [], [], []
for i in range(len(dates)):
    if kl[i] == 'dry': continue
    for h in range(24):
        r = inten[i, h]; b = lam[dow[i], h]
        if r >= 0.3 and b > 100:
            RS.append(r); ES.append(hc[i, h] / b - 1); WS.append(b)
RS, ES, WS = np.array(RS), np.array(ES), np.array(WS)

def grid_fit(m):
    best = None
    for RC in np.linspace(0.05, 30, 600):
        x = np.log1p(RS[m] / RC)
        th = (x * ES[m] * WS[m]).sum() / ((x * x * WS[m]).sum() + 1e-12)
        sse = float((WS[m] * (th * x - ES[m]) ** 2).sum())
        if best is None or sse < best[2]: best = (th, RC, sse)
    return best[0], best[1]

# 实测分箱(体积加权 + IQR)
BINS = [(0.3, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 30), (30, 60), (60, 140)]
b_r, b_e, b_lo, b_hi, b_n = [], [], [], [], []
for lo, hi in BINS:
    m = (RS >= lo) & (RS < hi)
    if m.sum() < 3: continue
    b_r.append(float(np.exp(np.mean(np.log(RS[m])))))
    b_e.append(float(np.average(ES[m], weights=WS[m])))
    b_lo.append(float(np.percentile(ES[m], 25))); b_hi.append(float(np.percentile(ES[m], 75)))
    b_n.append(int(m.sum()))
b_r, b_e = np.array(b_r), np.array(b_e)

# 极端雨时散点(≥30)
m_ext = RS >= 30
r_ext, e_ext = RS[m_ext], ES[m_ext]
LBL = {74.9: '7/17 夜间', 82.5: '8/18 晚峰', 61.2: '5/3 夜间', 134.1: '7/14 晚峰\n(历史极值)'}

# ===== 图 =====
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={'width_ratios': [1.45, 1]})
fig.suptitle('训练域边界敏感性：边界放到哪，曲线长什么样，盲测还剩多少',
             fontsize=14, fontweight='bold', y=1.02)

# ---- ① 主图: 曲线族 + 实测 ----
ax = axes[0]
colors = {4: '#9DC3E6', 8: '#4472C4', 16: NAVY, 30: '#ED7D31', 42: RED}
rr = np.geomspace(0.3, 140, 400)
for bnd in [4, 8, 16, 30, 42]:
    th, RC = grid_fit(RS < bnd)
    f = lambda r, t=th, c=RC: t * np.log1p(r / c)
    lw = 2.8 if bnd == 16 else 1.8
    lbl = f'边界{bnd}' + ('（推荐）' if bnd == 16 else '')
    ax.plot(rr, f(rr) * 100, lw=lw, color=colors[bnd], label=lbl, zorder=4 if bnd == 16 else 3)
# 实测箱
yerr = np.array([b_e - np.array(b_lo), np.array(b_hi) - b_e]) * 100
ax.errorbar(b_r, b_e * 100, yerr=yerr, fmt='o', ms=8, color='#1A1A1A',
            ecolor=GREY, elinewidth=2, capsize=5, capthick=2, zorder=6, label='实测(体积加权, IQR)')
for x_, y_, nn in zip(b_r, b_e, b_n):
    ax.annotate(f'n={nn}', (x_, np.array(b_hi)[list(b_r).index(x_)] * 100 + 1.5), fontsize=8, color=MUTED, ha='center')
# 极端雨时散点
ax.scatter(r_ext, e_ext * 100, s=70, marker='D', color=RED, zorder=7, label='极端雨时(≥30, 逐时)')
for r_, e_ in zip(r_ext, e_ext):
    key = min(LBL, key=lambda k: abs(k - r_))
    ax.annotate(LBL[key], (r_, e_ * 100), textcoords='offset points', xytext=(8, 6), fontsize=8.5, color=RED)
ax.axvspan(0.3, 16, color='#D6E4F0', alpha=0.30, zorder=1)
ax.axvline(16, color=NAVY, ls='--', lw=1.4)
ax.text(2.2, 55, '推荐训练域 r<16\n(吃满常态+中大雨)', fontsize=9.5, color=NAVY, ha='center')
ax.text(45, 3, '盲测域\n(极端才外推)', fontsize=9.5, color=MUTED, ha='center')
ax.set_xscale('log'); ax.set_xlim(0.3, 160)
ax.set_xlabel('雨强 r (mm/h, log)'); ax.set_ylabel('出发量效应 (%)')
ax.set_title('① 各边界拟合的曲线族 vs 实测：边界越低饱和越早', fontsize=11.5, loc='left', color=NAVY)
ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3, which='both')

# ---- ② 偏差热力 ----
ax = axes[1]
bnds = [4, 8, 16, 30, 42]
blind_bins = [(4, 8, '4-8'), (8, 16, '8-16'), (16, 30, '16-30'), (30, 60, '30-60'), (60, 140, '60-134')]
M = np.full((len(bnds), len(blind_bins)), np.nan)
for bi, bnd in enumerate(bnds):
    th, RC = grid_fit(RS < bnd)
    f = lambda r, t=th, c=RC: t * np.log1p(r / c)
    for ci, (lo, hi, nm) in enumerate(blind_bins):
        if bnd > lo: continue
        m = (RS >= lo) & (RS < hi)
        if m.sum() == 0: continue
        a = np.average(ES[m], weights=WS[m]); p = f(np.average(RS[m], weights=WS[m]))
        M[bi, ci] = abs(p - a) * 100
im = ax.imshow(M, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=20)
ax.set_xticks(range(len(blind_bins))); ax.set_xticklabels([b[2] for b in blind_bins], fontsize=9.5)
ax.set_yticks(range(len(bnds))); ax.set_yticklabels([f'边界{b}' for b in bnds], fontsize=10)
for bi in range(len(bnds)):
    for ci in range(len(blind_bins)):
        v = M[bi, ci]
        if np.isnan(v):
            ax.text(ci, bi, '训练域', ha='center', va='center', fontsize=8.5, color='#999')
        else:
            ax.text(ci, bi, f'{v:.1f}', ha='center', va='center', fontsize=10,
                    fontweight='bold' if v < 4 else 'normal', color='#1A1A1A')
ax.axvline(1.5, color=MUTED, ls='--', lw=1)
ax.set_xlabel('盲测箱 (mm/h)')
ax.set_title('② 盲测偏差(pp, 绿=准): 60-134箱\n   实测仅n=3高方差, 无判据价值', fontsize=11.5, loc='left', color=NAVY)
plt.colorbar(im, ax=ax, label='|预测-实测| (pp)')

fig.text(0.5, 0.004,
         '结论: 边界≤2会崩(外推+16~23pp); 4/8/16可用且曲线稳定(≥8后两估计器收敛); 推荐16——吃满常态, 只对30+/60+外推; 极端小时实测散布+10~+60%(混杂主导), 极值验证靠日级盲测(06: 偏差4%)',
         ha='center', fontsize=9.5, color=DGREEN, fontweight='bold', wrap=True)
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/训练域边界敏感性.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 训练域边界敏感性.png')
