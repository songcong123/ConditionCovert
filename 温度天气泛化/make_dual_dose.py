# -*- coding: utf-8 -*-
"""双泛化维剂量曲线: 雨强(已盲测) × 温度(已标定) —— 两个泛化维各自一条曲线
输出: 温度天气泛化/泛化剂量曲线.png
"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---- 365 个雨时标定样本（轻中雨日×雨时, 与标定同口径）----
Agg = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
Df = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
wdates = list(Agg['dates']); hour_cnt = Agg['hour_cnt'].astype(np.float64)
klmap = dict(zip(list(Df['dates']), Df['klass'].tolist()))
kl_w = np.array([klmap[d] for d in wdates]); nw = len(wdates)
yr_w = np.array([int(str(d)[:4]) for d in wdates])
dow_w = np.array([(np.datetime64(d, 'D').view('int64') + 3) % 7 for d in wdates])
inten_w = np.zeros((nw, 24))
didx = {d: i for i, d in enumerate(Df['dates'])}
for dd, hh, mm in zip(Df['rh_date'], Df['rh_hour'].astype(int), Df['rh_mm']):
    if str(dd) in didx: inten_w[didx[str(dd)], hh] = mm
dry_m = kl_w == 'dry'
lam0 = np.full((2, 7, 24), np.nan)
yidx = np.searchsorted([2024, 2025], yr_w)
for yi in range(2):
    for dw in range(7):
        for h in range(24):
            sel = np.where((yidx == yi) & (dow_w == dw) & dry_m)[0]
            vals = hour_cnt[sel, h]
            if len(vals) >= 2: lam0[yi, dw, h] = vals.mean()
rs, ys = [], []
for i in range(nw):
    if kl_w[i] not in ('light', 'medium'): continue
    for h in range(24):
        r_h = inten_w[i, h]
        L = lam0[yidx[i], dow_w[i], h]
        if r_h >= 0.3 and not np.isnan(L) and L > 0:
            rs.append(r_h); ys.append(hour_cnt[i, h] / L)
rs = np.array(rs); ys = np.array(ys)
print(f'雨时标定样本: {len(rs)} 个')

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
fig.suptitle('两个泛化维各自一条剂量曲线：推理时按场景取值连乘（观测域内插值，域外沿曲线延伸）',
             fontsize=13, fontweight='bold', y=1.02)

# ---------- ① 雨强剂量 ----------
ax = axes[0]
r = np.linspace(0, 150, 200)
mult = 1 + 0.052 * np.log1p(r / 0.2)
ax.scatter(rs, ys, s=13, color='#1F4E79', alpha=0.28, label=f'雨时样本 n={len(rs)}', zorder=3)
ax.plot(r, mult, lw=2.6, color='#1F4E79')
for rr, lab, off in [(10, '+20%', (-30, 10)), (30, '+26%', (5, 10)), (148, '+34%', (-16, 10))]:
    m = 1 + 0.052 * np.log1p(rr / 0.2)
    ax.plot([rr], [m], 'o', ms=8, color='#C00000', zorder=5)
    ax.annotate(f'{rr}mm/h\n{lab}', (rr, m), textcoords='offset points', xytext=off, fontsize=9, color='#C00000')
ax.axvline(148, color='#999', ls=':', lw=1.2)
ax.annotate('观测上限 148', (150, 1.0), fontsize=9, color='#666', ha='left')
ax.set_xlabel('雨强 r (mm/h)'); ax.set_ylabel('出发量乘子')
ax.set_title('① 泛化维=雨强：δvol(r)=0.052·ln(1+r/0.2)\n176 个同时钟点雨时样本标定 · 强雨盲测偏差 4%', fontsize=11.5, loc='left', color='#1F4E79')
ax.grid(alpha=0.3); ax.set_ylim(0.9, 1.48); ax.set_xlim(0, 165)
ax.legend(fontsize=9, loc='upper left')

# ---------- ② 温度剂量 ----------
ax = axes[1]
T = np.linspace(2, 42, 200)
mult_T = np.ones_like(T)                                    # 观测域内平坦
ax.plot(T, mult_T, lw=2.2, color='#7F7F7F', ls='--', label='干球拟合（≈平坦）')
ax.axhline(1, color='#999', lw=1)
pts = [(26.5, 1.015, '#1F4E79', (0, 10)), (28, 1.005, '#1F4E79', (0, -18)), (30.5, 1.051, '#1F4E79', (0, 10))]
for tt, mm, cc, off in pts:
    ax.plot([tt], [mm], 'o', ms=9, color=cc, zorder=5)
    ax.annotate(f'{mm:.3f}', (tt, mm), textcoords='offset points', xytext=off, ha='center', fontsize=9, color=cc)
ax.plot([34.5], [1.123], 'D', ms=12, color='#C00000', zorder=6)
ax.annotate('×1.12（方式转移）', (34.5, 1.123), textcoords='offset points', xytext=(-12, 12), fontsize=10, color='#C00000', fontweight='bold', ha='left')
ax.axvspan(33, 42, color='#C00000', alpha=0.06)
ax.annotate('外推区（挂低置信）', (43.5, 1.17), fontsize=8.5, color='#666', ha='right')
ax.axvline(40, color='#999', ls=':', lw=1.2)
ax.annotate('40 观测上限', (38.2, 0.975), fontsize=8.5, color='#666666', ha='right')
ax.set_xlabel('气温 T (°C)'); ax.set_ylabel('出发量乘子')
ax.set_title('② 泛化维=温度：观测域内 ≈0（平坦）\n热浪小时 ×1.12（方式转移）；闷热 +5pp（控温对比）', fontsize=11.5, loc='left', color='#ED7D31')
ax.grid(alpha=0.3); ax.set_ylim(0.9, 1.25); ax.set_xlim(2, 44)
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/温度天气泛化/泛化剂量曲线.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 泛化剂量曲线.png')
