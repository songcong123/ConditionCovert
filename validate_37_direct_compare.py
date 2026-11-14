# -*- coding: utf-8 -*-
"""最直白对比: 每个雨强箱 = 实际/RF/公式 三柱并排, 谁离实际柱近谁好
   内插域(训练见过 r<4)绿底 | 外推域(训练没见过 r≥4)橙底
   附小图: RF 对雨强的响应曲线(为什么外推不可用的根本原因)
输出: 内插外推_谁更接近实际.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestRegressor
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; GREY = '#404040'; MUTED = '#666666'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates'])
hc = A['hour_cnt'].astype(float)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
wd = (dow + 3) % 7
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)
inten = np.zeros((len(dates), 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

# 样本(特征同 validate_35)
X_all, y_all, r_all, w_all = [], [], [], []
for i in range(len(dates)):
    if kl[i] == 'dry': continue
    for h in range(24):
        base = lam[dow[i], h]
        if base < 200: continue
        r_now = inten[i, h]
        r_lag1 = inten[i, h - 1] if h >= 1 else 0
        is_peak = 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0
        is_we = 1 if wd[i] >= 5 else 0
        X_all.append([np.log1p(r_now), np.log1p(r_lag1), is_peak, is_we,
                      is_peak * np.log1p(r_now), is_we * np.log1p(r_now), h])
        y_all.append(hc[i, h] / base - 1)
        r_all.append(r_now); w_all.append(base)
X, y, r_raw, wts = np.array(X_all), np.array(y_all), np.array(r_all), np.array(w_all)

THETA, RC = 0.052, 0.2
formula = lambda r: THETA * np.log1p(np.asarray(r, float) / RC)

# RF: 只用 r<4 训练
m_lo = r_raw < 4
rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=5,
                           random_state=42, n_jobs=-1)
rf.fit(X[m_lo], y[m_lo])
rf_pred = rf.predict(X)

# ===== 分箱(体积加权) =====
BINS = [(0.3, 1, '0.3-1'), (1, 4, '1-4'), (4, 8, '4-8'), (8, 16, '8-16'), (16, 140, '≥16')]
ACT, RF_, F_ = [], [], []
for lo, hi, nm in BINS:
    m = (r_raw >= lo) & (r_raw < hi)
    ACT.append(float(np.average(y[m], weights=wts[m])))
    RF_.append(float(np.average(rf_pred[m], weights=wts[m])))
    F_.append(float(np.average(formula(r_raw[m]), weights=wts[m])))
ACT, RF_, F_ = np.array(ACT) * 100, np.array(RF_) * 100, np.array(F_) * 100
print('箱       实际    RF    公式   |  RF差  公式差  赢家')
for k, (_, _, nm) in enumerate(BINS):
    dr, df = abs(RF_[k] - ACT[k]), abs(F_[k] - ACT[k])
    print(f'{nm:6s} {ACT[k]:6.1f} {RF_[k]:6.1f} {F_[k]:6.1f} | {dr:5.1f} {df:6.1f}   {"RF" if dr < df else "公式"}')

# ===== 图 =====
fig = plt.figure(figsize=(15, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], width_ratios=[2.6, 1], hspace=0.42, wspace=0.22)
ax = fig.add_subplot(gs[:, 0])
fig.suptitle('内插 vs 外推：RF 和公式谁更接近实际值？（都只用 r<4mm/h 训练）',
             fontsize=15, fontweight='bold', y=0.99)

nb = len(BINS)
# 背景: 内插绿 / 外推橙
ax.add_patch(Rectangle((-0.5, 0), 2, 40, color='#E2F0D9', alpha=0.5, zorder=0))
ax.add_patch(Rectangle((1.5, 0), 3.5, 40, color='#FDEADA', alpha=0.55, zorder=0))
ax.text(0.5, 38.5, '内插域（训练见过）', ha='center', fontsize=12, color=DGREEN, fontweight='bold')
ax.text(3.2, 38.5, '外推域（训练没见过）', ha='center', fontsize=12, color='#843C0C', fontweight='bold')

x = np.arange(nb); bw = 0.26
ax.bar(x - bw, ACT, bw, color='#404040', label='实际值', zorder=3)
ax.bar(x, RF_, bw, color=RED, label='RF 预测', zorder=3)
ax.bar(x + bw, F_, bw, color=NAVY, label='公式预测', zorder=3)

for k in range(nb):
    ax.annotate(f'{ACT[k]:.1f}', (x[k] - bw, ACT[k] + 0.5), ha='center', fontsize=9.5, fontweight='bold', color='#404040')
    ax.annotate(f'{RF_[k]:.1f}', (x[k], RF_[k] + 0.5), ha='center', fontsize=9.5, color=RED)
    ax.annotate(f'{F_[k]:.1f}', (x[k] + bw, F_[k] + 0.5), ha='center', fontsize=9.5, color=NAVY)
    # 差距标注 + 赢家
    dr, df = abs(RF_[k] - ACT[k]), abs(F_[k] - ACT[k])
    win = 'RF✓' if dr < df else '公式✓'
    wc = RED if dr < df else NAVY
    ax.annotate(f'RF差{dr:.1f} 公式差{df:.1f}  {win}', (x[k], 30.5), ha='center', fontsize=8.5,
                color=wc, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels([f'{b[2]}\nmm/h' for b in BINS], fontsize=10.5)
ax.set_ylabel('出发量效应 (%)')
ax.set_ylim(0, 40.5)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(alpha=0.3, axis='y')

# ---- 右上: RF 响应曲线(为什么外推不可用) ----
ax2 = fig.add_subplot(gs[0, 1])
rr = np.linspace(0.05, 60, 200)
rf_curve = rf.predict(np.column_stack([np.log1p(rr), np.zeros_like(rr), np.zeros_like(rr),
                                       np.zeros_like(rr), np.zeros_like(rr), np.zeros_like(rr),
                                       np.full_like(rr, 13.0)]))
ax2.plot(rr, formula(rr) * 100, lw=2.4, color=NAVY, label='公式')
ax2.plot(rr, rf_curve * 100, lw=2.4, color=RED, label='RF')
ax2.axvline(4, color=MUTED, ls='--', lw=1.2)
ax2.text(4.5, 2, '训练边界', fontsize=8.5, color=MUTED)
ax2.set_xlabel('输入雨强 (mm/h)'); ax2.set_ylabel('预测效应 (%)')
ax2.set_title('RF 对雨强无响应：\nr=10 与 r=60 答案相同', fontsize=10.5, loc='left', color=NAVY)
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# ---- 右下: 结论卡 ----
ax3 = fig.add_subplot(gs[1, 1]); ax3.axis('off')
ax3.text(0.0, 0.95, '结论', fontsize=13, fontweight='bold', color=NAVY, va='top')
ax3.text(0.0, 0.62, '① 内插域: RF 更接近实际（2/2 箱）\n② 外推域: 逐箱差距互有胜负，但 RF 对\n   雨强零响应（右上），给暴雨分级时\n   r=10/30/60 全是同一答案 → 不可用',
         fontsize=10.5, va='top', color='#1A1A1A', linespacing=1.5)
ax3.text(0.0, 0.10, '→ 域内用 RF，域外用公式', fontsize=11.5, fontweight='bold', color=DGREEN)

fig.text(0.5, 0.005,
         '口径: RF 与公式都只用 r<4mm/h 训练(内插/外推唯一区别=测试数据在训练域内还是域外); 各箱数值为体积加权均值; 差=|预测-实际|',
         ha='center', fontsize=9, color=MUTED)
plt.savefig(r'D:/AI/条件转化/内插外推_谁更接近实际.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 内插外推_谁更接近实际.png')
