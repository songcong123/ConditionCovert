# -*- coding: utf-8 -*-
"""validate_50b: RF 只会"重放"训练分布 —— 构造数据示意版(美观优先)
   真实规律 = 饱和对数曲线(模拟雨强→出发量效应)
   观测点   = 规律 + 噪声(训练域内密, 域外稀但存在)
   RF       = 真的只用 r<4 的观测点训练(sklearn), 平台是真实模型行为
   公式     = 域内两箱均值两点标定 θ·ln(1+r/RC), 外推跟涨
   输出: RF重放训练分布_示意版.png
"""
import numpy as np
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY_D = '#17375E'; NAVY = '#1F4E79'; RED = '#C00000'; GREEN = '#2E7D32'; MUTED = '#666666'

rng = np.random.default_rng(50)
true_f = lambda r: 0.052 * np.log1p(np.asarray(r, float) / 0.2) * 100   # 真实规律(%)

# ---- 构造观测 ----
r_in = rng.uniform(0.1, 4.0, 320)
r_out = rng.uniform(4.2, 15, 36)
y_in = true_f(r_in) + rng.normal(0, 2.2, 320)
y_out = true_f(r_out) + rng.normal(0, 2.2, 36)

# ---- RF: 只训练 r<4 ----
rf = RandomForestRegressor(300, max_depth=8, min_samples_leaf=4, random_state=0, n_jobs=-1)
rf.fit(r_in.reshape(-1, 1), y_in)

# ---- DNN: 同一协议(只用 r<4)训 5 次, 展示外推走向不可控 ----
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(r_in.reshape(-1, 1))
dnns = []
for sd in [1, 2, 3, 4, 5]:
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam',
                     alpha=1e-3, max_iter=9000, random_state=sd)
    m.fit(sc.transform(r_in.reshape(-1, 1)), y_in)
    dnns.append(m)

rr = np.linspace(0.12, 15, 800)
rf_curve = rf.predict(rr.reshape(-1, 1))
dnn_curves = [m.predict(sc.transform(rr.reshape(-1, 1))) for m in dnns]
print(f'r=10: 真实{true_f(10):.1f} RF{rf.predict([[10]])[0]:.1f} DNN5次:' +
      ', '.join(f'{c[np.searchsorted(rr, 10)]:.1f}' for c in dnn_curves))
print(f'r=15: 真实{true_f(15):.1f} RF{rf.predict([[15]])[0]:.1f} DNN5次:' +
      ', '.join(f'{c[np.searchsorted(rr, 15)]:.1f}' for c in dnn_curves))

# ================= 图 =================
fig, ax = plt.subplots(figsize=(13, 7.4))
fig.suptitle('为什么外推不能用黑盒模型：RF 只会重放、DNN 沿惯性冲——没见过的世界都不行（构造数据演示）',
             fontsize=15, fontweight='bold', y=0.975)

# 分区底色 + 边界
ax.axvspan(-1.5, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 16.5, color='#FDF1E8', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2.6, zorder=2)

# 曲线
ax.plot(rr, true_f(rr), color=NAVY_D, lw=3.2, zorder=5)
for c in dnn_curves:
    ax.plot(rr, c, color='#7030A0', lw=1.4, alpha=0.75, ls='-.', zorder=4)
ax.plot(rr, rf_curve, color=RED, lw=2.8, zorder=5)

# 观测点
ax.scatter(r_in, y_in, s=17, color='#9DB4CE', alpha=0.55, edgecolor='none', zorder=3)
ax.scatter(r_out, y_out, s=26, color='#E8A47C', alpha=0.85, edgecolor='white', linewidth=0.4, zorder=3)

# ---- 分区标题 ----
ax.text(2.0, 55.5, '黑盒模型训练时见过的数据', ha='center', fontsize=13, fontweight='bold', color=NAVY)
ax.text(2.0, 52.6, '(雨强 0~4, 320 个点)', ha='center', fontsize=10.5, color=MUTED)
ax.text(10.3, 55.5, '从未见过的雨强', ha='center', fontsize=13, fontweight='bold', color='#843C0C')
ax.text(10.3, 52.6, '(> 4, 橙点 = 后来发生的真实)', ha='center', fontsize=10.5, color=MUTED)
ax.text(4.55, 0.8, '训练边界\nr = 4', fontsize=10.5, color=RED, fontweight='bold', va='bottom')

# ---- 曲线旁标签 ----
ax.text(1.15, 10.2, '真实规律', fontsize=12.5, fontweight='bold', color=NAVY_D, rotation=52)
ax.text(9.6, 16.4, 'RF', fontsize=15, fontweight='bold', color=RED)
ax.text(0.28, 26.5, 'DNN×5次训练', fontsize=11.5, fontweight='bold', color='#7030A0')

# ---- RF 平台标注 ----
plat = rf.predict([[15.0]])[0]
ax.annotate('RF 躺平: 卡在边界值\n(分裂点全在训练范围内\n→ 输出恒 = 边界值)',
            xy=(13, plat), xytext=(6.4, 1.2), fontsize=10, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.8),
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec=RED, lw=1.2))

# ---- DNN 标注 ----
dnn_end = [c[-1] for c in dnn_curves]
ax.annotate(f'DNN 冲过头: 沿边界斜率线性延伸\n同一数据训 5 次全部冲到 {min(dnn_end):.0f}~{max(dnn_end):.0f}%\n(真实只有 {true_f(15):.0f}% —— 更离谱)',
            xy=(12.2, dnn_curves[0][np.searchsorted(rr, 12.2)]), xytext=(0.6, 38.5), fontsize=10, color='#7030A0',
            arrowprops=dict(arrowstyle='->', color='#7030A0', lw=1.8),
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#7030A0', lw=1.2))

# ---- 差距双向箭头 ----
xg = 14.8
ax.annotate('', xy=(xg, true_f(14.8)), xytext=(xg, plat),
            arrowprops=dict(arrowstyle='<->', color='#843C0C', lw=2.4))
ax.text(9.0, 17.6, f'差 {true_f(14.8)-plat:.0f} 个百分点 (RF 没跟上)',
        ha='left', va='center', fontsize=11, fontweight='bold', color='#843C0C')

# 域内也标一个"贴合"
ax.text(1.35, -3.8, '域内: RF 贴合观测 ✓', fontsize=10.5, color=NAVY, fontweight='bold')

ax.set_xlim(-1.5, 17)
ax.set_ylim(-6.5, 60)
ax.set_xlabel('雨强 (mm/h)', fontsize=12)
ax.set_ylabel('出发量效应 (%, 相对晴天基线)', fontsize=12)
ax.tick_params(labelsize=11)
ax.grid(alpha=0.25)

fig.text(0.5, 0.008,
         '构造数据: 真实规律=饱和对数曲线, 观测=规律+噪声; RF(300棵树)与 DNN(64x64 ReLU MLP) 都只用蓝色区 320 个点训练——都是真 sklearn 模型跑出来的。\n'
         '域外两种失败方式: RF=所有分裂点都在训练范围内→输出恒为边界常数(躺平); DNN=ReLU 网络沿训练边界处的斜率线性延伸(冲过头)。"黑盒外推, 错法不同, 都不可靠"',
         ha='center', fontsize=9.5, color='#375623')
plt.tight_layout(rect=[0, 0.05, 1, 0.955])
fig.savefig(r'D:/AI/条件转化/RF重放训练分布_示意版.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved RF重放训练分布_示意版.png')
