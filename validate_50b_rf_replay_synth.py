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
r_out = rng.uniform(4.2, 40, 46)
y_in = true_f(r_in) + rng.normal(0, 2.2, 320)
y_out = true_f(r_out) + rng.normal(0, 2.2, 46)

# ---- RF: 只训练 r<4 ----
rf = RandomForestRegressor(300, max_depth=8, min_samples_leaf=4, random_state=0, n_jobs=-1)
rf.fit(r_in.reshape(-1, 1), y_in)

# ---- 公式: 域内两箱两点标定 ----
b1 = r_in < 1.0; b2 = r_in >= 1.0
r1, y1 = r_in[b1].mean(), y_in[b1].mean()
r2, y2 = r_in[b2].mean(), y_in[b2].mean()
f = lambda RC: np.log1p(r1 / RC) / np.log1p(r2 / RC) - y1 / y2
RC = brentq(f, 1e-3, 10)
th = y1 / np.log1p(r1 / RC)
formula = lambda r: th * np.log1p(np.asarray(r, float) / RC)

rr = np.linspace(0.12, 40, 800)
rf_curve = rf.predict(rr.reshape(-1, 1))
fm_curve = formula(rr)
print(f'两点标定: θ={th:.4f}, RC={RC:.3f}')
print(f'r=10: 真实{true_f(10):.1f} RF{rf.predict([[10]])[0]:.1f} 公式{formula(10):.1f}')
print(f'r=30: 真实{true_f(30):.1f} RF{rf.predict([[30]])[0]:.1f} 公式{formula(30):.1f}')

# ================= 图 =================
fig, ax = plt.subplots(figsize=(13, 7.4))
fig.suptitle('为什么外推不能用随机森林：训练之后，它只会"重放"见过的世界（构造数据演示）',
             fontsize=15, fontweight='bold', y=0.975)

# 分区底色 + 边界
ax.axvspan(-1.5, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 42, color='#FDF1E8', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2.6, zorder=2)

# 曲线
ax.plot(rr, true_f(rr), color=NAVY_D, lw=3.2, zorder=5)
ax.plot(rr, rf_curve, color=RED, lw=2.8, zorder=5)
ax.plot(rr, fm_curve, color=GREEN, lw=2.6, ls='--', zorder=4)

# 观测点
ax.scatter(r_in, y_in, s=17, color='#9DB4CE', alpha=0.55, edgecolor='none', zorder=3)
ax.scatter(r_out, y_out, s=26, color='#E8A47C', alpha=0.85, edgecolor='white', linewidth=0.4, zorder=3)

# ---- 分区标题 ----
ax.text(2.0, 31.8, 'RF 训练时见过的数据', ha='center', fontsize=13, fontweight='bold', color=NAVY)
ax.text(2.0, 29.6, '(雨强 0~4, 320 个点)', ha='center', fontsize=10.5, color=MUTED)
ax.text(23, 31.8, 'RF 从未见过的雨强', ha='center', fontsize=13, fontweight='bold', color='#843C0C')
ax.text(23, 29.6, '(> 4, 橙色点是后来才发生的真实情况)', ha='center', fontsize=10.5, color=MUTED)
ax.text(4.55, 0.8, '训练边界\nr = 4', fontsize=10.5, color=RED, fontweight='bold', va='bottom')

# ---- 曲线旁标签 ----
ax.text(1.15, 10.2, '真实规律', fontsize=12.5, fontweight='bold', color=NAVY_D, rotation=52)
ax.text(6.8, 19.0, '公式(两点标定)', fontsize=12, fontweight='bold', color=GREEN, rotation=25)
ax.text(15.5, 16.2, 'RF', fontsize=15, fontweight='bold', color=RED)

# ---- RF 平台标注 ----
plat = rf.predict([[40.0]])[0]
ax.annotate('RF 卡在边界值不动了\n(所有分裂点都在训练范围内,\n超范围 = 恒等输出边界值)',
            xy=(30, plat), xytext=(22.5, 4.2), fontsize=10.5, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.8),
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec=RED, lw=1.2))

# ---- 差距双向箭头 ----
xg = 38
ax.annotate('', xy=(xg, true_f(38)), xytext=(xg, plat),
            arrowprops=dict(arrowstyle='<->', color='#843C0C', lw=2.4))
ax.text(xg - 0.5, (true_f(38) + plat) / 2, f'差 {true_f(38)-plat:.0f} 个百分点\n(RF 没跟上)',
        ha='right', va='center', fontsize=11, fontweight='bold', color='#843C0C')

# 域内也标一个"贴合"
ax.text(1.35, -3.8, '域内: RF 贴合观测 ✓', fontsize=10.5, color=NAVY, fontweight='bold')

ax.set_xlim(-1.5, 42)
ax.set_ylim(-6.5, 34.5)
ax.set_xlabel('雨强 (mm/h)', fontsize=12)
ax.set_ylabel('出发量效应 (%, 相对晴天基线)', fontsize=12)
ax.tick_params(labelsize=11)
ax.grid(alpha=0.25)

fig.text(0.5, 0.008,
         '构造数据: 真实规律=饱和对数曲线, 观测=规律+噪声; RF(300棵树)只用蓝色区 320 个点训练——它是真的 sklearn 模型, 平台不是画出来的。\n'
         '域内 RF 贴合观测; 域外连续特征(雨强)超出所有分裂点 → 每棵树落到边界叶子, 输出恒为常数。"见过→重放, 没见过→答非所问"',
         ha='center', fontsize=9.5, color='#375623')
plt.tight_layout(rect=[0, 0.05, 1, 0.955])
fig.savefig(r'D:/AI/条件转化/RF重放训练分布_示意版.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved RF重放训练分布_示意版.png')
