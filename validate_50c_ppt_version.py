# -*- coding: utf-8 -*-
"""validate_50c: RF/DNN 外推对比 PPT 极简版
   原则: 零标注框/零箭头/零脚注/零标题, 标签=短词贴曲线, 观测点淡化
   输出: RF_DNN外推对比_PPT版.png
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY_D = '#17375E'; RED = '#C00000'; PURPLE = '#7030A0'

rng = np.random.default_rng(50)
true_f = lambda r: 0.052 * np.log1p(np.asarray(r, float) / 0.2) * 100

r_in = rng.uniform(0.1, 4.0, 320)
r_out = rng.uniform(4.2, 15, 36)
y_in = true_f(r_in) + rng.normal(0, 2.2, 320)
y_out = true_f(r_out) + rng.normal(0, 2.2, 36)

rf = RandomForestRegressor(300, max_depth=8, min_samples_leaf=4, random_state=0, n_jobs=-1)
rf.fit(r_in.reshape(-1, 1), y_in)
sc = StandardScaler().fit(r_in.reshape(-1, 1))
dnn = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam',
                   alpha=1e-3, max_iter=9000, random_state=1)
dnn.fit(sc.transform(r_in.reshape(-1, 1)), y_in)

rr = np.linspace(0.12, 15, 800)

fig, ax = plt.subplots(figsize=(12, 6.8))
ax.axvspan(-1, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 16.2, color='#FDF1E8', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2, zorder=2)

ax.scatter(r_in, y_in, s=14, color='#A7BCD6', alpha=0.45, edgecolor='none', zorder=2)
ax.scatter(r_out, y_out, s=20, color='#E8A47C', alpha=0.75, edgecolor='none', zorder=3)

ax.plot(rr, true_f(rr), color=NAVY_D, lw=3.4, zorder=5)
ax.plot(rr, dnn.predict(sc.transform(rr.reshape(-1, 1))), color=PURPLE, lw=2.6, ls='-.', zorder=4)
ax.plot(rr, rf.predict(rr.reshape(-1, 1)), color=RED, lw=3, zorder=5)

# 顶部区名
ax.text(1.8, 51.5, '训练时见过', ha='center', fontsize=15, fontweight='bold', color=NAVY_D)
ax.text(10.4, 51.5, '没见过', ha='center', fontsize=15, fontweight='bold', color='#843C0C')
ax.text(4.3, -3.2, '训练边界 r=4', fontsize=11, color=RED, fontweight='bold', va='bottom')

# 曲线短标签(贴末端)
ax.text(0.5, 9.8, '真实规律', fontsize=14, fontweight='bold', color=NAVY_D, rotation=50)
ax.text(13.1, 9.6, 'RF：躺平', fontsize=14, fontweight='bold', color=RED)
ax.text(11.0, 51.8, 'DNN：冲过头', fontsize=14, fontweight='bold', color=PURPLE)

ax.set_xlim(-1, 16.2)
ax.set_ylim(-5, 58)
ax.set_xlabel('雨强 (mm/h)', fontsize=13)
ax.set_ylabel('出发量效应 (%)', fontsize=13)
ax.tick_params(labelsize=12)
ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/RF_DNN外推对比_PPT版.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved RF_DNN外推对比_PPT版.png')
