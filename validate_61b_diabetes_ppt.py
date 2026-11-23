# -*- coding: utf-8 -*-
"""validate_61b: 糖尿病数据集复验 · PPT极简版(单面板, 零标注框/零脚注/零标题)
   输出: 公开数据集_糖尿病_PPT版.png
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#17375E'; RED = '#C00000'; PURPLE = '#7030A0'; MUTED = '#666666'

d = load_diabetes()
X = d.data[:, 2] * 50
y = d.target.astype(float)
BOUND = float(np.percentile(X, 60))
m_tr = X < BOUND

rf = RandomForestRegressor(500, max_depth=None, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf.fit(X[m_tr].reshape(-1, 1), y[m_tr])
sc = StandardScaler().fit(X[m_tr].reshape(-1, 1))
dnn = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation='relu', solver='adam',
                   alpha=1e-3, max_iter=30000, random_state=0)
dnn.fit(sc.transform(X[m_tr].reshape(-1, 1)), y[m_tr])
mae_in_rf = np.mean(np.abs(y[m_tr] - rf.predict(X[m_tr].reshape(-1, 1))))
mae_ex_rf = np.mean(np.abs(y[~m_tr] - rf.predict(X[~m_tr].reshape(-1, 1))))
mae_ex_dn = np.mean(np.abs(y[~m_tr] - dnn.predict(sc.transform(X[~m_tr].reshape(-1, 1)))))
mae_in_dn = np.mean(np.abs(y[m_tr] - dnn.predict(sc.transform(X[m_tr].reshape(-1, 1)))))

fig, ax = plt.subplots(figsize=(12, 6.6))
ax.axvspan(X.min(), BOUND, color='#EAF1F9', zorder=0)
ax.axvspan(BOUND, X.max(), color='#FDF1E8', zorder=0)
ax.axvline(BOUND, color=RED, ls='--', lw=2, zorder=2)
ax.scatter(X[m_tr], y[m_tr], s=18, color='#9DB4CE', alpha=0.55, edgecolor='none', zorder=2)
ax.scatter(X[~m_tr], y[~m_tr], s=22, color='#E8A47C', alpha=0.85, edgecolor='none', zorder=3)
gx = np.linspace(X.min(), X.max(), 400)
ax.plot(gx, rf.predict(gx.reshape(-1, 1)), lw=3, color=RED, zorder=5)
ax.plot(gx, dnn.predict(sc.transform(gx.reshape(-1, 1))), lw=2.8, color=PURPLE, ls='-.', zorder=4)

ax.text(BOUND - 2.4, 322, '见过', ha='center', fontsize=19, fontweight='bold', color=NAVY)
ax.text(BOUND + 4.2, 322, '没见过（对真实患者测误差）', ha='center', fontsize=15, fontweight='bold', color='#843C0C')
ax.text(3.4, 96, 'RF：躺平', fontsize=16, fontweight='bold', color=RED)
ax.text(4.6, 232, 'DNN：直线漂移', fontsize=16, fontweight='bold', color=PURPLE)
ax.text(BOUND + 0.3, 32, '训练边界', fontsize=12.5, color=RED, fontweight='bold')

# 域外误差倍数(两个短标注)
ax.text(BOUND + 1.0, 140, f'RF 误差 ×{mae_ex_rf/mae_in_rf:.1f}', fontsize=13.5, color=RED, fontweight='bold')
ax.text(BOUND + 1.0, 100, f'DNN 误差 ×{mae_ex_dn/mae_in_dn:.1f}', fontsize=13.5, color=PURPLE, fontweight='bold')

ax.set_xlim(X.min(), X.max()); ax.set_ylim(20, 345)
ax.set_xlabel('BMI（标准化刻度）', fontsize=15)
ax.set_ylabel('病情进展指数', fontsize=15)
ax.tick_params(labelsize=13)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/公开数据集_糖尿病_PPT版.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved 公开数据集_糖尿病_PPT版.png')
