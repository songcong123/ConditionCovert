# -*- coding: utf-8 -*-
"""挑战页配图: 模型只会"重放"训练分布 → 未覆盖极端场景严重失真
   基于真实训练的虚拟实验(validate_58 同款): 3000点只覆盖x∈[0,4], RF/DNN充分训练(域内MAE≈噪声底)
   单面板极简, PPT直插。输出: 重放训练分布_极端失真.png
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

NAVY = '#17375E'; RED = '#C00000'; PURPLE = '#7030A0'; MUTED = '#666666'

rng = np.random.default_rng(58)
f_true = lambda x: 30.0 * np.log1p(np.asarray(x, float) / 1.5) / np.log1p(10 / 1.5)
x_tr = rng.uniform(0, 4, 3000)
y_tr = f_true(x_tr) + rng.normal(0, 2.0, 3000)
rf = RandomForestRegressor(500, max_depth=None, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf.fit(x_tr.reshape(-1, 1), y_tr)
sc = StandardScaler().fit(x_tr.reshape(-1, 1))
dnn = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation='relu', solver='adam',
                   alpha=1e-4, max_iter=30000, random_state=0)
dnn.fit(sc.transform(x_tr.reshape(-1, 1)), y_tr)

gx = np.linspace(0, 10, 600)
fy = f_true(gx)
rf_y = rf.predict(gx.reshape(-1, 1))
dnn_y = dnn.predict(sc.transform(gx.reshape(-1, 1)))

fig, ax = plt.subplots(figsize=(13, 7.6))
fig.suptitle('模型只会"重放"训练分布：未覆盖的极端场景产生严重失真',
             fontsize=17, fontweight='bold', y=0.985)

ax.axvspan(0, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 10.6, color='#FDF1E8', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2.2, zorder=2)

idx = rng.choice(3000, 700, replace=False)
ax.scatter(x_tr[idx], y_tr[idx], s=10, color='#9DB4CE', alpha=0.45, edgecolor='none', zorder=2)
ax.plot(gx, fy, lw=3.6, color=NAVY, zorder=5)
ax.plot(gx, rf_y, lw=3, color=RED, zorder=5)
ax.plot(gx, dnn_y, lw=2.8, color=PURPLE, ls='-.', zorder=4)

# 失真区阴影(真实 − RF躺平线之间)
plat = rf_y[np.argmin(np.abs(gx - 4))]
mask = gx >= 4
ax.fill_between(gx[mask], plat, fy[mask], color=RED, alpha=0.13, zorder=1)

# 区域标签
ax.text(2.0, 36.2, '训练分布覆盖区', ha='center', fontsize=16, fontweight='bold', color=NAVY)
ax.text(2.0, 33.6, '(见过, 表现好)', ha='center', fontsize=12.5, color=MUTED)
ax.text(7.3, 36.2, '极端场景(未覆盖)', ha='center', fontsize=16, fontweight='bold', color='#843C0C')
ax.text(7.3, 33.6, '(失真随条件强度拉大)', ha='center', fontsize=12.5, color=MUTED)

# 曲线短标签
ax.text(6.9, 25.8, '真实响应(还在涨)', fontsize=14.5, fontweight='bold', color=NAVY)
ax.text(4.7, 13.6, 'RF：躺平\n(冻在边界值 = 重放)', fontsize=13.5, fontweight='bold', color=RED)
ax.text(8.3, 44.0, 'DNN：直线冲出', fontsize=13.5, fontweight='bold', color=PURPLE)

# 失真差标注
xg = 9.8
fv, rv = fy[np.argmin(np.abs(gx - xg))], rf_y[np.argmin(np.abs(gx - xg))]
ax.annotate('', xy=(xg, fv), xytext=(xg, rv),
            arrowprops=dict(arrowstyle='<->', color='#8B0000', lw=2.6))
ax.text(xg - 0.25, (fv + rv) / 2 + 1.2, f'差 {fv - rv:.0f}', ha='right',
        fontsize=14, fontweight='bold', color='#8B0000')

ax.text(4.2, -2.6, '训练边界', fontsize=12, color=RED, fontweight='bold')
ax.set_xlim(0, 10.6); ax.set_ylim(-4, 48)
ax.set_xlabel('条件强度 x（训练只覆盖 0~4）', fontsize=14.5)
ax.set_ylabel('系统响应 y', fontsize=14.5)
ax.tick_params(labelsize=13)
ax.grid(alpha=0.22)
fig.text(0.5, 0.012,
         '虚拟数据充分训练: 3000样本只覆盖 x∈[0,4], RF与DNN域内MAE≈噪声底(学满)。失真与训练充分性无关, 由模型结构决定:\n'
         '树=分段常数, 域外恒等输出边界叶子(躺平); 网络=分段线性, 域外沿边界斜率直线延伸(冲出) —— 都不知道"该弯"。',
         ha='center', fontsize=11.5, color='#375623', fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
fig.savefig(r'D:/AI/条件转化/重放训练分布_极端失真.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved 重放训练分布_极端失真.png')
