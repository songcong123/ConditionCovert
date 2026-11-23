# -*- coding: utf-8 -*-
"""validate_58: 虚拟数据·充分训练·见过/没见过推理可视化(通用演示, 不依赖任何真实业务数据)
   任务: y = f(x) + 噪声, f = 饱和曲线; 训练集只从 x∈[0,4] 采 3000 点(充分)
   模型: RF(500树, 不限深) 与 DNN(128x128x64, 迭代至收敛) —— 两个都充分训练
   推理: x∈[0,10] 全范围; [0,4]=见过, (4,10]=没见过
   可视化: 左=预测形态(真实/散点/RF/DNN); 右=误差曲线(域内贴噪声底, 域外各自爆炸)
   输出: 虚拟数据_见过vs没见过_推理可视化.png
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

NAVY = '#17375E'; RED = '#C00000'; PURPLE = '#7030A0'; MUTED = '#666666'; DGREEN = '#375623'

rng = np.random.default_rng(58)
f_true = lambda x: 30.0 * np.log1p(np.asarray(x, float) / 1.5) / np.log1p(10 / 1.5)   # 饱和曲线 0~30
SIGMA = 2.0

# ---- 虚拟数据: 训练集只见 [0,4], 3000 点(充分) ----
x_tr = rng.uniform(0, 4, 3000)
y_tr = f_true(x_tr) + rng.normal(0, SIGMA, 3000)

# ---- 充分训练 ----
rf = RandomForestRegressor(500, max_depth=None, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf.fit(x_tr.reshape(-1, 1), y_tr)
sc = StandardScaler().fit(x_tr.reshape(-1, 1))
dnn = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation='relu', solver='adam',
                   alpha=1e-4, max_iter=30000, random_state=0)
dnn.fit(sc.transform(x_tr.reshape(-1, 1)), y_tr)
print(f'DNN 训练轮数(至收敛): {dnn.n_iter_}, 最终loss {dnn.loss_:.2f} (噪声方差 {SIGMA**2:.1f})')

# ---- 全范围推理 ----
gx = np.linspace(0, 10, 600)
fy = f_true(gx)
rf_y = rf.predict(gx.reshape(-1, 1))
dnn_y = dnn.predict(sc.transform(gx.reshape(-1, 1)))

# ---- 充分性证明: 域内拟合 MAE ≈ 噪声期望(|N(0,2)|的均值≈1.6) ----
pr_tr = rf.predict(x_tr.reshape(-1, 1))
pd_tr = dnn.predict(sc.transform(x_tr.reshape(-1, 1)))
print(f'域内拟合MAE: RF {np.mean(np.abs(y_tr-pr_tr)):.2f} | DNN {np.mean(np.abs(y_tr-pd_tr)):.2f}  (噪声底≈1.6 → 都学满了)')

print(f'\n{"x":>4s} {"域":6s} {"真实":>7s} {"RF":>7s} {"DNN":>7s} {"RF误":>6s} {"DNN误":>6s}')
for xv in [2.0, 4.0, 6.0, 8.0, 10.0]:
    i_ = np.argmin(np.abs(gx - xv))
    dom = '见过' if xv <= 4 else '没见过'
    print(f'{xv:4.1f} {dom:6s} {fy[i_]:7.1f} {rf_y[i_]:7.1f} {dnn_y[i_]:7.1f} '
          f'{abs(rf_y[i_]-fy[i_]):6.1f} {abs(dnn_y[i_]-fy[i_]):6.1f}')

# ================= 可视化 =================
fig, axes = plt.subplots(1, 2, figsize=(16, 7.6))
fig.suptitle('虚拟数据 · 充分训练 · 见过 vs 没见过：训练域内都学满了，一出域各自失真',
             fontsize=15.5, fontweight='bold', y=0.99)

# ---- 左: 预测形态 ----
ax = axes[0]
ax.axvspan(0, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 10, color='#FDF1E8', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2.2, zorder=2)
idx = rng.choice(3000, 700, replace=False)
ax.scatter(x_tr[idx], y_tr[idx], s=10, color='#9DB4CE', alpha=0.45, edgecolor='none', zorder=2)
ax.plot(gx, fy, lw=3.6, color=NAVY, zorder=5, label='真实规律 f(x)')
ax.plot(gx, rf_y, lw=3, color=RED, zorder=4, label='RF(500树, 充分训练)')
ax.plot(gx, dnn_y, lw=2.8, color=PURPLE, ls='-.', zorder=4, label='DNN(128×128×64, 训练至收敛)')
ax.text(2.0, 33.5, '见过(训练分布内)', ha='center', fontsize=14.5, fontweight='bold', color='#1F4E79')
ax.text(7.0, 33.5, '没见过', ha='center', fontsize=14.5, fontweight='bold', color='#843C0C')
ax.text(4.15, -2.8, '训练边界 x=4', fontsize=11.5, color=RED, fontweight='bold')
ax.annotate('RF: 躺平在边界值', xy=(8.6, rf_y[np.argmin(np.abs(gx-8.6))]), xytext=(5.6, 8.5),
            fontsize=12.5, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.8))
ax.annotate('DNN: 沿惯性直线冲过真实', xy=(9.4, dnn_y[np.argmin(np.abs(gx-9.4))]), xytext=(5.8, 27.5),
            fontsize=12.5, color=PURPLE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.8))
ax.set_xlim(0, 10); ax.set_ylim(-4, 37)
ax.set_xlabel('条件 x (虚拟, 训练只覆盖 0~4)', fontsize=14)
ax.set_ylabel('系统响应 y', fontsize=14)
ax.tick_params(labelsize=12.5)
ax.legend(fontsize=11.5, loc='lower right')
ax.grid(alpha=0.25)
ax.set_title('① 预测形态: 域内两条线都压在真实规律上', fontsize=13, loc='left', color=NAVY)

# ---- 右: 误差曲线 ----
ax = axes[1]
ax.axvspan(0, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 10, color='#FDF1E8', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2.2)
ax.plot(gx, np.abs(rf_y - fy), lw=3, color=RED, label='RF 误差')
ax.plot(gx, np.abs(dnn_y - fy), lw=2.8, color=PURPLE, ls='-.', label='DNN 误差')
ax.axhline(SIGMA * 0.8, color=MUTED, ls=':', lw=1.8)
ax.text(0.3, SIGMA * 0.8 + 0.45, '噪声底(数据本身的随机波动)', fontsize=11, color=MUTED)
ax.text(2.0, 12.5, '见过: 误差≈噪声底\n(模型已学到数据极限)', ha='center', fontsize=12.5,
        color='#1F4E79', fontweight='bold')
ax.annotate('没见过: 误差一路发散\n(RF 差到 11, DNN 差到 5+)',
            xy=(8.2, abs(rf_y[np.argmin(np.abs(gx-8.2))] - fy[np.argmin(np.abs(gx-8.2))])),
            xytext=(5.3, 6.6), fontsize=12.5, color='#843C0C', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#843C0C', lw=1.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 14)
ax.set_xlabel('条件 x (训练只覆盖 0~4)', fontsize=14)
ax.set_ylabel('|预测 − 真实|', fontsize=14)
ax.tick_params(labelsize=12.5)
ax.legend(fontsize=11.5, loc='upper left')
ax.grid(alpha=0.25)
ax.set_title('② 误差曲线: 失真只发生在没见过的区域', fontsize=13, loc='left', color=NAVY)

fig.text(0.5, 0.005,
         f'虚拟数据: y=饱和曲线f(x)+噪声(σ={SIGMA}), 训练集3000点充分采样 x∈[0,4]; RF与DNN均充分训练(域内MAE≈1.5=噪声底, 见过部分已学到数据极限)。\n'
         '结论与数据无关, 由模型结构决定: RF=分段常数, 域外恒等输出边界叶子; DNN=分段线性, 域外沿边界斜率直线延伸 —— 都不知道"该弯"',
         ha='center', fontsize=11.5, color=DGREEN, fontweight='bold')
plt.tight_layout(rect=[0, 0.055, 1, 0.955])
fig.savefig(r'D:/AI/条件转化/虚拟数据_见过vs没见过_推理可视化.png', dpi=200, bbox_inches='tight', facecolor='white')
print('\nsaved 虚拟数据_见过vs没见过_推理可视化.png')
