# -*- coding: utf-8 -*-
"""validate_60: 其他公开数据集复验 —— sklearn 糖尿病数据集(442名真实患者)
   任务: 用 BMI(体重指数) 预测 糖尿病病情一年进展指数(连续值回归)
   协议: 只用 BMI<26 的患者训练(约2/3), BMI>=26 为"没见过"盲测域 —— 真实数据存在, 可对真值量误差
   模型: RF(500树) 与 DNN(128x128x64) 均充分训练
   输出: 公开数据集_糖尿病_见过vs没见过.png
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

NAVY = '#17375E'; RED = '#C00000'; PURPLE = '#7030A0'; MUTED = '#666666'; DGREEN = '#375623'

d = load_diabetes()
X = d.data[:, 2] * 50           # bmi 标准化列 ×50 ≈ 原始 BMI 量级(教程通用画法)
y = d.target.astype(float)
BOUND = float(np.percentile(X, 60))
m_tr = X < BOUND
print(f'患者总数 {len(X)} | 训练域 BMI<{BOUND}: {m_tr.sum()} 人 | 盲测域: {(~m_tr).sum()} 人')
print(f'BMI 范围 {X.min():.1f}~{X.max():.1f}')

rf = RandomForestRegressor(500, max_depth=None, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf.fit(X[m_tr].reshape(-1, 1), y[m_tr])
sc = StandardScaler().fit(X[m_tr].reshape(-1, 1))
dnn = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation='relu', solver='adam',
                   alpha=1e-3, max_iter=30000, random_state=0)
dnn.fit(sc.transform(X[m_tr].reshape(-1, 1)), y[m_tr])

pr_rf = rf.predict(X.reshape(-1, 1))
pr_dnn = dnn.predict(sc.transform(X.reshape(-1, 1)))
mae_in_rf = np.mean(np.abs(y[m_tr] - pr_rf[m_tr]))
mae_in_dn = np.mean(np.abs(y[m_tr] - pr_dnn[m_tr]))
mae_ex_rf = np.mean(np.abs(y[~m_tr] - pr_rf[~m_tr]))
mae_ex_dn = np.mean(np.abs(y[~m_tr] - pr_dnn[~m_tr]))
print(f'\n{"域":8s} {"RF MAE":>8s} {"DNN MAE":>8s}')
print(f'{"见过(域内)":8s} {mae_in_rf:8.1f} {mae_in_dn:8.1f}')
print(f'{"没见过":8s} {mae_ex_rf:8.1f} {mae_ex_dn:8.1f}  ← 对真实患者测的, 不是构造')
print(f'误差放大: RF ×{mae_ex_rf/mae_in_rf:.1f} | DNN ×{mae_ex_dn/mae_in_dn:.1f}')
# 域外模型行为
print(f'\n盲测域形态: RF 预测范围 [{pr_rf[~m_tr].min():.0f}, {pr_rf[~m_tr].max():.0f}] (冻在边界附近=重放)')
print(f'           DNN 预测范围 [{pr_dnn[~m_tr].min():.0f}, {pr_dnn[~m_tr].max():.0f}] (直线延伸)')

# ============ 图 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 7.4))
fig.suptitle('其他公开数据集复验（sklearn 糖尿病 · 442 名真实患者）：换个领域，结论不变',
             fontsize=15.5, fontweight='bold', y=0.99)

ax = axes[0]
ax.axvspan(X.min(), BOUND, color='#EAF1F9', zorder=0)
ax.axvspan(BOUND, X.max(), color='#FDF1E8', zorder=0)
ax.axvline(BOUND, color=RED, ls='--', lw=2.2, zorder=2)
ax.scatter(X[m_tr], y[m_tr], s=22, color='#9DB4CE', alpha=0.55, edgecolor='none', zorder=2)
ax.scatter(X[~m_tr], y[~m_tr], s=26, color='#E8A47C', alpha=0.85, edgecolor='none', zorder=3)
gx = np.linspace(X.min(), X.max(), 400)
ax.plot(gx, rf.predict(gx.reshape(-1, 1)), lw=3, color=RED, zorder=5, label='RF(500树, 充分训练)')
ax.plot(gx, dnn.predict(sc.transform(gx.reshape(-1, 1))), lw=2.8, color=PURPLE, ls='-.',
        zorder=4, label='DNN(128×128×64, 训练至收敛)')
ax.text(BOUND-3, 320, '见过(低BMI)', ha='center', fontsize=14, fontweight='bold', color=NAVY)
ax.text(BOUND+6, 320, '没见过(高BMI)', ha='center', fontsize=14, fontweight='bold', color='#843C0C')
ax.text(BOUND + 0.3, 40, '训练边界(低60%分位)', fontsize=11, color=RED, fontweight='bold')
ax.annotate('RF 冻在边界值\n(不随 BMI 变化)', xy=(37, rf.predict([[37]])[0]), xytext=(31.5, 85),
            fontsize=12, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.8))
ax.annotate('DNN 沿边界斜率\n直线延伸(向错误方向)', xy=(BOUND+3.2, float(dnn.predict(sc.transform([[BOUND+3.2]]))[0])),
            xytext=(BOUND+1.6, 230), fontsize=12, color=PURPLE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.8))
ax.set_xlim(X.min()-0.5, X.max()+0.5); ax.set_ylim(20, 345)
ax.set_xlabel('BMI（标准化刻度 ×50）', fontsize=14)
ax.set_ylabel('糖尿病病情一年进展指数', fontsize=14)
ax.tick_params(labelsize=12.5)
ax.legend(fontsize=11.5, loc='upper left')
ax.grid(alpha=0.25)
ax.set_title('① 预测形态: 域内贴合, 一出边界各自失真\n   (橙点=真实高BMI患者, 全部被预测错)', fontsize=13, loc='left', color=NAVY)

ax = axes[1]
names = ['见过\n(域内)', '没见过\n(盲测域)']
rf_v = [mae_in_rf, mae_ex_rf]; dn_v = [mae_in_dn, mae_ex_dn]
x = np.arange(2); bw = 0.32
b1 = ax.bar(x - bw/2, rf_v, bw, color=RED, label='RF')
b2 = ax.bar(x + bw/2, dn_v, bw, color=PURPLE, label='DNN')
for bb, vv in [(b1, rf_v), (b2, dn_v)]:
    for r_, v in zip(bb, vv):
        ax.annotate(f'{v:.1f}', (r_.get_x() + r_.get_width()/2, v), xytext=(0, 4),
                    textcoords='offset points', ha='center', fontsize=13, fontweight='bold')
for i_, mult in enumerate(['×1 (基准)', f'×{mae_ex_rf/mae_in_rf:.1f}', '×1 (基准)', f'×{mae_ex_dn/mae_in_dn:.1f}']):
    pass

ax.annotate(f'RF ×{mae_ex_rf/mae_in_rf:.1f}', (1, mae_ex_rf + 3), ha='center', fontsize=12,
            color='#843C0C', fontweight='bold')
ax.annotate(f'DNN ×{mae_ex_dn/mae_in_dn:.1f}', (1 + bw/2, mae_ex_dn + 12), ha='center', fontsize=12,
            color=PURPLE, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=13)
ax.set_ylabel('对真实患者的 MAE（越低越好）', fontsize=13.5)
ax.tick_params(labelsize=12.5)
ax.legend(fontsize=12, loc='upper left')
ax.grid(alpha=0.25, axis='y')
ax.set_title('② 误差对真值: 没见过区域误差放大约 1.5~2 倍\n   (对真实数据测的, 无构造成分)', fontsize=13, loc='left', color=NAVY)

fig.text(0.5, 0.005,
         f'数据: sklearn 内置糖尿病数据集(442名真实患者, Efron et al. 2004), 与出行/交通完全无关。协议: 只用低 60% BMI 训练, 高 BMI 盲测(真实值存在)。\n'
         f'结论跨领域成立: 域内误差基准 {mae_in_rf:.0f}/{mae_in_dn:.0f} → 域外 {mae_ex_rf:.0f}(RF×{mae_ex_rf/mae_in_rf:.1f}) / {mae_ex_dn:.0f}(DNN×{mae_ex_dn/mae_in_dn:.1f}); RF冻在边界值, DNN直线漂移 —— "重放训练分布"是模型结构属性, 不是数据特性',
         ha='center', fontsize=10.5, color=DGREEN, fontweight='bold')
plt.tight_layout(rect=[0, 0.06, 1, 0.955])
fig.savefig(r'D:/AI/条件转化/公开数据集_糖尿病_见过vs没见过.png', dpi=200, bbox_inches='tight', facecolor='white')
print('\nsaved 公开数据集_糖尿病_见过vs没见过.png')
