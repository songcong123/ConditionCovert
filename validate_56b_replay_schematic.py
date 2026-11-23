# -*- coding: utf-8 -*-
"""示意图: 模型只会'重放'训练分布 → 极端场景严重失真(概念图, 非真实数据)
   三要素: 真实系统响应(持续上升) / 训练分布覆盖区(只有左侧) / 模型输出(域内贴、域外躺平)
   红色阴影 = 失真区。输出: 重放训练分布_挑战示意图.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#17375E'; RED = '#C00000'; GREY = '#8C8C8C'

rng = np.random.default_rng(7)
r = np.linspace(0.2, 30, 400)
true_f = lambda x: 26 * np.log1p(x / 0.8) / np.log1p(30 / 0.8)          # 归一化到0~26

# 训练数据(只覆盖 0.2~4)
r_tr = rng.uniform(0.2, 4, 200)
y_tr = true_f(r_tr) + rng.normal(0, 1.1, 200)

fig, ax = plt.subplots(figsize=(13, 7.4))

# 训练覆盖区底色 + 极端区底色
ax.axvspan(-1, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 32, color='#FDECEC', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2.4, zorder=2)

# 失真区阴影(真实 − 模型 之间)
plat = true_f(4) * 0.97
ax.fill_between(r[r >= 4], plat, true_f(r[r >= 4]), color=RED, alpha=0.16, zorder=1)

# 真实系统响应
ax.plot(r, true_f(r), lw=3.6, color=NAVY, zorder=5)
# 模型输出1(RF类): 域内贴合, 域外躺平
r_in = r[r <= 4.05]
model_in = true_f(r_in) * 0.96
ax.plot(r_in, model_in, lw=3.2, color=RED, zorder=5)
ax.plot([4, 30], [plat, plat], lw=3.2, color=RED, zorder=5)
# 模型输出2(DNN类): 域内贴合, 域外沿边界斜率线性冲过头
PURPLE = '#7030A0'
slope = (38 - plat) / 26.0
dnn_out = plat + slope * np.maximum(r - 4, 0)
dnn_curve = np.where(r <= 4.05, true_f(r) * 0.95, dnn_out)
ax.plot(r, dnn_curve, lw=2.8, color=PURPLE, ls='-.', zorder=4)

# 训练散点
ax.scatter(r_tr, y_tr, s=22, color='#7FA6CC', alpha=0.6, edgecolor='none', zorder=3)

# 区域大标签
ax.text(1.5, 38.6, '训练分布覆盖区', ha='center', fontsize=17, fontweight='bold', color=NAVY)
ax.text(1.5, 35.9, '(模型在这里表现好)', ha='center', fontsize=13, color='#666666')
ax.text(17, 38.6, '极端场景(训练未覆盖)', ha='center', fontsize=17, fontweight='bold', color=RED)
ax.text(17, 35.9, '(两种黑盒都严重失真, 错法不同)', ha='center', fontsize=13, color='#666666')

# 曲线标签
ax.text(7.6, 21.5, '真实系统响应\n(还在持续上升)', fontsize=14.5, fontweight='bold', color=NAVY)
ax.text(6.2, 8.8, 'RF：躺平(冻在边界值)', fontsize=14, fontweight='bold', color=RED)
ax.text(15.5, 31.5, 'DNN：冲过头\n(沿边界斜率冲出去)', fontsize=14, fontweight='bold', color=PURPLE)

# 失真标注(右下空白区, 箭头指向失真区)
ax.annotate('严重失真区\n(RF 需要给 +26, 只给 +11\n差 15 个百分点)', xy=(20, (true_f(20) + plat) / 2),
            xytext=(15.2, 2.2), fontsize=13.5, fontweight='bold', color='#8B0000',
            arrowprops=dict(arrowstyle='->', color='#8B0000', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=RED, lw=1.6))
ax.annotate('', xy=(27.5, true_f(27.5)), xytext=(27.5, plat),
            arrowprops=dict(arrowstyle='<->', color='#8B0000', lw=3))

ax.text(4.4, -1.8, '训练边界', fontsize=13, color=RED, fontweight='bold')

ax.set_xlim(-1, 32)
ax.set_ylim(-3, 42)
ax.set_xlabel('条件强度(如雨强)', fontsize=15)
ax.set_ylabel('系统响应(如出行量效应 %)', fontsize=15)
ax.tick_params(labelsize=13)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/重放训练分布_挑战示意图.png', dpi=200, bbox_inches='tight', facecolor='white')
print('saved 重放训练分布_挑战示意图.png')
