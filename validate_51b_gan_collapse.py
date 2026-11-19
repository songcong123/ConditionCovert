# -*- coding: utf-8 -*-
"""validate_51b: GAN 模式坍塌 单图讲解版(从 validate_51 面板③抽出放大)
   真实分布 = NYC 干日距离四分量(三个人群) —— 模式坍塌的'模式'就是这三类人群
   输出: GAN坍塌_单图.png
"""
import numpy as np
import json
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
B3 = json.load(open(r'D:/AI/条件转化/.flow_scores.json'))['B3']
dates = list(A['dates'])
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
dry_i = np.where(kl == 'dry')[0]
dd = dist[np.isin(dist_day, dry_i)]; dd = dd[(dd > 0.3) & (dd < 60)]

w_sum = B3['w1'] + B3['w2'] + B3['w_ap'] + B3['w_pa']
w1, w2, wap = B3['w1']/w_sum, B3['w2']/w_sum, B3['w_ap']/w_sum
gx = np.linspace(0.3, 30, 600)
comp1 = w1 * stats.lognorm.pdf(gx, B3['sd1'], scale=np.exp(B3['mu1']))
comp2 = w2 * stats.lognorm.pdf(gx, B3['sd2'], scale=np.exp(B3['mu2']))
comp3 = wap * stats.lognorm.pdf(gx, B3['sd_ap'], scale=np.exp(B3['mu_ap']))
real = comp1 + comp2 + comp3

fig, ax = plt.subplots(figsize=(12, 7.2))
fig.suptitle('GAN 模式坍塌：只会生成"最常见的一类"，整个人群结构塌掉',
             fontsize=15, fontweight='bold', y=0.97)

ax.fill_between(gx, 0, real, color='#D6E4F0', alpha=0.55)
ax.plot(gx, real, lw=3.2, color=NAVY, label='真实分布：三类人群共存（NYC 实测四分量）', zorder=5)
peak_i = int(np.argmax(comp1))
coll = stats.lognorm.pdf(gx, 0.45, scale=np.exp(np.log(gx[peak_i]))) * w1 * 2.75
ax.fill_between(gx, 0, coll, color='#F4B6B6', alpha=0.6)
ax.plot(gx, coll, lw=2.8, color=RED, label='坍塌的 GAN：只覆盖最常见人群（短途）', zorder=6)
ax.plot(gx, real * 0.97, lw=2.2, ls='--', color=DGREEN, label='扩散模型(2023+, DiffTraj 等)：基本恢复全覆盖', zorder=4)

# 人群标注
ax.text(1.15, comp1.max()*0.55, f'短途人群 {w1*100:.0f}%', fontsize=11.5, color='#2E5A88', fontweight='bold')
ax.text(6.8, comp2.max()*0.9, f'中途 {w2*100:.0f}%', fontsize=11.5, color='#4E7C41', fontweight='bold')
ax.text(14.5, comp3.max()*0.75, f'长途/机场 {(1-w1-w2)*100:.0f}%', fontsize=11.5, color='#9C5708', fontweight='bold')

# 坍塌区标注
ax.annotate('模式坍塌:\n中/长途两个人群整体消失\n(生成 1000 条 → 几乎全是短途)',
            xy=(7.5, comp2.max()*0.45), xytext=(12.5, comp1.max()*0.42), fontsize=11.5, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=2.2),
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec=RED, lw=1.4))

ax.set_xlim(0, 30); ax.set_ylim(0, real.max() * 1.18)
ax.set_xlabel('单程距离 (mi)', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.grid(alpha=0.3)

fig.text(0.5, 0.008,
         '机制: GAN 的判别器只惩罚"不像真的", 生成器最安全的策略 = 只产最常见的那一类 → 长尾模式全部丢失; 训练还不稳定(判别器过强即崩)\n'
         '为什么扩散模型解决了: 去噪目标逐点学习全部数据、没有对抗博弈, 模式覆盖是结构保证 —— 所以这条挑战"真但已过时", 别当主矛盾;\n'
         '真正没解决的是: 生成模型只会重放训练分布, 没见过的条件(极端天气)生成不了 —— 那是条件外推问题, 与 GAN 坍塌是两回事',
         ha='center', fontsize=10, color=DGREEN)
plt.tight_layout(rect=[0, 0.075, 1, 0.955])
fig.savefig(r'D:/AI/条件转化/GAN坍塌_单图.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved GAN坍塌_单图.png')
