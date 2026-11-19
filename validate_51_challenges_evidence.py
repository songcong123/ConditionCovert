# -*- coding: utf-8 -*-
"""validate_51: 轨迹生成四大挑战 × 我们数据里的实证(配图版)
   A 起点热点集中/长尾覆盖不足 → NYC 264区出发量 Lorenz 曲线(真实数据)
   B 平均轨迹丢多样性        → 干日距离四分量 vs 平均距离竖线(真实数据)
   C GAN 模式坍塌            → 构造示意: 多峰真实 vs 坍塌单峰 vs 扩散恢复
   D 空间建模不足/POI        → 264区空间散点(点大小=出发量) + γ 实测标注
   输出: 轨迹生成挑战_实证配图.png
"""
import numpy as np
import json
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; DGREEN = '#375623'; MUTED = '#666666'; ORANGE = '#ED7D31'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
Z = np.load(r'D:/AI/条件转化/.flow_zones.npz')
SP = np.load(r'D:/AI/条件转化/.flow_spatial.npz')
B3 = json.load(open(r'D:/AI/条件转化/.flow_scores.json'))['B3']

dates = list(A['dates'])
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)

fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
fig.suptitle('轨迹生成四大挑战 · 我们数据里的实证：三个真、一个真但过时',
             fontsize=15, fontweight='bold', y=0.985)

# ============ A: 起点热点集中/长尾覆盖不足(真挑战) ============
ax = axes[0, 0]
pu = SP['nov_pu'].sum(axis=(0, 1))                    # 264 区出发量(5天池化)
pu_s = np.sort(pu)[::-1]
cum = np.cumsum(pu_s) / pu_s.sum()
xs = np.arange(1, len(pu_s) + 1) / len(pu_s)
ax.plot(xs, cum * 100, lw=3, color=NAVY)
ax.plot([0, 1], [0, 100], ls=':', color='#999999', lw=1.6)
k20 = int(len(pu_s) * 0.2)
share20 = cum[k20 - 1] * 100
share50 = cum[len(pu_s) // 2 - 1] * 100
ax.plot([0.2], [share20], 'o', ms=11, color=RED, zorder=5)
ax.annotate(f'前 20% 的区\n= {share20:.0f}% 出发量', (0.2, share20), xytext=(0.30, share20 - 22),
            fontsize=11, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.6))
ax.plot([0.5], [share50], 's', ms=10, color=ORANGE, zorder=5)
ax.annotate(f'后 50% 的区\n只占 {100 - share50:.0f}%', (0.5, share50), xytext=(0.55, share50 - 18),
            fontsize=11, color='#843C0C', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.6))
ax.text(0.42, 0.30, '"均匀假设"虚线\n(每区 0.4%)', fontsize=9.5, color=MUTED, ha='center')
ax.set_xlabel('区排名(从最热到最冷, 累计比例)'); ax.set_ylabel('累计出发量份额 (%)')
ax.set_title('① 起点热点集中、长尾覆盖不足 —— 真挑战 ✓\n   NYC 264 区出发量: 头部极重、半数区长尾', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.3)

# ============ B: 平均轨迹丢多样性(真挑战) ============
ax = axes[0, 1]
dry_i = np.where(kl == 'dry')[0]
dd = dist[np.isin(dist_day, dry_i)]; dd = dd[(dd > 0.3) & (dd < 60)]
w_sum = B3['w1'] + B3['w2'] + B3['w_ap'] + B3['w_pa']
w1, w2, wap = B3['w1']/w_sum, B3['w2']/w_sum, B3['w_ap']/w_sum
gx = np.linspace(0.3, 30, 500)
comp1 = w1 * stats.lognorm.pdf(gx, B3['sd1'], scale=np.exp(B3['mu1']))
comp2 = w2 * stats.lognorm.pdf(gx, B3['sd2'], scale=np.exp(B3['mu2']))
comp3 = wap * stats.lognorm.pdf(gx, B3['sd_ap'], scale=np.exp(B3['mu_ap']))
ax.hist(dd, bins=np.linspace(0.3, 30, 80), density=True, alpha=0.25, color='#8FAADC')
ax.plot(gx, comp1, lw=2, color='#4472C4'); ax.plot(gx, comp2, lw=2, color='#70AD47')
ax.plot(gx, comp3, lw=2, color=ORANGE)
ax.plot(gx, comp1 + comp2 + comp3, lw=3, color=NAVY)
mean_d = float(dd.mean())
ax.axvline(mean_d, color=RED, ls='--', lw=2.4)
ax.text(mean_d + 0.4, max(comp1 + comp2 + comp3) * 0.72,
        f'平均距离 {mean_d:.1f} mi\n落在谷底——\n那里几乎没人出发', fontsize=10.5, color=RED, fontweight='bold')
ax.text(1.1, max(comp1+comp2+comp3)*0.42, f'短途人群\n{w1*100:.0f}%', fontsize=10, color='#4472C4', fontweight='bold')
ax.text(4.6, max(comp1+comp2+comp3)*0.22, f'中途\n{w2*100:.0f}%', fontsize=10, color='#70AD47', fontweight='bold')
ax.text(13.5, max(comp1+comp2+comp3)*0.13, f'长途+机场\n{(1-w1-w2)*100:.0f}%', fontsize=10, color=ORANGE, fontweight='bold')
ax.set_xlim(0, 30)
ax.set_xlabel('单程距离 (mi)'); ax.set_ylabel('密度')
ax.set_title('② "平均轨迹"骗人 —— 真挑战 ✓\n   距离是混合分布: 平均值落在人群之间的谷底, 谁也不代表', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.3)

# ============ C: GAN 模式坍塌(真但已过时) ============
ax = axes[1, 0]
real = comp1 + comp2 + comp3
peak_i = int(np.argmax(comp1))
mu_c = gx[peak_i]
coll = stats.lognorm.pdf(gx, 0.45, scale=np.exp(np.log(mu_c))) * 1.0
diff = real * 1.0
ax.fill_between(gx, 0, real, color='#D6E4F0', alpha=0.6)
ax.plot(gx, real, lw=3, color=NAVY, label='真实分布(三个人群)')
ax.fill_between(gx, 0, coll * w1 * 2.6, color='#F4B6B6', alpha=0.55)
ax.plot(gx, coll * w1 * 2.6, lw=2.6, color=RED, label='GAN 坍塌(只生成最常见的一类)')
ax.plot(gx, diff * 0.98, lw=2.2, ls='--', color=DGREEN, label='扩散模型 2023+(基本恢复全覆盖)')
ax.text(0.55, real.max()*0.30, '中/长途人群\n整体消失', fontsize=10.5, color=RED, fontweight='bold')
ax.annotate('', xy=(6.5, comp2.max()*0.9), xytext=(1.6, comp1.max()*0.12),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2))
ax.set_xlim(0, 30); ax.set_ylim(0, real.max() * 1.15)
ax.set_xlabel('单程距离 (mi, 同上分布)'); ax.set_ylabel('密度')
ax.set_title('③ GAN 模式坍塌 —— 真挑战但已过时 ⚠\n   2023 后扩散路线(DiffTraj 等)基本解决; 剩下的真问题=条件外推', fontsize=12, loc='left', color=NAVY)
ax.legend(fontsize=9.5, loc='upper right')
ax.grid(alpha=0.3)

# ============ D: 空间建模不足/POI(场景相关) ============
ax = axes[1, 1]
lat = Z['lat']; lon = Z['lon']
sz = (pu / pu.max()) * 260 + 6
sc = ax.scatter(lon, lat, s=sz, c=np.log10(pu + 1), cmap='YlOrRd', edgecolor='#843C0C',
                linewidth=0.5, alpha=0.9)
plt.colorbar(sc, ax=ax, label='出发量 (log 刻度)', shrink=0.85)
ax.text(0.03, 0.97, '空间远非均匀网格:\n点大小/颜色=出发量\n热岛+大片长尾区',
        transform=ax.transAxes, fontsize=10.5, va='top', fontweight='bold', color='#843C0C',
        bbox=dict(boxstyle='round,pad=0.4', fc='#FFF2CC', ec='#D6B656', alpha=0.92))
ax.text(0.62, 0.10, 'POI 吸引力实证(雨时 γ):\n机场 −27% | 核心 +6% | 梯度 −0.66pp/km\n→ 均匀网格假设不成立',
        transform=ax.transAxes, fontsize=10, va='bottom', color=DGREEN, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=DGREEN, alpha=0.92))
ax.set_xlabel('经度'); ax.set_ylabel('纬度')
ax.set_title('④ 空间建模不足 / POI —— 场景相关 ⚠\n   POI 部分为真(有 γ 实测); "道路约束"对蜂窝级业务弱相关', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.3)

fig.text(0.5, 0.004,
         '数据: NYC 2.61 亿行真实行程(2024-2025)。①④为真实数据实测; ②为真实分布+四分量拟合(B3参数); ③为机制示意(基于②的真实多峰形态构造)。\n'
         '汇报口径: ①②是当前真痛点(我们的查表+混合分布建模直接回应); ③曾是痛点、扩散模型已缓解——别把过时问题当主矛盾; ④按业务取舍(POI真、路网弱)',
         ha='center', fontsize=9.5, color=DGREEN, fontweight='bold')
plt.tight_layout(rect=[0, 0.045, 1, 0.965])
fig.savefig(r'D:/AI/条件转化/轨迹生成挑战_实证配图.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f'A: 前20%区={share20:.0f}% | 后50%区={100-share50:.0f}%')
print(f'B: 平均距离={mean_d:.2f} mi | w1={w1:.2f} w2={w2:.2f}')
print('saved 轨迹生成挑战_实证配图.png')
