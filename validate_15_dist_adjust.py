# -*- coding: utf-8 -*-
"""距离分布的调整过程(白话版): 基线 → 三个旋钮逐个拧 → 合成暴雨 → 与实测暴雨对照
旋钮1: 短途的典型距离缩短 (μ₁ 左移)     —— 雨天短途更短
旋钮2: 短途的人变多 (w₁ 权重↑)          —— 原本走路的人改打车
旋钮3: 机场的人变少 (机场份额 ×ρ_ap)    —— 航班延误、长途受抑
"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats

A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
B = json.load(open('D:/AI/条件转化/.flow_scores.json'))['B3']

dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
dist = A['dist'].astype(float)
dd_dry = dist[np.isin(A['dist_day'].astype(int), np.where(kl == 'dry')[0])]
dd_heavy = dist[np.isin(A['dist_day'].astype(int), np.where(kl == 'heavy')[0])]

def mix_pdf(x, P):
    p = (P['w1'] * stats.lognorm.pdf(x, P['sd1'], scale=np.exp(P['mu1']))
         + P['w2'] * stats.lognorm.pdf(x, P['sd2'], scale=np.exp(P['mu2']))
         + P['w_ap'] * stats.lognorm.pdf(x, P['sd_ap'], scale=np.exp(P['mu_ap'])))
    p += np.where(x >= P['xmin'], P['w_pa'] * P['alpha'] / P['xmin'] * (x / P['xmin']) ** (-(P['alpha'] + 1)), 0)
    return p

def sample_mix(P, n, rng):
    u = rng.random(n)
    c1 = P['w1']; c2 = c1 + P['w2']; c3 = c2 + P['w_ap']
    x = np.empty(n)
    m1 = u < c1; m2 = (u >= c1) & (u < c2); m3 = (u >= c2) & (u < c3); m4 = ~(m1 | m2 | m3)
    x[m1] = rng.lognormal(P['mu1'], P['sd1'], m1.sum())
    x[m2] = rng.lognormal(P['mu2'], P['sd2'], m2.sum())
    x[m3] = rng.lognormal(P['mu_ap'], P['sd_ap'], m3.sum())
    x[m4] = P['xmin'] * (1 - rng.random(m4.sum())) ** (-1 / P['alpha'])
    return np.clip(x, 0.1, 100)

x = np.geomspace(0.05, 40, 700)
P0 = dict(B)
P1 = dict(B); P1['mu1'] -= 0.061                    # 旋钮1: μ₁ 左移(中位-6%)
P2 = dict(P1); P2['w1'] += 0.025                    # 旋钮2: 短途权重 +2.5pp
P3 = dict(P2); P3['w_ap'] *= 0.90                   # 旋钮3: 机场份额 ×0.90
for P in (P2, P3):                                   # 归一化
    s = P['w1'] + P['w2'] + P['w_ap'] + P['w_pa']
    P['w1'], P['w2'], P['w_ap'], P['w_pa'] = P['w1']/s, P['w2']/s, P['w_ap']/s, P['w_pa']/s

rng = np.random.default_rng(7)
ks = stats.ks_2samp(sample_mix(P3, 200000, rng),
                    rng.choice(dd_heavy[(dd_heavy > 0.05) & (dd_heavy < 80)], 200000, replace=False)).statistic

fig = plt.figure(figsize=(18, 9.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
fig.suptitle('行程距离的调整过程：不是整体平移，而是拧三个旋钮（晴天 → 暴雨）',
             fontsize=15, fontweight='bold', y=0.99)

# A 基线
ax = fig.add_subplot(gs[0, 0])
ax.hist(dd_dry[(dd_dry > 0.05) & (dd_dry < 40)], bins=np.geomspace(0.05, 40, 60), density=True, alpha=0.30, color='#BF9000')
ax.plot(x, mix_pdf(x, P0), lw=2.6, color='#833C0B')
ax.plot(x, P0['w1'] * stats.lognorm.pdf(x, P0['sd1'], scale=np.exp(P0['mu1'])), lw=1.4, ls='--', color='#C55A11')
ax.plot(x, P0['w_ap'] * stats.lognorm.pdf(x, P0['sd_ap'], scale=np.exp(P0['mu_ap'])), lw=1.6, ls='--', color='#C00000')
ax.text(13.5, 0.075, '机场', fontsize=10, color='#C00000')
ax.text(1.9, 0.42, '短途峰\n(61%)', fontsize=10, color='#C55A11')
ax.set_xscale('log')
ax.set_title('A. 晴天基线：四分量', fontsize=12, loc='left', color='#833C0B')
ax.set_xticks([0.1, 1, 10, 40]); ax.set_xticklabels(['0.1', '1', '10', '40'])
ax.grid(alpha=0.3, which='both'); ax.set_xlabel('距离 (mi)')

# B 旋钮1
ax = fig.add_subplot(gs[0, 1])
ax.plot(x, mix_pdf(x, P0), lw=1.8, color='#BFBFBF', ls='--', label='晴天')
ax.plot(x, mix_pdf(x, P1), lw=2.6, color='#ED7D31')
ax.axvline(1.76, color='#BFBFBF', ls=':'); ax.axvline(np.exp(B['mu1'] - 0.061), color='#C00000', ls=':')
ax.annotate('', xy=(np.exp(B['mu1'] - 0.061), 0.40), xytext=(1.76, 0.40),
            arrowprops=dict(arrowstyle='->', lw=2, color='#C00000'))
ax.text(1.5, 0.425, '峰左移 1.76→1.66mi', fontsize=10, color='#C00000')
ax.set_xscale('log')
ax.set_title('B. 旋钮①：短途典型距离缩短\n（μ₁ 左移，中位 −6%）', fontsize=12, loc='left', color='#833C0B')
ax.set_xticks([0.1, 1, 10, 40]); ax.set_xticklabels(['0.1', '1', '10', '40'])
ax.grid(alpha=0.3, which='both'); ax.set_xlabel('距离 (mi)')

# C 旋钮2
ax = fig.add_subplot(gs[0, 2])
ax.plot(x, mix_pdf(x, P1), lw=1.8, color='#BFBFBF', ls='--', label='只动旋钮①后')
ax.plot(x, mix_pdf(x, P2), lw=2.6, color='#ED7D31')
ax.fill_between(x[(x > 0.3) & (x < 2.2)], 0, mix_pdf(x, P2)[(x > 0.3) & (x < 2.2)] - mix_pdf(x, P1)[(x > 0.3) & (x < 2.2)],
                color='#ED7D31', alpha=0.35)
ax.text(0.65, 0.10, '短途份额\n61%→64.7%\n(+2.5pp)', fontsize=10, color='#C55A11')
ax.set_xscale('log')
ax.set_title('C. 旋钮②：短途的人变多\n（原走路改打车, w₁↑）', fontsize=12, loc='left', color='#833C0B')
ax.set_xticks([0.1, 1, 10, 40]); ax.set_xticklabels(['0.1', '1', '10', '40'])
ax.grid(alpha=0.3, which='both'); ax.set_xlabel('距离 (mi)')

# D 旋钮3
ax = fig.add_subplot(gs[0, 3])
ax.plot(x, mix_pdf(x, P2), lw=1.8, color='#BFBFBF', ls='--')
ax.plot(x, mix_pdf(x, P3), lw=2.6, color='#ED7D31')
ax.plot(x, P2['w_ap'] * stats.lognorm.pdf(x, P2['sd_ap'], scale=np.exp(P2['mu_ap'])), lw=1.5, ls='--', color='#C00000')
ax.plot(x, P3['w_ap'] * stats.lognorm.pdf(x, P3['sd_ap'], scale=np.exp(P3['mu_ap'])), lw=2.2, color='#C00000')
ax.annotate('机场鼓包压扁\n份额 ×0.90', xy=(13, 0.062), xytext=(3.2, 0.115), fontsize=10, color='#C00000',
            arrowprops=dict(arrowstyle='->', color='#C00000'))
ax.set_xscale('log')
ax.set_title('D. 旋钮③：机场的人变少\n（航班延误, ×ρ_ap）', fontsize=12, loc='left', color='#833C0B')
ax.set_xticks([0.1, 1, 10, 40]); ax.set_xticklabels(['0.1', '1', '10', '40'])
ax.grid(alpha=0.3, which='both'); ax.set_xlabel('距离 (mi)')

# E 合成 vs 实测
ax = fig.add_subplot(gs[1, :])
bins = np.geomspace(0.05, 40, 80)
ax.hist(dd_dry[(dd_dry > 0.05) & (dd_dry < 40)], bins=bins, density=True, alpha=0.30, color='#A9D18E', label='晴天（实测）')
ax.hist(dd_heavy[(dd_heavy > 0.05) & (dd_heavy < 40)], bins=bins, density=True, alpha=0.30, color='#4472C4', label='暴雨日（实测, 52天）')
ax.plot(x, mix_pdf(x, P3), lw=2.8, color='#C00000', label='暴雨（合成：三旋钮拧完的曲线）')
ax.set_xscale('log')
ax.set_xlabel('行程距离 (mi, log轴)'); ax.set_ylabel('密度')
ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20, 40]); ax.set_xticklabels(['0.1', '0.5', '1', '2', '5', '10', '20', '40'])
ax.legend(fontsize=11, loc='upper right'); ax.grid(alpha=0.3, which='both')
ax.set_title(f'E. 验证：三个旋钮合成暴雨曲线 vs 真实暴雨日 ——  KS = {ks:.3f}（虚线基线不调时 KS≈0.24）',
             fontsize=13, loc='left', color='#833C0B')

plt.tight_layout()
plt.savefig('D:/AI/条件转化/距离调整三旋钮.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f'saved 距离调整三旋钮.png | 合成vs实测 KS={ks:.4f}')
