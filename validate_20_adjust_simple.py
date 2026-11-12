# -*- coding: utf-8 -*-
"""调整过程示例（三行版）：每行一个量，左=基线、右=扰动后+验证"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats

A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load('D:/AI/条件转化/.flow_spd.npz')
B3 = json.load(open('D:/AI/条件转化/.flow_scores.json'))['B3']

dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
dry_i = np.where(kl == 'dry')[0]; he_i = np.where(kl == 'heavy')[0]
hour_cnt = A['hour_cnt'].astype(float)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)

fig, axes = plt.subplots(3, 2, figsize=(16, 13))
fig.suptitle('三个量的调整过程：基线 → 暴雨扰动', fontsize=16, fontweight='bold', y=0.995)
plt.subplots_adjust(hspace=0.38, wspace=0.25)

# ================= 第一行：出发时间 =================
hd = hour_cnt[dry_i].mean(0) / 1000
hh = hour_cnt[he_i].mean(0) / 1000
x = np.arange(24)
axL = axes[0][0]; axR = axes[0][1]
axL.bar(x, hd, color='#DAE8FC', edgecolor='#4472C4', lw=0.6)
axL.set_title('基线（晴天）', fontsize=12, loc='left', color='#4472C4')
axL.set_ylabel('出发量（千单/时）'); axL.set_xlabel('时刻'); axL.set_xticks(range(0,25,4))
axL.grid(alpha=0.3)
axR.bar(x, hd, color='#DAE8FC', edgecolor='#4472C4', lw=0.6, alpha=0.5, label='晴天基线')
axR.plot(x, hh, 'o-', color='#C00000', lw=2.5, ms=4, label='暴雨（实测52天）')
axR.fill_between(x, hd, hh, where=hh>hd, color='#C00000', alpha=0.15)
axR.set_title('暴雨：整体抬高（+6.8%），峰形不变', fontsize=12, loc='left', color='#C00000')
axR.set_xlabel('时刻'); axR.set_xticks(range(0,25,4)); axR.legend(fontsize=10); axR.grid(alpha=0.3)
axes[0][0].text(-0.1, 1.06, '① 出发时间', transform=axes[0][0].transAxes,
                fontsize=14, fontweight='bold', color='#4472C4')

# ================= 第二行：速度 =================
v = S['v'].astype(float); vd = S['day'].astype(int)
v_d = v[np.isin(vd, dry_i)]; v_h = v[np.isin(vd, he_i)]
rng = np.random.default_rng(2)
v_ds = rng.choice(v_d, 200000, replace=False)
v_hs = rng.choice(v_h, min(len(v_h), 200000), replace=False)
bins = np.arange(0, 40, 0.5)
axL = axes[1][0]; axR = axes[1][1]
axL.hist(v_ds, bins=bins, density=True, alpha=0.5, color='#70AD47', edgecolor='#507E32', lw=0.3)
axL.set_title('基线（晴天）', fontsize=12, loc='left', color='#70AD47')
axL.set_ylabel('密度'); axL.set_xlabel('全程平均速度（英里/时）'); axL.grid(alpha=0.3)
axR.hist(v_ds, bins=bins, density=True, alpha=0.3, color='#A9D18E', label='晴天基线')
axR.hist(v_hs, bins=bins, density=True, alpha=0.3, color='#C00000', label='暴雨')
for vv, c in [(v_ds, '#70AD47'), (v_hs, '#C00000')]:
    mu_, sd_ = stats.norm.fit(np.log(vv))
    xs = np.linspace(0.5, 40, 400)
    axR.plot(xs, stats.lognorm.pdf(xs, sd_, scale=np.exp(mu_)), lw=2.2, color=c)
med_d, med_h = np.median(v_ds), np.median(v_hs)
axR.axvline(med_d, color='#70AD47', ls=':', lw=1.8)
axR.axvline(med_h, color='#C00000', ls=':', lw=1.8)
axR.annotate('', xy=(med_h, 0.07), xytext=(med_d, 0.07), arrowprops=dict(arrowstyle='->', lw=2.2, color='#404040'))
axR.text((med_d+med_h)/2, 0.075, f'{med_d:.1f} → {med_h:.1f}', ha='center', fontsize=11, fontweight='bold')
axR.set_title('暴雨：整体左移（变慢），形状不变', fontsize=12, loc='left', color='#C00000')
axR.set_xlabel('全程平均速度（英里/时）'); axR.legend(fontsize=10); axR.grid(alpha=0.3)
axes[1][0].text(-0.1, 1.06, '② 速度', transform=axes[1][0].transAxes,
                fontsize=14, fontweight='bold', color='#70AD47')

# ================= 第三行：距离 =================
dd_dry = dist[np.isin(dist_day, dry_i)]
dd_dry = dd_dry[(dd_dry > 0.05) & (dd_dry < 40)]
dd_he = dist[np.isin(dist_day, he_i)]
dd_he = dd_he[(dd_he > 0.05) & (dd_he < 40)]
bins_d = np.geomspace(0.05, 40, 60)
axL = axes[2][0]; axR = axes[2][1]
axL.hist(dd_dry, bins=bins_d, density=True, alpha=0.5, color='#ED7D31', edgecolor='#C55A11', lw=0.3)
axL.set_xscale('log')
axL.set_xticks([0.1, 1, 10, 40]); axL.set_xticklabels(['0.1', '1', '10', '40'])
axL.set_title('基线（晴天）：短途峰 + 机场鼓包 + 长尾', fontsize=12, loc='left', color='#ED7D31')
axL.set_ylabel('密度'); axL.set_xlabel('行程距离（英里，对数轴）'); axL.grid(alpha=0.3, which='both')

# 右图: 暴雨 vs 晴天
def mix_pdf(x, P):
    p = (P['w1'] * stats.lognorm.pdf(x, P['sd1'], scale=np.exp(P['mu1']))
         + P['w2'] * stats.lognorm.pdf(x, P['sd2'], scale=np.exp(P['mu2']))
         + P['w_ap'] * stats.lognorm.pdf(x, P['sd_ap'], scale=np.exp(P['mu_ap'])))
    p += np.where(x >= P['xmin'], P['w_pa'] * P['alpha'] / P['xmin'] * (x / P['xmin']) ** (-(P['alpha'] + 1)), 0)
    return p
P0 = dict(B3)
P3 = dict(B3); P3['mu1'] -= 0.061; P3['w1'] += 0.025; P3['w_ap'] *= 0.90
s_ = P3['w1'] + P3['w2'] + P3['w_ap'] + P3['w_pa']
P3['w1'], P3['w2'], P3['w_ap'], P3['w_pa'] = P3['w1']/s_, P3['w2']/s_, P3['w_ap']/s_, P3['w_pa']/s_
xs = np.geomspace(0.05, 40, 600)
axR.hist(dd_he, bins=bins_d, density=True, alpha=0.25, color='#4472C4', label='真实暴雨日')
axR.plot(xs, mix_pdf(xs, P0), lw=1.8, color='#8C8C8C', ls='--', label='晴天基线')
axR.plot(xs, mix_pdf(xs, P3), lw=2.6, color='#C00000', label='暴雨（调整后）')
axR.set_xscale('log')
axR.set_xticks([0.1, 1, 10, 40]); axR.set_xticklabels(['0.1', '1', '10', '40'])
axR.set_title('暴雨：短途左移+变多，长途×0.9  →  KS=0.024', fontsize=12, loc='left', color='#C00000')
axR.set_xlabel('行程距离（英里，对数轴）'); axR.legend(fontsize=9.5); axR.grid(alpha=0.3, which='both')
axes[2][0].text(-0.1, 1.06, '③ 行程距离', transform=axes[2][0].transAxes,
                fontsize=14, fontweight='bold', color='#ED7D31')

plt.savefig('D:/AI/条件转化/调整过程示例.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 调整过程示例.png')
