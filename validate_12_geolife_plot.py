# -*- coding: utf-8 -*-
"""GeoLife(个体域) vs NYC HV FHV(网约车域) 对照图"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

G = np.load('D:/AI/条件转化/.geolife_val.npz')
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)

dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
hour_cnt = A['hour_cnt'].astype(float)
how = hour_cnt[kl == 'dry'].sum(0).reshape(-1, 24).mean(0)
how /= how.sum()
dist_nyc = A['dist'].astype(float)
dist_nyc = dist_nyc[(dist_nyc > 0.05) & (dist_nyc < 80)]
rng = np.random.default_rng(2)
dist_nyc_s = rng.choice(dist_nyc, 300000, replace=False)

d_geo = G['d_mi'].astype(float)
ks_tpl, r0, beta, kap = G['tpl']
ks_ln, mu_ln, sd_ln = G['ln']
alpha_geo = float(G['alpha'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
fig.suptitle('GeoLife（北京·182人·个体移动域）vs NYC HV FHV（网约车订单域）：三个量的跨域对照',
             fontsize=14, fontweight='bold', y=1.02)

# (1) 距离分布
ax = axes[0]
bins = np.geomspace(0.05, 60, 70)
ax.hist(d_geo, bins=bins, density=True, alpha=0.55, color='#7030A0', label=f'GeoLife 移动步 (n={len(d_geo):,})')
ax.hist(dist_nyc_s, bins=bins, density=True, alpha=0.45, color='#ED7D31', label='NYC 行程 (干日)')
g = np.geomspace(0.05, 60, 500)
pdf_tpl = (g + r0) ** (-beta) * np.exp(-g / kap); pdf_tpl *= (np.sum(np.diff(bins)[:1]) * 0 + 1)
pdf_tpl *= len(d_geo) * np.mean(np.diff(np.geomspace(0.05, 60, 500))[:1])  # rough scale
# 正确缩放: 按直方图bin宽积分匹配
pdf_tpl = (g + r0) ** (-beta) * np.exp(-g / kap)
norm = np.trapz(pdf_tpl, g); pdf_tpl /= norm
ax.plot(g, pdf_tpl, 'm-', lw=2, label=f'截断幂律拟合 (β={beta:.2f}, κ={kap:.0f}mi, KS={ks_tpl:.3f})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('移动步 / 行程距离 (mi, log轴)'); ax.set_ylabel('密度 (log轴)')
ax.set_title(f'(1) 距离分布：GeoLife 截断幂律 KS={ks_tpl:.3f} vs LogNormal KS={ks_ln:.3f}\n'
             f'GeoLife尾α={alpha_geo:.2f}；NYC主体 logNormal(峰1.76mi)+机场+Pareto(α=2.86)', fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which='both')

# (2) 速度 σ
ax = axes[1]
sig_h = G['sig_h']; cnt = G['cnt_h']
ok = cnt > 0
x = np.arange(24)
ax.bar(x[ok], sig_h[ok], color='#7030A0', alpha=0.7, width=0.6, label='GeoLife 分小时 σ(log v)')
ax.axhline(float(G['sig_all']), color='#7030A0', ls='--', lw=1.8, label=f'GeoLife 全体 σ={G["sig_all"]:.2f}')
ax.axhline(0.44, color='#ED7D31', ls='-', lw=2.2, label='NYC 网约车 形状参数$\,\sigma_0$=0.44（跨雨强/时段稳定）')
ax.set_xlabel('小时'); ax.set_ylabel('σ(log v)')
ax.set_title('(2) 速度分布形状：个体域(混合模式)明显更宽且不稳\n——"σ稳定"是订单域门到门速度的性质，不跨域', fontsize=10.5)
ax.set_xticks(range(0, 25, 4)); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(1.3, sig_h[ok].max() * 1.15))

# (3) 时间节律
ax = axes[2]
wd_g = G['wd_mean']
x = np.arange(24)
ax.bar(x, how * 100, alpha=0.45, color='#ED7D31', width=0.55, label='NYC HV FHV（干日出发）')
ax.plot(x, wd_g * 100, 'o-', color='#7030A0', lw=2.2, ms=4, label='GeoLife（移动点频）')
ax.set_xlabel('小时'); ax.set_ylabel('份额 (%)')
ax.set_title('(3) 时间节律：个体域通勤双峰 vs 网约车域晚峰主导\n（跨域可迁移的是节律"形态"，幅度需本地重建）', fontsize=10.5)
ax.set_xticks(range(0, 25, 4)); ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('D:/AI/条件转化/GeoLife验证.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved GeoLife验证.png')
