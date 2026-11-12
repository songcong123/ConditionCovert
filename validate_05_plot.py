# -*- coding: utf-8 -*-
"""验证汇总图: 6 面板
(a) 小时雨强剂量-响应(实测3箱+拟合曲线)  (b) σ(log v) 跨雨强稳定性
(c) λ(t): 干基线/真实中雨/生成中雨        (d) P(d): 真实 vs 生成 vs 纯基线
(e) P(v) 同                              (f) 三版四量得分对比(pass1/pass2b/基线)"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

G = np.load('D:/AI/条件转化/.flow_gen.npz', allow_pickle=True)
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
S = np.load('D:/AI/条件转化/.flow_spd.npz')
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
J = json.load(open('D:/AI/条件转化/.flow_scores.json'))
P1 = json.load(open('D:/AI/条件转化/.flow_scores_pass1.json'))

dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
me_i = np.where(kl == 'medium')[0]; dry_i = np.where(kl == 'dry')[0]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('极端天气轨迹生成流程 · 数据验证报告（中雨替身盲测, NYC HV FHV 2024.04–10, n=1.30亿行）',
             fontsize=14, fontweight='bold', y=0.99)

# (a) 剂量-响应
ax = axes[0][0]
mh = J['measured_hour']
rs = [mh[k]['r'] for k in mh]; dv = [mh[k]['dvol'] for k in mh]
dvv = [mh[k]['d_v'] for k in mh]
th, RC = J['theta']['vol_h'], J['theta']['RC_H']
xx = np.linspace(0.1, 25, 300)
ax.scatter(rs, [v * 100 for v in dv], s=90, c='#C00000', zorder=5, label='实测 δ_vol')
ax.plot(xx, th * np.log1p(xx / RC) * 100, 'r-', lw=2,
        label=f'拟合 θ·log(1+r/{RC:.1f})  R²=1.00')
ax2 = ax.twinx()
ax2.scatter(rs, [v * 100 for v in dvv], s=90, c='#1F4E79', marker='s', zorder=5, label='实测 δ_v')
ax2.plot(xx, J['theta']['v_h'] * np.log1p(xx / J['theta']['RC_H'] if 'RC_H' in J['theta'] else xx / 0.2) * 100,
         'b--', lw=2)
ax.set_xlabel('小时雨强 r (mm/h)'); ax.set_ylabel('λ 偏移 δ_vol (%)', color='#C00000')
ax2.set_ylabel('log v 偏移 δ_v (%)', color='#1F4E79')
ax.set_title('(a) 剂量-响应实测与拟合（标定段 0.3–1 mm/h）', fontsize=11)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper left')
ax.grid(alpha=0.3)

# (b) σ 稳定性
ax = axes[0][1]
sg = [mh[k]['sig'] for k in mh]
labs = ['干日\n(基线)'] + [f'{k} mm/h' for k in mh]
vals = [J['B4']['sigma0']] + sg
bars = ax.bar(labs, vals, color=['#A9D18E'] + ['#8FAADC'] * 3, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f'{v:.3f}', ha='center', fontsize=10)
ax.set_ylim(0.40, 0.47); ax.set_ylabel('σ(log v)')
ax.set_title('(b) 核心假设检验：分布形状跨雨强稳定（只平移 μ）', fontsize=11)
ax.grid(alpha=0.3, axis='y')

# (c) λ(t)
ax = axes[0][2]
x = np.arange(24)
ax.bar(x, G['lam0_t'] / G['lam0_t'].sum() * 100, alpha=0.45, color='#A9D18E', label='B1 干日基线')
ax.plot(x, G['hour_me'] / G['hour_me'].sum() * 100, 'k-o', ms=3.5, lw=1.8, label='真实中雨(n=17日)')
ax.plot(x, G['hour_gen'] / G['hour_gen'].sum() * 100, 'r-', lw=2.2, label='生成中雨(pass2b)')
ax.set_xlabel('出发时刻(时)'); ax.set_ylabel('份额 (%)')
ax.set_title(f"(c) λ(t)：JS(生成,真实)={J['scores']['lam_t']:.4f} vs 基线 {J['scores_ab']['lam_t']:.4f}", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(range(0, 25, 4))

# (d) P(d)
ax = axes[1][0]
dist_me = A['dist'][np.isin(A['dist_day'], me_i)].astype(float)
bins = np.arange(0, 15.25, 0.25)
sub = np.random.default_rng(1).choice(dist_me, 200000, replace=False)
ax.hist(sub, bins=bins, density=True, alpha=0.45, color='#404040', label='真实中雨')
ax.hist(G['d'], bins=bins, density=True, alpha=0.5, color='#C00000', label='生成(pass2b)')
from scipy import stats as st
B3, B3e = J['B3'], J['B3e']
xs = np.linspace(0.1, 15, 500)
pdf0 = (B3['w1'] * st.lognorm.pdf(xs, B3['sd1'], scale=np.exp(B3['mu1']))
        + B3['w2'] * st.lognorm.pdf(xs, B3['sd2'], scale=np.exp(B3['mu2']))
        + B3['w_ap'] * st.lognorm.pdf(xs, B3['sd_ap'], scale=np.exp(B3['mu_ap'])))
pdf0 += B3['w_pa'] * np.where(xs >= B3['xmin'], B3['alpha'] / B3['xmin'] * (xs / B3['xmin']) ** (-(B3['alpha'] + 1)), 0)
ax.plot(xs, pdf0, 'g--', lw=2, label='B3 干日基线(未扰动)')
med = np.median(sub); ax.axvline(med, color='#404040', ls=':', lw=1.6)
ax.axvline(np.median(G['d']), color='#C00000', ls=':', lw=1.6)
ax.set_xlabel('行程距离 (mi)'); ax.set_ylabel('密度')
ax.set_title(f"(d) P(d)：KS(生成,真实)={J['scores']['d']:.4f} vs 基线 {J['scores_ab']['d']:.4f}\n"
             f"中位 真实{med:.2f} / 生成{np.median(G['d']):.2f} mi（外推过冲约2×）", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 15)

# (e) P(v)
ax = axes[1][1]
spd_me = S['v'][np.isin(S['day'], me_i)].astype(float)
sub = np.random.default_rng(1).choice(spd_me, 200000, replace=False)
bins = np.arange(0, 40, 0.5)
ax.hist(sub, bins=bins, density=True, alpha=0.45, color='#404040', label='真实中雨')
ax.hist(G['v'], bins=bins, density=True, alpha=0.5, color='#ED7D31', label='生成(pass2b)')
ax.set_xlabel('平均速度 (mph)'); ax.set_ylabel('密度')
ax.set_title(f"(e) P(v)：KS(生成,真实)={J['scores']['v']:.4f} vs 基线 {J['scores_ab']['v']:.4f}", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 40)

# (f) 得分对比
ax = axes[1][2]
cats = ['JS λ(t)', 'KS P(d)', 'KS P(v)', 'JS PU', 'JS DO']
keys = ['lam_t', 'd', 'v', 'pu', 'do']
w = 0.25; xp = np.arange(len(cats))
base = [J['scores_ab'][k] for k in keys]
s1 = [P1['scores'][k] for k in keys]
s2 = [J['scores'][k] for k in keys]
ax.bar(xp - w, base, w, color='#A9D18E', label='纯基线(消融)')
ax.bar(xp, s1, w, color='#B4C7E7', label='pass1 按图原样')
ax.bar(xp + w, s2, w, color='#C00000', label='pass2b 修正回路')
ax.set_xticks(xp); ax.set_xticklabels(cats)
ax.set_ylabel('得分 (越低越好)')
ax.set_title('(f) 替身盲测四量：三版对比（log轴）', fontsize=11)
ax.set_yscale('log'); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/AI/条件转化/极端天气流程验证.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 极端天气流程验证.png')
