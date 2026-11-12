# -*- coding: utf-8 -*-
"""基线四分布实测图: 与汇报版流程图四色一一对应
① 🔵 时间节律 $ambda_0(t)$: 7×24 hour-of-week 热图(干日均值)
② 🟣 空间热力图 $ambda_0(s)$: NYC 263 zone 出发量 choropleth(log)
③ 🟠 距离分布 $P_0(d)$: 直方图 + 四分量分解曲线
④ 🟢 速度分布 $P_0(v,|,t)$: 直方图+LN拟合 + μ₀(t) 24时剖面
"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
import geopandas as gpd

A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load('D:/AI/条件转化/.flow_spd.npz')
J = json.load(open('D:/AI/条件转化/.flow_scores.json'))

dates = list(A['dates'])
klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
m_dry = kl == 'dry'
hour_cnt = A['hour_cnt'].astype(float)
pu_cnt = A['pu_cnt'].astype(float)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
dry_i = np.where(m_dry)[0]

fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 2)
fig.suptitle('流程图中四个基线部件的真实长相（NYC 网约车 · 2024+2025 年 · 晴天 288 天）',
             fontsize=15, fontweight='bold', y=0.99)

# ---------- ① 时间节律 $ambda_0(t)$ ----------
ax = fig.add_subplot(gs[0, 0])
how = np.zeros((7, 24)); hown = np.zeros(7)
dowv = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
for k in range(7):
    sel = m_dry & (dowv == k)
    how[k] = hour_cnt[sel].sum(0); hown[k] = sel.sum()
how /= np.maximum(hown, 1)[:, None] / 1000     # 千趟/日
wd_order = [(v + 3) % 7 for v in range(7)]      # epoch: 0=Thu → weekday Mon=0
labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
how_m = how[np.argsort(wd_order)]
im = ax.imshow(how_m, aspect='auto', cmap='Blues', origin='lower')
ax.set_yticks(range(7)); ax.set_yticklabels(labels)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel('小时'); ax.set_ylabel('')
mean_curve = how.mean(0)
ax2 = ax.twinx()
ax2.plot(range(24), mean_curve / mean_curve.sum() * 100, color='#C00000', lw=2.2, marker='o', ms=3.5)
ax2.set_ylabel('全周平均出发份额 (%)', color='#C00000')
ax2.set_ylim(0, mean_curve.max() / mean_curve.sum() * 100 * 1.35)
pk = int(np.argmax(mean_curve))
ax2.annotate(f'晚峰 {pk}时 ({mean_curve[pk]/mean_curve.sum()*100:.1f}%)',
             xy=(pk, mean_curve[pk] / mean_curve.sum() * 100), xytext=(pk - 6.5, mean_curve[pk] / mean_curve.sum() * 100 * 1.12),
             fontsize=10, color='#C00000', arrowprops=dict(arrowstyle='->', color='#C00000'))
plt.colorbar(im, ax=ax, shrink=0.85, label='出发量（千趟/小时）', pad=0.08)
ax.set_title('① 时间节律 $ambda_0(t)$：周内×小时剖面（三峰, 晚峰最大; 周六夜量抬升）', fontsize=12, loc='left', color='#1F4E79')

# ---------- ② 空间热力图 $ambda_0(s)$ ----------
ax = fig.add_subplot(gs[0, 1])
pu = pu_cnt[m_dry].sum(0); pu[0] = 0
share_pct = pu / pu.sum() * 100
g = gpd.read_file('D:/AI/条件转化/data/tlc/zones/_shp/taxi_zones/taxi_zones.shp').to_crs(4326)
g['share'] = [share_pct[l] if 1 <= l <= 263 else np.nan for l in g['LocationID']]
# 离散分档: 0 / <0.1 / 0.1-0.5 / 0.5-1 / 1-2 / 2-4 / >4 (%)
bins = [-0.01, 0.0001, 0.1, 0.5, 1.0, 2.0, 4.0, 100]
labels_val = [0, 1, 2, 3, 4, 5, 6]
g['level'] = [np.digitize(v, bins[1:-1], right=True) if v == v else np.nan for v in g['share']]
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
cmap_colors = ['#F5F5F5', '#E8E3F0', '#C9B8DE', '#A67FB8', '#7B4FA0', '#5B2C86', '#3D1A66']
cmap = mcolors.ListedColormap(cmap_colors)
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)
g.plot(ax=ax, column='level', cmap=cmap, norm=norm, missing_kwds=dict(color='#E0E0E0'),
       edgecolor='#BBBBBB', lw=0.3)
legend_labels = ['≈0%（郊区）', '<0.1%', '0.1~0.5%', '0.5~1%', '1~2%', '2~4%', '>4%（核心）']
patches = [Patch(fc=cmap_colors[i], ec='#888', label=legend_labels[i]) for i in range(7)]
ax.legend(handles=patches, loc='lower left', fontsize=8, title='出发量占比', title_fontsize=9,
          framealpha=0.9, ncol=2)
top = np.argsort(-pu[1:264])[:5] + 1
import csv
zn = {}
for r in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if r and r[0].isdigit(): zn[int(r[0])] = r[2][:10]
Zc = np.load('D:/AI/条件转化/.flow_zones.npz')
lat, lon = Zc['lat'], Zc['lon']
for i in top[:4]:
    ax.annotate(f'{zn[i]}\n{share_pct[i]:.1f}%', xy=(lon[i], lat[i]), fontsize=8.5,
                ha='center', color='#FFFFFF' if share_pct[i] > 2 else '#1a1a1a',
                fontweight='bold',
                xytext=(lon[i], lat[i] - 0.045),
                bbox=dict(boxstyle='round,pad=0.15', fc='#7B4FA0' if share_pct[i] > 2 else 'white',
                          ec='#999', alpha=0.9))
# 说明文字
ax.text(0.98, 0.98, '颜色越深＝打车出发的人越多\n曼哈顿市中心最集中（Top5 区域占约两成）\n郊区（斯塔滕岛等）几乎为零',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='white', ec='#9673A6', alpha=0.9))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('② 空间分布 $ambda_0(s)$：打车的人主要从哪里出发', fontsize=12, loc='left', color='#5B2C6F')

# ---------- ③ 距离分布 $P_0(d)$ ----------
ax = fig.add_subplot(gs[1, 0])
dd = dist[np.isin(dist_day, dry_i)]
dd = dd[(dd > 0.05) & (dd < 40)]
sub = np.random.default_rng(1).choice(dd, 300000, replace=False)
bins = np.geomspace(0.05, 40, 90)
ax.hist(sub, bins=bins, density=True, alpha=0.35, color='#ED7D31', label='实测（晴天样本）')
xs = np.geomspace(0.05, 40, 600)
B = J['B3']
def lnpdf(x, mu, sd): return stats.lognorm.pdf(x, sd, scale=np.exp(mu))
c1 = B['w1'] * lnpdf(xs, B['mu1'], B['sd1'])
c2 = B['w2'] * lnpdf(xs, B['mu2'], B['sd2'])
ca = B['w_ap'] * lnpdf(xs, B['mu_ap'], B['sd_ap'])
xmin, alpha = B['xmin'], B['alpha']
cp = np.where(xs >= xmin, B['w_pa'] * alpha / xmin * (xs / xmin) ** (-(alpha + 1)), 0)
ax.plot(xs, c1, lw=1.8, ls='--', color='#C55A11', label=f"短途LN（峰{np.exp(B['mu1']):.2f}mi, w={B['w1']:.2f}）")
ax.plot(xs, c2, lw=1.8, ls=':', color='#C55A11', label=f"中途LN（{np.exp(B['mu2']):.1f}mi, w={B['w2']:.2f}）")
ax.plot(xs, ca, lw=2.2, color='#C00000', label=f"机场LN（{np.exp(B['mu_ap']):.0f}mi, w={B['w_ap']:.2f}）")
ax.plot(xs, cp, lw=1.8, ls='-.', color='#7F7F7F', label=f"Pareto尾（α={alpha:.2f}, w={B['w_pa']:.2f}）")
ax.plot(xs, c1 + c2 + ca + cp, lw=2.4, color='#1a1a1a', label='四分量混合（KS=0.0135）')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('行程距离 (mi, log轴)'); ax.set_ylabel('密度 (log轴)')
ax.legend(fontsize=8.5, loc='lower left'); ax.grid(alpha=0.3, which='both')
ax.text(0.98, 0.03,
        r'$P_0(d)=\sum_k w_k\,\mathrm{LN}(\mu_k,\sigma_k)+w_p\,\mathrm{Pareto}(d;\alpha,d_{min})$' + '\n'
        r'$w{=}(0.61,0.16,0.10,0.13)$, LN 峰 $1.8/4.6/13$ mi,  $\alpha{=}2.89$, KS$=0.0135$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round', fc='#FBF4EE', ec='#ED7D31', alpha=0.95))
ax.set_title('③ 距离分布 $P_0(d)$：短途峰 + 中途 + 机场鼓包 + 幂尾（四分量）', fontsize=12, loc='left', color='#833C0B')

# ---------- ④ 速度分布 $P_0(v,|,t)$：左右两个子图 ----------
sub4 = gs[1, 1].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
axL = fig.add_subplot(sub4[0, 0])
axR = fig.add_subplot(sub4[0, 1])
v = S['v'].astype(float); vd = S['day'].astype(int); vh = S['hour'].astype(int)
v_d = v[np.isin(vd, dry_i)]; h_d = vh[np.isin(vd, dry_i)]
sub = np.random.default_rng(2).choice(v_d, 300000, replace=False)
bins = np.arange(0, 40, 0.5)
axL.hist(sub, bins=bins, density=True, alpha=0.35, color='#70AD47', label='实测（晴天样本）')
lv = np.log(v_d); mu_all, sd_all = stats.norm.fit(lv)
xs = np.linspace(0.2, 40, 500)
axL.plot(xs, stats.lognorm.pdf(xs, sd_all, scale=np.exp(mu_all)), lw=2.4, color='#1a1a1a',
         label=f'LogNormal 拟合（形状参数 $\sigma_0$={sd_all:.3f}, 中位 {np.exp(mu_all):.1f} 英里/时）')
axL.set_xlim(0, 40); axL.set_xlabel('全程平均速度（英里/时，含等红灯）'); axL.set_ylabel('密度')
axL.legend(fontsize=9, loc='upper right'); axL.grid(alpha=0.3)
axL.text(0.98, 0.04,
         r'$\log v=\mu_0(t)+\varepsilon,\ \varepsilon\sim\mathcal{N}(0,\sigma_0^2)$' + '\n'
         r'$\sigma_0=0.441$（实测）',
         transform=axL.transAxes, ha='right', va='bottom', fontsize=9.5,
         bbox=dict(boxstyle='round', fc='#F1F7EC', ec='#70AD47', alpha=0.95))
axL.set_title('④ 速度分布 $P_0(v,|,t)$：整体呈 LogNormal', fontsize=12, loc='left', color='#375623')

mu_t = np.array([np.exp(lv[h_d == h].mean()) for h in range(24)])
sig_t = np.array([lv[h_d == h].std() for h in range(24)])
axR.plot(range(24), mu_t, 'o-', color='#548235', lw=2.2, ms=3.5)
axR.set_xlabel('小时'); axR.set_ylabel('中位速度（英里/时）', color='#548235')
axR.tick_params(axis='y', labelcolor='#548235')
axR.set_xticks(range(0, 25, 4)); axR.grid(alpha=0.3)
axR.set_ylim(7, 21)
axR2 = axR.twinx()
axR2.plot(range(24), sig_t, 's--', color='#BF8F00', lw=1.8, ms=3.5)
axR2.set_ylabel('快慢差异（越小越均匀）', color='#BF8F00')
axR2.tick_params(axis='y', labelcolor='#BF8F00')
axR2.set_ylim(0.30, 0.60)
axR.annotate(f'凌晨4时\n{mu_t[4]:.1f} 英里/时\n（路空，最快）', xy=(4, mu_t[4]), xytext=(7, mu_t[4] + 1.2),
             fontsize=8.5, color='#548235', ha='left',
             arrowprops=dict(arrowstyle='->', color='#548235'))
axR.annotate(f'下午16时\n{mu_t[16]:.1f} 英里/时\n（最堵）', xy=(16, mu_t[16]), xytext=(18.5, mu_t[16] - 1.2),
             fontsize=8.5, color='#548235', ha='left',
             arrowprops=dict(arrowstyle='->', color='#548235'))
axR.set_title('逐时看：中位速度大起大落，快慢差异几乎不变', fontsize=10.5, loc='left')

plt.tight_layout()
plt.savefig('D:/AI/条件转化/基线四分布.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 基线四分布.png | B3来自.flow_scores.json: alpha', J['B3']['alpha'])
