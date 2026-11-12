# -*- coding: utf-8 -*-
"""方案流程 × 调整过程示例 合并图（简洁版：字少、数字准）"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
import geopandas as gpd

A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load('D:/AI/条件转化/.flow_spd.npz')
SP = np.load('D:/AI/条件转化/.flow_spatial.npz')
B3 = json.load(open('D:/AI/条件转化/.flow_scores.json'))['B3']

dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
dry_i = np.where(kl == 'dry')[0]; he_i = np.where(kl == 'heavy')[0]
hour_cnt = A['hour_cnt'].astype(float)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
pu_cnt = A['pu_cnt'].astype(float)

C_T = ('#4472C4', '#DAE8FC'); C_S = ('#9673A6', '#E1D5E7')
C_D = ('#ED7D31', '#FCE4D6'); C_V = ('#70AD47', '#E2F0D9')
GRAY = '#8C8C8C'

fig = plt.figure(figsize=(18, 17))
gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 1, 1.15], hspace=0.30, wspace=0.22)
fig.suptitle('方案流程 × 四个量的调整实例', fontsize=16, fontweight='bold', y=0.995)

# ================= 上：流程带 =================
axf = fig.add_subplot(gs[0, :]); axf.axis('off')
axf.set_xlim(0, 1800); axf.set_ylim(0, 230)

def box(x, y, w, h, fc, ec, label, fs=10.5, bold=False, lw=1.8):
    axf.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=3,rounding_size=7', fc=fc, ec=ec, lw=lw))
    axf.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fs,
             fontweight='bold' if bold else 'normal', linespacing=1.5)

def arrow(x0, x1, y=105, color='#595959'):
    axf.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=18, lw=2.0, color=color))

box(15, 55, 190, 100, '#F2F2F2', '#666', '常态行程数据\n＋\n外生条件 c', fs=10)
arrow(215, 255)
box(258, 25, 340, 170, '#FFF', '#82B366', '', lw=2)
axf.text(428, 182, 'Step 1  基线', ha='center', fontsize=11.5, fontweight='bold', color='#375623')
for j, (txt, (ec, fc)) in enumerate([
    ('① 出发时间', C_T), ('② 空间分布', C_S), ('③ 行程距离', C_D), ('④ 速度', C_V)]):
    box(270, 145 - j * 36, 316, 30, fc, ec, txt, fs=10)
arrow(602, 645)
box(648, 25, 400, 170, '#FFF', '#D79B00', '', lw=2)
axf.text(848, 182, 'Step 2  条件调整（只改参数，不改形状）', ha='center', fontsize=11.5, fontweight='bold', color='#8C5A00')
for j, (txt, (ec, fc)) in enumerate([
    ('① 总量 ×系数', C_T), ('② 空间 ×乘子', C_S), ('③ 距离 3参数', C_D), ('④ 速度 ×折扣', C_V)]):
    box(660, 145 - j * 36, 376, 30, fc, ec, txt, fs=10)
    axf.add_patch(Circle((672, 160 - j * 36), 9, fc=ec, ec='white', lw=1.2, zorder=5))
    axf.text(672, 160 - j * 36, '①②③④'[j], ha='center', va='center', fontsize=8, color='white', fontweight='bold', zorder=6)
arrow(1052, 1095)
box(1098, 65, 190, 80, '#F2F2F2', '#666', 'Step 3 采样\n→ Step 4 终点配对\n→ Step 5 校验输出', fs=10)
axf.text(900, 10, '（校验不通过→回 Step 2；盲测通过才放行外推）', ha='center', fontsize=9.5, color='#8C8C8C')

# ================= ① 出发时间 =================
ax = fig.add_subplot(gs[1, 0])
hd = hour_cnt[dry_i].mean(0) / 1000
hh = hour_cnt[he_i].mean(0) / 1000
x = np.arange(24)
ax.bar(x, hd, color='#DAE8FC', edgecolor=C_T[0], lw=0.6, label='晴天基线')
ax.plot(x, hh, 'o-', color='#C00000', lw=2.4, ms=4, label='暴雨日（52天实测）')
ax.set_xlabel('时刻（时）'); ax.set_ylabel('出发量（千单/小时）')
ax.set_xticks(range(0, 25, 4))
ax.set_title('① 出发时间', fontsize=13.5, fontweight='bold', color=C_T[0], loc='left', pad=12)
ax.text(0.03, 0.92, '总量 +6.8%，峰形不变\n逐时系数 +9%~+22%', transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', fc='#F4F8FC', ec=C_T[0], alpha=0.9))
ax.legend(fontsize=9.5, loc='upper right'); ax.grid(alpha=0.3)

# ================= ② 空间分布 =================
g = gpd.read_file('D:/AI/条件转化/data/tlc/zones/_shp/taxi_zones/taxi_zones.shp').to_crs(4326)
gsk = gs[1, 1].subgridspec(1, 2, wspace=0.05)
axL = fig.add_subplot(gsk[0]); axR = fig.add_subplot(gsk[1])
pu = pu_cnt[dry_i].sum(0); pu[0] = 0
g['pu'] = [pu[l] if 1 <= l <= 263 else np.nan for l in g['LocationID']]
g.plot(ax=axL, column=np.log10(g['pu']), cmap='Purples', missing_kwds=dict(color='#EFEFEF'),
       edgecolor='#B0B0B0', lw=0.3)
nov = SP['nov_pu'].astype(float)
mar = nov[0, 8:17].sum(axis=0); bas = nov[1:, 8:17].sum(axis=1).mean(axis=0)
chg = np.full(264, np.nan); m_ = bas >= 40
chg[1:264][m_[1:264]] = (mar[1:264][m_[1:264]] / bas[1:264][m_[1:264]]) - 1
g['chg'] = [chg[l] if 1 <= l <= 263 else np.nan for l in g['LocationID']]
g.plot(ax=axR, column='chg', cmap='RdBu_r', vmin=-0.7, vmax=0.7, missing_kwds=dict(color='#EFEFEF'),
       edgecolor='#B0B0B0', lw=0.3)
Zc = np.load('D:/AI/条件转化/.flow_zones.npz')
axR.annotate('中央公园 −48%', xy=(Zc['lon'][43], Zc['lat'][43]), fontsize=9.5, color='#404040',
             xytext=(Zc['lon'][43] - 0.17, Zc['lat'][43] + 0.07),
             arrowprops=dict(arrowstyle='->', color='#404040'))
for a_ in (axL, axR): a_.set_xticks([]); a_.set_yticks([])
axL.set_title('晴天基线（雨天近似不动）', fontsize=10.5, loc='left', color=C_S[0])
axR.set_title('马拉松日（事件才大动）', fontsize=10.5, loc='left', color=C_S[0])
axL.text(0.02, 1.25, '② 空间分布', transform=axL.transAxes, fontsize=13.5, fontweight='bold', color=C_S[0])

# ================= ③ 距离 =================
gsk3 = gs[2, 0].subgridspec(1, 2, wspace=0.25, width_ratios=[1.3, 1])
axA = fig.add_subplot(gsk3[0]); axB = fig.add_subplot(gsk3[1])
xs = np.geomspace(0.05, 40, 700)
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
axA.plot(xs, mix_pdf(xs, P0), lw=2, color=GRAY, ls='--', label='晴天基线')
axA.plot(xs, mix_pdf(xs, P3), lw=2.8, color='#C00000', label='暴雨（调整后）')
axA.annotate('①短途左移\n(−6%)', xy=(1.35, 0.40), xytext=(0.2, 0.55), fontsize=9.5,
             color='#833C0B', arrowprops=dict(arrowstyle='->', color='#833C0B'))
axA.annotate('②短途更多\n(+2.5pp)', xy=(0.75, 0.25), xytext=(2.5, 0.32), fontsize=9.5,
             color='#C55A11', arrowprops=dict(arrowstyle='->', color='#C55A11'))
axA.annotate('③长途×0.9', xy=(13, 0.06), xytext=(4, 0.14), fontsize=9.5,
             color='#C00000', arrowprops=dict(arrowstyle='->', color='#C00000'))
axA.set_xscale('log')
axA.set_xticks([0.1, 1, 10, 40]); axA.set_xticklabels(['0.1', '1', '10', '40'])
axA.set_xlabel('距离（英里，对数轴）'); axA.grid(alpha=0.3, which='both')
axA.legend(fontsize=9, loc='upper right')
axA.text(0.02, 1.10, '③ 行程距离：3 个可调参数', transform=axA.transAxes,
         fontsize=13.5, fontweight='bold', color=C_D[0])
bins = np.geomspace(0.05, 40, 70)
dd_he = dist[np.isin(dist_day, he_i)]; dd_he = dd_he[(dd_he > 0.05) & (dd_he < 40)]
axB.hist(dd_he, bins=bins, density=True, alpha=0.35, color='#4472C4', label='真实暴雨日')
axB.plot(xs, mix_pdf(xs, P3), lw=2.6, color='#C00000', label='生成曲线')
axB.set_xscale('log')
axB.set_xticks([0.1, 1, 10, 40]); axB.set_xticklabels(['0.1', '1', '10', '40'])
axB.set_xlabel('距离（英里，对数轴）'); axB.grid(alpha=0.3, which='both')
axB.legend(fontsize=9, loc='upper right')
axB.text(0.5, 0.5, 'KS=0.024\n（不调=0.24）', ha='center', fontsize=12, color='#C00000',
         fontweight='bold', transform=axB.transAxes,
         bbox=dict(boxstyle='round', fc='white', ec='#C00000', alpha=0.9))
axB.set_title('验证', fontsize=10.5, loc='left')

# ================= ④ 速度 =================
ax = fig.add_subplot(gs[2, 1])
v = S['v'].astype(float); vd = S['day'].astype(int)
v_d = v[np.isin(vd, dry_i)]; v_h = v[np.isin(vd, he_i)]
rng = np.random.default_rng(1)
v_ds = rng.choice(v_d, 200000, replace=False); v_hs = rng.choice(v_h, min(len(v_h), 200000), replace=False)
bins = np.arange(0, 40, 0.5)
ax.hist(v_ds, bins=bins, density=True, alpha=0.35, color='#A9D18E', label='晴天基线')
ax.hist(v_hs, bins=bins, density=True, alpha=0.35, color='#C00000', label='暴雨日')
xs2 = np.linspace(0.5, 40, 400)
for vv, c in [(v_ds, '#70AD47'), (v_hs, '#C00000')]:
    mu_, sd_ = stats.norm.fit(np.log(vv))
    ax.plot(xs2, stats.lognorm.pdf(xs2, sd_, scale=np.exp(mu_)), lw=2.2, color=c, alpha=0.9)
med_d, med_h = np.median(v_ds), np.median(v_hs)
ax.axvline(med_d, color='#70AD47', ls=':', lw=1.8); ax.axvline(med_h, color='#C00000', ls=':', lw=1.8)
ax.annotate('', xy=(med_h, 0.070), xytext=(med_d, 0.070), arrowprops=dict(arrowstyle='->', lw=2, color='#404040'))
ax.text((med_d + med_h) / 2, 0.074, f'{med_d:.1f} → {med_h:.1f} 英里/时', ha='center', fontsize=10)
ax.set_title('④ 速度', fontsize=13.5, fontweight='bold', color=C_V[0], loc='left', pad=12)
ax.text(0.55, 0.92, '整体×0.96（变慢 4%）\n形状不变（快慢比例恒定）', transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', fc='#F1F7EC', ec=C_V[0], alpha=0.9))
ax.set_xlabel('全程平均速度（英里/时）'); ax.set_ylabel('密度'); ax.set_xlim(0, 40)
ax.legend(fontsize=9.5, loc='upper right'); ax.grid(alpha=0.3)

plt.savefig('D:/AI/条件转化/方案流程×调整示例.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved')
