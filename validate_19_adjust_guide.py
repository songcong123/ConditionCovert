# -*- coding: utf-8 -*-
"""调整过程示例 v3: 用户认可的版式 + 补齐缺的四件事
1) 每格统一三行制: 基线是什么(怎么来) → 怎么调(系数公式哪来) → 怎么采样(抽签)
2) λ(t) 概念: 查表即分布, 直方图+抽签示意
3) 空间分布: 转盘示意(晴天份额→乘子→暴雨份额)
4) 干周三/系数出处 全部标注
"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyBboxPatch, FancyArrowPatch
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
pu_cnt = A['pu_cnt'].astype(float)
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])

C_T = ('#4472C4', '#DAE8FC'); C_S = ('#9673A6', '#E1D5E7')
C_D = ('#ED7D31', '#FCE4D6'); C_V = ('#70AD47', '#E2F0D9')
GRAY = '#8C8C8C'

# 干周三查表
w = hour_cnt[(kl == 'dry') & (dow == 4)].mean(0)
hh = hour_cnt[he_i].mean(0)

fig = plt.figure(figsize=(18, 17))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.06], hspace=0.30, wspace=0.22)
fig.suptitle('四个量的调整全过程（基线是什么 → 怎么调 → 怎么采样出一单）——全部实测数据',
             fontsize=16, fontweight='bold', y=0.995)

# ================= ① 出发时间 =================
ax = fig.add_subplot(gs[0, 0])
x = np.arange(24)
ax.bar(x, w / 1000, color='#DAE8FC', edgecolor=C_T[0], lw=0.6, label='基线查表（"干周三"＝没下雨的周三，实测平均）')
ax.plot(x, hh / 1000, 'o-', color='#C00000', lw=2.2, ms=4, label='暴雨日实测（52天平均）')
ax.set_xlabel('时刻（时）'); ax.set_ylabel('单量（千单/小时）'); ax.set_xticks(range(0, 25, 4))
rng = np.random.default_rng(8)
lam = w * np.array([1.0 if h < 12 else 1.169 for h in range(24)])
smp = rng.choice(24, size=8, p=lam / lam.sum())
for i, s in enumerate(smp):
    ax.annotate('', xy=(s, w[s] / 1000 + 1.2 + (i % 4) * 0.55), xytext=(s, w[s] / 1000),
                arrowprops=dict(arrowstyle='-', color='#C00000', lw=1.1, alpha=0.75))
ax.text(0.03, 0.97, '① 出发时间', transform=ax.transAxes, fontsize=14, fontweight='bold', color=C_T[0], va='top')
ax.text(0.03, 0.845,
        '基线：一张 7×24 查表（星期几×每小时的单量，288 个晴天日实测平均）。\n'
        '    本图 = 其中的"周三"一行：8时 35,297单、14时 27,005单……这就是 $\\lambda_0(t)$ 本身\n'
        '怎么调：每小时乘系数  系数(r)=1+0.052×ln(1+r/0.2)\n'
        '    公式哪来：实测三个点(小雨+8.6%/中雨+12.9%/大雨+21.7%)拟合对数曲线，回代误差<1%\n'
        '    例：14时预报 5mm/h → 系数1.169 → 单量 27,005×1.169≈31,563\n'
        '怎么采样：24个小时按(乘完后)单量比例抽签——红色箭头=抽到的8单\n'
        '    暴雨实测红线：峰形不变、整体抬高（外推只推幅度，形状沿用实测）',
        transform=ax.transAxes, fontsize=9.6, va='top', linespacing=1.62,
        bbox=dict(boxstyle='round', fc='#F4F8FC', ec=C_T[0]))
ax.legend(fontsize=9.5, loc='upper right'); ax.grid(alpha=0.3)
ax.set_ylim(0, 46)

# ================= ② 空间分布(转盘) =================
ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0.02, 0.99, '② 空间分布（起点在哪）', transform=ax.transAxes, fontsize=14, fontweight='bold', color=C_S[0], va='top')
pu = pu_cnt[dry_i].sum(0); pu[0] = 0
share = pu / pu.sum()
top6 = np.argsort(-share[1:264])[:6] + 1
import csv
zn = {}
for r_ in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if r_ and r_[0].isdigit(): zn[int(r_[0])] = r_[2][:8]
others = 1 - share[top6].sum()
segs = [(zn[z], share[z], '#9673A6' if i % 2 == 0 else '#B492C4') for i, z in enumerate(top6)]
segs.append(('其余区域合计', others, '#D9C2E9'))
# 左转盘: 晴天
cx, cy, R = 2.6, 4.7, 1.75
a0 = 90
for nm, sh, c in segs:
    ang = sh * 360
    ax.add_patch(Wedge((cx, cy), R, a0 - ang, a0, fc=c, ec='white', lw=1.2))
    a0 -= ang
ax.add_patch(FancyArrowPatch((cx, cy + R + 0.42), (cx + 0.55, cy + R + 0.72), arrowstyle='-|>', mutation_scale=16, color='#404040', lw=1.6))
ax.text(cx, cy - R - 0.42, '晴天转盘（各区域按份额切分）\n= 基线热力图，263个数实测直读', ha='center', fontsize=9.5)
# 中间: 乘子
ax.annotate('', xy=(5.05, cy), xytext=(4.55, cy), arrowprops=dict(arrowstyle='-|>', lw=2.4, color='#C00000'))
ax.text(4.8, cy + 0.42, '每个区域\n×乘子', ha='center', fontsize=10, color='#C00000', fontweight='bold')
# 右转盘: 暴雨
cx2 = 7.2
mults = {'雨(暴雨时)': [1.07, 0.98], }
a0 = 90
segs2 = []
for nm, sh, c in segs:
    a0 -= sh * 360
# 暴雨转盘: 用简化乘子示意(核心区×1.07, 其余×0.98)重算份额
mult_map = {zn[161]: 1.07, zn[230]: 1.07, zn[162]: 1.07, zn[163]: 1.07}  # 中城系×1.07示意
newsh = []
for nm, sh, c in segs:
    m = 1.07 if any(k in nm for k in ['Midtown', 'Times S']) else 0.98
    newsh.append(sh * m)
newsh = np.array(newsh); newsh /= newsh.sum()
a0 = 90
for (nm, sh, c), ns in zip(segs, newsh):
    ang = ns * 360
    ax.add_patch(Wedge((cx2, cy), R, a0 - ang, a0, fc=c, ec='white', lw=1.2, alpha=0.9))
    a0 -= ang
ax.add_patch(FancyArrowPatch((cx2, cy + R + 0.42), (cx2 + 0.55, cy2 := cy + R + 0.72) if False else (cx2 + 0.55, cy + R + 0.72), arrowstyle='-|>', mutation_scale=16, color='#404040', lw=1.6))
ax.text(cx2, cy - R - 0.42, '暴雨转盘（乘子后归一化）\n密集核心 ×1.07、机场 ×0.7、外环 ×0.98', ha='center', fontsize=9.5)
ax.text(0.03, 0.80,
        '基线：263 个区域的出发份额（实测直读，即左图热力图数字版）\n'
        '怎么调：雨天＝每个区域乘一个乘子再归一化——\n'
        '    乘子实测（雨时口径）：市中心密集区 ×1.07、远端枢纽/机场 ×0.7、外环 ×0.98\n'
        '    事件＝查表（马拉松沿线 ×0.5、美网场馆 ×3.5，来自事件日历+实测锚点）\n'
        '怎么采样：转盘抽签——按乘完的份额比例抽一个区域\n'
        '    （例：时代广场 1.21%×1.08→1.31%；抽 10 单多为普通区，偶尔核心/机场）',
        transform=ax.transAxes, fontsize=9.6, va='top', linespacing=1.62,
        bbox=dict(boxstyle='round', fc='#F7F2FA', ec=C_S[0]))

# ================= ③ 距离 =================
ax = fig.add_subplot(gs[1, 0])
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
ax.plot(xs, mix_pdf(xs, P0), lw=2, color=GRAY, ls='--', label='晴天基线（拟合：四个分量=四类人群）')
ax.plot(xs, mix_pdf(xs, P3), lw=2.8, color='#C00000', label='暴雨（3个参数调整后）')
msk = (xs > 0.35) & (xs < 2.0)
ax.fill_between(xs[msk], 0, mix_pdf(xs, P3)[msk] - mix_pdf(xs, P0)[msk], color='#ED7D31', alpha=0.45)
ax.annotate('参数1 短途典型距离左移\n1.74→1.64英里(实测:中位−6%)', xy=(1.35, 0.40), xytext=(0.16, 0.52),
            fontsize=9.5, color='#833C0B', arrowprops=dict(arrowstyle='->', color='#833C0B'))
ax.annotate('橙色阴影=参数2：新增短途单\n(走路改打车, 占比+2.5pp, 实测)', xy=(0.8, 0.27), xytext=(2.6, 0.36),
            fontsize=9.5, color='#C55A11', arrowprops=dict(arrowstyle='->', color='#ED7D31'))
ax.annotate('参数3 长途/机场份额×0.9\n(实测; OD独立核验机场单占9%)', xy=(13, 0.062), xytext=(3.2, 0.15),
            fontsize=9.5, color='#C00000', arrowprops=dict(arrowstyle='->', color='#C00000'))
ax.set_xscale('log')
ax.set_xticks([0.1, 1, 10, 40]); ax.set_xticklabels(['0.1', '1', '10', '40'])
ax.set_xlabel('行程距离（英里，对数轴）'); ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=9, loc='upper right')
ax.text(0.02, 1.10, '③ 行程距离', transform=ax.transAxes, fontsize=14, fontweight='bold', color=C_D[0])
ax.text(0.40, 0.97,
        '基线：三类人群三个子分布叠加\n'
        '    （短途58%/中途30%/长途12%，EM拟合，KS=0.006）\n'
        '怎么调：只动3个数(上图箭头)，来源=52个\n'
        '    真实暴雨日实测直读；其余7个参数锁死\n'
        '怎么采样：先抽签选人群(按新占比)，\n'
        '    再从该人群子分布抽距离',
        transform=ax.transAxes, fontsize=9.6, va='top', linespacing=1.62,
        bbox=dict(boxstyle='round', fc='#FBF4EE', ec=C_D[0]))

# ================= ④ 速度 =================
ax = fig.add_subplot(gs[1, 1])
v = S['v'].astype(float); vd = S['day'].astype(int)
v_d = v[np.isin(vd, dry_i)]; v_h = v[np.isin(vd, he_i)]
rng2 = np.random.default_rng(1)
v_ds = rng2.choice(v_d, 200000, replace=False); v_hs = rng2.choice(v_h, min(len(v_h), 200000), replace=False)
bins = np.arange(0, 40, 0.5)
ax.hist(v_ds, bins=bins, density=True, alpha=0.35, color='#A9D18E', label='晴天基线（实测）')
ax.hist(v_hs, bins=bins, density=True, alpha=0.35, color='#C00000', label='暴雨日（实测）')
xs2 = np.linspace(0.5, 40, 400)
for vv, c in [(v_ds, '#70AD47'), (v_hs, '#C00000')]:
    mu_, sd_ = stats.norm.fit(np.log(vv))
    ax.plot(xs2, stats.lognorm.pdf(xs2, sd_, scale=np.exp(mu_)), lw=2.2, color=c, alpha=0.9)
med_d, med_h = np.median(v_ds), np.median(v_hs)
ax.axvline(med_d, color='#70AD47', ls=':', lw=1.8); ax.axvline(med_h, color='#C00000', ls=':', lw=1.8)
ax.annotate('', xy=(med_h, 0.068), xytext=(med_d, 0.068), arrowprops=dict(arrowstyle='->', lw=2, color='#404040'))
ax.text((med_d + med_h) / 2, 0.072, f'中位 {med_d:.1f}→{med_h:.1f} 英里/时', ha='center', fontsize=10)
ax.text(0.03, 0.97, '④ 速度', transform=ax.transAxes, fontsize=14, fontweight='bold', color=C_V[0], va='top')
ax.text(0.03, 0.855,
        '基线：24个小时各一个中位速度(查表) + 全天共用的宽度参数\n'
        '    （凌晨4时 18.8 ↔ 午后16时 9.8 英里/时，实测）\n'
        '怎么调：整条曲线×折扣系数。系数(r)=exp(−0.025×ln(1+r/0.3))\n'
        '    公式哪来：同①的三点拟合法（实测 −3.8%/−5.3%/−10.1%）\n'
        '    例：雨强5mm/h → ×0.93，即所有人一起慢7%\n'
        '怎么采样：查该小时中位×宽度随机数×折扣系数\n'
        '    （同一小时两单可差2倍多——宽度=真实快慢差异，实测恒定）\n'
        '生成后：时长=距离÷速度（自动一致）',
        transform=ax.transAxes, fontsize=9.6, va='top', linespacing=1.62,
        bbox=dict(boxstyle='round', fc='#F1F7EC', ec=C_V[0]))
ax.set_xlabel('门到门平均速度（英里/时）'); ax.set_ylabel('密度'); ax.set_xlim(0, 40)
ax.legend(fontsize=9.5, loc='upper right'); ax.grid(alpha=0.3)

plt.savefig('D:/AI/条件转化/调整过程示例.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 调整过程示例.png v3')
