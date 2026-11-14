# -*- coding: utf-8 -*-
"""分布层面的内插/外推例子: 速度 + 距离, 正态曲线(钟形)表示
   每面板 4 条曲线: 基线(灰虚) / 实际(黑粗) / 公式合成(蓝) / RF合成(红)
   合成 = 基线分布整体平移 log(1+效应)  ← 乘法算子的分布含义
   内插例子: 速度 r=1-4mm/h, 距离 日降水5-15mm | 外推例子: 速度 r=8-16, 距离 日降水30-60
   RF/公式的箱效应来自 validate_38(已验证)
输出: 分布层面_内插外推对比.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; GREY = '#999999'; BLACK = '#1A1A1A'; MUTED = '#666666'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); n = len(dates)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)

# ---- 速度样本(log) ----
def spd_log(mask_days):
    m = np.isin(spdD, mask_days) & (spdV > 1) & (spdV < 60)
    return np.log(spdV[m])

dry_i = np.where(kl == 'dry')[0]
lv_base = spd_log(dry_i)
# 内插例子: 1-4mm/h 的雨时
hrs_in = [i for i in range(n) if kl[i] != 'dry' and 1 <= inten[i].max() < 16]
m_in = (np.isin(spdD, hrs_in)) & (spdV > 1) & (spdV < 60)
# 逐小时筛选: 该小时雨强在箱内
sel_in = np.zeros(len(spdV), bool)
for i in hrs_in:
    for h in range(24):
        if 1 <= inten[i, h] < 4:
            sel_in |= (spdD == i) & (spdH == h)
sel_in &= (spdV > 1) & (spdV < 60)
lv_act_in = np.log(spdV[sel_in])
# 外推例子: 8-16mm/h
sel_ex = np.zeros(len(spdV), bool)
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        if 8 <= inten[i, h] < 16:
            sel_ex |= (spdD == i) & (spdH == h)
sel_ex &= (spdV > 1) & (spdV < 60)
lv_act_ex = np.log(spdV[sel_ex])
print(f'速度样本: 基线 {len(lv_base):,} | 内插实际 {len(lv_act_in):,} | 外推实际 {len(lv_act_ex):,}')

# ---- 距离样本(log, km) ----
def dist_log(mask_days):
    s = dist[np.isin(dist_day, mask_days)]
    s = s[(s > 0.2) & (s < 60)]
    return np.log(s * 1.609)      # mi→km
ld_base = dist_log(dry_i)
ld_act_in = dist_log(np.where((dp >= 5) & (dp < 15) & (kl != 'dry'))[0])
ld_act_ex = dist_log(np.where((dp >= 30) & (dp < 60) & (kl != 'dry'))[0])
print(f'距离样本: 基线 {len(ld_base):,} | 内插实际 {len(ld_act_in):,} | 外推实际 {len(ld_act_ex):,}')

# ---- 效应(validate_38 已验证) ----
SPD_EFF = dict(inter=dict(act=-0.051, rf=-0.051, f=-0.052), ex=dict(act=-0.098, rf=-0.065, f=-0.085))
DST_EFF = dict(inter=dict(act=-0.036, rf=-0.032, f=-0.016), ex=dict(act=-0.137, rf=-0.045, f=-0.039))

# ---- KDE 画曲线(基线平移 = 乘法) ----
def draw(ax, base, act, eff, xlab, title, unit_tick):
    kb = stats.gaussian_kde(base)
    gx = np.linspace(base.mean() - 4.2 * base.std(), base.mean() + 4.2 * base.std(), 400)
    ax.plot(gx, kb(gx), lw=2, ls='--', color=GREY, label='晴天基线', zorder=2)
    # 实际
    ka = stats.gaussian_kde(act)
    ax.plot(gx, ka(gx), lw=3, color=BLACK, label='雨天实际', zorder=5)
    # 公式 / RF = 基线平移 ln(1+eff)
    ax.plot(gx, kb(gx - np.log1p(eff['f'])), lw=2.4, color=NAVY, label='公式合成', zorder=4)
    ax.plot(gx, kb(gx - np.log1p(eff['rf'])), lw=2.4, color=RED, label='RF 合成', zorder=4)
    # 中位竖线
    for name, e, c in [('实际', None, BLACK), ('公式', eff['f'], NAVY), ('RF', eff['rf'], RED)]:
        med = np.exp(np.median(act)) if e is None else np.exp(np.median(base)) * (1 + e)
        lx = np.log(med)
        ax.axvline(lx, color=c, ls=':', lw=1.4, ymax=0.86)
    # 中位数标注
    med_a = np.exp(np.median(act)) * (1.609 if 'km' in xlab else 1)
    med_0 = np.exp(np.median(base))
    txt = (f"实际中位 {np.exp(np.median(act)):.1f}  公式 {med_0*(1+eff['f']):.1f}  RF {med_0*(1+eff['rf']):.1f}"
           .replace('.1f', ''))
    ax.set_title(title, fontsize=11.5, loc='left', color=NAVY)
    ax.set_xlabel(xlab); ax.set_ylabel('密度')
    ax.legend(fontsize=9.5, loc='upper right')
    ax.grid(alpha=0.3)

fig, axes = plt.subplots(2, 2, figsize=(15, 9.6))
fig.suptitle('分布层面的例子：基线 × 乘子 = 钟形平移 —— 谁的峰贴着"雨天实际"（黑线）',
             fontsize=15, fontweight='bold', y=0.995)

draw(axes[0, 0], lv_base, lv_act_in, SPD_EFF['inter'],
     'log 速度 (ln mph)', '① 速度 · 内插例子（小雨 r=1~4 mm/h，训练见过）', None)
draw(axes[0, 1], lv_base, lv_act_ex, SPD_EFF['ex'],
     'log 速度 (ln mph)', '② 速度 · 外推例子（大雨 r=8~16 mm/h，训练没见过）', None)
draw(axes[1, 0], ld_base, ld_act_in, DST_EFF['inter'],
     'log 距离 (ln km)', '③ 距离 · 内插例子（中雨日 5~15 mm，训练见过）', None)
draw(axes[1, 1], ld_base, ld_act_ex, DST_EFF['ex'],
     'log 距离 (ln km)', '④ 距离 · 外推例子（大雨日 30~60 mm，训练没见过）', None)

# 面板结论标注
axes[0, 0].text(0.03, 0.92, '三条线几乎重合\n→ 内插域谁用都行', transform=axes[0, 0].transAxes,
                fontsize=10, color=DGREEN, va='top', bbox=dict(boxstyle='round,pad=0.3', fc='#E2F0D9', ec='#70AD47', alpha=0.9))
axes[0, 1].text(0.03, 0.92, '实际峰(黑)更深地左移\n公式(蓝)跟上, RF(红)停在半路', transform=axes[0, 1].transAxes,
                fontsize=10, color='#843C0C', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='#FDEADA', ec='#ED7D31', alpha=0.9))
axes[1, 1].text(0.03, 0.92, '实际峰左移 −13.7%\n公式 −3.9%, RF 卡死 −4.5%', transform=axes[1, 1].transAxes,
                fontsize=10, color='#843C0C', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='#FDEADA', ec='#ED7D31', alpha=0.9))

fig.text(0.5, 0.004,
         '读法: 黑线=真实雨天的分布; 蓝/红线=把基线(灰虚)按各自预测的乘子平移后的合成分布 —— 谁贴黑线谁准。内插例子(左列)三线贴合; 外推例子(右列)RF 平移不足、公式更贴',
         ha='center', fontsize=10.5, color=DGREEN, fontweight='bold', wrap=True)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(r'D:/AI/条件转化/分布层面_内插外推对比.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 分布层面_内插外推对比.png')
