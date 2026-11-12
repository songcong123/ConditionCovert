# -*- coding: utf-8 -*-
"""无线行人版·调整过程示例（三天数据 4/1~4/3, 演示数据）
   与 NYC 调整过程示例同构: 三行四列(基线→雨强乘子→调整后→对照)
   数据现实:
     - 数据集仅 3 天: 4/1 晴(=基线日) / 4/2 小雨 / 4/3 暴雨
     - 3 天无法标定剂量曲线 → 系数借 NYC 出租车标定值(可迁移假设, 行人需重标)
     - 演示数据: 形态按 v3 预处理输出(n_trips/h, dist_m, speed_kmh)仿制
输出: 无线行人_调整过程示例_演示版.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; OR = '#ED7D31'; GRN = '#70AD47'
GREY = '#7F7F7F'; MUTED = '#666666'; DGREEN = '#375623'

rng = np.random.default_rng(41)
H = np.arange(24)

# ================= 合成演示数据（三天） =================
# 4/1 晴（基线日）: 通勤双峰
base = (900 * np.exp(-((H - 8) / 2.2) ** 2) + 1100 * np.exp(-((H - 18) / 2.6) ** 2)
        + 180 * np.exp(-((H - 13) / 3.5) ** 2) + 90)
base_j = base * (1 + rng.normal(0, 0.03, 24))                      # 加轻微噪声

# 4/3 暴雨: 逐时雨强(午后峰值 15mm/h, 日累计 ~60mm)
rain43 = np.zeros(24)
rain43[2:6] = [1.0, 1.5, 2.0, 1.2]
rain43[12:20] = [3.0, 6.0, 10.0, 15.0, 12.0, 8.0, 4.0, 2.0]
# 4/2 小雨
rain42 = np.zeros(24)
rain42[6:12] = [0.5, 1.0, 1.5, 2.5, 2.0, 1.0]

# NYC 系数(可迁移假设)
theta_v, RC_v = 0.052, 0.2
mult43 = 1 + theta_v * np.log1p(rain43 / RC_v)                      # 4/3 逐时乘子
mult42 = 1 + theta_v * np.log1p(rain42 / RC_v)

# "实测"(演示): 4/3 = 基线×乘子×噪声(假设系数完全可迁移)
act43 = base_j * mult43 * (1 + rng.normal(0, 0.08, 24))
act42 = base_j * mult42 * (1 + rng.normal(0, 0.08, 24))

# 速度: 行人基线 ~12km/h(步行+骑行混合), 雨天 ×exp(−δv), δv=0.025·ln(1+r/0.3)
v_base = 12.5 - 1.8 * np.exp(-((H - 8.5) / 2.0) ** 2) - 1.2 * np.exp(-((H - 18) / 2.5) ** 2)
dv43 = np.exp(-0.025 * np.log1p(rain43 / 0.3))
v_act43 = v_base * dv43 * (1 + rng.normal(0, 0.04, 24))

# 距离: 行人短途为主(中位~1.2km), LN; 三旋钮: μ1−0.061, w1+0.025, 长途×0.9
d_grid = np.linspace(0.05, 6.0, 200)
pdf0 = np.exp(-((np.log(d_grid) - np.log(1.2)) ** 2) / (2 * 0.75 ** 2)) / (d_grid * 0.75 * np.sqrt(2 * np.pi))
pdf0 /= pdf0.max()
pdf3 = np.exp(-((np.log(d_grid) - np.log(1.2 * np.exp(-0.061))) ** 2) / (2 * 0.75 ** 2)) / (d_grid * 0.75 * np.sqrt(2 * np.pi))
pdf3 = pdf3 / pdf3.max() * (1 + 0.10 * np.exp(-((np.log(d_grid) - np.log(0.8)) ** 2)))
pdf3 = np.where(d_grid > 3.5, pdf3 * 0.9, pdf3)
pdf3 /= pdf3.max()

# ================= 画图: 3 行 4 列 =================
fig, axes = plt.subplots(3, 4, figsize=(20, 11))
fig.suptitle('无线行人版 · 调整过程示例（数据集仅 4/1~4/3 三天 · 演示数据，形态按 v3 预处理输出仿制）\n'
             '流程：4/1 晴天=基线日 → 借 NYC 标定系数做逐时/三旋钮调整 → 合成 4/3 暴雨日 → 与"实测"对照',
             fontsize=14.5, fontweight='bold', y=0.985)

def style(ax, title, xlab='小时', ylab=''):
    ax.set_title(title, fontsize=11, loc='left', color=NAVY)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)

# ---------- 行1 出发时间 ----------
ax = axes[0, 0]
ax.bar(H, base_j, color='#8FAADC', edgecolor=NAVY, lw=0.6)
style(ax, 'A. 4/1 晴天基线（逐时出发量）', ylab='趟/时')
ax.text(0.97, 0.95, '三天里唯一晴天\n= 基线日（n=1）', transform=ax.transAxes,
        fontsize=9, color=RED, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF2CC', ec='#D6B656'))

ax = axes[0, 1]
ax.bar(H, rain43, color='#4472C4', alpha=0.75)
ax2 = ax.twinx()
ax2.plot(H, mult43, lw=2.4, color=RED, marker='o', ms=4)
ax2.set_ylabel('乘子 (1+δvol)', color=RED)
ax.set_ylim(0, 18); ax2.set_ylim(0.9, 1.5)
style(ax, 'B. 4/3 逐时雨强 → 出发量乘子', ylab='雨强 (mm/h)')
ax.text(0.97, 0.92, 'δvol(r)=0.052·ln(1+r/0.2)\n15mm/h → ×1.26', transform=ax.transAxes,
        fontsize=9, color=NAVY, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=NAVY, alpha=0.9))

ax = axes[0, 2]
ax.bar(H, base_j, color=GREY, alpha=0.35, label='基线(4/1)')
ax.plot(H, base_j * mult43, lw=2.4, color=RED, marker='o', ms=4, label='调整后(合成 4/3)')
style(ax, 'C. 基线 × 逐时乘子 = 合成暴雨日', ylab='趟/时')
ax.legend(fontsize=9)

ax = axes[0, 3]
ax.plot(H, act43, lw=2.2, color='#4472C4', marker='s', ms=4, label='4/3 "实测"（演示）')
ax.plot(H, base_j * mult43, lw=2.2, color=RED, ls='--', label='调整后合成')
err = np.abs(act43 - base_j * mult43) / np.maximum(act43, 1)
style(ax, f'D. 对照：逐时偏差中位 {np.median(err)*100:.0f}%', ylab='趟/时')
ax.legend(fontsize=9)

# ---------- 行2 距离 ----------
ax = axes[1, 0]
ax.plot(d_grid, pdf0, lw=2.4, color='#8FAADC')
ax.fill_between(d_grid, 0, pdf0, color='#8FAADC', alpha=0.25)
ax.axvline(1.2, color=GREY, ls=':')
style(ax, 'A. 4/1 基线距离分布（短途为主）', xlab='行程距离 (km)', ylab='密度')
ax.text(1.28, 0.55, '中位 ≈1.2 km\n(步行/骑行混合)', fontsize=9.5, color=MUTED)

ax = axes[1, 1]
ax.plot(d_grid, pdf0, lw=1.2, color=GREY, ls='--', label='基线')
ax.plot(d_grid, pdf3, lw=2.4, color=OR, label='旋钮①②③后')
ax.annotate('①μ1左移 −6%\n(短途更短)', xy=(0.75, pdf3[35]), xytext=(1.6, 0.85),
            fontsize=9, color=RED, arrowprops=dict(arrowstyle='->', color=RED))
ax.annotate('③长途×0.9', xy=(4.2, float(np.interp(4.2, d_grid, pdf3))), xytext=(4.4, 0.35),
            fontsize=9, color=RED, arrowprops=dict(arrowstyle='->', color=RED))
style(ax, 'B. 三旋钮（同 NYC 标定值）', xlab='行程距离 (km)', ylab='密度')
ax.legend(fontsize=9)

ax = axes[1, 2]
ax.plot(d_grid, pdf3, lw=2.4, color=OR)
ax.fill_between(d_grid, 0, pdf3, color=OR, alpha=0.2)
style(ax, 'C. 合成暴雨日距离分布', xlab='行程距离 (km)', ylab='密度')

ax = axes[1, 3]
bins = np.linspace(0.1, 5, 16)
# 演示实测: 短途偏移 + 噪声
d_act = np.concatenate([rng.lognormal(np.log(0.85), 0.72, 4200), rng.lognormal(np.log(2.2), 0.6, 800)])
ax.hist(d_act, bins=bins, density=True, alpha=0.35, color='#4472C4', label='4/3 "实测"（演示）')
ax.plot(d_grid, pdf3 * 12, lw=2.2, color=OR, ls='--', label='合成(×12 缩放)')
style(ax, 'D. 对照：形态方向一致', xlab='行程距离 (km)', ylab='密度')
ax.legend(fontsize=9)

# ---------- 行3 速度 ----------
ax = axes[2, 0]
ax.plot(H, v_base, lw=2.4, color='#8FAADC', marker='o', ms=4)
style(ax, 'A. 4/1 基线逐时中位速度', ylab='km/h')
ax.text(0.97, 0.95, '行人 ~12.5 km/h\n(步行+骑行混合)', transform=ax.transAxes,
        fontsize=9, color=MUTED, ha='right', va='top')

ax = axes[2, 1]
ax.bar(H, rain43, color='#4472C4', alpha=0.75)
ax2 = ax.twinx()
ax2.plot(H, dv43, lw=2.4, color=RED, marker='o', ms=4)
ax2.set_ylabel('速度乘子 exp(−δv)', color=RED)
ax.set_ylim(0, 18); ax2.set_ylim(0.75, 1.05)
style(ax, 'B. 雨强 → 速度乘子', ylab='雨强 (mm/h)')
ax.text(0.97, 0.92, 'δv=0.025·ln(1+r/0.3)\n15mm/h → 慢 11%', transform=ax.transAxes,
        fontsize=9, color=NAVY, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=NAVY, alpha=0.9))

ax = axes[2, 2]
ax.plot(H, v_base, lw=1.2, color=GREY, ls='--', label='基线')
ax.plot(H, v_base * dv43, lw=2.4, color=OR, label='调整后')
style(ax, 'C. 基线 × exp(−δv) = 暴雨日速度', ylab='km/h')
ax.legend(fontsize=9)

ax = axes[2, 3]
ax.plot(H, v_act43, lw=2.2, color='#4472C4', marker='s', ms=4, label='4/3 "实测"（演示）')
ax.plot(H, v_base * dv43, lw=2.2, color=OR, ls='--', label='调整后合成')
style(ax, f'D. 对照：偏差中位 {np.median(np.abs(v_act43 - v_base*dv43)/v_act43)*100:.0f}%', ylab='km/h')
ax.legend(fontsize=9)

fig.text(0.5, 0.003,
         '数据现实与诚实边界：仅 3 天数据 → ① 基线 n=1（4/1 晴天，无跨周稳健性）② 剂量系数全部借 NYC 出租车标定值（δvol/δv/三旋钮）——'
         '行人方向可能相反（雨压制户外出行），上线前必须用自有数据重标；③ 本图演示"标签生成→系数应用→对照"流程，真实数据接上后仅替换数字',
         ha='center', fontsize=10, color=DGREEN, fontweight='bold', wrap=True)

plt.tight_layout(rect=[0, 0.035, 1, 0.945])
fig.savefig(r'D:/AI/条件转化/无线行人_调整过程示例_演示版.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 无线行人_调整过程示例_演示版.png')
