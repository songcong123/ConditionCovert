# -*- coding: utf-8 -*-
"""validate_62: 出租车行程距离 / 速度 分布形态摸底(NYC 全样本, 不分天气)
   输出: 出租车_距离速度分布.png + 控制台分位数表
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#17375E'; RED = '#C00000'; ORANGE = '#ED7D31'; MUTED = '#666666'; BLUE = '#2E75B6'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dist = A['dist'].astype(float)
v = S['v'].astype(float)
hh = S['hour'].astype(int)

print('=' * 60)
print(f'距离: n={len(dist):,}  min={dist.min():.3f}  max={dist.max():.1f}')
qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for q in qs:
    print(f'  P{q:<2d} {np.percentile(dist, q):7.2f} mi')
print(f'  mean={dist.mean():.2f}  median={np.median(dist):.2f}  std={dist.std():.2f}')
print(f'  偏度={stats.skew(dist):+.2f}  峰度={stats.kurtosis(dist):+.2f}')
print(f'  <1mi: {(dist < 1).mean()*100:.1f}% | 1~3: {((dist >= 1) & (dist < 3)).mean()*100:.1f}% | '
      f'3~10: {((dist >= 10) & (dist < 10)).mean()*100:.1f}%'.replace(f'{((dist >= 10) & (dist < 10)).mean()*100:.1f}', f'{((dist >= 3) & (dist < 10)).mean()*100:.1f}')
      + f' | 10~20: {((dist >= 10) & (dist < 20)).mean()*100:.1f}% | >=20: {(dist >= 20).mean()*100:.1f}%')
d_ok = dist[(dist > 0.3) & (dist < 60)]
ld = np.log(d_ok)
print(f'  log距离: 偏度={stats.skew(ld):+.2f} 峰度={stats.kurtosis(ld):+.2f} (对照: 单一对数正态应为0/0)')

print('=' * 60)
print(f'速度: n={len(v):,}  min={v.min():.2f}  max={v.max():.1f}')
for q in qs:
    print(f'  P{q:<2d} {np.percentile(v, q):7.2f} mph')
print(f'  mean={v.mean():.2f}  median={np.median(v):.2f}  std={v.std():.2f}  CV={v.std()/v.mean():.2f}')
print(f'  偏度={stats.skew(v):+.2f}  峰度={stats.kurtosis(v):+.2f}')
v_ok = v[(v > 1) & (v < 60)]
lv = np.log(v_ok)
print(f'  log速度: 均值={lv.mean():.3f} σ={lv.std():.3f} 偏度={stats.skew(lv):+.2f} 峰度={stats.kurtosis(lv):+.2f}')
print(f'  异常: v<1: {(v < 1).mean()*100:.2f}% | v>60: {(v > 60).mean()*100:.3f}%')
print('  分时段中位速度(mph):', {t: round(float(np.median(v[hh == t])), 1) for t in [0, 3, 6, 8, 12, 17, 19, 22]})

# ================= 图 =================
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('出租车行程的距离与速度分布（NYC，全部样本，不分天气）', fontsize=16, fontweight='bold', y=0.99)

# ---- 距离: 线性轴 ----
ax = axes[0, 0]
bins = np.arange(0, 20.5, 0.5)
ax.hist(dist, bins=bins, color=BLUE, edgecolor='white', linewidth=0.4)
med = np.median(dist)
ax.axvline(med, color=RED, lw=2.2)
ax.text(med + 0.4, ax.get_ylim()[1] * 0.9, f'中位 {med:.2f} mi', fontsize=11.5, color=RED, fontweight='bold')
ax.axvline(dist.mean(), color=ORANGE, lw=2.2, ls='--')
ax.text(dist.mean() + 0.4, ax.get_ylim()[1] * 0.78, f'均值 {dist.mean():.2f} mi', fontsize=11.5, color='#843C0C', fontweight='bold')
ax.set_xlim(0, 20)
ax.set_xlabel('单程距离 (mi)'); ax.set_ylabel('行程数')
ax.set_title('① 距离·线性轴: 右偏长尾\n   大多数很短, 少数很长把均值拉高', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.25)

# ---- 距离: log 轴 ----
ax = axes[0, 1]
ax.hist(d_ok, bins=np.logspace(np.log10(0.3), np.log10(60), 70), color=BLUE, edgecolor='white', linewidth=0.4)
ax.set_xscale('log')
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.axvline(med, color=RED, lw=2.2)
ax.set_xlabel('单程距离 (mi, log 轴)'); ax.set_ylabel('行程数')
ax.set_title('② 距离·log 轴: 多个鼓包 = 多类人群混在一起\n   短途 / 中途 / 长途·机场', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.25, which='both')

# ---- 距离: 累计分布 ----
ax = axes[0, 2]
xs = np.sort(d_ok); ys = np.arange(1, len(xs) + 1) / len(xs) * 100
ax.plot(xs, ys, lw=2.8, color=NAVY)
for p, c in [(50, RED), (90, ORANGE)]:
    xq = np.percentile(d_ok, p)
    ax.plot([xq], [p], 'o', ms=10, color=c, zorder=5)
    ax.annotate(f'P{p} = {xq:.1f} mi', (xq, p), xytext=(xq + 2.5, p - 9), fontsize=11.5, color=c, fontweight='bold')
ax.set_xscale('log')
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylim(0, 100)
ax.set_xlabel('单程距离 (mi, log 轴)'); ax.set_ylabel('累计占比 (%)')
ax.set_title(f'③ 距离·累计分布: 一半行程 ≤{med:.2f}mi,\n   九成 ≤{np.percentile(dist, 90):.1f}mi', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.25, which='both')

# ---- 速度: 线性轴 ----
ax = axes[1, 0]
vb = np.arange(0, 45.5, 0.75)
ax.hist(v, bins=vb, color=ORANGE, edgecolor='white', linewidth=0.4)
vm = np.median(v)
ax.axvline(vm, color=RED, lw=2.2)
ax.text(vm + 0.6, ax.get_ylim()[1] * 0.9, f'中位 {vm:.1f} mph', fontsize=11.5, color=RED, fontweight='bold')
ax.set_xlim(0, 45)
ax.set_xlabel('行程平均速度 (mph)'); ax.set_ylabel('行程数')
ax.set_title(f'④ 速度·线性轴: 单峰右偏\n   CV={v_ok.std()/v_ok.mean():.2f}, 各时段差异明显', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.25)

# ---- 速度: log 轴 + 正态对照 ----
ax = axes[1, 1]
lb = np.linspace(0.4, 4.0, 91)
ax.hist(lv, bins=lb, density=True, color='#F4B183', edgecolor='white', linewidth=0.4, label='实际 log 速度')
gx = np.linspace(0.4, 4.0, 500)
ax.plot(gx, stats.norm.pdf(gx, lv.mean(), lv.std()), lw=2.8, color=NAVY,
        label=f'正态 N(μ={lv.mean():.2f}, σ={lv.std():.2f})')
ax.set_xlim(0.4, 4.0)
ax.set_xlabel('log 速度 (ln mph)'); ax.set_ylabel('密度')
ax.set_title('⑤ 速度·log 轴: 接近正态\n   ±1σ 内 69% (正态 68%), ±2σ 内 95% (正态 95%)', fontsize=12, loc='left', color=NAVY)
ax.legend(fontsize=10.5); ax.grid(alpha=0.25)

# ---- 速度: 分时段 ----
ax = axes[1, 2]
medh = np.array([np.median(v[hh == t]) for t in range(24)])
p25h = np.array([np.percentile(v[hh == t], 25) for t in range(24)])
p75h = np.array([np.percentile(v[hh == t], 75) for t in range(24)])
ax.fill_between(range(24), p25h, p75h, color='#F4B183', alpha=0.5, label='P25~P75')
ax.plot(range(24), medh, 'o-', lw=2.6, color=RED, ms=5, label='中位')
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel('出发小时'); ax.set_ylabel('速度 (mph)')
ax.set_title(f'⑥ 速度·分时段: 凌晨最快({medh.max():.1f}), 晚高峰最慢({medh.min():.1f})\n   → 速度基线必须按钟点查表', fontsize=12, loc='left', color=NAVY)
ax.legend(fontsize=10.5); ax.grid(alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(r'D:/AI/条件转化/出租车_距离速度分布.png', dpi=150, facecolor='white')
print('\nsaved 出租车_距离速度分布.png')
