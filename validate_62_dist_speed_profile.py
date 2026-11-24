# -*- coding: utf-8 -*-
"""validate_62: 出租车行程距离 / 速度 分布形态摸底(NYC 全样本, 不分天气)
   单位: 距离 km, 速度 km/h (原始数据为 mi / mph, 加载后换算: 1 mi = 1.609344 km)
   输出: 出租车_距离速度分布.png + 控制台分位数表
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#17375E'; RED = '#C00000'; ORANGE = '#ED7D31'; MUTED = '#666666'; BLUE = '#2E75B6'
MI2KM = 1.609344

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dist = A['dist'].astype(float) * MI2KM          # mi -> km
v = S['v'].astype(float) * MI2KM               # mph -> km/h
hh = S['hour'].astype(int)
D_OK = (0.5, 100)                               # 距离有效范围 km (原 0.3~60 mi)
V_OK = (1.6, 96)                                # 速度有效范围 km/h (原 1~60 mph)

qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print('=' * 60)
print(f'距离(km): n={len(dist):,}  min={dist.min():.3f}  max={dist.max():.1f}')
for q in qs:
    print(f'  P{q:<2d} {np.percentile(dist, q):7.2f}')
print(f'  mean={dist.mean():.2f}  median={np.median(dist):.2f}  std={dist.std():.2f}')
print(f'  偏度={stats.skew(dist):+.2f}  峰度={stats.kurtosis(dist):+.2f}')
segs = [(0, 2, '<2'), (2, 5, '2~5'), (5, 16, '5~16'), (16, 32, '16~32'), (32, 1e9, '>=32')]
print('  分段占比: ' + ' | '.join(f'{nm}km {(dist[(dist >= lo) & (dist < hi)]).size / len(dist) * 100:.1f}%' for lo, hi, nm in segs))
d_ok = dist[(dist > D_OK[0]) & (dist < D_OK[1])]
ld = np.log(d_ok)
print(f'  log距离: 偏度={stats.skew(ld):+.2f} 峰度={stats.kurtosis(ld):+.2f} (对照: 单一对数正态应为0/0)')

print('=' * 60)
print(f'速度(km/h): n={len(v):,}  min={v.min():.2f}  max={v.max():.1f}')
for q in qs:
    print(f'  P{q:<2d} {np.percentile(v, q):7.2f}')
print(f'  mean={v.mean():.2f}  median={np.median(v):.2f}  std={v.std():.2f}  CV={v.std()/v.mean():.2f}')
print(f'  偏度={stats.skew(v):+.2f}  峰度={stats.kurtosis(v):+.2f}')
m_ok = (v > V_OK[0]) & (v < V_OK[1])
v_ok = v[m_ok]
lv = np.log(v_ok)
z = (lv - lv.mean()) / lv.std()
in1 = (np.abs(z) <= 1).mean() * 100; in2 = (np.abs(z) <= 2).mean() * 100
print(f'  log速度: 均值={lv.mean():.3f} σ={lv.std():.3f} 偏度={stats.skew(lv):+.2f} 峰度={stats.kurtosis(lv):+.2f}')
print(f'  ±1σ内 {in1:.1f}% (正态68.3%) | ±2σ内 {in2:.1f}% (正态95.4%)')
print(f'  异常: v<1.6: {(v < 1.6).mean()*100:.2f}% | v>96: {(v > 96).mean()*100:.3f}%')
medh = np.array([np.median(v[hh == t]) for t in range(24)])
print('  分时段中位速度(km/h):', {t: round(float(medh[t]), 1) for t in [0, 3, 6, 8, 12, 17, 19, 22]})

# ================= 图 =================
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('出租车行程的距离与速度分布（NYC，全部样本，不分天气）', fontsize=16, fontweight='bold', y=0.99)
km_tk = [0.5, 1, 2, 5, 10, 20, 50, 100]
fmt_km = matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:g}')

# ---- 距离: 线性轴 ----
ax = axes[0, 0]
XMAX = 32
ax.hist(dist, bins=np.arange(0, XMAX + 0.8, 0.8), color=BLUE, edgecolor='white', linewidth=0.4)
med = float(np.median(dist)); mean = float(dist.mean())
ax.axvline(med, color=RED, lw=2.2)
ax.text(med + 0.5, ax.get_ylim()[1] * 0.9, f'中位 {med:.2f} km', fontsize=11.5, color=RED, fontweight='bold')
ax.axvline(mean, color=ORANGE, lw=2.2, ls='--')
ax.text(mean + 0.5, ax.get_ylim()[1] * 0.78, f'均值 {mean:.2f} km', fontsize=11.5, color='#843C0C', fontweight='bold')
ax.set_xlim(0, XMAX)
ax.set_xlabel('单程距离 (km)'); ax.set_ylabel('行程数')
ax.set_title('① 距离·线性轴: 右偏长尾\n   大多数很短, 少数很长把均值拉高', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.25)

# ---- 距离: log 轴 ----
ax = axes[0, 1]
ax.hist(d_ok, bins=np.logspace(np.log10(D_OK[0]), np.log10(D_OK[1]), 70), color=BLUE, edgecolor='white', linewidth=0.4)
ax.set_xscale('log')
ax.set_xticks(km_tk); ax.get_xaxis().set_major_formatter(fmt_km)
ax.axvline(med, color=RED, lw=2.2)
ax.set_xlabel('单程距离 (km, log 轴)'); ax.set_ylabel('行程数')
ax.set_title('② 距离·log 轴: 多个鼓包 = 多类人群混在一起\n   短途 / 中途 / 长途·机场', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.25, which='both')

# ---- 距离: 累计分布 ----
ax = axes[0, 2]
xs = np.sort(d_ok); ys = np.arange(1, len(xs) + 1) / len(xs) * 100
ax.plot(xs, ys, lw=2.8, color=NAVY)
p90 = float(np.percentile(d_ok, 90))
for p, c, dx in [(50, RED, 1.6), (90, ORANGE, 3.5)]:
    xq = float(np.percentile(d_ok, p))
    ax.plot([xq], [p], 'o', ms=10, color=c, zorder=5)
    ax.annotate(f'P{p} = {xq:.1f} km', (xq, p), xytext=(xq + dx, p - 9), fontsize=11.5, color=c, fontweight='bold')
ax.set_xscale('log')
ax.set_xticks(km_tk); ax.get_xaxis().set_major_formatter(fmt_km)
ax.set_ylim(0, 100)
ax.set_xlabel('单程距离 (km, log 轴)'); ax.set_ylabel('累计占比 (%)')
ax.set_title(f'③ 距离·累计分布: 一半行程 ≤{med:.2f}km,\n   九成 ≤{p90:.1f}km', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.25, which='both')

# ---- 速度: 线性轴 ----
ax = axes[1, 0]
VMAX = 72
ax.hist(v, bins=np.arange(0, VMAX + 1.2, 1.2), color=ORANGE, edgecolor='white', linewidth=0.4)
vm = float(np.median(v))
ax.axvline(vm, color=RED, lw=2.2)
ax.text(vm + 1, ax.get_ylim()[1] * 0.9, f'中位 {vm:.1f} km/h', fontsize=11.5, color=RED, fontweight='bold')
ax.set_xlim(0, VMAX)
ax.set_xlabel('行程平均速度 (km/h)'); ax.set_ylabel('行程数')
ax.set_title(f'④ 速度·线性轴: 单峰右偏\n   CV={v_ok.std()/v_ok.mean():.2f}, 各时段差异明显', fontsize=12, loc='left', color=NAVY)
ax.grid(alpha=0.25)

# ---- 速度: log 轴 + 正态对照 ----
ax = axes[1, 1]
lo, hi = 0.5, 4.5
ax.hist(lv, bins=np.linspace(lo, hi, 91), density=True, color='#F4B183', edgecolor='white', linewidth=0.4,
        label='实际 log 速度')
gx = np.linspace(lo, hi, 500)
ax.plot(gx, stats.norm.pdf(gx, lv.mean(), lv.std()), lw=2.8, color=NAVY,
        label=f'正态 N(μ={lv.mean():.2f}, σ={lv.std():.2f})')
ax.set_xlim(lo, hi)
ax.set_xlabel('log 速度 (ln km/h)'); ax.set_ylabel('密度')
ax.set_title(f'⑤ 速度·log 轴: 接近正态\n   ±1σ 内 {in1:.0f}% (正态 68%), ±2σ 内 {in2:.0f}% (正态 95%)',
             fontsize=12, loc='left', color=NAVY)
ax.legend(fontsize=10.5); ax.grid(alpha=0.25)

# ---- 速度: 分时段 ----
ax = axes[1, 2]
p25h = np.array([np.percentile(v[hh == t], 25) for t in range(24)])
p75h = np.array([np.percentile(v[hh == t], 75) for t in range(24)])
ax.fill_between(range(24), p25h, p75h, color='#F4B183', alpha=0.5, label='P25~P75')
ax.plot(range(24), medh, 'o-', lw=2.6, color=RED, ms=5, label='中位')
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel('出发小时'); ax.set_ylabel('速度 (km/h)')
ax.set_title(f'⑥ 速度·分时段: 凌晨最快({medh.max():.1f}), 晚高峰最慢({medh.min():.1f})\n   → 速度基线必须按钟点查表',
             fontsize=12, loc='left', color=NAVY)
ax.legend(fontsize=10.5); ax.grid(alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(r'D:/AI/条件转化/出租车_距离速度分布.png', dpi=150, facecolor='white')
print('\nsaved 出租车_距离速度分布.png')
