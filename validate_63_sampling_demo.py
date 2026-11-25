# -*- coding: utf-8 -*-
"""validate_63: "从四个分布采样 → 一条轨迹" 形象展示图
   四个分布(左 2x2): 出发时间(蓝) / 空间网格(紫) / 距离(橙) / 速度(绿) —— 沿用汇报四色约定
   右边: 地图局部放大画出这条轨迹(红框=在②全城图上的位置) + "轨迹名片"
   示例取自真实数据:
     · 时间/距离/速度 = 从 627 万条真实行程里抽一条(三者天然一致; 为便于展示限定为白天 3~10km)
     · 网格           = 按全天各区出发份额抽取
     · 终点           = 该起点网格的真实去向中, 里程与抽到距离相近者(按真实次数加权)抽取
   输出: 四分布采样_一条轨迹.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E75B6'; PURPLE = '#7030A0'; ORANGE = '#ED7D31'; GREEN = '#2E7D32'
DORANGE = '#C55A11'
NAVY = '#1F4E79'; DARK = '#1A1A1A'; RED = '#C00000'
MI2KM = 1.609344
COSL = np.cos(np.radians(40.7))            # 经度 1 度 ≈ 纬度 cos(40.7°) 度

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
Z = np.load(r'D:/AI/条件转化/.flow_zones.npz')
M = np.load(r'D:/AI/条件转化/.flow_odm.npz', allow_pickle=True)

lat = Z['lat'].astype(float); lon = Z['lon'].astype(float)
ok = ~np.isnan(lat) & ~np.isnan(lon)
dist = A['dist'].astype(float) * MI2KM
spd = S['v'].astype(float) * MI2KM
hr = S['hour'].astype(int)
hour_cnt = A['hour_cnt'].astype(float).sum(0)
share_zone = A['pu_cnt'].astype(float).sum(0)

# ---- OD: 真实次数矩阵 + 真实平均里程矩阵(km) ----
op = A['od_pair'].astype(np.int64)
C = np.bincount(op, weights=A['od_cnt'].astype(np.float64), minlength=264 * 264).reshape(264, 264)
k_od = M['idx'][:, 1].astype(np.int64) * 264 + M['idx'][:, 2].astype(np.int64)
n_od = np.bincount(k_od, minlength=264 * 264).astype(float)
KM = (np.bincount(k_od, weights=M['miles'].astype(np.float64) * MI2KM, minlength=264 * 264) / np.maximum(n_od, 1)).reshape(264, 264)

# ---- 抽样: 抽一条白天的真实行程作为示例 ----
p = None
for seed in range(200):
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(dist)))
    if 7 <= hr[i] <= 19 and 3 <= dist[i] <= 10 and 10 <= spd[i] <= 35:
        o = int(rng.choice(np.where(ok)[0], p=share_zone[ok] / share_zone[ok].sum()))
        cand = np.where(ok & (C[o] > 0))[0]
        cand = cand[cand != o]
        bw = max(1.5, 0.3 * dist[i])
        w = C[o, cand] * np.exp(-((KM[o, cand] - dist[i]) / bw) ** 2)
        t = int(rng.choice(cand, p=w / w.sum()))
        p = dict(seed=seed, h=int(hr[i]), d=float(dist[i]), v=float(spd[i]), o=o, t=t)
        break
H, D, V, O, T = p['h'], p['d'], p['v'], p['o'], p['t']
mins = D / V * 60
print(f"示例(种子{p['seed']}): {H:02d}时出发 | 网格#{O} → #{T} | 行程 {D:.2f} km | 速度 {V:.1f} km/h | 用时 {mins:.1f} 分钟")
print(f"该去向真实平均里程 {KM[O, T]:.2f} km, 真实次数 {int(C[O, T])}")

# ================= 画布 =================
fig = plt.figure(figsize=(16, 9), facecolor='white')
fig.text(0.5, 0.955, '生成一条轨迹 = 从四个分布各抽一个数', ha='center', fontsize=22, fontweight='bold', color=NAVY)
fig.text(0.5, 0.912, '左边四个分布各抽一个数，拼成右边的一条轨迹；用时不用抽，由「距离 ÷ 速度」算出',
         ha='center', fontsize=13.5, color='#555555')

def tag(ax, x, y, txt, color, fs=13):
    ax.annotate(txt, (x, y), xycoords='axes fraction', ha='center', va='center', fontsize=fs, fontweight='bold',
                color=color, zorder=10,
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=color, lw=1.6))

def clean(ax, keep_bottom=True):
    for nm, s_ in ax.spines.items():
        s_.set_visible(keep_bottom and nm == 'bottom')

W, Hh = 0.19, 0.325
# ---- ① 出发时间 ----
ax = fig.add_axes([0.045, 0.545, W, Hh])
xs = np.arange(24)
ax.fill_between(xs, hour_cnt / hour_cnt.max(), color=BLUE, alpha=0.25)
ax.plot(xs, hour_cnt / hour_cnt.max(), lw=2.4, color=BLUE)
ax.axvline(H, color=RED, lw=1.4, ls='--', zorder=5)
ax.plot([H], [hour_cnt[H] / hour_cnt.max()], 'o', ms=13, color=RED, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
ax.set_xlim(0, 23); ax.set_ylim(0, 1.32)
ax.set_xticks(range(0, 24, 6)); ax.set_yticks([])
ax.set_xlabel('出发小时（时）', fontsize=12)
ax.tick_params(labelsize=11)
ax.set_title('① 出发时间分布', fontsize=14, color=BLUE, fontweight='bold', loc='left')
clean(ax)
tag(ax, 0.66, 0.90, f'抽到：{H:02d} 时', BLUE)

# ---- 右侧地图的视野(先算, 供②画红框) ----
x0, y0, x1, y1 = lon[O], lat[O], lon[T], lat[T]
mx, my = (x0 + x1) / 2, (y0 + y1) / 2
span = max(abs(y1 - y0), abs(x1 - x0) * COSL)
hy = max(0.012, span * 1.3); hx = hy / COSL
LON0, LON1 = lon[ok].min() - 0.02, lon[ok].max() + 0.02
LAT0, LAT1 = lat[ok].min() - 0.012, lat[ok].max() + 0.012

# ---- ② 空间网格 ----
ax = fig.add_axes([0.265, 0.545, W, Hh])
sh = share_zone.copy(); sh[~ok] = 0
sz = 8 + 70 * sh / sh.max()
ax.scatter(lon[ok], lat[ok], s=sz[ok], c=PURPLE, alpha=0.45, edgecolor='none', zorder=3)
ax.add_patch(Rectangle((mx - hx, my - hy), 2 * hx, 2 * hy, fill=False, ec=RED, lw=1.6, ls='--', zorder=7))
ax.plot([x0], [y0], '*', ms=19, c='#FFD966', markeredgecolor=RED, markeredgewidth=1.5, zorder=8)
ax.set_aspect(1 / COSL)
ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('② 空间网格分布（全城热力图）', fontsize=14, color=PURPLE, fontweight='bold', loc='left')
clean(ax, keep_bottom=False)
ax.text(0.5, 0.02, '点越大 = 这个网格出发的人越多', transform=ax.transAxes, ha='center', fontsize=10.5, color='#666666')
tag(ax, 0.70, 0.88, f'抽到：网格 #{O}', PURPLE, fs=12)

# ---- ③ 距离 ----
ax = fig.add_axes([0.045, 0.10, W, Hh])
bins = np.arange(0, 30.5, 1.0)
cnt, _ = np.histogram(dist, bins=bins)
ax.bar(bins[:-1], cnt, width=0.9, color=ORANGE, alpha=0.75, edgecolor='white', linewidth=0.4)
ax.axvline(D, color=RED, lw=2, ls='--', zorder=5)
ax.plot([D], [cnt[max(0, int(D) - 1)]], 'o', ms=13, color=RED, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
ax.set_xlim(0, 30); ax.set_ylim(0, cnt.max() * 1.30)
ax.set_xticks(range(0, 31, 10)); ax.set_yticks([])
ax.set_xlabel('单程距离（km）', fontsize=12)
ax.tick_params(labelsize=11)
ax.set_title('③ 距离分布', fontsize=14, color=DORANGE, fontweight='bold', loc='left')
clean(ax)
tag(ax, 0.66, 0.90, f'抽到：{D:.1f} km', DORANGE)

# ---- ④ 速度 ----
ax = fig.add_axes([0.265, 0.10, W, Hh])
bins = np.arange(0, 72.5, 2.0)
cnt, _ = np.histogram(spd, bins=bins)
ax.bar(bins[:-1], cnt, width=1.8, color=GREEN, alpha=0.75, edgecolor='white', linewidth=0.4)
ax.axvline(V, color=RED, lw=2, ls='--', zorder=5)
ax.plot([V], [cnt[max(0, int(V // 2))]], 'o', ms=13, color=RED, zorder=6, markeredgecolor='white', markeredgewidth=1.5)
ax.set_xlim(0, 70); ax.set_ylim(0, cnt.max() * 1.30)
ax.set_xticks(range(0, 71, 20)); ax.set_yticks([])
ax.set_xlabel('速度（km/h）', fontsize=12)
ax.tick_params(labelsize=11)
ax.set_title('④ 速度分布', fontsize=14, color=GREEN, fontweight='bold', loc='left')
clean(ax)
tag(ax, 0.66, 0.90, f'抽到：{V:.0f} km/h', GREEN)

# ---- 中间大箭头 ----
arr = FancyArrowPatch((0.468, 0.49), (0.512, 0.49), transform=fig.transFigure, arrowstyle='-|>',
                      mutation_scale=38, lw=4.5, color=NAVY)
fig.patches.append(arr)
fig.text(0.49, 0.545, '四个数', ha='center', fontsize=13.5, fontweight='bold', color=NAVY)
fig.text(0.49, 0.435, '一条轨迹', ha='center', fontsize=13.5, fontweight='bold', color=NAVY)

# ---- 右: 地图(红框内放大)上的轨迹 ----
ax = fig.add_axes([0.545, 0.40, 0.235, 0.47])
ax.scatter(lon[ok], lat[ok], marker='s', s=30, c='#D5D5D5', edgecolor='none', zorder=2)
rad = 0.22
ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
                             mutation_scale=28, lw=3.6, color=ORANGE, zorder=6, shrinkA=9, shrinkB=9))
ax.plot([x0], [y0], '*', ms=26, c='#FFD966', markeredgecolor=PURPLE, markeredgewidth=2.2, zorder=9)
ax.plot([x1], [y1], 'o', ms=13, c=ORANGE, markeredgecolor='white', markeredgewidth=1.5, zorder=9)
ax.set_aspect(1 / COSL)
ax.set_xlim(mx - hx, mx + hx); ax.set_ylim(my - hy, my + hy)
ax.set_xticks([]); ax.set_yticks([])
clean(ax, keep_bottom=False)
ax.set_title('拼成的轨迹（左图红框内，放大）', fontsize=13.5, color=NAVY, fontweight='bold', loc='center', pad=4)

# 标签: 屏幕方向上, u = 起点→终点, n = 弧线凸出的一侧
u = np.array([(x1 - x0) * COSL, y1 - y0]); u = u / np.linalg.norm(u)
n = np.array([u[1], -u[0]])
def lab(txt, xy, off, color, fs=11.5):
    ax.annotate(txt, xy, xytext=tuple(off), textcoords='offset points', fontsize=fs, fontweight='bold', color=color,
                ha='center', va='center', zorder=10, annotation_clip=False,
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=color, lw=1.4))
Lpt = np.hypot((x1 - x0) * COSL, y1 - y0) / (2 * hy) * 0.235 * 16 * 72    # 起终点连线的屏幕长度(pt)
lab(f'起点：网格 #{O}', (x0, y0), -u * 31 + n * 20, PURPLE)
lab(f'{H:02d} 时出发', (x0, y0), -u * 31 - n * 20, BLUE)
lab(f'全程 {D:.1f} km', (mx, my), n * (rad * Lpt / 2 + 21), DORANGE)       # 贴在弧线顶点外侧
lab(f'{V:.0f} km/h', (mx, my), -n * 21, GREEN)
lab(f'终点：网格 #{T}', (x1, y1), u * 26 + n * 6, DORANGE)

# ---- 右: 轨迹名片 ----
ax = fig.add_axes([0.795, 0.40, 0.185, 0.47]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.add_patch(FancyBboxPatch((0.02, 0.01), 0.96, 0.98, boxstyle='round,pad=0.01,rounding_size=0.05',
                            fc='#F7F9FC', ec=NAVY, lw=2))
ax.text(0.5, 0.93, '一条轨迹', ha='center', va='center', fontsize=17, fontweight='bold', color=NAVY)
rows = [('几点出发', f'{H:02d} 时', BLUE), ('从哪出发', f'网格 #{O}', PURPLE),
        ('走多远', f'{D:.1f} km', DORANGE), ('走多快', f'{V:.0f} km/h', GREEN)]
for k, (q, val, c) in enumerate(rows):
    y = 0.78 - k * 0.135
    ax.add_patch(Circle((0.10, y), 0.038, fc=c, ec='none', zorder=3))
    ax.text(0.10, y, str(k + 1), ha='center', va='center', fontsize=11.5, fontweight='bold', color='white', zorder=4)
    ax.text(0.17, y, q, ha='left', va='center', fontsize=13.5, color='#444444')
    ax.text(0.93, y, val, ha='right', va='center', fontsize=15, fontweight='bold', color=c)
ax.plot([0.08, 0.92], [0.245, 0.245], color='#BFBFBF', lw=1.2)
ax.text(0.10, 0.165, '用时多久', ha='left', va='center', fontsize=13.5, color='#444444')
ax.text(0.93, 0.165, f'约 {mins:.0f} 分钟', ha='right', va='center', fontsize=15, fontweight='bold', color=RED)
ax.text(0.5, 0.095, f'= {D:.1f} ÷ {V:.0f} × 60，不用抽', ha='center', va='center', fontsize=11.5, color='#666666')

# ---- 底部读法 ----
fig.text(0.545, 0.275, '怎么读', fontsize=14, fontweight='bold', color=NAVY, va='center')
fig.text(0.545, 0.218, '① 几点出发   ② 从哪个网格出发   ③ 走多远，决定终点   ④ 走多快',
         fontsize=12, color=DARK, va='center')
fig.text(0.545, 0.162, '四个数拼成一条轨迹的描述；重复抽样，就得到全城一天的轨迹。',
         fontsize=12, color=DARK, va='center')
fig.text(0.545, 0.107, '示例取自真实数据（纽约出租车 627 万条行程）中的一条：时间、距离、速度来自同一条行程；',
         fontsize=10.5, color='#666666', va='center')
fig.text(0.545, 0.073, '网格按各区出发份额抽取，终点按该网格真实去向中里程相近者抽取。',
         fontsize=10.5, color='#666666', va='center')
fig.text(0.545, 0.039, '图上的连线只是示意，实际沿道路走，里程更长。',
         fontsize=10.5, color='#666666', va='center')

fig.savefig(r'D:/AI/条件转化/四分布采样_一条轨迹.png', dpi=200, facecolor='white')
print('saved 四分布采样_一条轨迹.png')
