# -*- coding: utf-8 -*-
"""validate_63b: "从四个分布采样 → 一条轨迹" 大字版(供 PPT 缩放后仍可读)
   由 validate_63_sampling_demo.py 派生: 字号整体放大约 2 倍, 为腾出空间做了精简 ——
     去掉底部"怎么读"说明、热力图说明、箭头旁文字; 名片改为横条; 地图/标签文字缩短
   数据与抽样口径同 validate_63(示例取自真实数据的一条行程)
   输出: 四分布采样_一条轨迹_大字版.png
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
NAVY = '#1F4E79'; RED = '#C00000'
MI2KM = 1.609344
COSL = np.cos(np.radians(40.7))

# ---- 字号(原版 → 大字版) ----
FS_TITLE = 42                      # 原版 22
FS_PT = 26                         # 面板标题 14
FS_TAG, FS_TAG2 = 24, 22           # "抽到"标签 13 / 12
FS_AXIS, FS_TICK = 22, 20          # 轴标签 12 / 刻度 11
FS_MAP = 22                        # 地图标签 11.5
FS_CARD, FS_CARD_S = 24, 19        # 名片 13.5 / 说明

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

op = A['od_pair'].astype(np.int64)
C = np.bincount(op, weights=A['od_cnt'].astype(np.float64), minlength=264 * 264).reshape(264, 264)
k_od = M['idx'][:, 1].astype(np.int64) * 264 + M['idx'][:, 2].astype(np.int64)
n_od = np.bincount(k_od, minlength=264 * 264).astype(float)
KM = (np.bincount(k_od, weights=M['miles'].astype(np.float64) * MI2KM, minlength=264 * 264) / np.maximum(n_od, 1)).reshape(264, 264)

# ---- 抽样(与 validate_63 完全相同的种子与规则, 保证示例一致) ----
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
print(f"示例(种子{p['seed']}): {H:02d}时 | 网格#{O} → #{T} | {D:.2f} km | {V:.1f} km/h | {mins:.1f} 分钟")

# ================= 画布 =================
fig = plt.figure(figsize=(16, 9), facecolor='white')
fig.text(0.5, 0.950, '从四个分布各抽一个数，拼成一条轨迹', ha='center', va='center', fontsize=FS_TITLE, fontweight='bold', color=NAVY)

def tag(ax, x, y, txt, color, fs):
    ax.annotate(txt, (x, y), xycoords='axes fraction', ha='center', va='center', fontsize=fs, fontweight='bold',
                color=color, zorder=10, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=color, lw=2.2))

def clean(ax, keep_bottom=True):
    for nm, s_ in ax.spines.items():
        s_.set_visible(keep_bottom and nm == 'bottom')
        if s_.get_visible(): s_.set_linewidth(1.6)

# 纵向分配(英寸, 自上而下): 顶边距0.08 + 标题0.70 + 间隔0.08 + 面板标题区0.42 + 绘图区2.66 + 横轴区0.91
#                           + 行间隔0.08 + 面板标题区0.42 + 绘图区2.66 + 横轴区0.91 + 底边距0.08 = 9.00
W, Hh = 0.242, 2.66 / 9
X1, X2 = 0.02, 0.292
YU, YL = 0.562, 0.110

# ---- ① 出发时间 ----
ax = fig.add_axes([X1, YU, W, Hh])
xs = np.arange(24)
ax.fill_between(xs, hour_cnt / hour_cnt.max(), color=BLUE, alpha=0.25)
ax.plot(xs, hour_cnt / hour_cnt.max(), lw=4, color=BLUE)
ax.axvline(H, color=RED, lw=2.4, ls='--', zorder=5)
ax.plot([H], [hour_cnt[H] / hour_cnt.max()], 'o', ms=20, color=RED, zorder=6, markeredgecolor='white', markeredgewidth=2.5)
ax.set_xlim(0, 23); ax.set_ylim(0, 1.45)
ax.set_xticks(range(0, 24, 6)); ax.set_yticks([])
ax.set_xlabel('出发小时（时）', fontsize=FS_AXIS, labelpad=6)
ax.tick_params(labelsize=FS_TICK, length=6, width=1.2)
ax.set_title('① 出发时间分布', fontsize=FS_PT, color=BLUE, fontweight='bold', loc='left', pad=8)
clean(ax)
tag(ax, 0.66, 0.90, f'抽到：{H:02d} 时', BLUE, FS_TAG)

# ---- 右侧地图视野(供②画红框) ----
x0, y0, x1, y1 = lon[O], lat[O], lon[T], lat[T]
mx, my = (x0 + x1) / 2, (y0 + y1) / 2
span = max(abs(y1 - y0), abs(x1 - x0) * COSL)
hy = max(0.012, span * 1.5); hx = hy / COSL
LON0, LON1 = lon[ok].min() - 0.02, lon[ok].max() + 0.02
LAT0 = lat[ok].min() - 0.012
LAT1 = lat[ok].max() + 0.066                    # 顶部多留白, 给"抽到"标签让位

# ---- ② 空间网格 ----
ax = fig.add_axes([X2, YU, W, Hh])
sh = share_zone.copy(); sh[~ok] = 0
sz = 12 + 95 * sh / sh.max()
ax.scatter(lon[ok], lat[ok], s=sz[ok], c=PURPLE, alpha=0.45, edgecolor='none', zorder=3)
ax.add_patch(Rectangle((mx - hx, my - hy), 2 * hx, 2 * hy, fill=False, ec=RED, lw=2.6, ls='--', zorder=7))
ax.plot([x0], [y0], '*', ms=24, c='#FFD966', markeredgecolor=RED, markeredgewidth=2, zorder=8)
ax.set_aspect(1 / COSL)
ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('② 空间网格分布', fontsize=FS_PT, color=PURPLE, fontweight='bold', loc='left', pad=8)
clean(ax, keep_bottom=False)
tag(ax, 0.62, 0.912, f'抽到：网格 #{O}', PURPLE, FS_TAG2)

# ---- ③ 距离 ----
ax = fig.add_axes([X1, YL, W, Hh])
bins = np.arange(0, 30.5, 1.0)
cnt, _ = np.histogram(dist, bins=bins)
ax.bar(bins[:-1], cnt, width=0.9, color=ORANGE, alpha=0.75, edgecolor='white', linewidth=0.6)
ax.axvline(D, color=RED, lw=2.6, ls='--', zorder=5)
ax.plot([D], [cnt[max(0, int(D) - 1)]], 'o', ms=20, color=RED, zorder=6, markeredgecolor='white', markeredgewidth=2.5)
ax.set_xlim(0, 30); ax.set_ylim(0, cnt.max() * 1.45)
ax.set_xticks(range(0, 31, 10)); ax.set_yticks([])
ax.set_xlabel('单程距离（km）', fontsize=FS_AXIS, labelpad=6)
ax.tick_params(labelsize=FS_TICK, length=6, width=1.2)
ax.set_title('③ 距离分布', fontsize=FS_PT, color=DORANGE, fontweight='bold', loc='left', pad=8)
clean(ax)
tag(ax, 0.64, 0.90, f'抽到：{D:.1f} km', DORANGE, FS_TAG)

# ---- ④ 速度 ----
ax = fig.add_axes([X2, YL, W, Hh])
bins = np.arange(0, 72.5, 2.0)
cnt, _ = np.histogram(spd, bins=bins)
ax.bar(bins[:-1], cnt, width=1.8, color=GREEN, alpha=0.75, edgecolor='white', linewidth=0.6)
ax.axvline(V, color=RED, lw=2.6, ls='--', zorder=5)
ax.plot([V], [cnt[max(0, int(V // 2))]], 'o', ms=20, color=RED, zorder=6, markeredgecolor='white', markeredgewidth=2.5)
ax.set_xlim(0, 70); ax.set_ylim(0, cnt.max() * 1.45)
ax.set_xticks(range(0, 71, 20)); ax.set_yticks([])
ax.set_xlabel('速度（km/h）', fontsize=FS_AXIS, labelpad=6)
ax.tick_params(labelsize=FS_TICK, length=6, width=1.2)
ax.set_title('④ 速度分布', fontsize=FS_PT, color=GREEN, fontweight='bold', loc='left', pad=8)
clean(ax)
tag(ax, 0.63, 0.90, f'抽到：{V:.0f} km/h', GREEN, FS_TAG2)

# ---- 中间大箭头 ----
fig.patches.append(FancyArrowPatch((0.545, 0.49), (0.578, 0.49), transform=fig.transFigure,
                                   arrowstyle='-|>', mutation_scale=60, lw=7, color=NAVY))

# ---- 右: 地图(红框内放大) ----
AXW, AXH = 0.40 * 16, 0.588 * 9                      # 英寸
ax = fig.add_axes([0.585, 0.27, 0.40, 0.588])
ax.scatter(lon[ok], lat[ok], marker='s', s=70, c='#D5D5D5', edgecolor='none', zorder=2)
rad = 0.22
ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
                             mutation_scale=44, lw=6, color=ORANGE, zorder=6, shrinkA=15, shrinkB=15))
ax.plot([x0], [y0], '*', ms=40, c='#FFD966', markeredgecolor=PURPLE, markeredgewidth=3.2, zorder=9)
ax.plot([x1], [y1], 'o', ms=20, c=ORANGE, markeredgecolor='white', markeredgewidth=2.5, zorder=9)
ax.set_aspect(1 / COSL)
ax.set_xlim(mx - hx, mx + hx); ax.set_ylim(my - hy, my + hy)
ax.set_xticks([]); ax.set_yticks([])
clean(ax, keep_bottom=False)
ax.set_title('拼成的一条轨迹（红框内放大）', fontsize=FS_PT, color=NAVY, fontweight='bold', loc='center', pad=6)

u = np.array([(x1 - x0) * COSL, y1 - y0]); u = u / np.linalg.norm(u)
n = np.array([u[1], -u[0]])
def lab(txt, xy, off, color):
    ax.annotate(txt, xy, xytext=tuple(off), textcoords='offset points', fontsize=FS_MAP, fontweight='bold', color=color,
                ha='center', va='center', zorder=10, annotation_clip=False,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, lw=2.2))
Lpt = np.hypot((x1 - x0) * COSL, y1 - y0) / (2 * hy) * (AXH - 0.55) * 72    # 起终点连线的屏幕长度(pt)
lab(f'起点 #{O}', (x0, y0), -u * 54 + n * 37, PURPLE)
lab(f'{H:02d} 时', (x0, y0), -u * 54 - n * 37, BLUE)
lab(f'{D:.1f} km', (mx, my), n * (rad * Lpt / 2 + 28), DORANGE)
lab(f'{V:.0f} km/h', (mx, my), -n * 37, GREEN)
lab(f'终点 #{T}', (x1, y1), u * 54 + n * 10, DORANGE)

# ---- 右: 轨迹名片(横条, 坐标=英寸) ----
CW, CH = 0.40 * 16, 0.244 * 9
ax = fig.add_axes([0.585, 0.009, 0.40, 0.244]); ax.set_xlim(0, CW); ax.set_ylim(0, CH)
ax.set_aspect('equal'); ax.axis('off')
ax.add_patch(FancyBboxPatch((0.02, 0.02), CW - 0.04, CH - 0.04, boxstyle='round,pad=0,rounding_size=0.14',
                            fc='#F7F9FC', ec=NAVY, lw=3))
XC = 3.0                                              # 第二列文字起点(英寸)
items = [(1, 0.72, 1.74, f'{H:02d} 时出发', BLUE), (2, XC, 1.74, f'网格 #{O} 出发', PURPLE),
         (3, 0.72, 1.19, f'{D:.1f} km', DORANGE), (4, XC, 1.19, f'{V:.0f} km/h', GREEN)]
for k, x, y, txt, c in items:
    ax.add_patch(Circle((x - 0.30, y), 0.21, fc=c, ec='none', zorder=3))
    ax.text(x - 0.30, y - 0.015, str(k), ha='center', va='center', fontsize=18, fontweight='bold', color='white', zorder=4)
    ax.text(x, y, txt, ha='left', va='center', fontsize=FS_CARD, fontweight='bold', color=c)
ax.plot([0.25, CW - 0.25], [0.80, 0.80], color='#BFBFBF', lw=2)
ax.text(0.25, 0.40, '用时', ha='left', va='center', fontsize=FS_CARD, color='#555555')
ax.text(1.02, 0.40, f'约 {mins:.0f} 分钟', ha='left', va='center', fontsize=FS_CARD + 2, fontweight='bold', color=RED)
ax.text(XC, 0.40, f'（{D:.1f} ÷ {V:.0f} × 60，不用抽）', ha='left', va='center', fontsize=FS_CARD_S, color='#666666')

fig.savefig(r'D:/AI/条件转化/四分布采样_一条轨迹_大字版.png', dpi=200, facecolor='white')
print('saved 四分布采样_一条轨迹_大字版.png')
