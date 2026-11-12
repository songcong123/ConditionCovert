# -*- coding: utf-8 -*-
"""暴雨对起点(PU)分布的完整解剖(用 .rain_spatial.npz, 无需重扫):
1) 全城变化地图(≥4mm/h 雨时 vs 同钟点晴天, 相对份额变化%)
2) 分组统计: 机场/曼哈顿核心/曼哈顿其余/内环(<5km)/外环(≥5km) × 两剂量箱
3) 份额变化 ~ 离市中心距离 的梯度/相关
输出 暴雨起点分布分析.png"""
import numpy as np, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import geopandas as gpd

R = np.load('D:/AI/条件转化/.rain_spatial.npz')
puA, puB, howA, howB = R['puA'], R['puB'], R['howA'], R['howB']
dry = R['pu_dry_how'].reshape(7 * 24, 264)
Z = np.load('D:/AI/条件转化/.flow_zones.npz')
dist_km = Z['dist_km']; lat, lon = Z['lat'], Z['lon']
ds = dry / np.maximum(dry.sum(1, keepdims=True), 1)

def shares(pu, how):
    w = how / max(how.sum(), 1)
    b = (w[:, None] * ds).sum(0); b /= b.sum()
    r = pu / pu.sum()
    return r, b

rA, bA = shares(puA, howA)
rB, bB = shares(puB, howB)
relB = np.where(bB > 0.002, (rB / np.maximum(bB, 1e-9) - 1) * 100, np.nan)   # 相对变化%, 过滤基线<0.2%
relA = np.where(bA > 0.002, (rA / np.maximum(bA, 1e-9) - 1) * 100, np.nan)

# 分组
AIR = [1, 132, 138]
CORE = [161, 162, 163, 164, 230, 231, 79, 24, 225]      # 中城/时代广场/联合广场/东西村
d_cbd = dist_km[161, 1:264]
MANH = [i for i in range(1, 264) if i not in AIR + CORE and 48 <= i <= 216 and _manh(i)] if False else None
# 用 lookup 的 borough 判曼哈顿
bor = {}
zn = {}
for row in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if row and row[0].isdigit(): bor[int(row[0])] = row[1]; zn[int(row[0])] = row[2][:12]
MANH = [i for i in range(1, 264) if bor.get(i) == 'Manhattan' and i not in CORE + AIR]
INNER = [i for i in range(1, 264) if bor.get(i) != 'Manhattan' and i not in AIR and d_cbd[i - 1] < 5.0]
OUTER = [i for i in range(1, 264) if bor.get(i) != 'Manhattan' and i not in AIR and d_cbd[i - 1] >= 5.0]
CATS = [('机场(3个)', AIR), ('曼哈顿密集核心', CORE), ('曼哈顿其余', MANH), ('内环<5km', INNER), ('外环≥5km', OUTER)]

print('===== 暴雨时起点份额的分组变化(相对%) =====')
for tag, r, b in [('1-4mm/h', rA, bA), ('≥4mm/h', rB, bB)]:
    row = []
    for nm, zz in CATS:
        rr, bb = r[zz].sum(), b[zz].sum()
        row.append(f'{nm} {(rr/bb-1)*100:+.1f}%')
    print(f'  {tag}: ' + ' | '.join(row))

# 梯度: 相对变化 vs 距CBD
ok = ~np.isnan(relB)
ok[0] = False
cv = np.corrcoef(d_cbd[ok[1 - 1]], relB[1:][ok[1:]])[0, 1] if False else np.corrcoef(d_cbd[(ok)[1:]], relB[1:][ok[1:]])[0, 1]
print(f'\n相关: 份额相对变化 vs 距市中心距离  r = {cv:.3f}  (负=越远越降)')

# ================= 图 =================
fig = plt.figure(figsize=(17, 8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.85], wspace=0.25)
fig.suptitle('暴雨时起点分布解剖：谁升谁降（雨时≥4mm/h，154小时，vs 同星期同钟点晴天）',
             fontsize=14.5, fontweight='bold', y=0.99)

# (1) 地图
ax = fig.add_subplot(gs[0])
g = gpd.read_file('D:/AI/条件转化/data/tlc/zones/_shp/taxi_zones/taxi_zones.shp').to_crs(4324)
gv = np.full(264, np.nan); gv[1:264] = relB[1:264]
g['chg'] = [gv[l] if 1 <= l <= 263 else np.nan for l in g['LocationID']]
g.plot(ax=ax, column='chg', cmap='RdBu_r', vmin=-30, vmax=30, missing_kwds=dict(color='#EFEFEF'),
       edgecolor='#B0B0B0', lw=0.35, legend=True,
       legend_kwds={'label': '起点份额相对变化 %（红=升, 蓝=降）', 'shrink': 0.8})
for zid, nm, dx, dy in [(138, '拉瓜迪亚 −31%', -0.28, -0.10), (132, '肯尼迪 −32%', -0.30, 0.10),
                        (161, '中城 +9%', 0.05, 0.03)]:
    ax.annotate(nm, xy=(lon[zid], lat[zid]), xytext=(lon[zid] + dx, lat[zid] + dy), fontsize=9.5,
                color='#333', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#999', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#333'))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('① 全城地图：核心微红、机场深蓝', fontsize=12, loc='left')

# (2) 分组柱状
ax = fig.add_subplot(gs[1])
names = [c[0] for c in CATS]
xA = [((rA[zz].sum()) / (bA[zz].sum()) - 1) * 100 for _, zz in CATS]
xB = [((rB[zz].sum()) / (bB[zz].sum()) - 1) * 100 for _, zz in CATS]
yp = np.arange(len(names))
ax.barh(yp + 0.2, xA, height=0.36, color='#8FAADC', label='1–4 mm/h')
ax.barh(yp - 0.2, xB, height=0.36, color='#C00000', label='≥4 mm/h（暴雨）')
ax.set_yticks(yp); ax.set_yticklabels(names)
ax.axvline(0, color='#333', lw=1)
for y, v in zip(yp - 0.2, xB): ax.text(v + (1 if v >= 0 else -1), y, f'{v:+.0f}%', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
ax.set_xlabel('起点份额相对变化 (%)')
ax.set_title('② 分组：机场大降、密集核心小升\n（剂量越大效应越大）', fontsize=12, loc='left')
ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='x'); ax.invert_yaxis()

# (3) 距离梯度
ax = fig.add_subplot(gs[2])
dd = d_cbd          # 已是 1..263 共 263 个
rr2 = relB[1:]      # 263 个
ok2 = ~np.isnan(rr2)
ax.scatter(dd[ok2], rr2[ok2], s=14, alpha=0.5, color='#808080')
# 分箱均值
bins = [0, 2, 4, 6, 9, 30]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = ok2 & (dd >= lo) & (dd < hi)
    if m.sum() > 3:
        ax.scatter([(lo + hi) / 2], [np.nanmean(rr2[m])], s=130, color='#C00000', zorder=5, marker='D')
        ax.annotate(f'{np.nanmean(rr2[m]):+.0f}%', ((lo + hi) / 2, np.nanmean(rr2[m]) + 1.5), ha='center', fontsize=9, color='#C00000')
ax.axhline(0, color='#333', lw=1)
ax.set_xlabel('距市中心距离 (km)'); ax.set_ylabel('起点份额相对变化 (%)')
ax.set_title(f'③ 距离梯度：r={cv:.2f}\n近核心升、远端降', fontsize=12, loc='left')
ax.grid(alpha=0.3)

plt.savefig('D:/AI/条件转化/暴雨起点分布分析.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nsaved 暴雨起点分布分析.png')
