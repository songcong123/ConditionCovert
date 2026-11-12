# -*- coding: utf-8 -*-
"""元旦 48h 出行动画: 从冬季聚合抽样逐时 OD 流 + 12月/1月 parquet 速度曲线
   48 帧(每小时一帧) → GIF; 地图=263 zone 真实多边形, 线=OD 流(透明度=流量), 圈=出发热点
   四量联动可见: 顶部 48h 乘子进度 + 右侧速度曲线 + 仪表条(出发量/速度/距离)
输出: 元旦跨年分析/元旦_48h动画.gif + 元旦_spd48.json
"""
import numpy as np, json, os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import geopandas as gpd
import pyarrow.parquet as pq

FOLD = r'D:/AI/条件转化/元旦跨年分析'
os.makedirs(FOLD, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 数据 ----------
W = np.load(r'D:/AI/条件转化/.flow_agg_winter.npz', allow_pickle=True)
R = np.load(r'D:/AI/条件转化/.rain_spatial.npz')
Z = np.load(r'D:/AI/条件转化/.flow_zones.npz')
dates = list(W['dates']); hc = W['hour_cnt'].astype(np.float64)
didx = {d: i for i, d in enumerate(dates)}
pu = W['pu_cnt'].astype(np.float64); do = W['do_cnt'].astype(np.float64)
kl = W['klass']; i31 = dates.index('2024-12-31'); i01 = dates.index('2025-01-01')
d0 = i31                                     # 48h 起点(12/31 0时)在 dates 中的索引
lat, lon = Z['lat'], Z['lon']
dry_how = R['pu_dry_how'].reshape(7, 24, 264).astype(np.float64)

dec_dry = [i for i, d in enumerate(dates) if d[:7] == '2024-12' and kl[i] == 'dry']
base_hour = np.median(hc[dec_dry], axis=0)                    # 逐时基线
mult48 = np.array([hc[(i31 if k < 24 else i01), k % 24] / base_hour[k % 24] for k in range(48)])
dist_med = {}
for nm, i in [('12/31', i31), ('1/1', i01)]:
    s = W['dist'].astype(np.float64)[W['dist_day'] == i]
    s = s[(s > 0.1) & (s < 80)]
    dist_med[nm] = float(np.median(s)) if len(s) else 2.8
dist_ratio = {0: round(dist_med['12/31'] / 2.81, 2), 24: round(dist_med['1/1'] / 2.81, 2)}

# 12月干日逐区基线 → 日重分配系数
dec_zone = np.median(pu[dec_dry], axis=0)
dec31_adj = np.maximum(pu[i31] / np.maximum(dec_zone, 1), 0)
jan1_adj = np.maximum(pu[i01] / np.maximum(dec_zone, 1), 0)
dec31_adj /= dec31_adj.mean(); jan1_adj /= jan1_adj.mean()

# 速度 48h: 从 parquet 现算 (12月干日逐时中位基线 + 12/31、1/1 逐时中位)
spd_dry_acc = [[] for _ in range(24)]
spd_ny_dec = np.full(24, np.nan); spd_ny_jan = np.full(24, np.nan)
for path in [r'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_2024-12.parquet',
             r'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_2025-01.parquet']:
    t = pq.read_table(path, columns=['pickup_datetime', 'trip_miles', 'trip_time'])
    dt = t['pickup_datetime'].to_numpy(); mi = t['trip_miles'].to_numpy(); tt = t['trip_time'].to_numpy(); del t
    v = mi / (tt.astype(np.float64) / 3600.0)
    m = (v > 1) & (v < 60) & (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800)
    dt2 = dt[m].astype('datetime64[m]')
    dstr = dt2.astype('datetime64[D]').astype(str)
    hh = dt2.astype('datetime64[h]').astype(np.int64) % 24
    vv = v[m]
    key = np.array([didx[d] for d in dstr]) * 24 + hh
    order = np.argsort(key, kind='stable')
    k2, start = np.unique(key[order], return_index=True)
    counts = np.diff(np.append(start, len(k2)))
    for kk, s0, cnt in zip(k2, start, counts):
        sl = vv[order][s0:s0 + cnt]
        di, h = divmod(int(kk), 24)
        if cnt < 30: continue
        med = float(np.median(sl))
        if kl[di] == 'dry': spd_dry_acc[h].append(med)
        if dates[di] == '2024-12-31': spd_ny_dec[h] = med
        if dates[di] == '2025-01-01': spd_ny_jan[h] = med
    print(path.split('_')[-1], 'done', flush=True)
spd_base0 = np.array([float(np.median(a)) if len(a) >= 3 else np.nan for a in spd_dry_acc])
spd48 = np.concatenate([spd_ny_dec / spd_base0, spd_ny_jan / spd_base0])
nans = np.isnan(spd48)
if nans.any():
    idx_ok = np.where(~nans)[0]
    spd48[nans] = np.interp(np.where(nans)[0], idx_ok, spd48[idx_ok])
print('spd48 ready', np.round(spd48, 2))

# ---------- 48 帧预抽样 ----------
PHASES = [(0, 7, '白天 ×%.2f', '#8FAADC'), (7, 19, '白天 ×%.2f', '#8FAADC'),
          (19, 24, '跨年夜抬升 ×%.2f', '#7030A0'), (24, 30, '元旦凌晨脉冲 ×%.2f', '#C00000'),
          (30, 36, '上午塌陷 ×%.2f', '#7F7F7F'), (36, 48, '元旦午后恢复 ×%.2f', '#70AD47')]
def phase_of(f):
    for a, b, t, c in PHASES:
        if a <= f < b: return t % mult48[f], c
    return '', '#333'
SCALE = 1 / 90.0
frames = []
rng = np.random.default_rng(20250101)
for f in range(48):
    di = i31 if f < 24 else i01
    h = f % 24
    dow = int((np.datetime64(dates[di], 'D').view('int64') + 3) % 7)
    adj = dec31_adj if f < 24 else jan1_adj
    wo = np.maximum(dry_how[dow, h] * adj, 0) + 0.5
    wd = np.maximum(do[di], 0) + 0.5
    Nf = int(min(360, max(50, hc[di, h] * SCALE)))
    oi = rng.choice(264, size=Nf, p=wo / wo.sum())
    di_ = rng.choice(264, size=Nf, p=wd / wd.sum())
    pairs, counts = np.unique(oi.astype(np.int64) * 264 + di_, return_counts=True)
    oz = (pairs // 264).astype(int); dz = (pairs % 264).astype(int)
    segs = [[lon[o], lat[o], lon[d2], lat[d2]] for o, d2 in zip(oz, dz) if o != d2]
    cnts = counts[[k for k, (o, d2) in enumerate(zip(oz, dz)) if o != d2]]
    oz_cnt = np.bincount(oi, minlength=264)
    frames.append(dict(h=h, day=('12/31' if f < 24 else '1/1'), di=di,
                       N=int(hc[di, h]), mult=float(mult48[f]), spd=float(spd48[f]),
                       segs=np.array(segs), cnts=cnts, oz_cnt=oz_cnt,
                       dr=(dist_ratio[0] if f < 24 else dist_ratio[24])))
print('48 帧抽样完成')

# ---------- 底图 ----------
g = gpd.read_file(r'D:/AI/条件转化/data/tlc/zones/_shp/taxi_zones/taxi_zones.shp').to_crs(4326)

# ---------- 动画 ----------
fig = plt.figure(figsize=(16, 9), facecolor='white')
axm = fig.add_axes([0.02, 0.03, 0.60, 0.92])
axc = fig.add_axes([0.65, 0.62, 0.33, 0.30])
axs = fig.add_axes([0.65, 0.34, 0.33, 0.22])
axg = fig.add_axes([0.65, 0.06, 0.33, 0.20]); axg.axis('off')
title_tx = fig.text(0.5, 0.955, '', ha='center', fontsize=17, fontweight='bold', color='#1F4E79')

for a, b, lab, c in PHASES:
    axc.axvspan(a, b, color=c, alpha=0.10)
axc.plot(np.arange(48), mult48, lw=2.2, color='#333333')
axc.axhline(1, color='#999', lw=1, ls=':')
axc.set_xlim(0, 47); axc.set_ylim(0, 6.2)
axc.set_xticks([0, 6, 12, 18, 24, 30, 36, 42])
axc.set_xticklabels(['12/31\n0时', '6时', '12时', '18时', '1/1\n0时', '6时', '12时', '18时'], fontsize=9)
axc.set_title('出发量乘子（×基线）', fontsize=11, loc='left', color='#1F4E79')
dot_c, = axc.plot([], [], 'o', ms=13, color='#C00000')

axs.plot(np.arange(48), spd48, lw=2.2, color='#1F4E79')
axs.axhline(1, color='#999', lw=1, ls=':')
axs.set_xlim(0, 47); axs.set_ylim(0.7, 1.5)
axs.set_xticks([0, 6, 12, 18, 24, 30, 36, 42])
axs.set_xticklabels(['0时', '6时', '12时', '18时', '0时', '6时', '12时', '18时'], fontsize=9)
axs.set_title('中位速度比（×12月干日基线）', fontsize=11, loc='left', color='#1F4E79')
dot_s, = axs.plot([], [], 'o', ms=12, color='#C00000')

labels4 = ['出发量乘子', '速度比', '距离比', '事件']
ys4 = [0.75, 0.55, 0.35, 0.15]
bars = axg.barh(ys4, [1, 1, 1, 0.05], height=0.16, color=['#C00000', '#1F4E79', '#70AD47', '#ED7D31'])
axg.set_xlim(0, 6.2); axg.set_yticks(ys4); axg.set_yticklabels(labels4, fontsize=11)
axg.set_title('四量仪表（×基线）', fontsize=11, loc='left', color='#1F4E79')
val_tx = axg.text(4.9, 0.75, '', va='center', fontsize=11, fontweight='bold', color='#C00000')

g.plot(ax=axm, color='#FFFFFF', edgecolor='#B0B0B0', lw=0.4)
axm.set_xlim(lon[1:264].min() - 0.05, lon[1:264].max() + 0.05)
axm.set_ylim(lat[1:264].min() - 0.04, lat[1:264].max() + 0.04)
axm.axis('off')
leg = axm.legend(handles=[Patch(color='#C00000', label='跨年夜/凌晨脉冲'), Patch(color='#7030A0', label='跨年夜抬升'),
                          Patch(color='#8FAADC', label='白天'), Patch(color='#B0B0B0', label='OD 流向线')],
                 loc='lower left', fontsize=9, framealpha=0.9)
lc = None; bub = None

def update(f):
    global lc, bub
    fr = frames[f]
    ph_txt, ph_c = phase_of(f)
    for col in (lc, bub):
        if col is not None:
            try: col.remove()
            except Exception: pass
    if len(fr['segs']):
        lc = LineCollection([[ (s[0], s[1]), (s[2], s[3]) ] for s in fr['segs']],
                            array=fr['cnts'], cmap='Reds', linewidths=np.maximum(fr['cnts'] * 0.35, 0.4),
                            alpha=0.45, capstyle='round')
        axm.add_collection(lc)
    hot = np.where(fr['oz_cnt'] >= 3)[0]
    bub = axm.scatter(lon[hot], lat[hot], s=fr['oz_cnt'][hot] * 4.5, c=ph_c, alpha=0.45,
                      edgecolors='none', zorder=5)
    dot_c.set_data([f], [fr['mult']])
    dot_s.set_data([f], [fr['spd']])
    bars[0].set_width(fr['mult']); val_tx.set_text(f'×{fr["mult"]:.2f}')
    bars[1].set_width(fr['spd'] if fr['spd'] > 0 else 1)
    bars[2].set_width(fr['dr'])
    bars[3].set_width(0.08 if f < 19 else 1.0)
    tag = '跨年活动: 时代广场封街' if (19 <= f < 24 or 24 <= f < 30) else ('元旦假日' if f >= 24 else '工作日')
    title_tx.set_text(f'{fr["day"]} {fr["h"]:02d}:00  ｜  出发量 ×{fr["mult"]:.2f}   速度 ×{fr["spd"]:.2f}   ｜  {ph_txt}   ｜  {tag}')
    return []

anim = FuncAnimation(fig, update, frames=48, blit=False)
gif_path = os.path.join(FOLD, '元旦_48h动画.gif')
anim.save(gif_path, writer=PillowWriter(fps=3), dpi=90, savefig_kwargs={'facecolor': 'white'})
json.dump(dict(mult48=[round(float(x), 3) for x in mult48], spd48=[round(float(x), 3) for x in spd48]),
          open(os.path.join(FOLD, '元旦_spd48.json'), 'w'), ensure_ascii=False)
print('saved', gif_path, f'{os.path.getsize(gif_path)/1e6:.1f} MB')
