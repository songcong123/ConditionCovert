# -*- coding: utf-8 -*-
"""元旦(跨年)四量标定: 48h 窗口 (2024-12-31 00:00 → 2025-01-01 23:00) vs 12月干日同钟点基线
   ① 逐时出发量乘子(跨年脉冲)  ② 逐时中位速度响应(凌晨空路)  ③ 空间热点(Jan1 分区 vs 12月干日)  ④ 距离日级
   输出: 元旦跨年分析/ 文件夹 (总览图+空间图+CSV+JSON)
"""
import numpy as np, json, csv, os
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

FOLD = r'D:/AI/条件转化/元旦跨年分析'
os.makedirs(FOLD, exist_ok=True)

W = np.load(r'D:/AI/条件转化/.flow_agg_winter.npz', allow_pickle=True)
dates = list(W['dates']); hc = W['hour_cnt'].astype(np.float64)
pu = W['pu_cnt'].astype(np.float64); dist = W['dist'].astype(np.float64)
dist_day = W['dist_day'].astype(int); kl = W['klass']
did = {d: i for i, d in enumerate(dates)}

# 12月干日索引与基线
dec_dry = [did[d] for d in dates if d[:7] == '2024-12' and kl[did[d]] == 'dry']
base_hour = np.median(hc[dec_dry], axis=0)                    # 逐时基线
base_zone = np.median(pu[dec_dry], axis=0)                    # 逐区基线
dd_med = []
for i in dec_dry:
    s = dist[dist_day == i]; s = s[(s > 0.1) & (s < 80)]
    if s.size >= 300: dd_med.append(float(np.median(s)))
base_dist = float(np.median(dd_med))
print(f'12月干日 {len(dec_dry)} 天 | 基线日总量 {base_hour.sum():,.0f} | 基线中位里程 {base_dist:.2f} mi')

# 48h 窗口
i31, i01 = did['2024-12-31'], did['2025-01-01']
mult48 = np.concatenate([hc[i31] / base_hour, hc[i01] / base_hour])

# 基线速度(逐时): 12月干日 —— 需要小时级速度, 从 parquet 现算 (顺带 Dec31/Jan1 逐时)
spd_base = np.full((len(dec_dry), 24), np.nan)                # 每干日每小时中位
spd_ny = np.zeros(48); spd_n_ny = np.zeros(48, int)
KEYD = [dates[i] for i in dec_dry] + ['2024-12-31', '2025-01-01']
kid = {d: i for i, d in enumerate(KEYD)}
acc = {}
for path in [r'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_2024-12.parquet',
             r'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_2025-01.parquet']:
    t = pq.read_table(path, columns=['pickup_datetime', 'trip_miles', 'trip_time'])
    dt = t['pickup_datetime'].to_numpy(); mi = t['trip_miles'].to_numpy(); tt = t['trip_time'].to_numpy(); del t
    spd = mi / (tt.astype(np.float64) / 3600.0)
    m = (spd > 1) & (spd < 60) & (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800)
    dt2 = dt[m].astype('datetime64[m]')
    day = dt2.astype('datetime64[D]').astype(str)
    hr = dt2.astype('datetime64[h]').astype(np.int64) % 24
    vv = spd[m]
    for d_, h_, v_ in zip(day, hr, vv):
        if d_ in kid:
            acc.setdefault((d_, int(h_)), []).append(v_)
    print(os.path.basename(path), 'done', flush=True)
for d_ in KEYD:
    for h in range(24):
        vals = acc.get((d_, h))
        if vals and len(vals) >= 200:
            j = kid[d_]
            if j < len(dec_dry):
                spd_base[j, h] = float(np.median(vals))
            elif d_ == '2024-12-31':
                spd_ny[h] = float(np.median(vals)); spd_n_ny[h] = len(vals)
            else:
                spd_ny[24 + h] = float(np.median(vals)); spd_n_ny[24 + h] = len(vals)
spd_base_h = np.nanmedian(spd_base, axis=0)
spd_ratio48 = np.concatenate([spd_ny[:24] / spd_base_h, spd_ny[24:] / spd_base_h])
spd_ok = ~np.isnan(spd_ratio48)

# 距离
d31 = dist[dist_day == i31]; d31 = d31[(d31 > 0.1) & (d31 < 80)]
d01 = dist[dist_day == i01]; d01 = d01[(d01 > 0.1) & (d01 < 80)]
m31, m01 = float(np.median(d31)), float(np.median(d01))

# 关键数字
seg = [('12/31 白天(7-19)', 7, 20), ('跨年夜(19-24)', 19, 24), ('元旦凌晨(0-6)', 24, 30),
       ('元旦上午(6-12)', 30, 36), ('元旦午后(12-24)', 36, 48)]
print('\n=== 逐段出发量乘子 ===')
for nm, a, b in seg:
    r = float(np.mean(mult48[a:b]))
    print(f'  {nm}: ×{r:.2f}')
pk = int(np.argmax(mult48)); print(f'峰值: {"12/31" if pk<24 else "1/1"} {pk%24}时 ×{mult48[pk]:.2f}')
print(f'距离: 12/31 中位 {m31:.2f} mi | 1/1 中位 {m01:.2f} mi (基线 {base_dist:.2f})')
print(f'速度: 凌晨0-6时 比值 {np.nanmedian(spd_ratio48[24:30]):.2f} | 跨年夜19-23时 {np.nanmedian(spd_ratio48[19:24]):.2f}')

# 空间: 1/1 分区 vs 基线
zn = {}
for r in csv.reader(open(r'D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if r and r[0].isdigit(): zn[int(r[0])] = r[2][:14]
nz = np.maximum(base_zone, 1e-9)
rel = (pu[i01] / max(pu[i01].sum(), 1)) / (base_zone / base_zone.sum()) - 1
rel[0] = np.nan; rel[base_zone < 0.0015] = np.nan
idx = np.where(~np.isnan(rel))[0]
order = idx[np.argsort(-rel[idx])]
print('\n=== 空间热点 (1/1 份额相对变化, 基线≥0.15%) ===')
for z in order[:10]:
    print(f'  + {zn.get(int(z), z):14s} {rel[z]*100:+6.1f}%')
for z in order[-6:]:
    print(f'  − {zn.get(int(z), z):14s} {rel[z]*100:+6.1f}%')

# ---------- 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
fig.suptitle('元旦(跨年)四量解剖：2024-12-31 00时 → 2025-01-01 23时，vs 12月干日同钟点基线', fontsize=14, fontweight='bold', y=0.99)
xt = [0, 6, 12, 18, 24, 30, 36, 42]
xl = ['12/31\n0时', '6时', '12时', '18时', '1/1\n0时', '6时', '12时', '18时']

ax = axes[0, 0]
ax.plot(np.arange(48), mult48, lw=2.4, color='#C00000')
ax.axvspan(22, 30, color='#C00000', alpha=0.08)
ax.axhline(1, color='#999', lw=1, ls=':')
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=9)
ax.set_ylabel('出发量乘子'); ax.set_title('① 48h 乘子曲线：跨年夜抬升 → 凌晨脉冲 → 白天回落', fontsize=11.5, loc='left')
ax.grid(alpha=0.3); ax.set_ylim(0, max(mult48) * 1.25)

ax = axes[0, 1]
ax.plot(np.arange(48)[spd_ok], spd_ratio48[spd_ok], lw=2.2, color='#1F4E79')
ax.axvspan(22, 30, color='#C00000', alpha=0.08)
ax.axhline(1, color='#999', lw=1, ls=':')
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=9)
ax.set_ylabel('中位速度比'); ax.set_title('② 速度响应：凌晨路空 → 提速（蓝=更快）', fontsize=11.5, loc='left')
ax.grid(alpha=0.3); ax.set_ylim(0.7, 1.6)

ax = axes[1, 0]
nm10 = [zn.get(int(z), str(z)) for z in order[:10]]
vl10 = [rel[z] * 100 for z in order[:10]]
ax.barh(range(10)[::-1], vl10, color='#70AD47', alpha=0.85)
ax.set_yticks(range(10)[::-1]); ax.set_yticklabels(nm10, fontsize=9)
ax.set_xlabel('1/1 份额相对基线变化 %'); ax.set_title('③ 空间热点 TOP10（全城均值 ≈0，局部重分配）', fontsize=11.5, loc='left')
ax.grid(alpha=0.3, axis='x')

ax = axes[1, 1]
parts = ax.violinplot([np.clip(d01, 0, 12), np.clip(d31, 0, 12), np.clip(dd_med, 0, 12)],
                      [1, 2, 3], showmedians=True, widths=0.8)
for pc, cc in zip(parts['bodies'], ['#ED7D31', '#BFBFBF', '#70AD47']): pc.set_facecolor(cc); pc.set_alpha(0.6)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f'1/1\n{m01:.2f}mi', f'12/31\n{m31:.2f}mi', f'12月干日\n{base_dist:.2f}mi'], fontsize=9.5)
ax.set_ylabel('单程里程 (mi)'); ax.set_title('④ 距离分布：跨年夜行程更远（活动+深夜）', fontsize=11.5, loc='left')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(FOLD, '元旦_证据总览.png'), dpi=150, bbox_inches='tight', facecolor='white')
print('saved 元旦_证据总览.png')

json.dump(dict(mult48=[round(float(x), 3) for x in mult48],
               peak=dict(hour=("12/31" if pk < 24 else "1/1") + f" {pk % 24}时", mult=round(float(mult48[pk]), 2)),
               dist=dict(ny=m01, eve=m31, base=base_dist),
               spd=dict(night=round(float(np.nanmedian(spd_ratio48[24:30])), 3),
                        eve=round(float(np.nanmedian(spd_ratio48[19:24])), 3)),
               spatial_top=[(zn.get(int(z), str(z)), round(float(rel[z]) * 100, 1)) for z in order[:10]]),
          open(os.path.join(FOLD, '元旦_统计.json'), 'w'), ensure_ascii=False, indent=1)
with open(os.path.join(FOLD, '元旦_空间分区.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w_ = csv.writer(f); w_.writerow(['zone_id', 'zone_name', '相对变化%'])
    for z in idx[np.argsort(-np.abs(rel[idx]))]:
        w_.writerow([z, zn.get(int(z), z), round(float(rel[z]) * 100, 1)])
print('saved json/csv')
