# -*- coding: utf-8 -*-
"""补测: 雨时(小时级)空间分布是否变化 —— 回应"下雨会改变空间分布"的质疑
对比: 雨时(日∈中/强雨日 且 1mm/h≤时雨强)的 PU zone 分布
  vs  同 hour-of-week 干日同时段的 PU zone 分布(消掉早晚峰空间结构)
两剂量箱: A=1-4mm/h, B=≥4mm/h
输出 .rain_spatial.npz + 控制台 JS/涨幅榜"""
import numpy as np
import pyarrow.parquet as pq

D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(D['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
n = len(dates); didx = {d: i for i, d in enumerate(dates)}
kl = np.array([klass[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
inten = np.zeros((n, 24))
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

rain_mask = np.isin(kl, ['medium', 'heavy'])
boxA = np.zeros((n, 24), bool); boxB = np.zeros((n, 24), bool)
boxA[rain_mask] = (inten[rain_mask] >= 1) & (inten[rain_mask] < 4)
boxB[rain_mask] = inten[rain_mask] >= 4
dry = kl == 'dry'
print(f'雨时箱A(1-4mm/h) {boxA.sum()} 小时 | 箱B(≥4mm/h) {boxB.sum()} 小时 | 干日 {dry.sum()} 天', flush=True)

dates_np = np.array(dates, dtype='datetime64[D]')
puA = np.zeros(264, np.int64); puB = np.zeros(264, np.int64)
pu_dry_how = np.zeros(7 * 24 * 264, np.int64)      # 干日 hour-of-week × zone (flat)
howA = np.zeros(7 * 24, np.int64); howB = np.zeros(7 * 24, np.int64)  # 各箱命中的干日样本量权重用

for mo in sorted({str(d)[:7] for d in dates}):
    t = pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                      columns=['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time'])
    dt = t['pickup_datetime'].to_numpy()
    pu = t['PULocationID'].to_numpy().astype(np.int32)
    doo = t['DOLocationID'].to_numpy().astype(np.int32)
    mi = t['trip_miles'].to_numpy(); tt = t['trip_time'].to_numpy()
    del t
    day = dt.astype('datetime64[D]')
    pos = np.clip(np.searchsorted(dates_np, day), 0, n - 1)
    spd = mi / (tt.astype(np.float64) / 3600.0)
    m = (dates_np[pos] == day) & (pu >= 1) & (pu <= 263) & (doo >= 1) & (doo <= 263) & \
        (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800) & (spd > 0) & (spd < 100)
    di = pos[m]; pu = pu[m]; dt2 = dt[m]
    hour = (dt2.astype('datetime64[h]').astype(np.int64) % 24).astype(np.int64)
    for box, acc, howacc in [(boxA, puA, howA), (boxB, puB, howB)]:
        sel = box[di, hour]
        if sel.any():
            acc += np.bincount(pu[sel], minlength=264)
            how_ = dow[di[sel]].astype(np.int64) * 24 + hour[sel]
            howacc += np.bincount(how_, minlength=7 * 24)
    selD = dry[di]
    if selD.any():
        flat = ((dow[di[selD]].astype(np.int64) * 24 + hour[selD]) * 264 + pu[selD])
        pu_dry_how += np.bincount(flat, minlength=7 * 24 * 264)
    print(f'{mo} done', flush=True)

pu_dry_how = pu_dry_how.reshape(7, 24, 264)
np.savez_compressed('D:/AI/条件转化/.rain_spatial.npz',
                    puA=puA, puB=puB, pu_dry_how=pu_dry_how, howA=howA, howB=howB)
print('saved .rain_spatial.npz', flush=True)

# ---------- 分析 ----------
import csv
zn = {}
for r in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if r and r[0].isdigit(): zn[int(r[0])] = r[2][:14]

def analyze(pu_rain, how_hit, tag):
    # 干日匹配基线: 对每个命中的 hour-of-week, 取干日该(how)的 zone 分布, 按 how_hit 权重加权
    howw = how_hit / max(how_hit.sum(), 1)
    dry_share_how = pu_dry_how.reshape(7 * 24, 264) / np.maximum(
        pu_dry_how.reshape(7 * 24, 264).sum(1, keepdims=True), 1)
    base = (howw[:, None] * dry_share_how).sum(0)      # 加权干日份额
    base /= base.sum()
    rs = pu_rain / max(pu_rain.sum(), 1)
    js = 0.5 * np.sum(rs * np.log(rs / base)) + 0.5 * np.sum(base * np.log(base / rs))
    js = 0.5 * (np.sum(rs[rs > 0] * np.log(rs[rs > 0] / base[rs > 0])) +
                np.sum(base[base > 0] * np.log(base[base > 0] / rs[base > 0])))
    vol = pu_rain.sum() / max((how_hit > 0).sum(), 1)
    diff = (rs - base) * 100
    idx = np.argsort(-np.abs(diff))[:8] + 0
    print(f'\n[{tag}] 雨时 vs 同钟点干日: JS={js:.5f} | 雨时趟/时={vol:,.0f}')
    for i in sorted(idx, key=lambda i: -diff[i]):
        if diff[i] == 0: continue
        print(f'   {zn.get(int(i), i):16s} {diff[i]:+.2f}pp  ({base[i]*100:.1f}%→{rs[i]*100:.1f}%)')
    return js, rs, base

jsA, rsA, baseA = analyze(puA, howA, '箱A 1-4mm/h')
jsB, rsB, baseB = analyze(puB, howB, '箱B ≥4mm/h')
print('\n参照: 强雨日·日级 JS=0.0002 (此前结论)')
