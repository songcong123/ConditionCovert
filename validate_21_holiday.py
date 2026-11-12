# -*- coding: utf-8 -*-
"""NYC 假日补扫：10 天假日 × 四量效应（vs 同星期非假日基线）"""
import numpy as np, csv, json
from datetime import date
import pyarrow.parquet as pq

# 假日列表（窗口内 10 天）
HOLIDAYS = {
    '2024-05-27': 'Memorial Day', '2024-06-19': 'Juneteenth', '2024-07-04': 'Independence Day',
    '2024-09-02': 'Labor Day', '2024-10-14': 'Columbus Day',
    '2025-05-26': 'Memorial Day', '2025-06-19': 'Juneteenth', '2025-07-04': 'Independence Day',
    '2025-09-01': 'Labor Day', '2025-10-13': 'Columbus Day',
}
dates_hol = sorted(HOLIDAYS.keys())
dow_hol = {d: date(int(d[:4]), int(d[5:7]), int(d[8:10])).weekday() for d in dates_hol}

# 天气：判断假日是否下雨
D0 = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dprecip_all = dict(zip(D0['dates'].tolist(), D0['dprecip'].tolist()))
# 假日不在分类日里(被剔了)，从原始逐时降水重建
import pandas as pd
from datetime import datetime, timedelta

def nth_sunday(y, m, n):
    d = datetime(y, m, 1); c = 0
    while True:
        if d.weekday() == 6:
            c += 1
            if c == n: return d
        d += timedelta(days=1)

def is_edt(dt):
    y = dt.year
    return nth_sunday(y, 3, 2) <= dt < nth_sunday(y, 11, 1)

hol_precip = {}
for y in [2024, 2025]:
    df = pd.read_csv(f'D:/AI/条件转化/data/weather/isd/72505394728_{y}.csv',
                     usecols=['DATE', 'REPORT_TYPE', 'AA1'], dtype=str)
    df = df[df['REPORT_TYPE'] == 'FM-15']
    for s, aa in zip(df['DATE'], df['AA1']):
        try: utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
        except: continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        dstr = loc.strftime('%Y-%m-%d')
        if dstr in HOLIDAYS and isinstance(aa, str) and ',' in aa:
            p = aa.split(',')
            if p[0] == '01' and p[1] not in ('9999', '99'):
                hol_precip[dstr] = hol_precip.get(dstr, 0) + float(p[1]) * 0.254
# 同小时取 max 去重
print('假日天气:')
for d in dates_hol:
    mm = hol_precip.get(d, 0)
    print(f'  {d} ({HOLIDAYS[d]:15s}) {dow_hol[d]} 降水 {mm:.1f}mm → {"雨日" if mm >= 0.5 else "晴天" if mm < 0.5 else "微雨"}')

# 基线：已有非假日干日基线
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
dates_base = list(A['dates'])
klass_base = dict(zip(D0['dates'].tolist(), D0['klass'].tolist()))
kl = np.array([klass_base[d] for d in dates_base])
dow_base = np.array([(date(int(d[:4]), int(d[5:7]), int(d[8:10])).weekday()) for d in dates_base])
hc = A['hour_cnt'].astype(float)
pu = A['pu_cnt'].astype(float)

# 星期基线（非假日干日）
lam_dow = {}
for k in range(7):
    sel = (kl == 'dry') & (dow_base == k)
    if sel.sum() > 0: lam_dow[k] = hc[sel].mean(0)
pu_dow = {}
for k in range(7):
    sel = (kl == 'dry') & (dow_base == k)
    if sel.sum() > 0: pu_dow[k] = pu[sel].sum(0) / sel.sum()

# 扫假日行程
hol_dates_np = np.array(dates_hol, dtype='datetime64[D]')
hour_cnt_h = np.zeros((len(dates_hol), 24), np.int64)
pu_h = np.zeros((len(dates_hol), 264), np.int64)
months = sorted({d[:7] for d in dates_hol})
for mo in months:
    t = pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                      columns=['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time'])
    dt = t['pickup_datetime'].to_numpy()
    p = t['PULocationID'].to_numpy().astype(np.int32)
    doo = t['DOLocationID'].to_numpy().astype(np.int32)
    mi = t['trip_miles'].to_numpy(); tt = t['trip_time'].to_numpy()
    del t
    day = dt.astype('datetime64[D]')
    pos = np.clip(np.searchsorted(hol_dates_np, day), 0, len(dates_hol) - 1)
    spd = mi / (tt.astype(np.float64) / 3600.0)
    m = (hol_dates_np[pos] == day) & (p >= 1) & (p <= 263) & (doo >= 1) & (doo <= 263) & \
        (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800) & (spd > 0) & (spd < 100)
    di = pos[m]; p2 = p[m]
    hour = (dt[m].astype('datetime64[h]').astype(np.int64) % 24).astype(np.int64)
    np.add.at(hour_cnt_h, (di, hour), 1)
    np.add.at(pu_h, (di, np.clip(p2, 0, 263)), 1)
    print(f'{mo} scanned', flush=True)

# 分析
print('\n===== 假日 vs 同星期非假日基线 =====')
results = {}
for j, d in enumerate(dates_hol):
    k = dow_hol[d]
    if k not in lam_dow: continue
    base = lam_dow[k]
    if base.sum() < 1000: continue
    vol = hour_cnt_h[j].sum() / base.sum() - 1
    # λ(t) 形态：峰位对比
    pk_h = np.argmax(hour_cnt_h[j])
    pk_b = np.argmax(base)
    # 空间
    pu_j = pu_h[j].astype(float); pu_j[0] = 0
    if pu_j.sum() < 100: continue
    pu_js = pu_j / pu_j.sum()
    pu_bs = pu_dow[k] / pu_dow[k].sum()
    mm = pu_bs > 1e-6
    js = 0.5 * (np.sum(pu_js[mm] * np.log(np.maximum(pu_js[mm], 1e-12) / pu_bs[mm])) +
                np.sum(pu_bs[mm] * np.log(pu_bs[mm] / np.maximum(pu_js[mm], 1e-12))))
    results[d] = dict(name=HOLIDAYS[d], dow=k, vol=float(vol), peak_h=pk_h, peak_b=pk_b, js=float(js),
                      precip=hol_precip.get(d, 0), n=int(hour_cnt_h[j].sum()))
    print(f'  {d} {HOLIDAYS[d]:15s} 周{"一二三四五六日"[k]} '
          f'总量{vol*100:+6.1f}%  峰位{pk_b}→{pk_h}时  空间JS={js:.4f}  n={hour_cnt_h[j].sum():,}')

# 晴天假日 vs 雨天假日
dry_hol = [d for d in results if results[d]['precip'] < 0.5]
rain_hol = [d for d in results if results[d]['precip'] >= 0.5]
if dry_hol:
    vols = [results[d]['vol'] for d in dry_hol]
    print(f'\n晴天假日 ({len(dry_hol)}天): 平均总量 {np.mean(vols)*100:+.1f}%')
if rain_hol:
    vols = [results[d]['vol'] for d in rain_hol]
    print(f'雨天假日 ({len(rain_hol)}天): 平均总量 {np.mean(vols)*100:+.1f}%')

json.dump(results, open('D:/AI/条件转化/.holiday_results.json', 'w'), indent=1, default=str)
print('\nsaved .holiday_results.json')
