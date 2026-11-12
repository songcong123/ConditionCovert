# -*- coding: utf-8 -*-
"""冬季补扫: 2024-11 ~ 2025-03 (感恩节/圣诞/跨年/寒潮/降雪/拥堵收费窗口)
   口径与 validate_02 完全一致(清洗/抽样/CAP), 但独立输出 .flow_agg_winter.npz —— 不触碰已标定的暖季数据
   附带: is_holiday 旗标(含感恩节/圣诞/元旦/MLK/总统日/选举日等) / 雪日分类(temp<=1°C 且有降水) / 逐日均温 / 拥堵收费旗标(≥2025-01-05)
"""
import numpy as np, pyarrow.parquet as pq, csv
from datetime import datetime, timedelta
import pandas as pd

MONTHS = ['2024-11', '2024-12', '2025-01', '2025-02', '2025-03']
PRICING_START = '2025-01-05'

# ---------- 1) 冬季假日表 ----------
holidays = {}
for row in csv.reader(open(r'D:/AI/条件转化/data/exogenous/holidays_us_ny.csv', encoding='utf-8-sig')):
    if row and row[0][:4].isdigit():
        if row[0][:7] in MONTHS or row[0] in ('2025-01-01',):
            holidays[row[0]] = row[4] if len(row) > 4 else row[1]
print(f'冬季假日 {len(holidays)} 个:', holidays)

# ---------- 2) ISD 逐时降水+温度 (FM-15, UTC→本地, 冬季=EST UTC-5 / 11月初与3月初含夏令时边界) ----------
def is_edt(dt):
    y = dt.year
    def nth_dow(m, dow, n):                     # 该月第 n 个星期X(dow)
        d0 = datetime(y, m, 1); c = 0
        while True:
            if d0.weekday() == dow:
                c += 1
                if c == n: return d0
            d0 += timedelta(days=1)

    return datetime(y, 3, 1) + timedelta((6 - datetime(y, 3, 1).weekday()) % 7) + timedelta(7 * 1) >= dt and \
           dt < datetime(y, 11, 1) + timedelta((6 - datetime(y, 11, 1).weekday()) % 7)

rows_p, rows_t = [], []
for yr in [2024, 2025]:
    df = pd.read_csv(f'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                     usecols=['DATE', 'REPORT_TYPE', 'AA1', 'TMP'], dtype=str)
    fm15 = df[df['REPORT_TYPE'] == 'FM-15']
    for s, aa, tmp in zip(fm15['DATE'], fm15['AA1'], fm15['TMP']):
        try: utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
        except: continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        if loc.strftime('%Y-%m') not in MONTHS and loc.strftime('%Y-%m-%d') not in ('2025-01-01',): continue
        if isinstance(aa, str) and ',' in aa:
            p = aa.split(',')
            if p[0] == '01' and p[1] not in ('9999', '99'):
                rows_p.append((loc, float(p[1]) * 0.254))          # 百分之一英寸→mm
        if isinstance(tmp, str) and ',' in tmp:
            v = float(tmp.split(',')[0]) / 10.0
            if -40 < v < 55: rows_t.append((loc, v))
w = pd.DataFrame(rows_p, columns=['dt', 'mm'])
w['date'] = w['dt'].dt.strftime('%Y-%m-%d'); w['hour'] = w['dt'].dt.hour
hourly = w.groupby(['date', 'hour'])['mm'].max()
daily = hourly.groupby(level=0).sum()
t_df = pd.DataFrame(rows_t, columns=['dt', 't'])
t_df['date'] = t_df['dt'].dt.strftime('%Y-%m-%d'); t_df['hour'] = t_df['dt'].dt.hour
t_hourly = t_df.groupby(['date', 'hour'])['t'].mean()
t_daily = t_df.groupby('date')['t'].mean()
print(f'降水: {len(hourly)} 雨时 | 温度: {len(t_hourly)} 时点 | 降水日 {daily.index.size}')

# ---------- 3) 日期表与分类 ----------
all_dates = []
for mo in MONTHS:
    y, m = int(mo[:4]), int(mo[5:])
    d = datetime(y, m, 1)
    while d.month == m:
        all_dates.append(d.strftime('%Y-%m-%d')); d += timedelta(days=1)
dates_np = np.array(all_dates, dtype='datetime64[D]'); n = len(dates_np)
did_of = {d: i for i, d in enumerate(all_dates)}

klass = np.empty(n, object); is_hol = np.zeros(n, bool); dprecip = np.zeros(n); tmean = np.full(n, np.nan)
snow_flag = np.zeros(n, bool); price_flag = dates_np >= np.datetime64(PRICING_START)
hol_names = np.full(n, '', object)
for i, d in enumerate(all_dates):
    mm = float(daily.get(d, 0.0))
    t = float(t_daily.get(d, np.nan)) if d in t_daily.index else np.nan
    dprecip[i] = mm
    if not np.isnan(t): tmean[i] = t
    if d in holidays:
        klass[i] = 'holiday'; is_hol[i] = True; hol_names[i] = holidays[d]
    elif mm < 0.5:
        klass[i] = 'dry'
    elif t == t and t <= 1.0:                                   # 有降水且均温≤1°C → 雪
        klass[i] = 'snow'; snow_flag[i] = True
    else:
        klass[i] = 'light' if mm < 5 else 'medium' if mm < 15 else 'heavy'
import collections
print('分类:', dict(collections.Counter(klass)))
print(f'雪日: {snow_flag.sum()} | 拥堵收费后天数: {price_flag.sum()} | 假日: {is_hol.sum()}')

# ---------- 4) 扫描 parquet (与 validate_02 同口径) ----------
import numpy.random as npr
rng = np.random.default_rng(44)
CAP_D, CAP_V, CAP_M = 15000, 15000, 8000
hour_cnt = np.zeros((n, 24), np.int64)
pu_cnt = np.zeros((n, 264), np.int64); do_cnt = np.zeros((n, 264), np.int64)
dist_l, dist_day_l, spd_l, spd_day_l = [], [], [], []
odm_l = []
for mo in MONTHS:
    t = pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                      columns=['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time'])
    dt = t['pickup_datetime'].to_numpy()
    pu = t['PULocationID'].to_numpy().astype(np.int32)
    doo = t['DOLocationID'].to_numpy().astype(np.int32)
    mi = t['trip_miles'].to_numpy(); tt = t['trip_time'].to_numpy(); del t
    day = dt.astype('datetime64[D]')
    pos = np.clip(np.searchsorted(dates_np, day), 0, n - 1)
    valid = dates_np[pos] == day
    spd_all = mi / (tt.astype(np.float64) / 3600.0)
    m = valid & (pu >= 1) & (pu <= 263) & (doo >= 1) & (doo <= 263) & \
        (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800) & (spd_all > 0) & (spd_all < 100)
    di = pos[m]; pu, doo, mi, tt = pu[m], doo[m], mi[m], tt[m]; dt2 = dt[m]
    hour = (dt2.astype('datetime64[h]').astype(np.int64) % 24)
    np.add.at(hour_cnt, (di, hour), 1)
    np.add.at(pu_cnt, (di, pu), 1); np.add.at(do_cnt, (di, doo), 1)
    nday = np.bincount(di, minlength=n)
    pkeep = np.minimum(1.0, CAP_D / np.maximum(nday, 1))[di]
    keep = rng.random(len(di)) < pkeep
    dist_l.append(mi[keep]); dist_day_l.append(di[keep])
    spd = mi / (tt / 3600.0)
    spd_l.append(spd[keep]); spd_day_l.append(di[keep])
    pkeep2 = np.minimum(1.0, CAP_M / np.maximum(nday, 1))[di]
    keep2 = rng.random(len(di)) < pkeep2
    odm_l.append(np.column_stack([di[keep2], pu[keep2], doo[keep2], mi[keep2]]).astype(np.float32))
    print(f'{mo}: 清洗后 {m.sum():,} 行', flush=True)

dist = np.concatenate(dist_l); dist_day = np.concatenate(dist_day_l)
spd = np.concatenate(spd_l); spd_day = np.concatenate(spd_day_l)
odm = np.concatenate(odm_l)

np.savez_compressed(r'D:/AI/条件转化/.flow_agg_winter.npz',
    dates=dates_np.astype(str), klass=klass, is_holiday=is_hol, holiday_name=hol_names,
    dprecip=dprecip, tmean=tmean, snow=snow_flag, pricing=price_flag,
    hour_cnt=hour_cnt.astype(np.int32), pu_cnt=pu_cnt.astype(np.int32), do_cnt=do_cnt.astype(np.int32),
    dist=dist.astype(np.float32), dist_day=dist_day.astype(np.int32),
    spd=spd.astype(np.float32), spd_day=spd_day.astype(np.int32), odm=odm.astype(np.float32))
np.savez(r'D:/AI/条件转化/.flow_days_winter.npz',
         dates=dates_np.astype(str), klass=klass, is_holiday=is_hol, holiday_name=hol_names,
         dprecip=dprecip, tmean=tmean, snow=snow_flag, pricing=price_flag)
print(f'total cleaned trips {hour_cnt.sum():,} | dist {len(dist):,} | spd {len(spd):,} | odm {len(odm):,}')
print('saved .flow_agg_winter.npz / .flow_days_winter.npz')
