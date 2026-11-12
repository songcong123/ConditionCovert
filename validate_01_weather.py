# -*- coding: utf-8 -*-
"""流程验证 Step0: 解码 NOAA ISD 中央公园站 2024-04~10 逐时降水(UTC→NY本地),
按日降水划分 干/轻雨/中雨/强雨 四档, 排除NY假日。输出 .flow_days.npz"""
import numpy as np, pandas as pd
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

MO_START, MO_END = 4, 10  # 2024-04 ~ 2024-10 暖季(无雪)

# ---- 1) 解码逐小时降水 ----
# ⚠️ 只用 FM-15 常规整点报, 同一(日,时)取 max ——
#    FM-16(SPECI)与 FM-15 同小时重复报 AA1 会重复累加(实测 11 时 12mm 被算成 63mm)
rows = []
df = pd.read_csv('D:/AI/条件转化/data/weather/isd/72505394728_2024.csv',
                 usecols=['DATE', 'REPORT_TYPE', 'AA1'], dtype=str)
df = df[df['REPORT_TYPE'] == 'FM-15']
for s, aa in zip(df['DATE'], df['AA1']):
    mo = int(s[5:7])
    if not (MO_START - 1 <= mo <= MO_END + 1): continue  # 留出 UTC↔本地跨月边界
    try: utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
    except: continue
    loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
    if not (MO_START <= loc.month <= MO_END): continue  # 按纽约本地月过滤
    if not isinstance(aa, str) or ',' not in aa: continue
    p = aa.split(',')
    if p[0] != '01' or p[1] in ('9999', '99'): continue
    mm = float(p[1]) * 0.254  # 百分之一英寸 → mm
    rows.append((loc, mm))
w = pd.DataFrame(rows, columns=['dt', 'precip_mm'])
w['date'] = w['dt'].dt.strftime('%Y-%m-%d')
w['hour'] = w['dt'].dt.hour
daily = w.groupby('date')['precip_mm'].max().groupby(level=0).sum().sort_index()  # 同日本地过滤
# 同(日,时)多报取 max(防重复), 再按日求和
hourly = w.groupby(['date', 'hour'])['precip_mm'].max()
daily = hourly.groupby(level=0).sum().sort_index()
rain_hours = hourly

# ---- 2) 假日排除 ----
hol = set()
for line in open('D:/AI/条件转化/data/exogenous/holidays_us_ny.csv', encoding='utf-8-sig'):
    if line[:4].isdigit(): hol.add(line.split(',')[0])

# ---- 3) 日类型划分 ----
dates, klasses, dprecip = [], [], []
for d, mm in daily.items():
    if d in hol: continue
    k = ('dry' if mm < 0.5 else 'light' if mm < 5 else 'medium' if mm < 15 else 'heavy')
    dates.append(d); klasses.append(k); dprecip.append(mm)
kc = pd.Series(klasses).value_counts()
print('日类型统计(2024-04~10, 排除假日):', dict(kc))
for k in ['dry', 'light', 'medium', 'heavy']:
    ds = [d for d, kk, m in zip(dates, klasses, dprecip) if kk == k]
    print(f'  {k:6s} n={len(ds):3d}  降水范围 {min(m for m,kk,m in zip(dprecip,klasses,dprecip) if kk==k):.1f}~'
          f'{max(m for m,kk,m in zip(dprecip,klasses,dprecip) if kk==k):.1f}mm  样例 {ds[:4]}')

# 逐日雨强(供剂量轴)与逐小时雨(供强度分层)
dates_arr = np.array(dates)
np.savez('D:/AI/条件转化/.flow_days.npz',
         dates=dates_arr, klass=np.array(klasses), dprecip=np.array(dprecip, float),
         rh_date=np.array([k for k, _ in rain_hours.index]),
         rh_hour=np.array([int(h) for _, h in rain_hours.index]),
         rh_mm=np.array(rain_hours.values, float))
print('saved .flow_days.npz')
