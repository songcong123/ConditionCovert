# -*- coding: utf-8 -*-
"""特征升级前后效果对比(同一RF同一协议):
   旧特征(7个: r/lag/peak/we/交互/hour)  vs  新三组特征(17个: 时段+天气+节假日)
   内插 = 域内随机20%留出 | 外推 = r>=4 盲测(训练从未见过)
   附消融: 新特征去掉节假日组 / 去掉天气组, 看各组贡献"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from feature_builder import build_features, build_holiday_table

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates_all = list(A['dates'])
hc = A['hour_cnt'].astype(float)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates_all])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates_all])
wd = (dow + 3) % 7
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)
inten = np.zeros((len(dates_all), 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

# 天气表(温度/湿度/风速)
def is_edt(utc):
    y = utc.year
    d0 = datetime(y, 3, 1); ss = d0 + timedelta((6 - d0.weekday()) % 7) + timedelta(7)
    d1 = datetime(y, 11, 1); se = d1 + timedelta((6 - d1.weekday()) % 7)
    return ss <= utc < se
acc = {}
for yr in [2024, 2025]:
    df = pd.read_csv(rf'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                     usecols=['DATE', 'REPORT_TYPE', 'AA1', 'TMP', 'DEW', 'WND'], dtype=str)
    fm = df[df['REPORT_TYPE'] == 'FM-15']
    for _, row in fm.iterrows():
        try: utc = datetime.strptime(row['DATE'], '%Y-%m-%dT%H:%M:%S')
        except: continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        key = (loc.strftime('%Y-%m-%d'), loc.hour)
        e = acc.setdefault(key, dict(rain=[], temp=[], dew=[], wind=[]))
        if isinstance(row['AA1'], str) and ',' in row['AA1']:
            p = row['AA1'].split(',')
            if p[0] == '01' and p[1] not in ('9999', '99'): e['rain'].append(float(p[1]) * 0.254)
        if isinstance(row['TMP'], str) and ',' in row['TMP']:
            v = float(row['TMP'].split(',')[0]) / 10
            if -40 < v < 55: e['temp'].append(v)
        if isinstance(row['DEW'], str) and ',' in row['DEW']:
            v = float(row['DEW'].split(',')[0]) / 10
            if -40 < v < 40: e['dew'].append(v)
        if isinstance(row['WND'], str) and ',' in row['WND']:
            v = float(row['WND'].split(',')[3]) / 10
            if 0 <= v < 60: e['wind'].append(v)
def rh_of(t, td):
    if td is None or t is None or td > t: return None
    et = 6.112 * np.exp(17.67 * t / (243.5 + t)); ed = 6.112 * np.exp(17.67 * td / (243.5 + td))
    return min(max(ed / et * 100, 5), 100)
wx = {}
for key, e in acc.items():
    t = float(np.mean(e['temp'])) if e['temp'] else None
    td = float(np.mean(e['dew'])) if e['dew'] else None
    wx[key] = dict(rain=float(np.mean(e['rain'])) if e['rain'] else 0.0, temp=t,
                   humidity=rh_of(t, td), wind=float(np.mean(e['wind'])) if e['wind'] else np.nan)

import csv as _csv
hol_rows = []
for row in _csv.reader(open(r'D:/AI/条件转化/data/exogenous/holidays_us_ny.csv', encoding='utf-8-sig')):
    if row and row[0][:4].isdigit() and '2024-04' <= row[0][:7] <= '2025-11':
        hol_rows.append((row[0], row[4] if len(row) > 4 else row[1], 1))
hol = build_holiday_table(hol_rows, dates_all)

# 样本
rows_d, rows_h, ys = [], [], []
for i, d in enumerate(dates_all):
    for h in range(24):
        base = lam[dow[i], h]
        if base < 200: continue
        rows_d.append(d); rows_h.append(int(h))
        ys.append(hc[i, h] / base - 1)
ys = np.array(ys)
F = build_features(rows_d, rows_h, wx, hol)
DF = pd.DataFrame(F)
r_raw = np.expm1(DF['rain_now'].values)

# ===== 特征集定义 =====
OLD = ['rain_now', 'rain_lag1', 'peak_dummy', 'we_dummy', 'rXpeak', 'rXwe', 'hour']
# 旧集的 peak/we/交互 从 builder 字段重建
DF['peak_dummy'] = DF['segment'].isin([1, 3]).astype(float)
DF['we_dummy'] = DF['is_weekend']
DF['rXpeak'] = DF['peak_dummy'] * DF['rain_now']
DF['rXwe'] = DF['we_dummy'] * DF['rain_now']

NEW_ALL = [c for c in DF.columns if c not in ('peak_dummy', 'we_dummy', 'rXpeak', 'rXwe')]
NEW_TW = [c for c in NEW_ALL if not c.startswith(('hol', 'is_holiday', 'weekend_share'))]   # 时段+天气(无节假日)
NEW_TH = [c for c in NEW_ALL if c not in ('temp', 'humidity', 'wind')]                      # 时段+节假日(无天气扩展)

# NaN: 温湿风缺失小时填中位数
for c in ['temp', 'humidity', 'wind']:
    DF[c] = DF[c].fillna(DF[c].median())

SETS = [('旧特征(7个, r+滞后+峰/周末+交互)', OLD),
        ('新: 时段+天气(13个)', NEW_TW),
        ('新: 时段+节假日(14个)', NEW_TH),
        ('新: 三组全特征(17个)', NEW_ALL)]

rng = np.random.default_rng(42)
idx_in = np.where(r_raw < 4)[0]
perm = rng.permutation(idx_in)
te_in = perm[len(perm) * 4 // 5:]          # 内插测试(域内20%)
m_ex = r_raw >= 4

print(f'{"特征集":34s} {"内插MAE":>8s} {"外推MAE":>8s} {"外推4-8":>8s} {"外推8-16":>8s} {"外推≥16":>8s}')
print('-' * 80)
for nm, cols in SETS:
    X = DF[cols].values
    m_tr = (r_raw < 4) & ~np.isin(np.arange(len(ys)), te_in)
    rf = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X[m_tr], ys[m_tr])
    p_in = rf.predict(X[te_in])
    mae_in = np.mean(np.abs(ys[te_in] - p_in)) * 100
    mae_ex = np.mean(np.abs(ys[m_ex] - rf.predict(X[m_ex]))) * 100
    dev = []
    for lo, hi in [(4, 8), (8, 16), (16, 1e9)]:
        m = m_ex & (r_raw >= lo) & (r_raw < hi)
        a = np.average(ys[m], weights=np.ones(m.sum()))
        p = np.average(rf.predict(X[m]), weights=np.ones(m.sum()))
        dev.append(abs(p - a) * 100)
    print(f'{nm:34s} {mae_in:7.1f}pp {mae_ex:7.1f}pp {dev[0]:7.1f}pp {dev[1]:7.1f}pp {dev[2]:7.1f}pp')
