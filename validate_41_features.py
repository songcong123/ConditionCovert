# -*- coding: utf-8 -*-
"""validate_41: 在 NYC 数据上验证特征构造器(feature_builder)
   构造 (日,时) 级全特征表 + 标签(出发量效应), 打印:
   ① 特征表预览  ② 各特征与标签的相关性排序  ③ RF(全特征) 内插/外推快速检验"""
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
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)

# ---- 天气查找表: ISD 2024/2025 (rain=AA1, temp/humidity=TMP/DEW→RH, wind=WND) ----
def is_edt(utc):
    y = utc.year
    d0 = datetime(y, 3, 1)
    ssn_start = d0 + timedelta((6 - d0.weekday()) % 7) + timedelta(7)
    d1 = datetime(y, 11, 1)
    ssn_end = d1 + timedelta((6 - d1.weekday()) % 7)
    return ssn_start <= utc < ssn_end

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
            p = row['WND'].split(',')
            v = float(p[3]) / 10
            if 0 <= v < 60: e['wind'].append(v)

def rh_of(t, td):
    if td is None or t is None or td > t: return None
    e_t = 6.112 * np.exp(17.67 * t / (243.5 + t))
    e_d = 6.112 * np.exp(17.67 * td / (243.5 + td))
    return min(max(e_d / e_t * 100, 5), 100)

wx = {}
for key, e in acc.items():
    t = float(np.mean(e['temp'])) if e['temp'] else None
    td = float(np.mean(e['dew'])) if e['dew'] else None
    wx[key] = dict(rain=float(np.mean(e['rain'])) if e['rain'] else 0.0,
                   temp=t, humidity=rh_of(t, td),
                   wind=float(np.mean(e['wind'])) if e['wind'] else np.nan)
print(f'天气表: {len(wx)} 个(日,时)点')

# ---- 假日表 ----
hol_rows = []
import csv as _csv
for row in _csv.reader(open(r'D:/AI/条件转化/data/exogenous/holidays_us_ny.csv', encoding='utf-8-sig')):
    if row and row[0][:4].isdigit():
        if '2024-04' <= row[0][:7] <= '2025-11':
            hol_rows.append((row[0], row[4] if len(row) > 4 else row[1], 1))
hol = build_holiday_table(hol_rows, dates_all)
print(f'假日表: {len(hol_rows)} 个假日日')

# ---- 样本: 全部(日,时)非干日 + 干日抽样(标签基准) ----
rows_d, rows_h, ys = [], [], []
for i, d in enumerate(dates_all):
    for h in range(24):
        base = lam[dow[i], h]
        if base < 200: continue
        rows_d.append(d); rows_h.append(int(h))
        ys.append(hc[i, h] / base - 1)
ys = np.array(ys)
F = build_features(rows_d, rows_h, wx, hol)
df = pd.DataFrame(F)
df['y'] = ys
df['date'] = rows_d
print(f'\n样本 {len(df)} 个(日,时) | 特征列: {[c for c in df.columns if c not in ("y", "date")]}')
print('\n特征表预览(前5行):')
print(df[['hour', 'segment', 'is_weekend', 'rain_now', 'temp', 'humidity', 'wind',
          'is_holiday', 'hol_dist', 'weekend_share', 'y']].head(5).round(3).to_string(index=False))

# ---- 相关性 ----
print('\n与出发量效应 y 的 Pearson 相关(样本=全部日,含干日):')
feat_cols = [c for c in df.columns if c not in ('y', 'date')]
cors = []
for c in feat_cols:
    m = df[c].notna()
    if m.sum() < 100: continue
    cors.append((c, float(np.corrcoef(df[c][m].to_numpy(dtype=float), df['y'][m].to_numpy(dtype=float))[0, 1])))
for c, r_ in sorted(cors, key=lambda t: -abs(t[1])):
    print(f'  {c:14s} {r_:+.3f}')

# ---- RF 快检: 内插 vs 外推(以雨强4为界) ----
r_raw = np.expm1(df['rain_now'].values)
Xf = df[feat_cols].fillna(0).values
m_tr = r_raw < 4
rf = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(Xf[m_tr], ys[m_tr])
mae_in = np.mean(np.abs(ys[m_tr] - rf.predict(Xf[m_tr]))) * 100
m_ex = r_raw >= 4
mae_ex = np.mean(np.abs(ys[m_ex] - rf.predict(Xf[m_ex]))) * 100
print(f'\nRF(三组全特征) 拟合内MAE {mae_in:.1f}pp | 外推MAE(r>=4) {mae_ex:.1f}pp')
imp = sorted(zip(feat_cols, rf.feature_importances_), key=lambda t: -t[1])
print('特征重要性 TOP8:', [(c, round(v, 3)) for c, v in imp[:8]])
