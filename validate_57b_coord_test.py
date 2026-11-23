# -*- coding: utf-8 -*-
"""validate_57b: DNN 外推"贴不贴"取决于输入坐标系(同数据同模型同训练集, 只换特征变换)
   结论存档(2026-11-23 实测):
     NYC 出发量, DNN(64x64 ReLU, r<4 训练, seed=1), ≥16 箱实测 +23.8:
       线性 r 输入  → +36.8 (冲过头 13pp)   ← 真实规律在线性轴是饱和曲线(在弯), DNN 直线不弯
       log(r) 输入  → +19.4 (差 4.4pp)      ← 真实规律=log 形式, 在 log 轴上恰好近直线, 直线碰巧贴住
     (8-16 箱: 线性 +16.8 / log +15.5 vs 实测 +26.6 —— 两版都跟不上加速段)
   规律统一: DNN 域外=直线(结构决定, 斜率锁死); 贴住=坐标系运气, 不是外推能力。
   引申: 知道正确函数形式(log)才是外推抓手 —— 公式族 θ·ln(1+r/RC) 的存在理由。
   复现: python validate_57b_coord_test.py (需 .flow_agg/.flow_days/ISD 天气, 约5分钟)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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

def is_edt(utc):
    y = utc.year
    d0 = datetime(y, 3, 1)
    s0 = d0 + timedelta((6 - d0.weekday()) % 7) + timedelta(7)
    d1 = datetime(y, 11, 1)
    return s0 <= utc < d1 + timedelta((6 - d1.weekday()) % 7)

acc = {}
for yr in [2024, 2025]:
    df = pd.read_csv(rf'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                     usecols=['DATE', 'REPORT_TYPE', 'AA1'], dtype=str)
    for _, row in df[df['REPORT_TYPE'] == 'FM-15'].iterrows():
        try: utc = datetime.strptime(row['DATE'], '%Y-%m-%dT%H:%M:%S')
        except Exception: continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        key = (loc.strftime('%Y-%m-%d'), loc.hour)
        e = acc.setdefault(key, [])
        if isinstance(row['AA1'], str) and ',' in row['AA1']:
            p = row['AA1'].split(',')
            if p[0] == '01' and p[1] not in ('9999', '99'): e.append(float(p[1]) * 0.254)

ry, rr = [], []
for i, d in enumerate(dates_all):
    if kl[i] == 'dry': continue
    for h in range(24):
        if lam[dow[i], h] < 200: continue
        w = acc.get((d, h))
        if w is None: continue
        ry.append(hc[i, h] / lam[dow[i], h] - 1)
        rr.append(float(np.mean(w)))
ry = np.array(ry) * 100; rr = np.array(rr)
m_tr = rr < 4

sc_lin = StandardScaler().fit(rr[m_tr].reshape(-1, 1))
sc_log = StandardScaler().fit(np.log1p(rr[m_tr]).reshape(-1, 1))
outs = {}
for nm, sc, tx in [('线性r', sc_lin, lambda v: v.reshape(-1, 1)),
                   ('log(r)', sc_log, lambda v: np.log1p(v).reshape(-1, 1))]:
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam',
                     alpha=1e-3, max_iter=9000, random_state=1)
    m.fit(sc.transform(tx(rr[m_tr])), ry[m_tr])
    outs[nm] = (m, sc, tx)

print(f'{"箱":8s} {"实测":>7s} {"DNN线性r":>9s} {"DNNlog(r)":>9s}')
for lo, hi, nm in [(0.3, 1, '0.3-1'), (1, 4, '1-4'), (4, 8, '4-8'), (8, 16, '8-16'), (16, 1e9, '≥16')]:
    m = (rr >= lo) & (rr < hi)
    row = f'{nm:8s} {ry[m].mean():+7.1f}'
    for nm2 in ['线性r', 'log(r)']:
        mm, sc, tx = outs[nm2]
        row += f' {mm.predict(sc.transform(tx(rr[m]))).mean():+9.1f}'
    print(row)
