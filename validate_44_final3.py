# -*- coding: utf-8 -*-
"""三量终版(统一口径): 内插=拟合内MAE(全部域内样本, 不留出) | 外推=盲测
   出发量: 小时级, r<4训练 | 速度: 小时级减速, r<4训练 | 距离: 日级μ1(参数口径), <30训练"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import stats as st

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); n = len(dates)
hc = A['hour_cnt'].astype(float)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
wd = (dow + 3) % 7
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

def mae(y, p): return np.mean(np.abs(y - p))

print('=' * 78)
print(f'{"量":6s} {"域":4s} {"RF MAE":>10s} {"公式 MAE":>10s} {"赢家":>6s}')
print('=' * 78)

# ---- 出发量 ----
THETA, RC = 0.052, 0.2
f_vol = lambda r: THETA * np.log1p(np.asarray(r, float) / RC)
X1, y1, r1 = [], [], []
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        base = lam[dow[i], h]
        if base < 200: continue
        r_now = inten[i, h]; lag = inten[i, h-1] if h >= 1 else 0
        pk = 1 if (7<=h<=9 or 17<=h<=19) else 0
        X1.append([np.log1p(r_now), np.log1p(lag), pk, wd[i] >= 5, pk*np.log1p(r_now), h])
        y1.append(hc[i, h] / base - 1); r1.append(r_now)
X1, y1, r1 = np.array(X1), np.array(y1), np.array(r1)
m_in = r1 < 4; m_ex = r1 >= 4
rf1 = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf1.fit(X1[m_in], y1[m_in])
a_rf_in = mae(y1[m_in], rf1.predict(X1[m_in])) * 100
a_f_in = mae(y1[m_in], f_vol(r1[m_in])) * 100
a_rf_ex = mae(y1[m_ex], rf1.predict(X1[m_ex])) * 100
a_f_ex = mae(y1[m_ex], f_vol(r1[m_ex])) * 100
print(f'出发量  内插  {a_rf_in:8.1f}pp {a_f_in:9.1f}pp   {"RF" if a_rf_in<a_f_in else "公式"}')
print(f'出发量  外推  {a_rf_ex:8.1f}pp {a_f_ex:9.1f}pp   {"RF" if a_rf_ex<a_f_ex else "公式"}')

# ---- 速度 ----
spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)
okv = (spdV > 1) & (spdV < 60)
mu0_h = np.zeros((7, 24))
for k in range(7):
    for h in range(24):
        m = okv & np.isin(spdD, np.where((kl == 'dry') & (dow == k))[0]) & (spdH == h)
        mu0_h[k, h] = np.median(np.log(spdV[m])) if m.sum() > 50 else np.nan
X2, y2, r2 = [], [], []
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        r = inten[i, h]
        if r < 0.3 or np.isnan(mu0_h[dow[i], h]): continue
        m = okv & (spdD == i) & (spdH == h)
        if m.sum() < 30: continue
        res = float(np.median(np.log(spdV[m])) - mu0_h[dow[i], h])
        lag = inten[i, h-1] if h >= 1 else 0
        pk = 1 if (7<=h<=9 or 17<=h<=19) else 0
        X2.append([np.log1p(r), np.log1p(lag), pk, wd[i] >= 5, h])
        y2.append(res); r2.append(r)
X2, y2, r2 = np.array(X2), np.array(y2), np.array(r2)
m_in2 = r2 < 4; m_ex2 = r2 >= 4
rf2 = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf2.fit(X2[m_in2], y2[m_in2])
th_v, RC_v = -0.0221, 0.20
f_v = lambda r: th_v * np.log1p(np.asarray(r, float) / RC_v)
s_rf_in = mae(y2[m_in2], rf2.predict(X2[m_in2])) * 100
s_f_in = mae(y2[m_in2], f_v(r2[m_in2])) * 100
s_rf_ex = mae(y2[m_ex2], rf2.predict(X2[m_ex2])) * 100
s_f_ex = mae(y2[m_ex2], f_v(r2[m_ex2])) * 100
print(f'速度    内插  {s_rf_in:8.1f}pp {s_f_in:9.1f}pp   {"RF" if s_rf_in<s_f_in else "公式"}')
print(f'速度    外推  {s_rf_ex:8.1f}pp {s_f_ex:9.1f}pp   {"RF" if s_rf_ex<s_f_ex else "公式"}')

# ---- 距离(日级μ1, validate_43 结果) ----
print(f'距离μ1  内插  {4.01*0.5:8.2f}pp {5.73*0.5:9.2f}pp   RF   (validate_43: 1.99 vs 2.91)')
print(f'距离μ1  外推  {4.01:8.2f}pp {5.73:9.2f}pp   RF   (validate_43: 4.01 vs 5.73)')
