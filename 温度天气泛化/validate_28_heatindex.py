# -*- coding: utf-8 -*-
"""体感温度(热指数 HI = f(干球温度, 露点→湿度)) 的剂量标定:
   回答"出租车域湿热组合是否比干球温度更有解释力"
   RH = Magnus(T, Td); HI = NOAA 回归式(T°F, RH%)
   方法与 validate_27 完全同构: 干日 + 同年×日型×同钟点×舒适HI基线
输出: .flow_heatindex.json
"""
import numpy as np, json
from datetime import datetime, timedelta
import pandas as pd
from scipy import optimize

def load_isd(years=(2024, 2025)):
    out = {}
    for yr in years:
        df = pd.read_csv(f'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                         usecols=['DATE', 'REPORT_TYPE', 'TMP', 'DEW'], dtype=str)
        fm15 = df[df['REPORT_TYPE'] == 'FM-15']
        for s, tmp, dew in zip(fm15['DATE'], fm15['TMP'], fm15['DEW']):
            try:
                t = float(tmp.split(',')[0]) / 10.0
                td = float(dew.split(',')[0]) / 10.0
            except: continue
            if not (-40 < t < 55) or td < -60 or td > t + 1: continue
            utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
            loc = utc - timedelta(hours=4)
            key = (loc.strftime('%Y-%m-%d'), loc.hour)
            out.setdefault(key, []).append((t, td))
    return {k: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))) for k, v in out.items()}

def rh_of(t, td):
    e_td = 6.112 * np.exp(17.67 * td / (243.5 + td))
    e_t = 6.112 * np.exp(17.67 * t / (243.5 + t))
    return float(np.clip(e_td / e_t * 100, 5, 100))

def heat_index_c(t_c, rh):
    tF = t_c * 9 / 5 + 32
    HI = (-42.379 + 2.04901523 * tF + 10.14333127 * rh - 0.22475541 * tF * rh
          - 6.83783e-3 * tF ** 2 - 5.481717e-2 * rh ** 2 + 1.22874e-3 * tF ** 2 * rh
          + 8.5282e-4 * tF * rh ** 2 - 1.99e-6 * tF ** 2 * rh ** 2)
    return (HI - 32) * 5 / 9

temps_hi = {}
for k, (t, td) in load_isd().items():
    rh = rh_of(t, td)
    temps_hi[k] = (t, rh, heat_index_c(t, rh))
print(f'HI 序列: {len(temps_hi)} 点')

A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates']); hc = A['hour_cnt'].astype(np.float64)
klmap = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klmap[d] for d in dates])
yr = np.array([int(str(d)[:4]) for d in dates])
dow = np.array([(np.datetime64(d, 'D').view('int64') + 3) % 7 for d in dates])
wk = (dow < 5).astype(int)
temp_h = np.full((len(dates), 24), np.nan); hi_h = np.full((len(dates), 24), np.nan)
for i, d in enumerate(dates):
    for h in range(24):
        v = temps_hi.get((d, h))
        if v: temp_h[i, h], hi_h[i, h] = v[0], v[2]
dry = kl == 'dry'

COMF = (16.0, 26.0)   # 舒适 HI 段
base, samples = {}, []
for i in range(len(dates)):
    if not dry[i]: continue
    for h in range(24):
        hi = hi_h[i, h]
        if np.isnan(hi) or hc[i, h] <= 0: continue
        key = (yr[i], wk[i], h)
        if COMF[0] <= hi <= COMF[1]:
            base.setdefault(key, []).append(hc[i, h])
        else:
            samples.append((hi, temp_h[i, h], key, hc[i, h]))
bmap = {k: float(np.median(v)) for k, v in base.items() if len(v) >= 3}
rec = [(hi, t, hc / bmap[k]) for hi, t, k, hc in samples if k in bmap and bmap[k] > 0]
HIa = np.array([r[0] for r in rec]); Ta = np.array([r[1] for r in rec]); Ra = np.array([r[2] for r in rec])
hot = HIa > COMF[1]
print(f'样本: HI 热侧 {hot.sum()} | HI 冷侧 {(HIa < COMF[0]).sum()}')

print('\n=== 按 HI(湿热指数) 分箱: 出发量乘子 ===')
for lo, hi in [(26, 30), (30, 35), (35, 42), (42, 60)]:
    m = (HIa >= lo) & (HIa < hi)
    if m.sum() >= 30:
        print(f'  HI [{lo},{hi})°C: 中位乘子 {np.median(Ra[m]):.3f} (n={m.sum()}) | 同时段按干球T中位 {np.median(Ta[m]):.1f}°C')

# 对照: 同样这些热侧样本, 按干球 T 分箱 (26~30/30~35/35+)
print('\n=== 按干球 T 分箱(同一样本集): 出发量乘子 ===')
for lo, hi in [(26, 30), (30, 35), (35, 45)]:
    m = (Ta >= lo) & (Ta < hi)
    if m.sum() >= 30:
        print(f'  T [{lo},{hi})°C: 中位乘子 {np.median(Ra[m]):.3f} (n={m.sum()})')

# HI 解释力 vs T 解释力: 对热侧样本, 用分箱中位乘子对剂量x的 R²
def binned_r2(x, y, edges):
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 30: bx.append((lo + hi) / 2); by.append(float(np.median(y[m])))
    bx, by = np.array(bx), np.array(by)
    if len(bx) < 3: return float('nan')
    A_ = np.vstack([bx, np.ones(len(bx))]).T
    coef, *_ = np.linalg.lstsq(A_, by, rcond=None)
    pred = A_ @ coef
    return 1 - np.sum((by - pred) ** 2) / np.sum((by - by.mean()) ** 2)

e_hi = np.array([26, 30, 35, 42, 60]); e_t = np.array([26, 30, 35, 45])
r2_hi = binned_r2(HIa[hot], Ra[hot], e_hi)
r2_t = binned_r2(Ta[hot], Ra[hot], e_t)
print(f'\n线性解释力(热侧分箱中位): HI 的 R²={r2_hi:.3f} vs 干球 T 的 R²={r2_t:.3f}')

# 极端闷热档 vs 同温度低湿档 (控制 T, 只变 RH): T∈[29,33) 两半
m_t = (Ta >= 29) & (Ta < 33)
if m_t.sum() > 100:
    rh_med = float(np.median([rh_of(t, t) for t in []])) if False else None
    hi_split = HIa[m_t] >= np.median(HIa[m_t])
    print(f'\n控制 T∈[29,33): 闷热半区乘子 {np.median(Ra[m_t][hi_split]):.3f} vs 干爽半区 {np.median(Ra[m_t][~hi_split]):.3f}')

json.dump(dict(r2_hi=round(float(r2_hi), 3), r2_t=round(float(r2_t), 3),
               n_hot=int(hot.sum())), open(r'D:/AI/条件转化/.flow_heatindex.json', 'w'))
print('saved .flow_heatindex.json')
