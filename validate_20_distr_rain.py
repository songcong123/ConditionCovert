# -*- coding: utf-8 -*-
"""NYC 极端天气下 出行时间/距离/速度/时长 四量分布变化(直接验证)
按小时雨强分箱(干/0.3-1/1-4/≥4 mm/h), 各箱抽距离/速度/时长样本,
报分位数 + vs 干箱 KS。出发时间(λ(t)形态)用逐时计数比。
输出 .nyc_rain_dists.json"""
import numpy as np, json
import pyarrow.parquet as pq
from scipy import stats

rng = np.random.default_rng(7)
MONTHS = [f'2024-{m:02d}' for m in range(4, 11)]
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
day_klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
inten = {}
for dd_, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    inten[(str(dd_), hh)] = mm

BIN_DEFS = [('dry', None, None), ('r_0.3_1', 0.3, 1.0), ('r_1_4', 1.0, 4.0), ('r_ge4', 4.0, 1e9)]
CAP = 300_000
samp = {k: dict(d=[], v=[], t=[], cnt=np.zeros(24)) for k, _, _ in BIN_DEFS}

for mo in MONTHS:
    tb = pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                       columns=['pickup_datetime', 'trip_miles', 'trip_time'])
    dt = np.asarray(tb['pickup_datetime']); mi = np.asarray(tb['trip_miles']); tt = np.asarray(tb['trip_time'])
    del tb
    dts = dt.astype('datetime64[s]')
    dstr = dts.astype('datetime64[D]').astype(str)
    h = (dts.astype('datetime64[h]').astype(int)) % 24
    m = (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800)
    dstr, h, mi, tt = dstr[m], h[m], mi[m], tt[m]
    v = mi / (tt / 3600.0)
    vm = (v > 0.5) & (v < 100)
    for name, lo, hi in BIN_DEFS:
        if name == 'dry':
            sel = np.array([day_klass.get(d, 'other') == 'dry' for d in np.unique(dstr)])
            dry_dates = set(np.unique(dstr)[sel].tolist())
            mask = np.fromiter((d in dry_dates for d in dstr), bool, len(dstr))
        else:
            mask = np.fromiter((inten.get((d, hh__), 0) >= lo and inten.get((d, hh__), 0) < hi
                                for d, hh__ in zip(dstr, h)), bool, len(dstr))
        S = samp[name]
        S['cnt'] += np.bincount(h[mask], minlength=24)
        idx = np.where(mask & vm)[0]
        take = rng.choice(idx, size=min(CAP, len(idx)), replace=False) if len(idx) > CAP else idx
        S['d'].append(mi[take].astype(np.float32)); S['v'].append(v[take].astype(np.float32)); S['t'].append((tt[take] / 60).astype(np.float32))
    print(f'{mo} done')

res = {}
dry = {q: np.concatenate(samp['dry'][q]) for q in ('d', 'v', 't')}
def qs(a): return [float(np.percentile(a, p)) for p in (25, 50, 75, 95)]
print('\n===== NYC 极端天气四量分布变化 (干日 vs 雨强箱) =====')
print(f'{"箱":10s} {"距离中位mi":>9s} {"p95":>6s} {"速度中位mph":>10s} {"p95":>6s} {"时长中位min":>10s} {"p95":>7s} {"KS_d":>7s} {"KS_v":>7s} {"KS_t":>7s}')
for name, _, _ in BIN_DEFS:
    S = {q: np.concatenate(samp[name][q]) for q in ('d', 'v', 't')}
    ks = lambda a, b: float(stats.ks_2samp(rng.choice(a, 200_000, replace=len(a) < 200_000),
                                           rng.choice(b, 200_000, replace=len(b) < 200_000)).statistic)
    qd, qv, qt = qs(S['d']), qs(S['v']), qs(S['t'])
    row = dict(n=int(len(S['d'])), d=qd, v=qv, t=qt,
               ks_d=ks(S['d'], dry['d']), ks_v=ks(S['v'], dry['v']), ks_t=ks(S['t'], dry['t']))
    res[name] = row
    print(f'{name:10s} {qd[1]:9.2f} {qd[3]:6.1f} {qv[1]:10.1f} {qv[3]:6.1f} {qt[1]:10.1f} {qt[3]:7.1f} '
          f'{row["ks_d"]:7.3f} {row["ks_v"]:7.3f} {row["ks_t"]:7.3f}')

# 出发时间形态: 雨强箱的逐时份额 vs 干日份额
share_dry = samp['dry']['cnt'] / samp['dry']['cnt'].sum()
res['lambda_profile'] = {}
print('\nλ(t) 逐时份额比(雨箱/干):')
for name, _, _ in BIN_DEFS:
    if name == 'dry': continue
    s = samp[name]['cnt'] / samp[name]['cnt'].sum()
    ratio = s / share_dry
    res['lambda_profile'][name] = [float(x) for x in ratio]
    hi5 = np.argsort(-ratio)[:3]; lo5 = np.argsort(ratio)[:3]
    print(f'  {name:8s}: 最强时段 ' + ', '.join(f'{h:02d}时×{ratio[h]:.2f}' for h in hi5) +
          ' | 最弱 ' + ', '.join(f'{h:02d}时×{ratio[h]:.2f}' for h in lo5))

json.dump(res, open('D:/AI/条件转化/.nyc_rain_dists.json', 'w'), indent=1, ensure_ascii=False)
print('\nsaved .nyc_rain_dists.json')
