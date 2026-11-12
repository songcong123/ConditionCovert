# -*- coding: utf-8 -*-
"""扩展天气-出行库: 2024+2025 两个暖季(04~10月)
A) 天气解码(两年份, FM-15去重) → 日分类 .flow_days.npz (覆盖旧版)
B) 行程单次扫描(14个月) → .flow_agg.npz + .flow_odm.npz + .flow_spd.npz (覆盖旧版)
C) 汇总: 新日构成 / 小时雨强剂量-响应 / 强雨日四量效应(同年同星期基线, 规避2025拥堵收费政策位移) / top单日
输出 .flow_expand_summary.json"""
import numpy as np, pandas as pd, json
from datetime import datetime, timedelta
import pyarrow.parquet as pq
from scipy import stats

# ============ A) 天气 ============
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

rows = []
for y in [2024, 2025]:
    df = pd.read_csv(f'D:/AI/条件转化/data/weather/isd/72505394728_{y}.csv',
                     usecols=['DATE', 'REPORT_TYPE', 'AA1'], dtype=str)
    df = df[df['REPORT_TYPE'] == 'FM-15']
    for s, aa in zip(df['DATE'], df['AA1']):
        mo = int(s[5:7])
        if not (3 <= mo <= 11): continue
        try: utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
        except: continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        if not (4 <= loc.month <= 10): continue
        if not isinstance(aa, str) or ',' not in aa: continue
        p = aa.split(',')
        if p[0] != '01' or p[1] in ('9999', '99'): continue
        rows.append((loc, float(p[1]) * 0.254))
w = pd.DataFrame(rows, columns=['dt', 'mm'])
w['date'] = w['dt'].dt.strftime('%Y-%m-%d'); w['hour'] = w['dt'].dt.hour
hourly = w.groupby(['date', 'hour'])['mm'].max()
daily = hourly.groupby(level=0).sum().sort_index()

hol = set()
for line in open('D:/AI/条件转化/data/exogenous/holidays_us_ny.csv', encoding='utf-8-sig'):
    if line[:4].isdigit(): hol.add(line.split(',')[0])

# 补充源: GHCN 日值(中央公园 94728) — 覆盖 ISD 年度文件缺失的 2025-08-28 之后
import os
SUPP = 'D:/AI/条件转化/data/weather/isd/ghcn_daily_94728_2025supp.csv'
supp = {}
has_hourly_set = set(daily.index)
if os.path.exists(SUPP):
    for _, r in pd.read_csv(SUPP).iterrows():
        d = str(r['DATE']); supp[d] = float(r['PRCP'])
    for d, mm_ in supp.items():
        if d not in has_hourly_set: daily.loc[d] = mm_
    daily = daily.sort_index()
    print(f'[A] GHCN 补充 {sum(1 for d in supp if d not in has_hourly_set)} 天(仅日总量, 无小时线)')

dates, klasses, dprecip = [], [], []
for d, mm in daily.items():
    if d in hol: continue
    k = 'dry' if mm < 0.5 else 'light' if mm < 5 else 'medium' if mm < 15 else 'heavy'
    dates.append(d); klasses.append(k); dprecip.append(mm)
print(f'[A] 两暖季日分类: ' + str(pd.Series(klasses).value_counts().to_dict()))
np.savez('D:/AI/条件转化/.flow_days.npz',
         dates=np.array(dates), klass=np.array(klasses), dprecip=np.array(dprecip, float),
         has_hourly=np.array([d in has_hourly_set for d in dates]),
         rh_date=np.array([k for k, _ in hourly.index]),
         rh_hour=np.array([int(h) for _, h in hourly.index]),
         rh_mm=np.array(hourly.values, float))

# ============ B) 行程扫描 ============
rng = np.random.default_rng(42)
dates_np = np.array(dates, dtype='datetime64[D]')
n_days = len(dates); didx = {d: i for i, d in enumerate(dates)}
CAP_D, CAP_M = 15000, 8000
hour_cnt = np.zeros((n_days, 24), np.int64)
pu_cnt = np.zeros((n_days, 264), np.int64); do_cnt = np.zeros((n_days, 264), np.int64)
od_day_l, od_pair_l, od_cnt_l = [], [], []
dist_l, dist_day_l = [], []
spd_l, spd_day_l, spd_hr_l = [], [], []
odm_l = []
months = [f'{y}-{m:02d}' for y in (2024, 2025) for m in range(4, 11)]
for mo in months:
    t = pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                      columns=['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time'])
    dt = t['pickup_datetime'].to_numpy()
    pu = t['PULocationID'].to_numpy().astype(np.int32)
    doo = t['DOLocationID'].to_numpy().astype(np.int32)
    mi = t['trip_miles'].to_numpy(); tt = t['trip_time'].to_numpy()
    del t
    day = dt.astype('datetime64[D]')
    pos = np.clip(np.searchsorted(dates_np, day), 0, n_days - 1)
    spd_all = mi / (tt.astype(np.float64) / 3600.0)
    m = (dates_np[pos] == day) & (pu >= 1) & (pu <= 263) & (doo >= 1) & (doo <= 263) & \
        (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800) & (spd_all > 0) & (spd_all < 100)
    di = pos[m]; pu2, doo2, mi2 = pu[m], doo[m], mi[m]; dt2 = dt[m]
    hour = (dt2.astype('datetime64[h]').astype(np.int64) % 24).astype(np.int8)
    np.add.at(hour_cnt, (di, hour), 1)
    np.add.at(pu_cnt, (di, pu2), 1); np.add.at(do_cnt, (di, doo2), 1)
    key = di.astype(np.int64) * 69696 + pu2.astype(np.int64) * 264 + doo2
    uk, uc = np.unique(key, return_counts=True)
    od_day_l.append((uk // 69696).astype(np.int32)); od_pair_l.append((uk % 69696).astype(np.int32))
    od_cnt_l.append(uc.astype(np.int32))
    nday = np.bincount(di, minlength=n_days)
    keep = rng.random(len(di)) < np.minimum(1.0, CAP_D / np.maximum(nday, 1))[di]
    dist_l.append(mi2[keep]); dist_day_l.append(di[keep])
    spd_l.append(spd_all[m][keep]); spd_day_l.append(di[keep]); spd_hr_l.append(hour[keep])
    keep2 = rng.random(len(di)) < np.minimum(1.0, CAP_M / np.maximum(nday, 1))[di]
    odm_l.append(np.column_stack([di[keep2], pu2[keep2], doo2[keep2]]).astype(np.int32))
    odm_l.append(mi2[keep2].astype(np.float32))
    print(f'[B] {mo}: 清洗 {m.sum():,}', flush=True)
# odm_l 交替 idx/miles, 拆开
odm_idx = np.concatenate(odm_l[0::2])
odm_mi = np.concatenate(odm_l[1::2])
np.savez_compressed('D:/AI/条件转化/.flow_agg.npz',
    dates=dates_np.astype(str), hour_cnt=hour_cnt.astype(np.int32),
    pu_cnt=pu_cnt.astype(np.int32), do_cnt=do_cnt.astype(np.int32),
    od_day=np.concatenate(od_day_l), od_pair=np.concatenate(od_pair_l), od_cnt=np.concatenate(od_cnt_l),
    dist=np.concatenate(dist_l).astype(np.float32), dist_day=np.concatenate(dist_day_l).astype(np.int32))
np.savez_compressed('D:/AI/条件转化/.flow_spd.npz',
    day=np.concatenate(spd_day_l).astype(np.int32), hour=np.concatenate(spd_hr_l).astype(np.int8),
    v=np.concatenate(spd_l).astype(np.float32))
np.savez_compressed('D:/AI/条件转化/.flow_odm.npz', idx=odm_idx, miles=odm_mi)
print(f'[B] 总计 {hour_cnt.sum():,} 行 → 三个npz已覆盖保存', flush=True)

# ============ C) 汇总(同年同星期基线) ============
kl = np.array(klasses); dp = np.array(dprecip)
year = np.array([int(d[:4]) for d in dates])
dow = np.array([np.datetime64(d, 'D').view('int64') % 7 for d in dates])
m_dry = kl == 'dry'
# lam0[year, dow, hour]
lam0 = np.zeros((2, 7, 24)); lam0n = np.zeros((2, 7))
for yi, y in enumerate([2024, 2025]):
    for k in range(7):
        sel = m_dry & (year == y) & (dow == k)
        lam0[yi, k] = hour_cnt[sel].sum(0); lam0n[yi, k] = sel.sum()
lam0 /= np.maximum(lam0n, 1)[:, :, None]
yidx = np.where(year == 2024, 0, 1)
def exp_dry(i): return lam0[yidx[i], dow[i]].sum()

# 小时雨强矩阵
rhd = {d: np.zeros(24) for d in dates}
for (d, h), v in hourly.items():
    if d in didx: rhd[d][h] = v
inten = np.array([rhd[d] for d in dates])

spd_v = np.concatenate(spd_l); spd_d = np.concatenate(spd_day_l); spd_h = np.concatenate(spd_hr_l)
lsv_all = np.log(spd_v)
mu0 = np.zeros(24)
hsel_dry = spd_h[m_dry[spd_d]]
lsv_dry = lsv_all[m_dry[spd_d]]
for h in range(24): mu0[h] = lsv_dry[hsel_dry == h].mean()
sigma0 = float(np.std(lsv_dry - mu0[hsel_dry]))

# 剂量-响应(小时箱) — 仅限有小时线的日子(ISD), GHCN 补充日跳过
has_hourly = np.array([d in has_hourly_set for d in dates])
bins_h = [(0.3, 1.0, '0.3-1'), (1.0, 4.0, '1-4'), (4.0, 1e9, '>=4')]
measured = {}
rain_i = np.where((kl != 'dry') & has_hourly)[0]
for lo, hi, nm in bins_h:
    bm = (inten >= lo) & (inten < hi)
    bm[~np.isin(np.arange(n_days), rain_i)] = False
    rat, wts = [], []
    for i in rain_i:
        for h in range(24):
            if bm[i, h] and lam0[yidx[i], dow[i], h] > 100:
                rat.append(hour_cnt[i, h] / lam0[yidx[i], dow[i], h] - 1)
                wts.append(lam0[yidx[i], dow[i], h])
    sm = bm[spd_d, spd_h]
    measured[nm] = dict(r=float(inten[bm].mean()), dvol=float(np.average(rat, weights=wts)),
                        d_v=float((lsv_all[sm] - mu0[spd_h[sm]]).mean()) if sm.sum() > 200 else None,
                        sig=float(np.std(lsv_all[sm] - mu0[spd_h[sm]])) if sm.sum() > 200 else None,
                        nh=int(bm.sum()))
print('[C] 小时剂量-响应(同年基线):')
for nm in measured:
    mm_ = measured[nm]
    print(f"  {nm:5s}mm/h n={mm_['nh']:4d} r̄={mm_['r']:.2f} δ_vol={mm_['dvol']:+.4f} "
          f"δ_v={mm_['d_v']:+.4f} σ={mm_['sig']:.3f}")

# 强雨日四量(同年基线)
dist_a = np.concatenate(dist_l); dist_day_a = np.concatenate(dist_day_l)
dd_all = dist_a[(dist_a > 0.1) & (dist_a < 80)]
dd_dry = dist_a[np.isin(dist_day_a, np.where(m_dry)[0])]
dd_dry = dd_dry[(dd_dry > 0.1) & (dd_dry < 80)]
med_dry = np.median(np.log(dd_dry)); sh_dry = (dd_dry < 2).mean()
he_i = np.where(kl == 'heavy')[0]; me_i = np.where(kl == 'medium')[0]
def day_fx(idx):
    dvol = float(np.mean([hour_cnt[i].sum() / exp_dry(i) - 1 for i in idx]))
    d_d = float(np.mean([np.median(np.log(dist_a[dist_day_a == i])) - med_dry for i in idx]))
    sh = float(np.mean([(dist_a[dist_day_a == i] < 2).mean() - sh_dry for i in idx]))
    d_v = float(np.mean([np.mean(np.log(spd_v[spd_d == i]) - mu0[spd_h[spd_d == i]]) for i in idx]))
    return dict(dvol=dvol, d_d=d_d, sh=sh, d_v=d_v, n=len(idx))
E_he, E_me = day_fx(he_i), day_fx(me_i)
print(f"[C] 强雨日(同年基线): n={E_he['n']} δ_vol={E_he['dvol']:+.3f} δ_d={E_he['d_d']:+.3f} "
      f"短途{E_he['sh']*100:+.1f}pp δ_v={E_he['d_v']:+.3f}")

# top 单日
top = he_i[np.argsort(-dp[he_i])][:8]
single = {}
print('[C] top8 单日:')
for i in top:
    dv = hour_cnt[i].sum() / exp_dry(i) - 1
    dmed = np.median(np.log(dist_a[dist_day_a == i])) - med_dry
    single[dates[i]] = dict(mm=float(dp[i]), dvol=float(dv), d_d=float(dmed), year=int(year[i]))
    print(f'  {dates[i]} ({dp[i]:.0f}mm): δ_vol={dv*100:+.1f}% δ_d={dmed:+.3f}')

# 中国口径构成
zh_bins = [(0, 0.5, '微雨'), (0.5, 5, '小雨'), (5, 15, '中雨'), (15, 25, '大雨'),
           (25, 50, '暴雨'), (50, 100, '大暴雨'), (100, 1e9, '特大暴雨')]
zh_cnt = {nm: int(((dp >= lo) & (dp < hi)).sum()) for lo, hi, nm in zh_bins}
print('[C] 中国口径:', zh_cnt)

json.dump(dict(zh_cnt=zh_cnt, measured=measured, E_he=E_he, E_me=E_me, single=single,
               sigma0=sigma0, n_days=int(n_days),
               klass_cnt={k: int((kl == k).sum()) for k in ['dry', 'light', 'medium', 'heavy']}),
          open('D:/AI/条件转化/.flow_expand_summary.json', 'w'), indent=1, ensure_ascii=False)
print('saved .flow_expand_summary.json')
