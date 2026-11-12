# -*- coding: utf-8 -*-
"""补扫: (day, hour, spd) 三元组, float32 — B4 的 μ₀(t) 与 δ_v 残差需要逐样本小时"""
import numpy as np, pyarrow.parquet as pq

rng = np.random.default_rng(44)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(D['dates']); n_days = len(dates)
dates_np = np.array(dates, dtype='datetime64[D]')
CAP = 15000
day_l, hr_l, v_l = [], [], []

for mo in range(4, 11):
    path = f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_2024-{mo:02d}.parquet'
    t = pq.read_table(path, columns=['pickup_datetime', 'PULocationID', 'DOLocationID',
                                     'trip_miles', 'trip_time'])
    dt = t['pickup_datetime'].to_numpy()
    pu = t['PULocationID'].to_numpy().astype(np.int32)
    doo = t['DOLocationID'].to_numpy().astype(np.int32)
    mi = t['trip_miles'].to_numpy()
    tt = t['trip_time'].to_numpy()
    del t
    day = dt.astype('datetime64[D]')
    pos = np.clip(np.searchsorted(dates_np, day), 0, n_days - 1)
    spd_all = mi / (tt.astype(np.float64) / 3600.0)
    m = (dates_np[pos] == day) & (pu >= 1) & (pu <= 263) & (doo >= 1) & (doo <= 263) & \
        (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800) & (spd_all > 0) & (spd_all < 100)
    di = pos[m]; mi = mi[m]; tt = tt[m]; dt = dt[m]
    spd = mi / (tt / 3600.0)
    hour = (dt.astype('datetime64[h]').astype(np.int64) % 24).astype(np.int16)
    nday = np.bincount(di, minlength=n_days)
    keep = rng.random(len(di)) < np.minimum(1.0, CAP / np.maximum(nday, 1))[di]
    day_l.append(di[keep]); hr_l.append(hour[keep]); v_l.append(spd[keep])
    print(f'2024-{mo:02d} ok', flush=True)

np.savez_compressed('D:/AI/条件转化/.flow_spd.npz',
                    day=np.concatenate(day_l).astype(np.int32),
                    hour=np.concatenate(hr_l).astype(np.int8),
                    v=np.concatenate(v_l).astype(np.float32))
print('saved .flow_spd.npz')
