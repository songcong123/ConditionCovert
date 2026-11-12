# -*- coding: utf-8 -*-
"""补扫: 只重采 (day, pu, do, miles) 三元组, miles 存 float32 (validate_02 误存 int32)"""
import numpy as np, pyarrow.parquet as pq

rng = np.random.default_rng(43)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(D['dates']); n_days = len(dates)
dates_np = np.array(dates, dtype='datetime64[D]')
CAP_M = 8000
day_l, pu_l, do_l, mi_l = [], [], [], []

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
    di = pos[m]; pu = pu[m]; doo = doo[m]; mi = mi[m]
    nday = np.bincount(di, minlength=n_days)
    keep = rng.random(len(di)) < np.minimum(1.0, CAP_M / np.maximum(nday, 1))[di]
    day_l.append(di[keep]); pu_l.append(pu[keep]); do_l.append(doo[keep]); mi_l.append(mi[keep])
    print(f'2024-{mo:02d} ok', flush=True)

odm = np.column_stack([np.concatenate(day_l), np.concatenate(pu_l), np.concatenate(do_l)]).astype(np.int32)
np.savez_compressed('D:/AI/条件转化/.flow_odm.npz', idx=odm, miles=np.concatenate(mi_l).astype(np.float32))
print('saved .flow_odm.npz', len(odm))
