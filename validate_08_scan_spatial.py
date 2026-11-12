# -*- coding: utf-8 -*-
"""空间分布案例数据补扫:
A) 美网2024(8/26-9/8): 逐日"到达zone93"的起点分布(POI=93的PU按zone计数) + 前后各2周基线
B) 马拉松2025(11/2) vs 普通周日: 逐日×逐小时×zone 的PU计数
输出 .flow_spatial.npz"""
import numpy as np, pyarrow.parquet as pq
from datetime import date, timedelta

USO_EVT = [(date(2024, 8, 26) + timedelta(days=k)).isoformat() for k in range(14)]
USO_BASE = [(date(2024, 8, 12) + timedelta(days=k)).isoformat() for k in range(14)] + \
           [(date(2024, 9, 9) + timedelta(days=k)).isoformat() for k in range(14)]
MAR_DAY = '2025-11-02'
MAR_BASE = ['2025-11-09', '2025-11-16', '2025-11-23', '2025-11-30']

uso_arr = {}   # date -> PU-zone counts of trips with DO==93
nov_pu = {}    # date -> (24,264) hourly PU counts (all trips)
nov_tot = {}   # date -> total cleaned trips

def scan(mo, want_dates, uso=None, nov=None):
    path = f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet'
    t = pq.read_table(path, columns=['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time'])
    dt = t['pickup_datetime'].to_numpy()
    pu = t['PULocationID'].to_numpy().astype(np.int32)
    doo = t['DOLocationID'].to_numpy().astype(np.int32)
    mi = t['trip_miles'].to_numpy(); tt = t['trip_time'].to_numpy()
    del t
    spd = mi / (tt.astype(np.float64) / 3600.0)
    m = (pu >= 1) & (pu <= 263) & (doo >= 1) & (doo <= 263) & \
        (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800) & (spd > 0) & (spd < 100)
    pu, doo, dt = pu[m], doo[m], dt[m]
    day = dt.astype('datetime64[D]').astype(str)
    hour = (dt.astype('datetime64[h]').astype(np.int64) % 24).astype(np.int8)
    for d in want_dates:
        sel = day == d
        if not sel.any(): continue
        if uso is not None:
            a = sel & (doo == 93)
            uso[d] = np.bincount(pu[a], minlength=264)
        if nov is not None:
            h = hour[sel]
            z = pu[sel]
            grid = np.zeros((24, 264), np.int64)
            np.add.at(grid, (h, z), 1)
            nov[d] = grid
    print(f'{mo} done', flush=True)

alld = sorted(set(USO_EVT + USO_BASE))
for d in alld:
    uso_arr.setdefault(d, np.zeros(264, np.int64))
scan('2024-08', [d for d in alld if d.startswith('2024-08')], uso=uso_arr)
scan('2024-09', [d for d in alld if d.startswith('2024-09')], uso=uso_arr)

for d in [MAR_DAY] + MAR_BASE:
    nov_pu.setdefault(d, np.zeros((24, 264), np.int64))
scan('2025-11', [MAR_DAY] + MAR_BASE, nov=nov_pu)

np.savez_compressed('D:/AI/条件转化/.flow_spatial.npz',
    dates_uso=np.array(alld), uso_arr=np.stack([uso_arr[d] for d in alld]),
    is_evt=np.array([d in USO_EVT for d in alld]),
    mar_days=np.array([MAR_DAY] + MAR_BASE),
    nov_pu=np.stack([nov_pu[d] for d in [MAR_DAY] + MAR_BASE]))
print('uso evt日均到达93:', uso_arr[USO_EVT[0]].sum(), '... saved .flow_spatial.npz')
