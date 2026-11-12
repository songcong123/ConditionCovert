# -*- coding: utf-8 -*-
"""流程验证 Step0b: 扫描 TLC HV FHV 2024-04~10, 按本地日聚合:
逐日 24h 计数 / PU/DO 逐zone计数 / OD矩阵(稀疏) / 距离样本 / 速度样本 / (PU,DO,miles)绕路样本
清洗: 0.1<miles<100, 60s<time<10800s, 0<v<100mph, zone 1..263
输出 .flow_agg.npz"""
import numpy as np, pyarrow.parquet as pq

rng = np.random.default_rng(42)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(D['dates'])                      # 已分类的本地日期(214天)
didx_of = {d: i for i, d in enumerate(dates)}
n_days = len(dates)
dates_np = np.array(dates, dtype='datetime64[D]')

CAP_D = 15000   # 每日距离样本上限
CAP_V = 15000   # 每日速度样本上限
CAP_M = 8000    # 每日绕路三元组上限

hour_cnt = np.zeros((n_days, 24), np.int64)
pu_cnt   = np.zeros((n_days, 264), np.int64)
do_cnt   = np.zeros((n_days, 264), np.int64)
od_day_l, od_pair_l, od_cnt_l = [], [], []
dist_l, dist_day_l = [], []
spd_l, spd_day_l = [], []
odm_day_l, odm_pu_l, odm_do_l, odm_mi_l = [], [], [], []

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
    # 仅保留已分类日期
    pos = np.searchsorted(dates_np, day)
    pos_c = np.clip(pos, 0, n_days - 1)
    valid_date = (dates_np[pos_c] == day)
    spd_all = mi / (tt.astype(np.float64) / 3600.0)
    m = valid_date & (pu >= 1) & (pu <= 263) & (doo >= 1) & (doo <= 263) & \
        (mi > 0.1) & (mi < 100) & (tt > 60) & (tt < 10800) & (spd_all > 0) & (spd_all < 100)
    di = pos_c[m]; pu = pu[m]; doo = doo[m]; mi = mi[m]; tt = tt[m]
    dt = dt[m]

    hour = (dt.astype('datetime64[h]').astype(np.int64) % 24)
    np.add.at(hour_cnt, (di, hour), 1)
    np.add.at(pu_cnt, (di, pu), 1)
    np.add.at(do_cnt, (di, doo), 1)
    # OD 稀疏
    pair = pu.astype(np.int64) * 264 + doo
    key = di.astype(np.int64) * 69696 + pair
    uk, uc = np.unique(key, return_counts=True)
    od_day_l.append((uk // 69696).astype(np.int32)); od_pair_l.append((uk % 69696).astype(np.int32))
    od_cnt_l.append(uc.astype(np.int32))

    # 每日抽样(向量化: 按日keep概率)
    nday = np.bincount(di, minlength=n_days)
    pkeep = np.minimum(1.0, CAP_D / np.maximum(nday, 1))[di]
    keep = rng.random(len(di)) < pkeep
    dist_l.append(mi[keep]); dist_day_l.append(di[keep])
    spd = mi / (tt / 3600.0)
    spd_l.append(spd[keep]); spd_day_l.append(di[keep])
    pkeep2 = np.minimum(1.0, CAP_M / np.maximum(nday, 1))[di]
    keep2 = rng.random(len(di)) < pkeep2
    odm_day_l.append(di[keep2]); odm_pu_l.append(pu[keep2])
    odm_do_l.append(doo[keep2]); odm_mi_l.append(mi[keep2])
    print(f'2024-{mo:02d}: 清洗后 {m.sum():,} 行', flush=True)

dist = np.concatenate(dist_l); dist_day = np.concatenate(dist_day_l)
spd = np.concatenate(spd_l); spd_day = np.concatenate(spd_day_l)
odm = np.column_stack([np.concatenate(odm_day_l), np.concatenate(odm_pu_l),
                       np.concatenate(odm_do_l), np.concatenate(odm_mi_l)])
np.savez_compressed('D:/AI/条件转化/.flow_agg.npz',
    dates=dates_np.astype(str), hour_cnt=hour_cnt.astype(np.int32),
    pu_cnt=pu_cnt.astype(np.int32), do_cnt=do_cnt.astype(np.int32),
    od_day=np.concatenate(od_day_l), od_pair=np.concatenate(od_pair_l), od_cnt=np.concatenate(od_cnt_l),
    dist=dist.astype(np.float32), dist_day=dist_day.astype(np.int32),
    spd=spd.astype(np.float32), spd_day=spd_day.astype(np.int32),
    odm=odm.astype(np.int32))
print(f'total cleaned trips {hour_cnt.sum():,} | dist samples {len(dist):,} | spd {len(spd):,} | odm {len(odm):,}')
print('saved .flow_agg.npz')
