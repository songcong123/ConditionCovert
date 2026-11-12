# -*- coding: utf-8 -*-
"""
第一步：从原始轨迹 CSV 提取逐日统计并关联天气（在有数据的机器上运行）
======================================================================
不做任何行程切割 —— 轨迹 id 在数据里已存在，每行 = (轨迹id, 时间, 经纬度)。

输入：
  raw_csv    原始 CSV，四列即可（列名可配）：
             轨迹id   str/int   如 AmfUEID 或 traj_id
             时间     int/str   13 位 UTC 毫秒（--tz 8 转北京时间）
             经度     float     LongitudeAVG
             纬度     float     LatitudeAVG
  weather    逐时天气 CSV，两种格式自动识别：
             ① 数值版（推荐, 天气数值/chuzhou2_数值天气.csv）：
                date,hour,text,rain(mm/h),snowfall,temperature_2m,
                apparent_temperature,relative_humidity_2m,wind_speed_10m,...
                → 雨强直接用 rain 列（连续值），同时输出温度/风速供多维标定
             ② 文本版（v3 data/<city>/weather.csv）：date,hour,text
                → text→雨强档位映射（小雨1.5/中雨5/...）

处理（每条轨迹只算一次）：
  出发时刻 = 该轨迹最早点（本地时）；行程距离 = 逐点 haversine 累计；
  速度 = 距离 ÷ (末点−首点)；距离单位 m，速度 km/h。
  最简质量过滤：点数 ≥2、距离 >50 m、时长 >60 s（阈值可配）。

输出（一个小 JSON）：
  {
    "dates":  ["2024-04-01",...,"2024-04-07"],
    "weather":   {date: [24 个天气文本]},
    "rain_mmh":  {date: [24 个估算雨强 mm/h]},   # text→雨强映射(小雨1.5/中雨5/大雨12/暴雨25)
    "n_trips":   {date: [24 个整数]},
    "speed_median": {date: [24 个数或 null]},
    "dist_median":  {date: 当日中位距离 m},
    "dist_samples": {date: [≤3000 个抽样距离 m]},
    "trip_count":   {date: 当日轨迹数}
  }

用法：
  python extract_trip_stats.py raw.csv --weather weather.csv \
      --id-col AmfUEID --time-col TimeStamp --lon-col LongitudeAVG --lat-col LatitudeAVG \
      --tz 8 --dates 2024-04-01,2024-04-07 --out wireless_stats_0401_0407.json
"""
import argparse
import json
import numpy as np
import pandas as pd

TEXT_TO_MMH = {'晴': 0, '多云': 0, '阴': 0, '雾': 0,
               '小雨': 1.5, '阵雨': 3, '中雨': 5, '大雨': 12,
               '暴雨': 25, '大暴雨': 40, '雷阵雨': 8, '小雪': 0, '中雪': 0, '大雪': 0}


def hav_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_csv')
    ap.add_argument('--weather', required=True, help='逐时天气 CSV (date,hour,text)')
    ap.add_argument('--id-col', default='AmfUEID')
    ap.add_argument('--time-col', default='TimeStamp')
    ap.add_argument('--lon-col', default='LongitudeAVG')
    ap.add_argument('--lat-col', default='LatitudeAVG')
    ap.add_argument('--tz', type=int, default=8)
    ap.add_argument('--dates', default='2024-04-01,2024-04-07')
    ap.add_argument('--min-dist', type=float, default=50)
    ap.add_argument('--min-dur', type=float, default=60)
    ap.add_argument('--out', default='wireless_stats_0401_0407.json')
    a = ap.parse_args()

    # ---- 天气（自动识别数值版/文本版）----
    w = pd.read_csv(a.weather)
    numeric = 'rain' in w.columns
    if numeric:
        print(f'数值天气: rain 列连续雨强 (max {w["rain"].max():.1f} mm/h)')
    wmap, rmap, tmap, vmap = {}, {}, {}, {}
    for _, r in w.iterrows():
        d = str(r['date']); h = int(r['hour'])
        wmap.setdefault(d, [None] * 24)[h] = str(r.get('text', ''))
        rr = float(r['rain']) if numeric else TEXT_TO_MMH.get(str(r['text']), 0.0)
        rmap.setdefault(d, [0.0] * 24)[h] = rr
        if numeric and 'temperature_2m' in w.columns:
            tmap.setdefault(d, [None] * 24)[h] = None if pd.isna(r['temperature_2m']) else float(r['temperature_2m'])
        if numeric and 'wind_speed_10m' in w.columns:
            vmap.setdefault(d, [None] * 24)[h] = None if pd.isna(r['wind_speed_10m']) else float(r['wind_speed_10m'])

    # ---- 轨迹 ----
    print(f'读取 {a.raw_csv}')
    df = pd.read_csv(a.raw_csv, usecols=[a.id_col, a.time_col, a.lon_col, a.lat_col],
                     dtype={a.id_col: str})
    df.columns = ['tid', 'ts', 'lon', 'lat']
    df = df.sort_values(['tid', 'ts'])
    ts = df['ts'].values.astype(np.float64)
    local = np.datetime64('1970-01-01T00:00:00') + (ts / 1000.0 + a.tz * 3600.0).astype('timedelta64[s]')
    dstr = local.astype('datetime64[D]').astype(str)
    hour = local.astype('datetime64[h]').astype(int) % 24
    d0, d1 = a.dates.split(',')
    m = (dstr >= d0) & (dstr <= d1)
    df = df[m]; dstr, hour = dstr[m], hour[m]
    ts, lon, lat, tid = ts[m], df['lon'].values, df['lat'].values, df['tid'].values
    del df
    print(f'窗口内点数 {len(tid):,}')

    # ---- 按轨迹聚合（sortkey 保证首末点正确）----
    order = np.lexsort((ts, tid))
    ts, lon, lat, tid, dstr, hour = ts[order], lon[order], lat[order], tid[order], dstr[order], hour[order]
    tid_change = tid[1:] != tid[:-1]
    bounds = np.where(tid_change)[0] + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(tid)]])
    print(f'轨迹数 {len(starts):,}')

    dates = sorted(set(dstr))
    out = {'dates': dates,
           'weather': {d: wmap.get(d, [None] * 24) for d in dates},
           'rain_mmh': {d: rmap.get(d, [0.0] * 24) for d in dates},
           'temperature': {d: tmap.get(d, [None] * 24) for d in dates},
           'wind_speed': {d: vmap.get(d, [None] * 24) for d in dates},
           'n_trips': {d: [0] * 24 for d in dates},
           'speed_median': {d: [None] * 24 for d in dates},
           'dist_median': {}, 'dist_samples': {d: [] for d in dates},
           'trip_count': {d: 0 for d in dates}}
    spd_by = {d: [[] for _ in range(24)] for d in dates}
    rng = np.random.default_rng(7)

    for s, e in zip(starts, ends):
        dur = ts[e - 1] - ts[s]
        if dur <= a.min_dur or (e - s) < 2: continue
        d_step = hav_m(lon[s:e-1], lat[s:e-1], lon[s:e][1:], lat[s:e][1:]).sum()  # 相邻点段长累加
        if d_step <= a.min_dist: continue
        d = dstr[s]
        out['n_trips'][d][hour[s]] += 1
        out['trip_count'][d] += 1
        spd_by[d][hour[s]].append(d_step / (dur / 1000.0) * 3.6)   # dur 是 ms → 先转秒
        out['dist_samples'][d].append(float(d_step))

    for d in dates:
        for h in range(24):
            v = spd_by[d][h]
            if len(v) >= 3: out['speed_median'][d][h] = round(float(np.median(v)), 2)
        ss = np.array(out['dist_samples'][d])
        if len(ss):
            out['dist_median'][d] = round(float(np.median(ss)), 1)
            out['dist_samples'][d] = [round(x, 1) for x in
                                      rng.choice(ss, min(3000, len(ss)), replace=False).tolist()]

    json.dump(out, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)
    print('\n逐日汇总:')
    for d in dates:
        wt = [t for t in out['weather'][d] if t]
        wmain = max(set(wt), key=wt.count) if wt else '?'
        print(f"  {d}: {out['trip_count'][d]:>7,} 条 | 主天气 {wmain} | 中位距离 {out['dist_median'].get(d, float('nan')):>6.0f} m")
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
