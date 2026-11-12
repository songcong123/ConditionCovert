# -*- coding: utf-8 -*-
"""自检：合成 4/1~4/7 一周演示数据（轨迹 CSV + 天气 CSV）→ 跑通 extract→绘图全管线
输出: wireless_adjust/demo_raw.csv, demo_weather.csv, wireless_stats_demo.json, 演示图
用法: python make_demo_data.py && python extract_trip_stats.py demo_raw.csv --weather demo_weather.csv \
      --dates 2024-04-01,2024-04-07 --out wireless_stats_demo.json && python make_adjust_example.py wireless_stats_demo.json
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(4141)
DAYS = ['2023-04-01', '2023-04-02', '2023-04-03', '2023-04-04',
        '2023-04-05', '2023-04-06', '2023-04-07']          # 一~日 完整一周
# 天气: 直接读 chuzhou2 数值天气(真实逐时 rain)
WX = pd.read_csv(r'D:/AI/条件转化/天气数值/chuzhou2_数值天气.csv')
WX = WX[WX['date'].isin(DAYS)]
RAIN = {d: dict(zip(g['hour'], g['rain'])) for d, g in WX.groupby('date')}
rows = []
tid = 0
for d in DAYS:
    dow = (pd.Timestamp(d).dayofweek)
    wknd = dow >= 5
    for h in range(24):
        # 一周形态: 工作日双峰 / 周末单峰偏晚
        n = (520 * np.exp(-((h - 8) / 2.0) ** 2) + 640 * np.exp(-((h - 18) / 2.4) ** 2) + 120) \
            if not wknd else (380 * np.exp(-((h - 11) / 3) ** 2) + 560 * np.exp(-((h - 20) / 2.6) ** 2) + 150)
        n = int(n * rng.uniform(0.9, 1.1))
        r = float(RAIN.get(d, {}).get(h, 0.0))
        n = int(n * (1 + 0.052 * np.log1p(r / 0.2))) if r else n   # 演示: 假设 NYC 系数成立
        n = max(n, 3)
        for _ in range(n):
            tid += 1
            t0 = pd.Timestamp(d).value // 10**6 + h * 3600_000 + int(rng.uniform(0, 55) * 60_000)
            npts = int(rng.integers(6, 40))
            dist = float(rng.lognormal(np.log(850), 0.75))
            v_kmh = float(rng.lognormal(np.log(11.5), 0.28)) * (np.exp(-0.025 * np.log1p(r / 0.3)) if r else 1)
            dur = dist / (v_kmh / 3.6)
            brg = rng.uniform(0, 2 * np.pi)
            for k in range(npts):
                frac = k / (npts - 1)
                dd = dist * frac
                lat = 31.2304 + (dd * np.cos(brg)) / 111320
                lon = 121.4737 + (dd * np.sin(brg)) / (111320 * np.cos(np.radians(31.23)))
                rows.append((f'UE_{tid:07d}', int(t0 + dur * frac * 1000), round(lon, 6), round(lat, 6)))
df = pd.DataFrame(rows, columns=['AmfUEID', 'TimeStamp', 'LongitudeAVG', 'LatitudeAVG'])
# 打乱顺序模拟原始数据(未排序)
df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
df.to_csv('demo_raw.csv', index=False, encoding='utf-8-sig')
print(f'demo_raw.csv: {len(df):,} 行 | {tid:,} 条轨迹 | 天气源: chuzhou2_数值天气.csv(真实)')
