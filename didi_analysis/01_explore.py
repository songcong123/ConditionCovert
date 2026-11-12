# -*- coding: utf-8 -*-
"""DiDi RideBench(合成) 数据探察: 结构/各内生变量形态/外生分布"""
import pandas as pd, numpy as np
df = pd.read_csv('D:/AI/条件转化/didi_data/ride_hailing.csv')
print('行数', len(df), '| 区域数', df.area_id.nunique(), '| 天数', df.day_index.nunique(),
      '| 半小时槽数', df.half_hour_of_day.nunique())
print('\n--- endo_1..9 概览(判断谁像需求量) ---')
for i in range(1, 10):
    c = f'endo_{i}'
    s = df[c]
    print(f'{c}: 类型{"整" if s.dtype.kind=="i" else "浮"} 均值{s.mean():9.1f} 中位{s.median():9.1f} '
          f'p99{s.quantile(.99):9.1f} 零占比{(s==0).mean()*100:5.1f}% 日内节律{"有" if s.groupby(df.half_hour_of_day).mean().max()/max(s.groupby(df.half_hour_of_day).mean().min(),1e-9)>2 else "弱"}')
print('\n--- 外生变量分布 ---')
print('weather_factor: 非零占比 %.1f%% 分位数' % ((df.weather_factor>0).mean()*100),
      np.round(df.weather_factor[df.weather_factor>0].quantile([.5,.9,.99]).values,3))
print('public_holiday 取值分布:', df.public_holiday.value_counts().head(8).to_dict())
print('traditional_festival:', df.traditional_festival.value_counts().head(5).to_dict())
print('western_festival:', df.western_festival.value_counts().head(5).to_dict())
print('large_scale_event_1 非零占比 %.2f%% | event_2 %.2f%%' % (
    (df.large_scale_event_1>0).mean()*100,(df.large_scale_event_2>0).mean()*100))
print('\n--- 区域异质性(以 endo_2=候选需求) ---')
g = df.groupby('area_id')['endo_2'].mean()
print('区域均值 p5/p50/p95: %.0f / %.0f / %.0f  最大/最小=%.1f' % (
    g.quantile(.05), g.median(), g.quantile(.95), g.max()/max(g.min(),1e-9)))
