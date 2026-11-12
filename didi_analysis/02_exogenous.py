# -*- coding: utf-8 -*-
"""DiDi(合成) 外生效应预演: 复用 NYC 管线口径
- 基线: hour-of-week × endo_2 (需求型候选, 匿名)
- weather_factor 剂量-响应(分箱, 同钟点匹配)
- 假日/传统节日/西方节日/大型事件 效应
输出 02_exogenous_results.json + 控制台"""
import pandas as pd, numpy as np, json

df = pd.read_csv('D:/AI/条件转化/didi_data/ride_hailing.csv',
                 usecols=['day_index','half_hour_of_day','day_of_week','area_id','endo_2',
                          'weather_factor','public_holiday','traditional_festival',
                          'western_festival','large_scale_event_1','large_scale_event_2'])
df['how'] = df.day_of_week * 48 + df.half_hour_of_day
# 基线: 无任何外生(全零)的 how × area 平均
clean = df[(df.weather_factor==0)&(df.public_holiday==0)&(df.traditional_festival==0)&
           (df.western_festival==0)&(df.large_scale_event_1==0)&(df.large_scale_event_2==0)]
base_how = clean.groupby('how')['endo_2'].mean()
base_how_area = clean.groupby(['how','area_id'])['endo_2'].mean()

def effect(mask, tag, area_level=False):
    sub = df[mask]
    if len(sub)==0: return
    if area_level:
        m = sub.merge(base_how_area.rename('base'), on=['how','area_id'], how='left')
    else:
        m = sub.merge(base_how.rename('base'), on='how', how='left')
    m = m.dropna()
    rel = (m.endo_2/m.base - 1)
    print(f'{tag}: n={len(m):,} 效应={rel.mean()*100:+.1f}% (p5~p95: {rel.quantile(.05)*100:+.0f}%~{rel.quantile(.95)*100:+.0f}%)')
    return float(rel.mean())

print('===== weather_factor 剂量-响应 (vs 同钟点干净基线) =====')
bins=[(0.0001,2),(2,5),(5,10),(10,99)]
res={}
for lo,hi in bins:
    r=effect((df.weather_factor>=lo)&(df.weather_factor<hi), f'  wf∈[{lo},{hi})')
    res[f'wf_{lo}_{hi}']=r
print('\n===== 节假日/事件效应 =====')
res['holiday']=effect(df.public_holiday>0,'公共假日(所有类别)')
res['trad']=effect(df.traditional_festival>0,'传统节日')
res['west']=effect(df.western_festival>0,'西方节日')
res['evt1']=effect(df.large_scale_event_1>0,'大型事件1')
res['evt2']=effect(df.large_scale_event_2>0,'大型事件2')
print('\n===== 假日分时段形态(传统节日: 各半时桶效应) =====')
sub=df[df.traditional_festival>0].merge(base_how.rename('base'),on='how')
sub=(sub.endo_2/sub.base-1).groupby(sub.half_hour_of_day).mean()*100
peak,low=sub.idxmax(),sub.idxmin()
print(f'  峰值时段 半小时槽{peak}({peak/2:.1f}时): {sub.max():+.0f}% | 谷值 槽{low}({low/2:.1f}时): {sub.min():+.0f}%')
json.dump(res,open('02_exogenous_results.json','w'),indent=1)
print('saved 02_exogenous_results.json')
