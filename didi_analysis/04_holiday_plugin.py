# -*- coding: utf-8 -*-
"""DiDi(合成) 节假日插件完整预演 —— 提取插件参数并检验两个假设
插件 = ① λ(t) 48槽乘子向量(假日/非假日) ② 区域 γ 向量(200, 哪些区域假日敏感)

检验:
  H1 形态重排: 传统节日"晚起晚散"(晨谷晚峰)、公共假日总量+形态
  H2 区域重排: 效应跨区域离散度(γ 有没有信号)、敏感度与基线份额的关系、热力图熵变化
口径: vs 同 hour-of-week 干净基线(与 02 一致); 区域级同时报 dow 匹配稳健性
输出 04_holiday_results.json(含可直接当插件用的乘子向量) + 控制台"""
import pandas as pd, numpy as np, json

df = pd.read_csv('D:/AI/条件转化/didi_data/ride_hailing.csv',
                 usecols=['day_index', 'half_hour_of_day', 'day_of_week', 'area_id', 'endo_2',
                          'weather_factor', 'public_holiday', 'traditional_festival',
                          'western_festival', 'large_scale_event_1', 'large_scale_event_2'])
df['how'] = df.day_of_week * 48 + df.half_hour_of_day
clean_mask = (df.weather_factor == 0) & (df.public_holiday == 0) & (df.traditional_festival == 0) & \
             (df.western_festival == 0) & (df.large_scale_event_1 == 0) & (df.large_scale_event_2 == 0)
clean = df[clean_mask]
base_how = clean.groupby('how')['endo_2'].mean()
base_how_area = clean.groupby(['how', 'area_id'])['endo_2'].mean()

def slot_profile(mask, name, res):
    """λ(t) 48槽乘子 = 假日各槽效应(vs 同how干净基线)"""
    sub = df[mask & (df.weather_factor == 0)]   # 假日里挑干净天气, 隔离假日效应
    m = sub.merge(base_how.rename('base'), on='how')
    prof = (m.endo_2 / m.base - 1).groupby(m.half_hour_of_day).mean()
    prof = prof.reindex(range(48), fill_value=np.nan)
    v = prof.values
    ok = ~np.isnan(v)
    mor = float(np.nanmean(v[12:24]))    # 06-12时
    eve = float(np.nanmean(v[36:46]))    # 18-23时
    res[name] = dict(profile=[None if np.isnan(x) else float(x) for x in v],
                     mean=float(np.nanmean(v)), morning=mor, evening=eve,
                     peak_slot=int(np.nanargmax(v)), peak_val=float(np.nanmax(v)),
                     trough_slot=int(np.nanargmin(v)), trough_val=float(np.nanmin(v)), n=int(len(m)))
    r = res[name]
    print(f'  {name}: 均值 {r["mean"]*100:+.1f}% | 晨(6-12时) {mor*100:+.1f}% vs 晚(18-23时) {eve*100:+.1f}% | '
          f'峰 槽{r["peak_slot"]}({r["peak_slot"]/2:.1f}时) {r["peak_val"]*100:+.0f}% | 谷 槽{r["trough_slot"]}({r["trough_slot"]/2:.1f}时) {r["trough_val"]*100:+.0f}%')

res = {}
print('===== H1 λ(t) 形态乘子(假日中天气干净行) =====')
slot_profile(df.public_holiday > 0, '公共假日(合并)', res)
slot_profile(df.traditional_festival > 0, '传统节日(合并)', res)
slot_profile(df.western_festival > 0, '西方节日(合并)', res)
# 公共假日分类别(码1-9, n>=5000 才报)
print('  -- 公共假日分类别 --')
res['public_by_code'] = {}
for c, g in df[df.public_holiday > 0].groupby('public_holiday'):
    if len(g) < 5000: continue
    gg = g[g.weather_factor == 0].merge(base_how.rename('base'), on='how')
    e = float((gg.endo_2 / gg.base - 1).mean())
    res['public_by_code'][int(c)] = dict(effect=e, n=int(len(gg)))
    print(f'     code{c}: {e*100:+.1f}% (n={len(gg):,})')

print('\n===== H2 区域 γ 重排(哪些区域假日敏感) =====')
def gamma_vec(mask, name):
    sub = df[mask & (df.weather_factor == 0)]
    m = sub.merge(base_how_area.rename('base'), on=['how', 'area_id']).dropna()
    eff = (m.endo_2 / m.base - 1).groupby(m.area_id)
    mean = eff.mean(); n = eff.size()
    mean = mean[n >= 200]                      # 每区域至少200观测
    # 熵: 假日份额 vs 干净份额
    hol_share = sub.groupby('area_id')['endo_2'].sum(); hol_share /= hol_share.sum()
    cln_share = clean.groupby('area_id')['endo_2'].sum(); cln_share /= cln_share.sum()
    Hh = float(-(hol_share*np.log(hol_share)).sum()); Hc = float(-(cln_share*np.log(cln_share)).sum())
    # 敏感度 vs 基线份额相关
    corr = float(np.corrcoef(np.log(mean.values), np.log(cln_share.reindex(mean.index).values))[0, 1])
    top = mean.nlargest(5); bot = mean.nsmallest(5)
    res[name] = dict(gamma={int(k): float(v) for k, v in mean.items()},
                     std=float(mean.std()), p5=float(mean.quantile(.05)), p95=float(mean.quantile(.95)),
                     ent_holiday=Hh, ent_clean=Hc, corr_logshare=corr)
    print(f'  {name}: 跨区域 std={mean.std()*100:.1f}pp p5~p95 [{mean.quantile(.05)*100:+.0f}%,{mean.quantile(.95)*100:+.0f}%] | '
          f'熵 {Hc:.3f}→{Hh:.3f}({"降" if Hh<Hc else "升"}) | γ~份额相关 {corr:+.2f}')
    print(f'     最敏感: ' + ', '.join(f'区{int(k)}({v*100:+.0f}%)' for k, v in top.items()))
    print(f'     最不敏感: ' + ', '.join(f'区{int(k)}({v*100:+.0f}%)' for k, v in bot.items()))
    return mean

g_pub = gamma_vec(df.public_holiday > 0, 'gamma_public')
g_trd = gamma_vec(df.traditional_festival > 0, 'gamma_traditional')
# 两类假日的区域敏感度是否一致(插件通用性)
common = g_pub.index.intersection(g_trd.index)
if len(common) > 20:
    c = float(np.corrcoef(g_pub[common].values, g_trd[common].values)[0, 1])
    res['gamma_corr_pub_trad'] = c
    print(f'  公共假日 γ vs 传统节日 γ 的相关: r={c:+.2f} ({"可共用一套γ" if c>0.7 else "需分假日类型"})')

json.dump(res, open('04_holiday_results.json', 'w'), indent=1, ensure_ascii=False)
print('\nsaved 04_holiday_results.json (含48槽乘子与γ向量, 可直接当插件参数)')
