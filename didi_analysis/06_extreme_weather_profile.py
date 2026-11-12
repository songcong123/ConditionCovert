# -*- coding: utf-8 -*-
"""DiDi(合成) 极端天气下的 出行时间(λ(t)形态)变化 + 9个endo身份识别
1) weather_factor 剂量分箱 × 48槽形态效应(vs 同how干净基线): 总量/峰位/晨晚结构/形态JS
2) endo_1~9 剂量-响应 + 日节律强度 → 识别哪个像需求(+雨弹性)/哪个像速度或供给(−)
输出 06_extreme_weather_results.json"""
import pandas as pd, numpy as np, json

df = pd.read_csv('D:/AI/条件转化/didi_data/ride_hailing.csv')
df['how'] = df.day_of_week * 48 + df.half_hour_of_day
is_clean = (df.weather_factor == 0) & (df.public_holiday == 0) & (df.traditional_festival == 0) & \
           (df.western_festival == 0) & (df.large_scale_event_1 == 0) & (df.large_scale_event_2 == 0)
clean = df[is_clean]
res = {}

# ===== 1) 极端天气下 λ(t) 形态变化 =====
bins = [(0.0001, 2, 'wf(0,2] 轻度'), (2, 5, 'wf(2,5] 中度'), (5, 10, 'wf(5,10] 重度'), (10, 99, 'wf>10 极端')]
print('===== weather_factor 剂量 × λ(t) 形态效应 (endo_2, vs 同how干净基线) =====')
base_how = clean.groupby('how')['endo_2'].mean()
slot_share_clean = clean.groupby('half_hour_of_day')['endo_2'].sum().values.astype(float)
slot_share_clean /= slot_share_clean.sum()
rows = []
for lo, hi, nm in bins:
    sub = df[(df.weather_factor >= lo) & (df.weather_factor < hi)].merge(base_how.rename('base'), on='how')
    eff = (sub.endo_2 / sub.base - 1)
    prof = eff.groupby(sub.half_hour_of_day).mean().reindex(range(48), fill_value=np.nan)
    v = prof.values * 100
    share = sub.groupby('half_hour_of_day')['endo_2'].sum().values.astype(float)
    share /= share.sum()
    js_profile = float(0.5 * (np.sum(share*np.log(share/slot_share_clean)) + np.sum(slot_share_clean*np.log(slot_share_clean/share))))
    row = dict(dose=nm, mean=float(np.nanmean(v)),
               night=float(np.nanmean(v[0:12])), morning=float(np.nanmean(v[12:24])),
               afternoon=float(np.nanmean(v[24:36])), evening=float(np.nanmean(v[36:48])),
               peak_slot=int(np.nanargmax(v)), peak=float(np.nanmax(v)),
               trough_slot=int(np.nanargmin(v)), trough=float(np.nanmin(v)),
               profile_js=js_profile, n=len(sub))
    rows.append(row)
    print(f'  {nm}: 均值{row["mean"]:+6.1f}% | 夜{row["night"]:+5.1f} 晨{row["morning"]:+5.1f} 午{row["afternoon"]:+5.1f} 晚{row["evening"]:+5.1f} | '
          f'峰槽{row["peak_slot"]/2:.1f}时{row["peak"]:+.0f}% 谷槽{row["trough_slot"]/2:.1f}时{row["trough"]:+.0f}% | 形态JS={js_profile:.4f}')
res['lambda_profile'] = rows

# ===== 2) endo_1~9 身份识别 =====
print('\n===== endo_1~9: 剂量-响应 × 日节律 → 身份识别 =====')
ident = []
clean_slot_mean = clean.groupby('half_hour_of_day').mean(numeric_only=True)
for k in range(1, 10):
    col = f'endo_{k}'
    m = df[df.weather_factor > 0].merge(base_how.rename('base'), on='how')
    rain_eff = float((m[col] / m.base - 1).mean())
    m10 = df[df.weather_factor > 10].merge(base_how.rename('base'), on='how')
    rain10 = float((m10[col] / m10.base - 1).mean())
    hol = df[df.public_holiday > 0].merge(base_how.rename('base'), on='how')
    hol_eff = float((hol[col] / hol.base - 1).mean())
    prof48 = clean_slot_mean[col].values
    rhythm = float(prof48.max() / max(prof48.min(), 1e-9))
    ident.append(dict(endo=k, rain_all=rain_eff, rain_extreme=rain10, holiday=hol_eff,
                      rhythm_ratio=rhythm, integral=bool((df[col] % 1 == 0).mean() > 0.9)))
    print(f'  endo_{k}: 雨效应{rain_eff*100:+6.1f}% (极端{rain10*100:+6.1f}%) 假日{hol_eff*100:+6.1f}% 日节律峰谷比{rhythm:4.1f} 整数型{ident[-1]["integral"]}')
res['endo_identity'] = ident
print('\n判读: 雨效应大幅为正=需求型; 明显为负=供给/效率型(速度/接驾时长代理); 假日弹性区分通勤/休闲')

json.dump(res, open('06_extreme_weather_results.json', 'w'), indent=1, ensure_ascii=False)
print('saved 06_extreme_weather_results.json')
