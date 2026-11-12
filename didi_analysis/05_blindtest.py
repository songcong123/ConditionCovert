# -*- coding: utf-8 -*-
"""DiDi(合成) 盲测闭环预演 —— "基线×乘子"管线在合成数据上端到端跑通
设计:
  训练 = 前 3 年(day_index<=1095) 干净行: 估 336×200 基线 + 雨剂量曲线 + 假日插件(48槽乘子+区域γ)
  测试 = 第 4 年(day_index>=1096) 全部行: 预测 endo_2, 与真实对比
  对照 = 纯基线(不带任何乘子)
打分: 分行型 MAPE(干净/雨/假日) + 时间剖面 JS + 区域剖面 JS + 两道闸(份额/熵)
输出 05_blindtest_results.json"""
import pandas as pd, numpy as np, json

df = pd.read_csv('D:/AI/条件转化/didi_data/ride_hailing.csv',
                 usecols=['day_index', 'half_hour_of_day', 'day_of_week', 'area_id', 'endo_2',
                          'weather_factor', 'public_holiday', 'traditional_festival',
                          'western_festival', 'large_scale_event_1', 'large_scale_event_2'])
df['how'] = df.day_of_week * 48 + df.half_hour_of_day
is_clean = (df.weather_factor == 0) & (df.public_holiday == 0) & (df.traditional_festival == 0) & \
           (df.western_festival == 0) & (df.large_scale_event_1 == 0) & (df.large_scale_event_2 == 0)

tr = df.day_index <= 1095
te = df.day_index >= 1096
clean_tr = df[tr & is_clean]

# ---- 训练侧参数 ----
base_how_area = clean_tr.groupby(['how', 'area_id'])['endo_2'].mean()
base_how = clean_tr.groupby('how')['endo_2'].mean()

# 雨剂量乘子(小时级, 分箱查表——合成数据超线性, 查表最诚实)
wf_bins = [(0.0001, 2), (2, 5), (5, 10), (10, 99)]
rain_mult = {}
m = df[tr & (df.weather_factor > 0)].merge(base_how.rename('base'), on='how')
rel = m.endo_2 / m.base - 1
for lo, hi in wf_bins:
    sel = (m.weather_factor >= lo) & (m.weather_factor < hi)
    rain_mult[f'{lo}_{hi}'] = float(1 + rel[sel].mean())
    print(f'  雨乘子 wf∈[{lo},{hi}): ×{rain_mult[f"{lo}_{hi}"]:.3f} (n={sel.sum():,})')

def rain_multiplier(wf):
    out = np.ones(len(wf))
    for i, (lo, hi) in enumerate(wf_bins):
        msk = (wf >= lo) & (wf < hi)
        out[msk] = rain_mult[f'{lo}_{hi}']
    return out

# 假日插件(训练侧重提: 48槽乘子 + 区域γ, 与04同法)
plug = {}
hol_defs = [('public_holiday', df.public_holiday > 0), ('traditional_festival', df.traditional_festival > 0)]
for name, mask in hol_defs:
    sub = df[tr & mask & (df.weather_factor == 0)]
    mm = sub.merge(base_how.rename('base'), on='how')
    prof = ((mm.endo_2 / mm.base - 1).groupby(mm.half_hour_of_day).mean()
            .reindex(range(48), fill_value=0.0))
    ma = sub.merge(base_how_area.rename('base'), on=['how', 'area_id']).dropna()
    eff = (ma.endo_2 / ma.base - 1).groupby(ma.area_id)
    g = eff.mean()[eff.size() >= 200]
    plug[name] = dict(prof=(1 + prof).values, gamma=dict({int(k): 1 + float(v) for k, v in g.items()}))
    print(f'  假日插件 {name}: 槽乘子范围 [{(1+prof).min():.2f},{(1+prof).max():.2f}] | γ p5~p95 '
          f'[{g.quantile(.05):+.2f},{g.quantile(.95):+.2f}]')

# ---- 测试侧预测 ----
test = df[te].copy()
test = test.set_index(['how', 'area_id'])
test['base'] = base_how_area
test = test.dropna(subset=['base']).reset_index()
pred_base = test.base.values
mult = rain_multiplier(test.weather_factor.values)
for name, mask_col in [('public_holiday', 'public_holiday'), ('traditional_festival', 'traditional_festival')]:
    msk = test[mask_col].values > 0
    P = plug[name]
    slot_mult = P['prof'][test.half_hour_of_day.values[msk]]
    gam = np.array([P['gamma'].get(int(a), 1.0) for a in test.area_id.values[msk]])
    mult[msk] = mult[msk] * slot_mult * gam
pred_full = pred_base * mult
y = test.endo_2.values

def mape(p, y_): return float(np.mean(np.abs(p / y_ - 1)))
res = {}
print('\n===== 测试年打分 (MAPE, 越低越好) =====')
seg_defs = [('全部行', np.ones(len(test), bool)),
            ('干净行', is_clean[te].reindex(test.index, fill_value=False).values if hasattr(is_clean[te], 'reindex') else None)]
# 干净行 mask 直接重算(避免索引对齐问题)
clean_te = (test.weather_factor == 0) & (test.public_holiday == 0) & (test.traditional_festival == 0) & \
           (test.western_festival == 0) & (test.large_scale_event_1 == 0) & (test.large_scale_event_2 == 0)
for name, msk in [('全部行', np.ones(len(test), bool)), ('干净行', clean_te.values),
                  ('雨行(wf>0)', (test.weather_factor > 0).values),
                  ('公共假日行', (test.public_holiday > 0).values),
                  ('传统节日行', (test.traditional_festival > 0).values)]:
    mb, mf = mape(pred_base[msk], y[msk]), mape(pred_full[msk], y[msk])
    res[f'mape_{name}'] = dict(base=mb, full=mf)
    print(f'  {name:8s}: 纯基线 {mb*100:6.1f}% → 带乘子 {mf*100:6.1f}%  ({(1-mf/mb)*100:+.0f}%)')

# 剖面 JS (时段×区域 联合)
def js(p, q):
    p = np.maximum(p / p.sum(), 1e-12); q = np.maximum(q / q.sum(), 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))
jt = pd.crosstab(test.half_hour_of_day, test.area_id, values=y, aggfunc='sum').reindex(index=range(48), columns=range(1, 201), fill_value=0).values.flatten()
jb = pd.crosstab(test.half_hour_of_day, test.area_id, values=pred_base, aggfunc='sum').reindex(index=range(48), columns=range(1, 201), fill_value=0).values.flatten()
jf = pd.crosstab(test.half_hour_of_day, test.area_id, values=pred_full, aggfunc='sum').reindex(index=range(48), columns=range(1, 201), fill_value=0).values.flatten()
res['js_joint'] = dict(base=js(jb, jt), full=js(jf, jt))
print(f'  (时段×区域)联合 JS: 基线 {js(jb, jt):.5f} → 带乘子 {js(jf, jt):.5f}  ({(1-js(jf,jt)/js(jb,jt))*100:+.0f}%)')

# 两道闸(容量/份额 + 熵)
codes, uniques = pd.factorize(test.area_id.values)
K = len(uniques)
sh_r = np.bincount(codes, weights=np.asarray(y, float), minlength=K); sh_r /= sh_r.sum()
sh_b = np.bincount(codes, weights=np.asarray(pred_base, float), minlength=K); sh_b /= sh_b.sum()
sh_f = np.bincount(codes, weights=np.asarray(pred_full, float), minlength=K); sh_f /= sh_f.sum()
cap_b = np.percentile(sh_b / np.maximum(sh_r, 1e-9), 99)
cap_f = np.percentile(sh_f / np.maximum(sh_r, 1e-9), 99)
Hr = float(-(sh_r*np.log(sh_r)).sum()); Hf = float(-(sh_f*np.log(sh_f)).sum())
res['gates'] = dict(share_p99_base=float(cap_b), share_p99_full=float(cap_f), ent_real=Hr, ent_full=Hf)
print(f'  闸-份额: 生成/真实 p99 基线 {cap_b:.2f} → 带乘子 {cap_f:.2f} (阈值1.3)')
print(f'  闸-熵: 真实 {Hr:.3f} vs 带乘子 {Hf:.3f} ({"✓" if abs(Hf-Hr)<0.01 else "⚠"})')

# ---- 诊断: 测试年效应 vs 训练年效应(分清"应用方式错"还是"模拟器非平稳") ----
print('\n===== 诊断: 测试年实测效应 vs 训练年乘子 =====')
mt = test.merge(base_how.rename('base2'), on='how')
rel_t = mt.endo_2 / mt.base2 - 1
for lo, hi in wf_bins:
    sel = (mt.weather_factor >= lo) & (mt.weather_factor < hi)
    if sel.sum() < 100: continue
    tr_v = rain_mult[f'{lo}_{hi}']
    te_v = 1 + rel_t[sel].mean()
    print(f'  雨 wf∈[{lo},{hi}): 训练×{tr_v:.3f} vs 测试×{te_v:.3f} ({"平稳" if abs(te_v-tr_v)<0.05 else "⚠非平稳"})')
for name, col in [('public_holiday', 'public_holiday'), ('traditional_festival', 'traditional_festival')]:
    sel = mt[col] > 0
    tr_mean = float((plug[name]['prof'] - 1).mean() + np.mean([g - 1 for g in plug[name]['gamma'].values()]))
    te_mean = float(rel_t[sel].mean())
    print(f'  {name}: 训练均值效应 {tr_mean*100:+.0f}% vs 测试 {te_mean*100:+.0f}% ({"平稳" if abs(te_mean-tr_mean)<0.05 else "⚠非平稳"})')

json.dump(res, open('05_blindtest_results.json', 'w'), indent=1, ensure_ascii=False)
print('\nsaved 05_blindtest_results.json')
