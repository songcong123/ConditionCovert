# -*- coding: utf-8 -*-
"""DiDi(合成) P(s|t) 可分性检验 —— 回答组合方案第③步: 起点要不要条件于时刻
问题: 用 P(s)×λ(t) (可分离) 近似联合 P(s,t), 损失多少? 是否值得上 P(s|t)?

检验一 互信息(信息论口径):
  I(T;S) = D(J(t,s) || P(t)P(s)),  U(S|T) = I / H(S)  ∈[0,1]
  U=0 完全可分离(乘积即真); U 大 → 必须条件化
  含: 半小时粒度 T vs hour-of-week 粒度 T(336) 对比; 计数加权 vs 需求加权
  零假设: 从可分离分布重抽同量样本(参数化 bootstrap)得 I_null

检验二 留出日外样本(预测口径, 参数量公平性的实战检验):
  奇数日估计 P(s) 与 P(s|t) → 预测偶数日各时段份额, 逐时段 JS, 对比平均

检验三 条件放大(插件口径): 干净 vs 降水行(wf>0) vs 假日行 的 U(S|T) 是否变化
输出 03_separability_results.json"""
import pandas as pd, numpy as np, json

rng = np.random.default_rng(7)
df = pd.read_csv('D:/AI/条件转化/didi_data/ride_hailing.csv',
                 usecols=['day_index', 'half_hour_of_day', 'day_of_week', 'area_id', 'endo_2',
                          'weather_factor', 'public_holiday', 'traditional_festival',
                          'western_festival', 'large_scale_event_1', 'large_scale_event_2'])
df['how'] = df.day_of_week * 48 + df.half_hour_of_day
clean_mask = (df.weather_factor == 0) & (df.public_holiday == 0) & (df.traditional_festival == 0) & \
             (df.western_festival == 0) & (df.large_scale_event_1 == 0) & (df.large_scale_event_2 == 0)

def joint(sub, tcol, weight=None):
    """t×s 联合(按 weight 加权, 默认计数)"""
    g = sub.groupby([tcol, 'area_id'])
    W = (g['endo_2'].sum() if weight == 'demand' else g.size()).unstack(fill_value=0)
    return (W.values).astype(float), W.index.values, W.columns.values

def mi_stats(J):
    J = J / J.sum()
    Pt = J.sum(1, keepdims=True); Ps = J.sum(0, keepdims=True)
    S_ = Pt @ Ps
    mask = J > 0
    I = float((J[mask] * np.log(J[mask] / S_[mask])).sum())          # nats
    Hs = float(-(Ps[Ps > 0] * np.log(Ps[Ps > 0])).sum())
    return I, Hs, I / Hs

def js_vec(p, q):
    p = np.maximum(p / p.sum(), 1e-12); q = np.maximum(q / q.sum(), 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

res = {}
clean = df[clean_mask]

# ===== 检验一: 互信息 / 不确定性系数 =====
print('===== 检验一 I(T;S) 与 U(S|T) (干净日, nats) =====')
for tcol, tname in [('half_hour_of_day', '半小时(48)'), ('how', 'hour-of-week(336)')]:
    for wname, w in [('计数', None), ('需求endo_2', 'demand')]:
        J, _, _ = joint(clean, tcol, w)
        I, Hs, U = mi_stats(J)
        # 参数化bootstrap零假设: 可分离分布重抽同量样本
        N = int(J.sum())
        Pt = (J.sum(1) / J.sum()); Ps0 = (J.sum(0) / J.sum())
        Sep = np.outer(Pt, Ps0).ravel()
        draws = rng.multinomial(N, Sep / Sep.sum(), size=20).reshape(20, *J.shape)
        I_null = np.array([mi_stats(d)[0] for d in draws])
        z = (I - I_null.mean()) / max(I_null.std(), 1e-12)
        key = f'{tname}_{wname}'
        res[key] = dict(I=I, H_S=Hs, U=U, I_null_mean=float(I_null.mean()), z=float(z))
        print(f'  T={tname:16s} {wname}: I={I:.4f} H(S)={Hs:.3f} U(S|T)={U*100:.2f}% | 零假设Ī={I_null.mean():.4f} z={z:.0f}')

# ===== 检验二: 留出日外样本 =====
print('\n===== 检验二 留出日预测(奇数日训练→偶数日各时段份额, JS 越低越好) =====')
days = clean.day_index.unique()
train = clean[clean.day_index % 2 == 1]; test = clean[clean.day_index % 2 == 0]
Jtr, _, areas_tr = joint(train, 'half_hour_of_day', 'demand')
Ps_tr = Jtr.sum(0) / Jtr.sum()
Pst_tr = Jtr / Jtr.sum(1, keepdims=True)
test_day_time = test.groupby(['day_index', 'half_hour_of_day'])['endo_2'].sum()
shares = (test.groupby(['day_index', 'half_hour_of_day', 'area_id'])['endo_2'].sum()
          / test_day_time).rename('share').reset_index()
jsA, jsB = [], []
for (d, t), g in shares.groupby(['day_index', 'half_hour_of_day']):
    v = np.zeros(200); v[g.area_id.values - 1] = g.share.values
    if v.sum() <= 0: continue
    jsA.append(js_vec(Ps_tr, v))          # 模型A: 全局P(s)
    jsB.append(js_vec(Pst_tr[t], v))      # 模型B: P(s|t)
jsA, jsB = np.array(jsA), np.array(jsB)
gain = (1 - jsB.mean() / jsA.mean()) * 100
res['oos'] = dict(js_global=float(jsA.mean()), js_cond=float(jsB.mean()), gain_pct=float(gain), n=int(len(jsA)))
print(f'  模型A P(s)全局:     JS={jsA.mean():.5f}')
print(f'  模型B P(s|t)条件:   JS={jsB.mean():.5f}  → 增益 {gain:+.1f}%')

# ===== 逐时段: P(s|t) 偏离全局 P(s) 的剖面 =====
Js, _, _ = joint(clean, 'half_hour_of_day', 'demand')
Ps_g = Js.sum(0) / Js.sum()
prof = np.array([js_vec(Ps_g, Js[t]) for t in range(48)])
res['hourly_js'] = [float(x) for x in prof]
hi = np.argsort(prof)[-5:][::-1]; lo = np.argsort(prof)[:3]
print('  P(s|t) 偏离 P(s) 最大时段: ' + ', '.join(f'{t/2:05.2f}时 JS={prof[t]:.4f}' for t in hi))
print('  最小时段: ' + ', '.join(f'{t/2:05.2f}时 JS={prof[t]:.4f}' for t in lo))

# ===== 检验三: 外生条件下的 U 是否放大 =====
print('\n===== 检验三 外生条件下的 U(S|T) (半小时粒度, 需求加权) =====')
for name, mask in [('干净', clean_mask), ('降水 wf>0', df.weather_factor > 0),
                   ('公共假日', df.public_holiday > 0), ('事件1/2', (df.large_scale_event_1 > 0) | (df.large_scale_event_2 > 0))]:
    sub = df[mask]
    if len(sub) < 1000: continue
    J, _, _ = joint(sub, 'half_hour_of_day', 'demand')
    I, Hs, U = mi_stats(J)
    res[f'U_{name}'] = dict(U=U, n=len(sub))
    print(f'  {name:8s}: U(S|T)={U*100:.2f}%  (n={len(sub):,})')

json.dump(res, open('03_separability_results.json', 'w'), indent=1, ensure_ascii=False)
print('\nsaved 03_separability_results.json')
