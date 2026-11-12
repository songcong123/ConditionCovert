# -*- coding: utf-8 -*-
"""空间 γ 的剂量公式拟合(真实数据):
   γ(s,r) = 1 + Δ̂(s) · ln(1+r/RC) / ln(1+r_B/RC)
   Δ̂(s) = a_air·1_air(s) + c + b·d_cbd(s)   (机场项组级直读 + 距核心距离梯度 OLS)
   数据: .rain_spatial.npz (雨时×zone 实测 vs 同钟点干日基线, 2024-04~10)
输出: .flow_space_dose.json
"""
import numpy as np, csv, json
from scipy import optimize

R = np.load(r'D:/AI/条件转化/.rain_spatial.npz')
puA, puB, howA, howB = R['puA'], R['puB'], R['howA'], R['howB']
dry = R['pu_dry_how'].reshape(7 * 24, 264)
Z = np.load(r'D:/AI/条件转化/.flow_zones.npz')
d_cbd = np.zeros(264); d_cbd[1:264] = Z['dist_km'][161, 1:264]
ds = dry / np.maximum(dry.sum(1, keepdims=True), 1)

def shares(pu, how):
    w = how / max(how.sum(), 1)
    b = (w[:, None] * ds).sum(0); b /= b.sum()
    r = pu / max(pu.sum(), 1)
    return r, b

rA, bA = shares(puA, howA)
rB, bB = shares(puB, howB)
AIR = np.zeros(264, bool); AIR[[1, 132, 138]] = True
DB = rB / np.maximum(bB, 1e-9) - 1
DA = rA / np.maximum(bA, 1e-9) - 1

# 机场: 组级直读(与 validate_18 解剖口径一致)
aB = rB[AIR].sum() / bB[AIR].sum() - 1
aA = rA[AIR].sum() / bA[AIR].sum() - 1
print(f'机场组级 Δ: 暴雨箱 {aB:+.3f} ({aB*100:+.1f}%) | 1-4箱 {aA:+.3f} ({aA*100:+.1f}%)')

# 非机场区: 过滤小基线 → OLS 距离梯度
good = (bB > 0.004) & (~AIR) & (bA > 0.002)
zz = np.where(good)[0]
X = np.column_stack([np.ones(zz.size), d_cbd[zz]])
betaB, *_ = np.linalg.lstsq(X, DB[zz], rcond=None)
pred = X @ betaB
r2 = 1 - np.sum((DB[zz] - pred) ** 2) / np.sum((DB[zz] - DB[zz].mean()) ** 2)
corr = np.corrcoef(d_cbd[zz], DB[zz])[0, 1]
c, b = float(betaB[0]), float(betaB[1])
print(f'非机场梯度: c(0km截距)={c:+.4f} ({c*100:+.2f}%) | b={b:+.5f}/km (每远1km {b*100:+.2f}pp)')
print(f'R²={r2:.3f} | 梯度相关 r={corr:.3f} | n={zz.size}')
CORE = [161, 162, 163, 164, 230, 231, 79, 24, 225]
core_pred = c + b * float(np.mean(d_cbd[CORE]))
core_act = rB[CORE].sum() / bB[CORE].sum() - 1
print(f'核心组校验: 模型 {core_pred:+.3f} vs 实际 {core_act:+.3f} ({core_act*100:+.1f}%)')

# 剂量 RC: 用机场对(组级, 信噪比最高)反解; 非机场区弱识别仅作一致性检查
D2 = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(D2['dates']); didx = {d: i for i, d in enumerate(dates)}
inten = np.zeros((len(dates), 24))
for dd, hh, mm in zip(D2['rh_date'], D2['rh_hour'].astype(int), D2['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm
klmap = dict(zip(D2['dates'].tolist(), D2['klass'].tolist()))
kl = np.array([klmap[d] for d in dates])
hA = inten[(kl[:, None] != 'dry') & (inten >= 1) & (inten < 4)]
hB = inten[(kl[:, None] != 'dry') & (inten >= 4)]
rA_med, rB_med = float(np.median(hA)), float(np.median(hB))
ratio_air = aA / aB
sol = optimize.brentq(lambda rc: np.log1p(rA_med / rc) / np.log1p(rB_med / rc) - ratio_air, 0.05, 500)
RC_use = round(sol, 1)
print(f'机场剂量对: {aA:+.3f}/{aB:+.3f} = {ratio_air:.3f} → 解得 RC = {sol:.2f} (取 {RC_use} mm/h)')
sig = good & (np.abs(DB) > 0.05) & (np.sign(DB) == np.sign(DA))
ratio_zone = float(np.median(DA[sig] / DB[sig])) if sig.sum() > 5 else float('nan')
print(f'非机场显著区一致性: DA/DB 中位 = {ratio_zone:.2f} (弱识别, 采用机场对结果)')

# JS 对照: 不调 vs 公式拟合 vs 分档直读
def js(p, q):
    m = (p > 0) & (q > 0)
    return 0.5 * (np.sum(p[m] * np.log(p[m] / q[m])) + np.sum(q[m] * np.log(q[m] / p[m])))
Dn = lambda r: np.log1p(r / RC_use) / np.log1p(rB_med / RC_use)
gm = np.ones(264)
gm[1:264] = np.maximum(1 + ((aB - c) * AIR[1:264] + c + b * d_cbd[1:264]) * Dn(rB_med), 0.05)
gm[AIR] = 1 + aB
gm_s = gm / gm.sum()
rBs, bBs = rB / rB.sum(), bB / bB.sum()
js_base = js(rBs, bBs)
mix_s = gm_s * bBs; mix_s /= mix_s.sum()
js_model = js(rBs, mix_s)
gm_d = np.ones(264); gm_d[1:264] = np.maximum(1 + DB[1:264], 0.05); gm_d[AIR] = 1 + aB
gm_ds = gm_d / gm_d.sum()
mix_d = gm_ds * bBs; mix_d /= mix_d.sum()
js_direct = js(rBs, mix_d)
print(f'\n=== 暴雨小时级 JS 对照 ===')
print(f'不调(基线直用): {js_base:.5f}')
print(f'公式拟合 γ:     {js_model:.5f}  (降幅 {(1 - js_model / js_base) * 100:.0f}%)')
print(f'分档直读上限:   {js_direct:.5f}')

# 外推样例
for r in [10, 30, 60]:
    dd = Dn(r)
    print(f'外推 r={r}mm/h: 机场 γ={max(1 + aB * dd, 0.05):.2f} | 核心(2km) γ={1 + (c + b * 2) * dd:.3f} | '
          f'远端(15km) γ={max(1 + (c + b * 15) * dd, 0.05):.2f}')

res = dict(a_air=round(float(aB), 4), a_air_A=round(float(aA), 4), c=round(c, 4),
           b_km=round(b, 5), r2=round(float(r2), 3), corr=round(float(corr), 3),
           RC=RC_use, ratio_air=round(float(ratio_air), 3), ratio_zone=round(float(ratio_zone), 3),
           rA_med=rA_med, rB_med=rB_med, n_grad=int(zz.size),
           js_base=round(float(js_base), 5), js_model=round(float(js_model), 5),
           js_direct=round(float(js_direct), 5),
           core_check=dict(pred=round(core_pred, 4), actual=round(float(core_act), 4)))
json.dump(res, open(r'D:/AI/条件转化/.flow_space_dose.json', 'w'), ensure_ascii=False, indent=1)
print('saved .flow_space_dose.json')
