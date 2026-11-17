# -*- coding: utf-8 -*-
"""validate_48b: 形状层诊断 —— 采样 KS 没贴底, 差在哪?
   ① 各雨强档实测残差σ vs 假设的锁死σ₀ (σ 是否随雨变)
   ② 各档残差偏度 vs 正态假设的 0
   ③ 换实测σ采样 / 用经验残差自举采样 → KS 各降多少(归因)
"""
import numpy as np
from scipy import stats

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); n = len(dates)
klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
rainy = np.where(kl != 'dry')[0]
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)
OKV = (spdV > 1) & (spdV < 60)

mu0_tb = np.full((7, 24), np.nan)
for k7 in range(7):
    sel = np.where((kl == 'dry') & (dow == k7))[0]
    for h in range(24):
        mm = np.isin(spdD, sel) & (spdH == h) & OKV
        if mm.sum() > 50:
            mu0_tb[k7, h] = np.median(np.log(spdV[mm]))

# 干日残差(经验形状 + σ₀)
m_dry = np.isin(spdD, np.where(kl == 'dry')[0]) & OKV
b = mu0_tb[dow[spdD[m_dry]], spdH[m_dry]]
kd = m_dry.copy(); kd[m_dry] = ~np.isnan(b)
res_dry = np.log(spdV[kd]) - mu0_tb[dow[spdD[kd]], spdH[kd]]
sig0 = float(np.std(res_dry))
print(f'σ₀={sig0:.4f} | 干日残差偏度={stats.skew(res_dry):+.3f} | 干日残差超额峰度={stats.kurtosis(res_dry):+.3f}')

th_v, RC_v = -0.0221, 0.20
dv = lambda r: th_v * np.log1p(np.asarray(r, float) / RC_v)
rng = np.random.default_rng(48)

# 雨日逐时残差
key = np.isin(spdD, rainy) & OKV
bk = mu0_tb[dow[spdD[key]], spdH[key]]
keep = key.copy(); keep[key] = ~np.isnan(bk)
dR, hR, vR = spdD[keep], spdH[keep], spdV[keep]
rR = inten[dR, hR]
resid = np.log(vR) - mu0_tb[dow[dR], hR]
m3 = rR >= 0.3
resid, rR = resid[m3], rR[m3]

BINS = [(0.3, 1.0, '0.3-1内插'), (1.0, 4.0, '1-4内插'), (4.0, 12.0, '4-12外推'), (12.0, 1e9, '≥12外推')]
print(f'\n{"档":10s} {"n":>9s} {"实测σ":>7s} {"vsσ₀":>7s} {"偏度":>7s} {"峰度":>7s} | {"KS[σ₀锁]":>9s} {"KS[实测σ]":>9s} {"KS[经验自举]":>11s} {"底":>6s}')
for lo, hi, nm in BINS:
    mm = (rR >= lo) & (rR < hi)
    x = resid[mm]
    if len(x) < 200: continue
    dv_bin = dv(rR[mm])
    sig_bin = float(np.std(x - dv_bin))
    skew = stats.skew(x - dv_bin); kurt = stats.kurtosis(x - dv_bin)
    # 三个采样版
    n_s = len(x)
    g1 = np.exp(dv_bin + rng.normal(0, sig0, n_s)) * 0 + dv_bin + rng.normal(0, sig0, n_s)      # σ₀
    g2 = dv_bin + rng.normal(0, sig_bin, n_s)                                                    # 实测σ
    g3 = dv_bin + rng.choice(res_dry, n_s, replace=True)                                         # 经验残差自举
    ks1 = stats.ks_2samp(x, g1).statistic
    ks2 = stats.ks_2samp(x, g2).statistic
    ks3 = stats.ks_2samp(x, g3).statistic
    perm = rng.permutation(n_s); hlf = n_s // 2
    fl = stats.ks_2samp(x[perm[:hlf]], x[perm[hlf:2*hlf]]).statistic
    print(f'{nm:10s} {len(x):9,} {sig_bin:7.3f} {sig_bin/sig0:7.3f} {skew:+7.3f} {kurt:+7.3f} | {ks1:9.3f} {ks2:9.3f} {ks3:11.3f} {fl:6.3f}')
print('\n读法: KS[σ₀锁]=现在的假设; KS[实测σ]=只放一个σ旋钮能到哪; KS[经验自举]=完全放开形状(上限)')
