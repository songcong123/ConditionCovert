# -*- coding: utf-8 -*-
"""方案A 第一步: 生成伪极端训练集 (方法.md §9-A)
用途: 混入条件生成器训练 → 让生成器"见过"连续条件, 极端不再是 OOD
条件向量 = (天气类型, 日降水mm, 峰值小时雨强mm/h, 逐时雨强r_h[24])  训练时照常条件 dropout

参数来源(不许拍脑勺, 全部可追溯):
  · 晴天基线 B1/B3/B4/B5 + 热力图γ: 2024-04~10 干日/轻雨日实测(pass2b 拟合, .flow_scores.json)
  · 小时级剂量 θ_vol/RC_vol, θ_v/RC_v: 三箱实测拟合(δ_vol R²=1.00)
  · 日级 δ_d/短途/ρ_ap: 轻雨标定 × log(1+r/R_C) 外推, R_C=1.0(v1.1 下调后)
  · 暴雪参数: 方法.md §3.1/§10 文献先验区间(暖季无雪数据, 属"文献为辅"通道, 已在场景表中注明)
输出: data/pseudo_extreme/pseudo_extreme_train.parquet + 场景表 json"""
import numpy as np, json, os
import pandas as pd
from scipy import stats

rng = np.random.default_rng(2026)
OUT = 'D:/AI/条件转化/data/pseudo_extreme'
os.makedirs(OUT, exist_ok=True)

# ---- 载入 pass2b 拟合参数与聚合资产 ----
F = json.load(open('D:/AI/条件转化/.flow_scores.json'))
B3 = F['B3']; theta = F['theta']; B5 = F['B5']; sigma0 = F['B4']['sigma0']
TH_med = F['TH']; r_li = F['r_li']; R_C = 1.0
th_vol, RC_vol, th_v, RC_v = theta['vol_h'], theta['RC_H'], theta['v_h'], theta['RC_H']
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
G = np.load('D:/AI/条件转化/.flow_gen.npz', allow_pickle=True)
Z = np.load('D:/AI/条件转化/.flow_zones.npz')
dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
dow = np.array([np.datetime64(d, 'D').view('int64') % 7 for d in dates])
hour_cnt = A['hour_cnt'].astype(np.int64)
pu_cnt = A['pu_cnt'].astype(np.int64); do_cnt = A['do_cnt'].astype(np.int64)
m_dry, m_li = kl == 'dry', kl == 'light'
dry_i, li_i = np.where(m_dry)[0], np.where(m_li)[0]
how_sum = np.zeros((7, 24)); how_n = np.zeros(7)
for k in range(7):
    sel = m_dry & (dow == k)
    how_sum[k] = hour_cnt[sel].sum(0); how_n[k] = sel.sum()
lam0_how = how_sum / np.maximum(how_n, 1)[:, None]
mu0_t = G['mu0_t']; pu_share0 = G['pu_share0']
pu_dry = pu_cnt[m_dry].sum(0).astype(float); do_dry = do_cnt[m_dry].sum(0).astype(float)
pu_r = pu_cnt[li_i].sum(0).astype(float); do_r = do_cnt[li_i].sum(0).astype(float)
pu_dry_f = np.maximum(pu_dry, 20.0); do_dry_f = np.maximum(do_dry, 20.0)
gamma_pu = np.log((pu_r + 0.05 * pu_dry_f) / (1.05 * pu_dry_f))
gamma_do = np.log((do_r + 0.05 * do_dry_f) / (1.05 * do_dry_f))
lam_ext_s = pu_share0 * np.exp(gamma_pu); lam_ext_s /= lam_ext_s.sum()
do_share0 = do_dry / do_dry.sum()
D_ext = do_share0 * np.exp(gamma_do); D_ext /= D_ext.sum()
dist_km = Z['dist_km'].astype(np.float64)
XMIN = B3['xmin']

def extrap_day(eff_li, r): return float(eff_li * np.log(1 + r / R_C) / np.log(1 + r_li / R_C))

def sample_B3(B, n, r_):
    u = r_.random(n)
    c1 = B['w1']; c2 = c1 + B['w2']; c3 = c2 + B['w_ap']
    m1 = u < c1; m2 = (u >= c1) & (u < c2); m3 = (u >= c2) & (u < c3); m4 = ~(m1 | m2 | m3)
    x = np.empty(n)
    x[m1] = np.exp(r_.normal(B['mu1'], B['sd1'], m1.sum()))
    x[m2] = np.exp(r_.normal(B['mu2'], B['sd2'], m2.sum()))
    x[m3] = np.exp(r_.normal(B['mu_ap'], B['sd_ap'], m3.sum()))
    x[m4] = XMIN * (1 - r_.random(m4.sum())) ** (-1 / B['alpha'])
    return np.clip(x, 0.1, 100)

def gen_scenario(name, wtype, r_h, rho_ap_scale, n_target, dow_k=2, extra_mu_shift=0.0, scale_override=None):
    """按逐时雨强时间线生成一个场景; scale_override 直接给定逐时总量倍率(暴雪用)"""
    r_day = float(r_h.sum())
    B3e = dict(B3)
    if wtype in ('rainstorm', 'rain'):
        d_d = extrap_day(F['day_li']['d_d'], r_day)      # 轻雨日级效应 → 场景日降水外推
        sh = extrap_day(F['day_li']['sh'], r_day)
        B3e['mu1'] = B3['mu1'] - d_d; B3e['w1'] = B3['w1'] + sh
    elif wtype == 'snow':                                 # 文献先验(方法.md §10)
        B3e['mu1'] = B3['mu1'] - 0.25; B3e['w1'] = B3['w1'] + 0.10
    # wtype=='dry': 距离分布不做任何扰动(纯基线对照)
    B3e['w_ap'] = B3['w_ap'] * rho_ap_scale
    s_ = B3e['w1'] + B3e['w2'] + B3e['w_ap'] + B3e['w_pa']
    B3e['w1'] /= s_; B3e['w2'] /= s_; B3e['w_ap'] /= s_; B3e['w_pa'] /= s_
    if scale_override is not None:
        scale_h = scale_override
    else:
        scale_h = np.clip(1 + th_vol * np.log1p(np.maximum(r_h, 0) / RC_vol), 0.3, 3.0)
    mu_shift = th_v * np.log1p(np.maximum(r_h, 0) / RC_v) - extra_mu_shift  # 带符号: th_v<0 减速; extra_mu_shift 为雪天额外减速幅度
    prof = lam0_how[dow_k] * scale_h
    n = n_target
    hrs = rng.choice(24, size=n, p=prof / prof.sum())
    dp = sample_B3(B3e, n, rng)
    vv = np.clip(np.exp(rng.normal(mu0_t[hrs] + mu_shift[hrs], sigma0, n)), 0.5, 99)  # 带符号效应: 雨雪均减速
    ss = rng.choice(264, size=n, p=lam_ext_s)
    # 终点: 环带重力(无容量约束——训练样本不替代闸; 极端场景另配配额)
    lam_det_samp = np.clip(rng.normal(B5['med'], B5['std'], n), 1.0, 3.0)
    tgt = dp / lam_det_samp
    band_w = np.maximum(tgt * (B5['std'] / B5['med']), 0.8)
    do_gen = np.empty(n, np.int32)
    d_matrix = dist_km[1:264, 1:264] / 1.60934
    logD = np.log(np.maximum(D_ext[1:264], 1e-9))
    CH = 50_000
    for st in range(0, n, CH):
        en = min(st + CH, n)
        dij = d_matrix[ss[st:en] - 1]
        hw = band_w[st:en][:, None]
        logw = logD[None, :] - ((dij - tgt[st:en][:, None]) ** 2) / (2 * hw ** 2)
        logw -= logw.max(1, keepdims=True)
        cump = np.cumsum(np.exp(logw), 1); cump /= cump[:, -1:]
        do_gen[st:en] = (np.argmax(cump > rng.random(en - st)[:, None], 1) + 1).astype(np.int32)
    dur = dp / vv
    df = pd.DataFrame(dict(scenario=name, weather_type=wtype,
                           precip_day=r_day, max_r_h=float(r_h.max()),
                           hour=hrs.astype(np.int16), pu=ss.astype(np.int16), do=do_gen,
                           d_path=dp.astype(np.float32), v_mph=vv.astype(np.float32),
                           dur_h=dur.astype(np.float32)))
    meta = dict(scenario=name, weather_type=wtype, precip_day_mm=round(r_day, 1),
                max_r_h_mm=round(float(r_h.max()), 1), n=n,
                scale_h=[round(float(x), 3) for x in scale_h],
                param_source=('剂量外推(实测θ/RC)' if wtype in ('rainstorm', 'rain') else '文献先验(方法§10)'))
    return df, meta

# ---- 场景设计 ----
h = np.zeros(24)
# S1 设计暴雨: 3h 峰值 30mm/h(纽约百年一遇小时雨强量级), 前后拖尾
s1 = np.zeros(24); s1[15] = 8; s1[16] = 20; s1[17] = 30; s1[18] = 22; s1[19] = 10; s1[20] = 4
# S2 持续性大雨: 10mm/h × 6h(长锋面雨)
s2 = np.zeros(24); s2[8:14] = 10; s2[14] = 5
# S3 一般暴雨: 峰 15mm/h × 2h
s3 = np.zeros(24); s3[13] = 5; s3[14] = 15; s3[15] = 12; s3[16] = 3
# S4/S5 暴雪: 早期抢购(白天+40%, 晚峰+60%) / 后期瘫痪(全天−70%)
s4_scale = np.ones(24); s4_scale[8:17] = 1.4; s4_scale[17:21] = 1.6
s5_scale = np.full(24, 0.30); s5_scale[7:10] = 0.20
frames, metas = [], []
for args in [
    dict(name='design_rainstorm_30mmh', wtype='rainstorm', r_h=s1, rho_ap_scale=0.7, n_target=200_000),
    dict(name='frontal_rain_10mmh6h', wtype='rainstorm', r_h=s2, rho_ap_scale=0.8, n_target=200_000),
    dict(name='severe_rain_15mmh', wtype='rainstorm', r_h=s3, rho_ap_scale=0.85, n_target=150_000),
    dict(name='snowstorm_early_surge', wtype='snow', r_h=np.zeros(24), rho_ap_scale=0.05, n_target=150_000, scale_override=s4_scale, extra_mu_shift=0.25),
    dict(name='snowstorm_collapse', wtype='snow', r_h=np.zeros(24), rho_ap_scale=0.05, n_target=150_000, scale_override=s5_scale, extra_mu_shift=0.15),
    dict(name='dry_control', wtype='dry', r_h=np.zeros(24), rho_ap_scale=1.0, n_target=150_000),
]:
    d_, m_ = gen_scenario(**args)
    frames.append(d_); metas.append(m_)
df = pd.concat(frames, ignore_index=True)
df.to_parquet(f'{OUT}/pseudo_extreme_train.parquet', index=False)
json.dump(metas, open(f'{OUT}/scenarios.json', 'w'), indent=1, ensure_ascii=False)
print(df.groupby('scenario').agg(n=('d_path', 'size'), d_med=('d_path', 'median'),
                                 v_med=('v_mph', 'median'), precip=('precip_day', 'first')).round(2))
print(f'\nsaved {OUT}/pseudo_extreme_train.parquet  {len(df):,} rows')
