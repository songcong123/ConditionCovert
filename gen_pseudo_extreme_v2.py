# -*- coding: utf-8 -*-
"""伪极端训练集 v2 —— 用 v1.2b 定稿参数 + 最新建模决策重生成
与 v1(data/pseudo_extreme, 旧参数四分量+P(s)×γ) 的区别:
  1) 起点 P(s|t): 24×263 逐时查表(validate_19 定稿, 留出日增益 +53.3%), 替代 P(s)×γ
  2) 空间低维乘子场(v1.2b 雨时级补测): 机场zone ×(1-0.27·dose), 密集核心 ×(1+0.07·dose), 外圈 ×(1-0.02·dose)
  3) 剂量曲线 v1.2: δ_vol=0.0523·ln(1+r/0.2), δ_v=-0.0254·ln(1+r/0.3), σ₀=0.438
  4) 距离三旋钮剂量缩放: μ₁ -3~6%·dose, w₁ +2.5pp·dose, 机场w ×(1-0.08·dose); 雪: 机场w→0.05, μ₁-6%, w₁+10pp
  5) 场景带 dow(§7.4 单日异质性): 闪洪=工作日, 长雨=周日
  6) 基线用两暖季 419 天(v1.2)
输出 data/pseudo_extreme_v2/"""
import numpy as np, json, os, csv
import pandas as pd

rng = np.random.default_rng(2027)
OUT = 'D:/AI/条件转化/data/pseudo_extreme_v2'
os.makedirs(OUT, exist_ok=True)

# ---- v1.2 定稿参数 ----
F = json.load(open('D:/AI/条件转化/.flow_scores.json'))
B3, B5, sigma0 = F['B3'], F['B5'], F['B4']['sigma0']
TH_VOL, RC_VOL = 0.0523, 0.2
TH_V, RC_V = -0.0254, 0.3

# ---- 两暖季基线 ----
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
G = np.load('D:/AI/条件转化/.flow_gen.npz', allow_pickle=True)
Z = np.load('D:/AI/条件转化/.flow_zones.npz')
dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
dow_a = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
hour_cnt = A['hour_cnt'].astype(np.int64)
pu_cnt = A['pu_cnt'].astype(np.int64); do_cnt = A['do_cnt'].astype(np.int64)
m_dry = kl == 'dry'
how_sum = np.zeros((7, 24))
for k in range(7):
    how_sum[k] = hour_cnt[m_dry & (dow_a == k)].sum(0)
lam0_how = how_sum / np.maximum((m_dry & (dow_a == None)[None, :].sum(0), 1), 1) if False else how_sum / np.maximum([(m_dry & (dow_a == k)).sum() for k in range(7)], 1)[:, None]
mu0_t = G['mu0_t']
do_dry = do_cnt[m_dry].sum(0).astype(float); do_share0 = do_dry / do_dry.sum()

# ---- P(s|hour) 查表(2024 干日, validate_19 缓存) ----
C = np.load('D:/AI/条件转化/.nyc_hh_zone.npz', allow_pickle=True)
c_dates = [str(x) for x in C['dates']]; cnt = C['cnt']      # (n,24,264)
c_kl = np.array([klass.get(d, 'na') for d in c_dates])
Ps_t = cnt[c_kl == 'dry'].sum(0).astype(float)              # (24,264)
Ps_t /= Ps_t.sum(1, keepdims=True)

# ---- zone 三分类(机场/密集核心/外圈) ----
borough = {}
for r in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if r and r[0].isdigit(): borough[int(r[0])] = r[1]
AIRPORTS = {1, 132, 138}
pu_dry = pu_cnt[m_dry].sum(0).astype(float)
top30 = set(np.argsort(-pu_dry)[1:31].tolist())
core = [z for z in top30 if borough.get(z) == 'Manhattan']
zclass = np.zeros(264, int)                                   # 0外圈 1核心 2机场
zclass[list(core)] = 1
zclass[list(AIRPORTS)] = 2
print(f'zone 分类: 核心 {len(core)} | 机场 {len(AIRPORTS)} | 外圈 {263-len(core)-len(AIRPORTS)}')

def m_zone(dose):
    m = np.ones(264)
    m[zclass == 2] *= (1 - 0.27 * dose)
    m[zclass == 1] *= (1 + 0.07 * dose)
    m[zclass == 0] *= (1 - 0.02 * dose)
    return m

def dose_of(r_h):
    return np.clip(np.log1p(np.maximum(r_h, 0) / 0.3) / np.log1p(12.64 / 0.3), 0, 1.2)

def sample_B3(B, n, r_):
    u = r_.random(n)
    c1 = B['w1']; c2 = c1 + B['w2']; c3 = c2 + B['w_ap']
    m1 = u < c1; m2 = (u >= c1) & (u < c2); m3 = (u >= c2) & (u < c3); m4 = ~(m1 | m2 | m3)
    x = np.empty(n)
    x[m1] = np.exp(r_.normal(B['mu1'], B['sd1'], m1.sum()))
    x[m2] = np.exp(r_.normal(B['mu2'], B['sd2'], m2.sum()))
    x[m3] = np.exp(r_.normal(B['mu_ap'], B['sd_ap'], m3.sum()))
    x[m4] = B['xmin'] * (1 - r_.random(m4.sum())) ** (-1 / B['alpha'])
    return np.clip(x, 0.1, 100)

dist_km = Z['dist_km'].astype(float)

def gen_scenario(name, wtype, r_h, n, dow_k, scale_override=None, snow=False, slow=0.0):
    dose = dose_of(r_h)
    if snow: dose = np.ones(24)
    scale_h = scale_override if scale_override is not None else np.clip(1 + TH_VOL * np.log1p(np.maximum(r_h, 0) / RC_VOL), 0.3, 3.0)
    mu_shift = TH_V * np.log1p(np.maximum(r_h, 0) / RC_V) - slow
    B3e = dict(B3)
    dm = dose.max()
    if snow:
        B3e['mu1'] = B3['mu1'] - 0.06; B3e['w1'] = B3['w1'] + 0.10; B3e['w_ap'] = B3['w_ap'] * 0.05
    else:
        B3e['mu1'] = B3['mu1'] - 0.06 * dm
        B3e['w1'] = B3['w1'] + 0.025 * dm
        B3e['w_ap'] = B3['w_ap'] * (1 - 0.08 * dm)
    s_ = B3e['w1'] + B3e['w2'] + B3e['w_ap'] + B3e['w_pa']
    B3e['w1'] /= s_; B3e['w2'] /= s_; B3e['w_ap'] /= s_; B3e['w_pa'] /= s_
    prof = lam0_how[dow_k] * scale_h; prof /= prof.sum()
    hrs = rng.choice(24, size=n, p=prof)
    dp = sample_B3(B3e, n, rng)
    vv = np.clip(np.exp(rng.normal(mu0_t[hrs] + mu_shift[hrs], sigma0, n)), 0.5, 99)
    # 起点 P(s|hour) × 乘子场(逐时剂量)
    mz = np.stack([m_zone(dose[h]) for h in range(24)])       # (24,264)
    ss = np.empty(n, np.int16)
    for h in range(24):
        mh = hrs == h
        if mh.sum() == 0: continue
        p = Ps_t[h] * mz[h]; p /= p.sum()
        ss[mh] = rng.choice(264, size=int(mh.sum()), p=p)
    # 终点: 环带重力, D_ext = 干日DO份额 × 场景剂量乘子
    Dz = do_share0 * m_zone(max(dose.max(), 0.3) if not snow else 1.0); Dz /= Dz.sum()
    lam_s = np.clip(rng.normal(B5['med'], B5['std'], n), 1.0, 3.0)
    tgt = dp / lam_s
    band = np.maximum(tgt * (B5['std'] / B5['med']), 0.8)
    d_matrix = dist_km[1:264, 1:264] / 1.60934
    logD = np.log(np.maximum(Dz[1:264], 1e-9))
    do_gen = np.empty(n, np.int32)
    CH = 50_000
    for st in range(0, n, CH):
        en = min(st + CH, n)
        dij = d_matrix[ss[st:en] - 1]
        hw = band[st:en][:, None]
        logw = logD[None, :] - ((dij - tgt[st:en][:, None]) ** 2) / (2 * hw ** 2)
        logw -= logw.max(1, keepdims=True)
        cump = np.cumsum(np.exp(logw), 1); cump /= cump[:, -1:]
        do_gen[st:en] = (np.argmax(cump > rng.random(en - st)[:, None], 1) + 1).astype(np.int32)
    df = pd.DataFrame(dict(scenario=name, weather_type=wtype, dow=dow_k,
                           precip_day=float(r_h.sum()), max_r_h=float(r_h.max()),
                           hour=hrs.astype(np.int16), pu=ss, do=do_gen,
                           d_path=dp.astype(np.float32), v_mph=vv.astype(np.float32),
                           dur_h=(dp / vv).astype(np.float32)))
    meta = dict(scenario=name, weather_type=wtype, dow=dow_k,
                precip_day_mm=round(float(r_h.sum()), 1), max_r_h_mm=round(float(r_h.max()), 1), n=n,
                param_source='v1.2b 定稿(剂量曲线RC 0.2/0.3, 三旋钮, P(s|t)查表, 乘子场)')
    return df, meta

# ---- 场景(§7.4: 闪洪=工作日, 长雨=周日) ----
s_flash = np.zeros(24); s_flash[15]=8; s_flash[16]=20; s_flash[17]=30; s_flash[18]=22; s_flash[19]=10; s_flash[20]=4
s_allday = np.zeros(24); s_allday[8:14]=10; s_allday[14]=5
s_sev = np.zeros(24); s_sev[13]=5; s_sev[14]=15; s_sev[15]=12; s_sev[16]=3
s4_scale = np.ones(24); s4_scale[8:17]=1.4; s4_scale[17:21]=1.6
s5_scale = np.full(24, 0.30); s5_scale[7:10]=0.20
jobs = [
    dict(name='flash_flood_weekday', wtype='rainstorm', r_h=s_flash, n=200_000, dow_k=2),
    dict(name='all_day_rain_sunday', wtype='rainstorm', r_h=s_allday, n=200_000, dow_k=0),
    dict(name='severe_rain_friday', wtype='rainstorm', r_h=s_sev, n=150_000, dow_k=4),
    dict(name='snowstorm_early_surge', wtype='snow', r_h=np.zeros(24), n=150_000, dow_k=2, scale_override=s4_scale, snow=True, slow=0.25),
    dict(name='snowstorm_collapse', wtype='snow', r_h=np.zeros(24), n=150_000, dow_k=2, scale_override=s5_scale, snow=True, slow=0.15),
    dict(name='dry_control_weekday', wtype='dry', r_h=np.zeros(24), n=100_000, dow_k=2),
    dict(name='dry_control_sunday', wtype='dry', r_h=np.zeros(24), n=100_000, dow_k=0),
]
frames, metas = [], []
for j in jobs:
    d_, m_ = gen_scenario(**j)
    frames.append(d_); metas.append(m_)
df = pd.concat(frames, ignore_index=True)
df.to_parquet(f'{OUT}/pseudo_extreme_train_v2.parquet', index=False)
json.dump(metas, open(f'{OUT}/scenarios_v2.json', 'w'), indent=1, ensure_ascii=False)
print(df.groupby('scenario').agg(n=('d_path', 'size'), d_med=('d_path', 'median'), v_med=('v_mph', 'median'),
                                 ap_share=('d_path', lambda s: float(((s >= 12) & (s <= 22)).mean() * 100))).round(2))
print(f'saved {OUT}/pseudo_extreme_train_v2.parquet  {len(df):,} rows')
