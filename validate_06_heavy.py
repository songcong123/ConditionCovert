# -*- coding: utf-8 -*-
"""流程验证 第二档盲测: 强雨日 (v1.1 方法.md §8 升级项)
与中雨档(pass2b)的区别:
  1) 盲测目标 = 强雨日 (日降水≥15mm, n≈24, 含 147.6mm 极值日)
  2) 小时级剂量-响应只用 0.3-1 / 1-4 mm/h 两个低箱标定(θ,RC 两点定解),
     ≥4mm/h 箱完全不参与标定 → 对强雨是纯外推(无泄漏)
  3) 日级 RC 按方法 v1.1 建议取 1.0(原 5.0 已证过冲 ~2×)
输出 .flow_scores_heavy.json"""
import numpy as np, json
from scipy import stats

rng = np.random.default_rng(7)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
M = np.load('D:/AI/条件转化/.flow_odm.npz')
S = np.load('D:/AI/条件转化/.flow_spd.npz')
Z = np.load('D:/AI/条件转化/.flow_zones.npz')

dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
dprecip = dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))
n_days = len(dates); day_idx = {d: i for i, d in enumerate(dates)}
kl = np.array([klass[d] for d in dates])
dow = np.array([np.datetime64(d, 'D').view('int64') % 7 for d in dates])
dprec = np.array([dprecip[d] for d in dates])
inten = np.zeros((n_days, 24))
for dd_, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if dd_ in day_idx: inten[day_idx[dd_], hh] = mm

hour_cnt = A['hour_cnt'].astype(np.int64)
pu_cnt = A['pu_cnt'].astype(np.int64); do_cnt = A['do_cnt'].astype(np.int64)
dist = A['dist'].astype(np.float64); dist_day = A['dist_day'].astype(np.int64)
spdV = S['v'].astype(np.float64); spdDay = S['day'].astype(np.int64); spdHr = S['hour'].astype(np.int64)
odm_idx = M['idx'].astype(np.int64); odm_mi = M['miles'].astype(np.float64)
dist_km = Z['dist_km'].astype(np.float64)

DRY, LIGHT, MED, HEAVY = 'dry', 'light', 'medium', 'heavy'
m_dry, m_li, m_me, m_he = kl == DRY, kl == LIGHT, kl == MED, kl == HEAVY
dry_i, li_i, me_i, he_i = (np.where(m)[0] for m in (m_dry, m_li, m_me, m_he))
print(f'干 {len(dry_i)} | 轻 {len(li_i)} | 中 {len(me_i)} | 强 {len(he_i)} 天 (强雨档目标)')

# ================= Step 1 晴天基线 (与 pass2b 相同, 只用干日) =================
how_sum = np.zeros((7, 24)); how_n = np.zeros(7)
for k in range(7):
    sel = m_dry & (dow == k)
    how_sum[k] = hour_cnt[sel].sum(0); how_n[k] = sel.sum()
lam0_how = how_sum / np.maximum(how_n, 1)[:, None]
lam0_t = how_sum.sum(0) / how_sum.sum()
pu_dry = pu_cnt[m_dry].sum(0).astype(float); pu_share0 = pu_dry / pu_dry.sum()
do_dry = do_cnt[m_dry].sum(0).astype(float); do_share0 = do_dry / do_dry.sum()

XMIN = 6.5
dd = dist[np.isin(dist_day, dry_i)]; dd = dd[(dd > 0.1) & (dd < 80)]
body = dd[dd < XMIN]; tail = dd[dd >= XMIN]
lx = np.log(body)
mu = np.array([np.log(1.2), np.log(4.5)]); sd = np.array([0.6, 0.4]); w = np.array([0.7, 0.3])
for _ in range(300):
    p = w[0] * stats.norm.pdf(lx, mu[0], sd[0]) + w[1] * stats.norm.pdf(lx, mu[1], sd[1])
    r1 = w[0] * stats.norm.pdf(lx, mu[0], sd[0]) / p; r2 = 1 - r1
    n1, n2 = r1.sum(), r2.sum()
    mu = np.array([(r1 * lx).sum() / n1, (r2 * lx).sum() / n2])
    sd = np.array([np.sqrt((r1 * (lx - mu[0]) ** 2).sum() / n1), np.sqrt((r2 * (lx - mu[1]) ** 2).sum() / n2)])
    w = np.array([n1, n2]) / len(lx)
lt = tail; ltx = np.log(lt)
mu_ap, sd_ap, w_ap, alpha = np.log(15.0), 0.22, 0.35, 2.7
for _ in range(300):
    f_ap = stats.norm.pdf(ltx, mu_ap, sd_ap) / lt
    f_pa = alpha / XMIN * (lt / XMIN) ** (-(alpha + 1))
    r = w_ap * f_ap / np.maximum(w_ap * f_ap + (1 - w_ap) * f_pa, 1e-300)
    nr = r.sum()
    mu_ap = np.clip((r * ltx).sum() / nr, np.log(10), np.log(20))
    sd_ap = np.clip(np.sqrt((r * (ltx - mu_ap) ** 2).sum() / nr), 0.1, 0.5)
    w_ap = float(np.clip(nr / len(lt), 0.05, 0.8))
    alpha = float(np.clip((1 - r).sum() / ((1 - r) * np.log(lt / XMIN)).sum(), 1.5, 5))
w_body = len(body) / len(dd)
B3 = dict(mu1=float(mu[0]), sd1=float(sd[0]), mu2=float(mu[1]), sd2=float(sd[1]),
          w1=float(w[0] * w_body), w2=float(w[1] * w_body),
          w_ap=float(w_ap * (1 - w_body)), w_pa=float((1 - w_ap) * (1 - w_body)),
          mu_ap=float(mu_ap), sd_ap=float(sd_ap), alpha=alpha, xmin=XMIN)

sel_s = np.isin(spdDay, dry_i)
lsv = np.log(spdV[sel_s]); hsel = spdHr[sel_s]
mu0_t = np.zeros(24)
for h in range(24):
    s_ = lsv[hsel == h]
    mu0_t[h] = s_.mean() if len(s_) >= 50 else lsv.mean()
sigma0 = float(np.std(lsv - mu0_t[hsel]))

sel = np.isin(odm_idx[:, 0], dry_i)
pu_o, do_o, mi_o = odm_idx[sel, 1], odm_idx[sel, 2], odm_mi[sel]
st_mi = dist_km[pu_o, do_o] / 1.60934
ok = (st_mi > 0.2) & (mi_o > 0.5)
lam_det = mi_o[ok] / st_mi[ok]; lam_det = lam_det[(lam_det > 0.8) & (lam_det < 4)]
B5_med, B5_std = float(np.median(lam_det)), float(np.std(lam_det))
print('Step1 基线完成 (与 pass2b 同参数)')

def expected_dry_how(i, h): return lam0_how[dow[i], h]

# ================= P1 剂量-响应: 只用低雨强两箱标定 (无泄漏) =================
def bin_stats(lo, hi):
    bm = np.zeros((n_days, 24), bool)
    for i in np.where(kl != DRY)[0]:
        bm[i] = (inten[i] >= lo) & (inten[i] < hi)
    rat, wts = [], []
    for i in np.where(kl != DRY)[0]:
        for h in range(24):
            if bm[i, h] and expected_dry_how(i, h) > 100:
                rat.append(hour_cnt[i, h] / expected_dry_how(i, h) - 1); wts.append(expected_dry_how(i, h))
    dvol = float(np.average(rat, weights=wts)) if rat else np.nan
    sm2 = bm[spdDay, spdHr]
    res = np.log(spdV[sm2]) - mu0_t[spdHr[sm2]]
    return dict(r=float(inten[bm].mean()), dvol=dvol,
                d_v=float(res.mean()) if sm2.sum() > 200 else np.nan,
                sig=float(res.std()) if sm2.sum() > 200 else np.nan, nh=int(bm.sum()))

binA = bin_stats(0.3, 1.0)   # 低箱1 (标定)
binB = bin_stats(1.0, 4.0)   # 低箱2 (标定)
binC = bin_stats(4.0, 1e9)   # 高箱 (仅用于事后核对, 不进标定)
print(f"标定箱: 0.3-1 r̄={binA['r']:.2f} δ_vol={binA['dvol']:+.4f} δ_v={binA['d_v']:+.4f} | "
      f"1-4 r̄={binB['r']:.2f} δ_vol={binB['dvol']:+.4f} δ_v={binB['d_v']:+.4f}")
print(f"事后核对箱: ≥4 r̄={binC['r']:.2f} δ_vol={binC['dvol']:+.4f} δ_v={binC['d_v']:+.4f} (σ={binC['sig']:.3f})")

def fit_theta_rc(rs, es):
    rs = np.array(rs); es = np.array(es)
    best = None
    for RC in np.linspace(0.2, 30, 2800):
        x = np.log1p(rs / RC)
        th = (x * es).sum() / (x * x).sum()
        sse = float(((th * x - es) ** 2).sum())
        if best is None or sse < best[0]: best = (sse, th, RC)
    sst = float(((es - es.mean()) ** 2).sum())
    return best[1], best[2], 1 - best[0] / sst

th_vol, RC_vol, _ = fit_theta_rc([binA['r'], binB['r']], [binA['dvol'], binB['dvol']])
th_v, RC_v, _ = fit_theta_rc([binA['r'], binB['r']], [binA['d_v'], binB['d_v']])
pred_C_vol = th_vol * np.log1p(binC['r'] / RC_vol)
pred_C_v = th_v * np.log1p(binC['r'] / RC_v)
print(f'两点标定: δ_vol={th_vol:.4f}·log(1+r/{RC_vol:.2f}) | δ_v={th_v:.4f}·log(1+r/{RC_v:.2f})')
print(f'外推预测 ≥4mm/h: δ_vol={pred_C_vol:+.4f} (实测{binC["dvol"]:+.4f}) | δ_v={pred_C_v:+.4f} (实测{binC["d_v"]:+.4f})')

# --- 日级: 轻雨标定 → 外推强雨, R_C=1.0 (v1.1 建议) ---
R_C = 1.0
def measure_day(idx):
    med_dry = np.median(np.log(dd))
    d_d = float(np.mean([np.median(np.log(dist[dist_day == i])) - med_dry for i in idx]))
    sh = float(np.mean([(dist[dist_day == i] < 2).mean() - (dd < 2).mean() for i in idx]))
    ap_dry = ((dd >= 12) & (dd <= 22)).mean()
    rho = float(np.mean([((dist[dist_day == i] >= 12) & (dist[dist_day == i] <= 22)).mean() / ap_dry for i in idx]))
    return d_d, sh, rho
d_d_li, sh_li, rho_li = measure_day(li_i)
r_li, r_he = float(dprec[li_i].mean()), float(dprec[he_i].mean())
def extrap_day(eff_li, r): return float(eff_li * np.log(1 + r / R_C) / np.log(1 + r_li / R_C))
TH = dict(d_d=extrap_day(d_d_li, r_he), sh=extrap_day(sh_li, r_he),
          rho_ap=1 + extrap_day(rho_li - 1, r_he))
d_d_he, sh_he, rho_he = measure_day(he_i)
print(f'日级(R_C={R_C}): δ_d pred{TH["d_d"]:+.4f} vs 实测{d_d_he:+.4f} | 短途 pred{TH["sh"]*100:+.2f}pp vs 实测{sh_he*100:+.2f}pp | ρ_ap pred{TH["rho_ap"]:.3f} vs 实测{rho_he:.3f}')

# γ 热力图 (伪计数平滑 + 零流量地板, 与 pass2b 相同)
pu_r = pu_cnt[li_i].sum(0).astype(float); do_r = do_cnt[li_i].sum(0).astype(float)
pu_dry_f = np.maximum(pu_dry, 20.0); do_dry_f = np.maximum(do_dry, 20.0)
gamma_pu = np.log((pu_r + 0.05 * pu_dry_f) / (1.05 * pu_dry_f))
gamma_do = np.log((do_r + 0.05 * do_dry_f) / (1.05 * do_dry_f))

# ================= Step 2 扰动 → 强雨日逐小时外推 =================
lam_ext_s = pu_share0 * np.exp(gamma_pu); lam_ext_s /= lam_ext_s.sum()
H0 = float(-(pu_share0[pu_share0 > 0] * np.log(pu_share0[pu_share0 > 0])).sum())
H1 = float(-(lam_ext_s[lam_ext_s > 0] * np.log(lam_ext_s[lam_ext_s > 0])).sum())
D_ext = do_share0 * np.exp(gamma_do); D_ext /= D_ext.sum()
B3e = dict(B3)
B3e['mu1'] = B3['mu1'] - TH['d_d']; B3e['w1'] = B3['w1'] + TH['sh']; B3e['w_ap'] = B3['w_ap'] * TH['rho_ap']
s_ = B3e['w1'] + B3e['w2'] + B3e['w_ap'] + B3e['w_pa']
B3e['w1'] /= s_; B3e['w2'] /= s_; B3e['w_ap'] /= s_; B3e['w_pa'] /= s_
eta = 0.0; sig_ext = sigma0 * (1 + eta)

scale_he = np.zeros((len(he_i), 24)); mu_shift_he = np.zeros((len(he_i), 24))
for k, i in enumerate(he_i):
    r_h = inten[i]
    scale_he[k] = np.clip(1 + th_vol * np.log1p(r_h / RC_vol), 0.3, 3.0)
    mu_shift_he[k] = th_v * np.log1p(r_h / RC_v)
exp_days = np.array([(lam0_how[dow[i]] * scale_he[k]).sum() for k, i in enumerate(he_i)])
real_tot = hour_cnt[he_i].sum(1).astype(float)
print(f'\n强雨日总量: 预测 {exp_days.sum():,.0f} vs 实测 {real_tot.sum():,.0f} 趟')

# ================= Step 3 采样 =================
NGEN = 1_200_000
counts_day = rng.multinomial(NGEN, exp_days / exp_days.sum())
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
d_l, t_l, s_l, v_l = [], [], [], []
for k, i in enumerate(he_i):
    n = int(counts_day[k])
    if n == 0: continue
    prof = lam0_how[dow[i]] * scale_he[k]; prof = prof / prof.sum()
    hrs = rng.choice(24, size=n, p=prof)
    dp = sample_B3(B3e, n, rng)
    vv = np.clip(np.exp(rng.normal(mu0_t[hrs] + mu_shift_he[k][hrs], sig_ext, n)), 0.5, 99)  # 带符号效应(th_v<0 → 减速)
    ss = rng.choice(264, size=n, p=lam_ext_s)
    d_l.append(dp); t_l.append(hrs); s_l.append(ss); v_l.append(vv)
d_gen = np.concatenate(d_l); t_gen = np.concatenate(t_l)
s_gen = np.concatenate(s_l); v_gen = np.concatenate(v_l)
dur_gen = d_gen / v_gen
print(f'生成 {len(d_gen):,} 条')

# ================= Step 4 终点 (环带重力 + 容量约束) =================
lam_det_samp = np.clip(rng.normal(B5_med, B5_std, len(d_gen)), 1.0, 3.0)
tgt = d_gen / lam_det_samp
band_w = np.maximum(tgt * (B5_std / B5_med), 0.8)
d_matrix = dist_km[1:264, 1:264] / 1.60934
logD = np.log(np.maximum(D_ext[1:264], 1e-9))
quota_share = do_dry[1:] / do_dry.sum()
quota = np.maximum(quota_share * len(d_gen) * 1.3, 30.0)
quota_zone = np.concatenate([[0.0], quota])
used = np.zeros(264); do_gen = np.empty(len(d_gen), np.int32); n_rej = 0
CH = 50_000
for st in range(0, len(d_gen), CH):
    en = min(st + CH, len(d_gen))
    dij = d_matrix[s_gen[st:en] - 1]
    hw = band_w[st:en][:, None]
    logw = logD[None, :] - ((dij - tgt[st:en][:, None]) ** 2) / (2 * hw ** 2)
    logw -= logw.max(1, keepdims=True)
    cump = np.cumsum(np.exp(logw), 1); cump /= cump[:, -1:]
    cur = (np.argmax(cump > rng.random(en - st)[:, None], 1) + 1).astype(np.int32)
    for _ in range(5):
        over = used[cur] >= quota_zone[cur]
        if not over.any(): break
        idxs = np.where(over)[0]; n_rej += len(idxs)
        rem = np.maximum(quota - used[1:], 0)
        p_rem = rem * D_ext[1:264]; p_rem /= p_rem.sum()
        cur[idxs] = rng.choice(np.arange(1, 264), size=len(idxs), p=p_rem)
    used += np.bincount(cur, minlength=264)
    do_gen[st:en] = cur
print(f'终点完成 (容量重采样 {n_rej/len(d_gen)*100:.1f}%)')

# ================= Step 5 三道闸 + 强雨档打分 =================
tri = np.abs(v_gen * dur_gen - d_gen) / d_gen
g_tri = float(np.percentile(tri, 99.9))
inflow_gen = np.bincount(do_gen, minlength=264)[1:].astype(float)
dry_do_share = do_dry[1:] / do_dry.sum()
okz = dry_do_share > 1e-6
cap_ratio = (inflow_gen[okz] / len(d_gen)) / dry_do_share[okz]
g_cap = float(np.percentile(cap_ratio, 99))
# 参照: 真实强雨日的同口径比值(检验闸阈值对强雨是否过紧)
do_he = do_cnt[he_i].sum(0).astype(float)
cap_ratio_real = (do_he[1:][okz] / do_he.sum()) / dry_do_share[okz]
g_cap_real = float(np.percentile(cap_ratio_real, 99))
sig_gen = float(np.std(np.log(v_gen)))
print(f'闸1 三角 p99.9 {g_tri:.1e} | 闸2 容量 p99={g_cap:.2f} (真实强雨参照 {g_cap_real:.2f}) | 闸3 σ比 {sig_gen/sig_ext:.3f} 熵 {H0:.3f}→{H1:.3f}')

dist_he = dist[np.isin(dist_day, he_i)]
spd_he = spdV[np.isin(spdDay, he_i)]
hour_he = hour_cnt[he_i].sum(0).astype(float); hour_he /= hour_he.sum()
pu_he = pu_cnt[he_i].sum(0).astype(float); pu_he /= pu_he.sum()
do_he_s = do_he / do_he.sum()
def js(p, q):
    p = np.maximum(p, 1e-12); p /= p.sum(); q = np.maximum(q, 1e-12); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))
def ks_sym(gen, real):
    n = min(len(gen), len(real))
    g_ = rng.choice(gen, n, replace=len(gen) < n)
    r_ = rng.choice(real, n, replace=len(real) < n)
    return float(stats.ks_2samp(g_, r_).statistic)
hour_gen = np.bincount(t_gen, minlength=24).astype(float); hour_gen /= hour_gen.sum()
pu_gen = np.bincount(s_gen, minlength=264).astype(float); pu_gen /= pu_gen.sum()
do_gs = np.bincount(do_gen, minlength=264).astype(float); do_gs /= do_gs.sum()
scores = dict(lam_t=js(hour_gen, hour_he), d=ks_sym(d_gen, dist_he),
              v=ks_sym(v_gen, spd_he), pu=js(pu_gen, pu_he), do=js(do_gs, do_he_s))
# 纯基线消融
rng2 = np.random.default_rng(11)
d_ab = sample_B3(B3, NGEN, rng2)
hrs_ab = rng2.choice(24, size=NGEN, p=lam0_t)
v_ab = np.clip(np.exp(rng2.normal(mu0_t[hrs_ab], sigma0, NGEN)), 0.5, 99)
hour_ab = np.bincount(hrs_ab, minlength=24).astype(float); hour_ab /= hour_ab.sum()
scores_ab = dict(lam_t=js(hour_ab, hour_he), d=ks_sym(d_ab, dist_he),
                 v=ks_sym(v_ab, spd_he), pu=js(pu_share0, pu_he), do=js(do_share0, do_he_s))
print('\n强雨档得分(越低越好):    扰动管线   纯基线   改善')
lbl = dict(lam_t='JS λ(t)', d='KS P(d)', v='KS P(v)', pu='JS PU', do='JS DO')
for k in lbl:
    imp = (1 - scores[k] / scores_ab[k]) * 100
    print(f'  {lbl[k]:9s} {scores[k]:.4f}   {scores_ab[k]:.4f}  {imp:+.0f}%')
vol_mape = float(np.mean(np.abs(exp_days / real_tot - 1)))
vol_mape_base = float(np.mean(np.abs(lam0_how[dow[he_i]].sum(1) / real_tot - 1)))
print(f'日总量 MAPE: 扰动 {vol_mape*100:.1f}% vs 基线 {vol_mape_base*100:.1f}%')
# DO 条件距离 MAPE
sel_od = np.isin(A['od_day'].astype(int), he_i)
r_pu = (A['od_pair'][sel_od] // 264).astype(int); r_do = (A['od_pair'][sel_od] % 264).astype(int)
r_cnt = A['od_cnt'][sel_od].astype(float)
r_straight = dist_km[r_pu, r_do] / 1.60934
top_pu = np.argsort(-pu_cnt[he_i].sum(0))[1:31]
g_straight = dist_km[s_gen, do_gen] / 1.60934
errs = []
for z in top_pu:
    mr_ = r_straight[r_pu == z]; cr_ = r_cnt[r_pu == z]
    mg_ = g_straight[s_gen == z]
    if len(mr_) < 50 or len(mg_) < 50: continue
    errs.append(abs(np.average(mr_, weights=cr_) - mg_.mean()) / np.average(mr_, weights=cr_))
cond_mape = float(np.mean(errs))
print(f'DO条件距离 MAPE(top30 PU): {cond_mape*100:.1f}%')

json.dump(dict(tier='heavy', n_heavy=int(len(he_i)), NGEN=NGEN,
               scores=scores, scores_ab=scores_ab, cond_dist_mape=cond_mape,
               vol=dict(mape=vol_mape, mape_base=vol_mape_base),
               hourly_cal=dict(binA=binA, binB=binB, binC_holdout=binC,
                               theta_vol=th_vol, RC_vol=RC_vol, theta_v=th_v, RC_v=RC_v,
                               pred_heavy=dict(dvol=pred_C_vol, d_v=pred_C_v)),
               day=dict(R_C=R_C, TH_pred=TH, measured=dict(d_d=d_d_he, sh=sh_he, rho=rho_he),
                        cal=dict(d_d=d_d_li, sh=sh_li, rho=rho_li, r_li=r_li, r_he=r_he)),
               gates=dict(tri=g_tri, cap_p99=g_cap, cap_p99_real=g_cap_real,
                          sig_ratio=sig_gen / sig_ext, H0=H0, H1=H1),
               B3=B3, B3e=B3e, B5=dict(med=B5_med, std=B5_std), B4=dict(sigma0=sigma0)),
          open('D:/AI/条件转化/.flow_scores_heavy.json', 'w'), indent=1, ensure_ascii=False)
print('saved .flow_scores_heavy.json')
