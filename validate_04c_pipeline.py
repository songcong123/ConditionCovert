# -*- coding: utf-8 -*-
"""流程验证 第二遍(修正回路, 对应流程图 D1"不通过→修正扰动"):
pass1 暴露的问题与本遍修正:
  1) γ 热力图公式错误(把平滑雨日份额当份额比) → 改为逐zone伪计数比
  2) ρ_airport 对比值直接外推爆炸(×2.8) → 对 (ρ-1) 外推
  3) 剂量用日降水 → 按文档§7改用小时雨强(0.3-1标定, >1外推), λ(t)与μ(t)逐小时用当天雨强时间线
  4) η=0.1 拍脑袋 → 从小雨时段残差标定(实测≈0)
  5) 闸2 零流量zone产生nan → 过滤干日流入<1趟/日的zone
其余与 pass1 相同, 输出 .flow_gen.npz/.flow_scores.json(pass2)"""
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
n_days = len(dates)
day_idx = {d: i for i, d in enumerate(dates)}
kl = np.array([klass[d] for d in dates])
dow = np.array([np.datetime64(d, 'D').view('int64') % 7 for d in dates])
dprec = np.array([dprecip[d] for d in dates])

# 逐(日,时)雨强矩阵 (mm/h, 本地时间, FM-15去重后)
inten = np.zeros((n_days, 24))
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if dd in day_idx: inten[day_idx[dd], hh] = mm

hour_cnt = A['hour_cnt'].astype(np.int64)
pu_cnt = A['pu_cnt'].astype(np.int64); do_cnt = A['do_cnt'].astype(np.int64)
dist = A['dist'].astype(np.float64); dist_day = A['dist_day'].astype(np.int64)
spdV = S['v'].astype(np.float64); spdDay = S['day'].astype(np.int64); spdHr = S['hour'].astype(np.int64)
odm_idx = M['idx'].astype(np.int64); odm_mi = M['miles'].astype(np.float64)
dist_km = Z['dist_km'].astype(np.float64)

DRY, LIGHT, MED, HEAVY = 'dry', 'light', 'medium', 'heavy'
m_dry, m_li, m_me, m_he = kl == DRY, kl == LIGHT, kl == MED, kl == HEAVY
dry_i, li_i, me_i, he_i = (np.where(m)[0] for m in (m_dry, m_li, m_me, m_he))
print(f'干 {len(dry_i)} | 轻 {len(li_i)} | 中 {len(me_i)} | 强 {len(he_i)} 天')

# ================= Step 1 晴天基线 (只用干日) =================
print('\n===== Step 1 晴天基线 B1-B5 =====')
how_sum = np.zeros((7, 24)); how_n = np.zeros(7)
for k in range(7):
    sel = m_dry & (dow == k)
    how_sum[k] = hour_cnt[sel].sum(0); how_n[k] = sel.sum()
lam0_how = how_sum / np.maximum(how_n, 1)[:, None]
lam0_t = how_sum.sum(0) / how_sum.sum()
print(f'B1 干日平均 {lam0_how.sum(1).mean():,.0f} 趟/日')

pu_dry = pu_cnt[m_dry].sum(0).astype(float); pu_share0 = pu_dry / pu_dry.sum()
do_dry = do_cnt[m_dry].sum(0).astype(float); do_share0 = do_dry / do_dry.sum()

XMIN = 6.5
dd = dist[np.isin(dist_day, dry_i)]
dd = dd[(dd > 0.1) & (dd < 80)]
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
print(f"B3 距离: 短LN({np.exp(mu[0]):.2f}mi)w={B3['w1']:.3f} | 中LN({np.exp(mu[1]):.1f}mi)w={B3['w2']:.3f} | "
      f"机场LN({np.exp(mu_ap):.1f}mi)w={B3['w_ap']:.3f} | Pareto(α={alpha:.2f})w={B3['w_pa']:.3f}")

sel_s = np.isin(spdDay, dry_i)
lsv = np.log(spdV[sel_s]); hsel = spdHr[sel_s]
mu0_t = np.zeros(24)
for h in range(24):
    s_ = lsv[hsel == h]
    mu0_t[h] = s_.mean() if len(s_) >= 50 else lsv.mean()
sigma0 = float(np.std(lsv - mu0_t[hsel]))
print(f'B4 速度: μ₀ 04时 {np.exp(mu0_t[4]):.1f}mph / 16时 {np.exp(mu0_t[16]):.1f}mph | σ₀={sigma0:.3f}')

sel = np.isin(odm_idx[:, 0], dry_i)
pu_o, do_o, mi_o = odm_idx[sel, 1], odm_idx[sel, 2], odm_mi[sel]
st_mi = dist_km[pu_o, do_o] / 1.60934
ok = (st_mi > 0.2) & (mi_o > 0.5)
lam_det = mi_o[ok] / st_mi[ok]
lam_det = lam_det[(lam_det > 0.8) & (lam_det < 4)]
B5_med, B5_std = float(np.median(lam_det)), float(np.std(lam_det))
print(f'B5 绕路: 中位 {B5_med:.2f} | IQR [{np.percentile(lam_det,25):.2f},{np.percentile(lam_det,75):.2f}] n={len(lam_det):,}')

# ================= P1 剂量-响应 (修正版) =================
print('\n===== P1 剂量-响应标定 (小时雨强, 标定段 0.3-1 mm/h) =====')
RC_H = 1.0      # 小时雨强阈值 mm/h (文档分层口径)
R_C = 5.0       # 日降水阈值 mm (日级效应用)

def expected_dry_how(i, h): return lam0_how[dow[i], h]

# --- 小时级: δ_vol(r_h), δ_v(r_h), η --- (只用轻雨日 0.3-1 mm/h 时段标定)
cal_mask = np.zeros((n_days, 24), bool)
for i in li_i:
    cal_mask[i] = (inten[i] >= 0.3) & (inten[i] < 1.0)
# δ_vol: 加权平均 count/expected
rat, wts = [], []
for i in np.where(m_li)[0]:
    for h in range(24):
        if cal_mask[i, h] and expected_dry_how(i, h) > 100:
            rat.append(hour_cnt[i, h] / expected_dry_how(i, h) - 1)
            wts.append(expected_dry_how(i, h))
dvol_h_cal = float(np.average(rat, weights=wts))
r_cal_h = float(inten[cal_mask].mean())
th_vol = dvol_h_cal / np.log(1 + r_cal_h / RC_H)
th_v = None; eta = 0.0
sm = cal_mask[spdDay, spdHr]
if sm.sum() > 200:
    res_cal = np.log(spdV[sm]) - mu0_t[spdHr[sm]]
    d_v_cal = float(res_cal.mean())
    th_v = d_v_cal / np.log(1 + r_cal_h / RC_H)
    eta = float(np.clip(np.std(res_cal) / sigma0 - 1, 0, 0.3))
print(f'标定段(轻雨日 r∈[0.3,1)mm/h, {cal_mask.sum()}小时): δ_vol={dvol_h_cal:+.4f} δ_v={d_v_cal:+.4f} η={eta:.3f}')

# --- 日级: δ_d, 短途, ρ_ap(-1), γ (轻雨日) ---
def measure_day(idx):
    med_dry = np.median(np.log(dd))
    d_d = float(np.mean([np.median(np.log(dist[dist_day == i])) - med_dry for i in idx]))
    sh = float(np.mean([(dist[dist_day == i] < 2).mean() - (dd < 2).mean() for i in idx]))
    ap_dry = ((dd >= 12) & (dd <= 22)).mean()
    rho = float(np.mean([((dist[dist_day == i] >= 12) & (dist[dist_day == i] <= 22)).mean() / ap_dry for i in idx]))
    return d_d, sh, rho

d_d_li, sh_li, rho_li = measure_day(li_i)
r_li = float(dprec[li_i].mean())
r_med, r_he = float(dprec[me_i].mean()), float(dprec[he_i].mean())
def extrap_day(eff_li, r): return float(eff_li * np.log(1 + r / R_C) / np.log(1 + r_li / R_C))
TH = dict(d_d=extrap_day(d_d_li, r_med), sh=extrap_day(sh_li, r_med),
          rho_ap=1 + extrap_day(rho_li - 1, r_med))
print(f'轻雨日级: δ_d={d_d_li:+.4f} 短途{sh_li*100:+.2f}pp ρ_ap={rho_li:.3f}')
print(f'外推中雨(r̄={r_med:.1f}): δ_d={TH["d_d"]:+.4f} 短途{TH["sh"]*100:+.2f}pp ρ_ap={TH["rho_ap"]:.3f}')

# γ 热力图(修正公式: 逐zone伪计数平滑比; 零流量zone地板20趟防除零)
pu_r = pu_cnt[li_i].sum(0).astype(float); do_r = do_cnt[li_i].sum(0).astype(float)
pu_dry_f = np.maximum(pu_dry, 20.0); do_dry_f = np.maximum(do_dry, 20.0)
gamma_pu = np.log((pu_r + 0.05 * pu_dry_f) / (1.05 * pu_dry_f))
gamma_do = np.log((do_r + 0.05 * do_dry_f) / (1.05 * do_dry_f))

# 实测剂量-响应曲线: 按小时雨强分箱(全雨日, 标定+检验)
print('\n实测小时级剂量-响应:')
bin_defs = [(0.3, 1.0, '0.3-1'), (1.0, 4.0, '1-4'), (4.0, 1e9, '≥4')]
measured = {}
rain_days_all = np.where(kl != DRY)[0]
for lo, hi, nm in bin_defs:
    bm = np.zeros((n_days, 24), bool)
    for i in rain_days_all:
        bm[i] = (inten[i] >= lo) & (inten[i] < hi)
    rat, wts = [], []
    for i in rain_days_all:
        for h in range(24):
            if bm[i, h] and expected_dry_how(i, h) > 100:
                rat.append(hour_cnt[i, h] / expected_dry_how(i, h) - 1); wts.append(expected_dry_how(i, h))
    dvol_m = float(np.average(rat, weights=wts)) if rat else np.nan
    sm2 = bm[spdDay, spdHr]
    dv_m = float((np.log(spdV[sm2]) - mu0_t[spdHr[sm2]]).mean()) if sm2.sum() > 200 else np.nan
    sg_m = float(np.std(np.log(spdV[sm2]) - mu0_t[spdHr[sm2]])) if sm2.sum() > 200 else np.nan
    r̄ = float(inten[bm].mean()) if bm.sum() else np.nan
    measured[nm] = dict(r=r̄, dvol=dvol_m, d_v=dv_m, sig=sg_m, nh=int(bm.sum()))
    print(f'  {nm:5s}mm/h: n={bm.sum():4d}时 r̄={r̄:.2f} δ_vol={dvol_m:+.4f} δ_v={dv_m:+.4f} σ={sg_m:.3f}')

# 拟合 effect(r)=θ·log(1+r/RC) 于三箱(文档§7形式, 同时拟θ与RC)
def fit_theta_rc(rs, es):
    rs = np.array(rs); es = np.array(es)
    best = None
    for RC in np.linspace(0.2, 30, 1500):
        x = np.log1p(rs / RC)
        th = (x * es).sum() / (x * x).sum()
        sse = float(((th * x - es) ** 2).sum())
        if best is None or sse < best[0]: best = (sse, th, RC)
    sst = float(((es - es.mean()) ** 2).sum())
    return best[1], best[2], 1 - best[0] / sst
rs = [measured[nm]['r'] for nm, _, _ in [(b[2], 0, 0) for b in bin_defs]]
th_vol_f, RC_vol_f, r2_vol = fit_theta_rc(rs, [measured[b[2]]['dvol'] for b in bin_defs])
th_v_f, RC_v_f, r2_v = fit_theta_rc(rs, [measured[b[2]]['d_v'] for b in bin_defs])
print(f'拟合: δ_vol={th_vol_f:.4f}·log(1+r/{RC_vol_f:.1f}) R²={r2_vol:.3f} | δ_v={th_v_f:.4f}·log(1+r/{RC_v_f:.1f}) R²={r2_v:.3f}')
# 用拟合版替换单点标定(生成器用拟合参数)
th_vol, RC_H, th_v = th_vol_f, RC_vol_f, th_v_f

# ================= Step 2 扰动(小时雨强版) =================
print('\n===== Step 2 扰动(λ与μ逐小时跟随当天雨强时间线) =====')
lam_ext_s = pu_share0 * np.exp(gamma_pu); lam_ext_s /= lam_ext_s.sum()
H0 = float(-(pu_share0[pu_share0 > 0] * np.log(pu_share0[pu_share0 > 0])).sum())
H1 = float(-(lam_ext_s[lam_ext_s > 0] * np.log(lam_ext_s[lam_ext_s > 0])).sum())
print(f'热力图熵: {H0:.3f} → {H1:.3f} ({"✓" if H1 < H0 else "✗"})')
D_ext = do_share0 * np.exp(gamma_do); D_ext /= D_ext.sum()
B3e = dict(B3)
B3e['mu1'] = B3['mu1'] - TH['d_d']
B3e['w1'] = B3['w1'] + TH['sh']
B3e['w_ap'] = B3['w_ap'] * TH['rho_ap']
s_ = B3e['w1'] + B3e['w2'] + B3e['w_ap'] + B3e['w_pa']
B3e['w1'] /= s_; B3e['w2'] /= s_; B3e['w_ap'] /= s_; B3e['w_pa'] /= s_
sig_ext = sigma0 * (1 + eta)

# 每中雨日逐小时外推: scale_h = 1 + θ_vol·log(1+r_h/RC), μ偏移同理
scale_me = np.zeros((len(me_i), 24)); mu_shift_me = np.zeros((len(me_i), 24))
for k, i in enumerate(me_i):
    r_h = inten[i]
    scale_me[k] = np.clip(1 + th_vol * np.log1p(r_h / RC_H), 0.3, 3.0)
    mu_shift_me[k] = (th_v * np.log1p(r_h / RC_H)) if th_v is not None else 0.0
exp_days = np.array([ (lam0_how[dow[i]] * scale_me[k]).sum() for k, i in enumerate(me_i)])

# ================= Step 3 采样 =================
print('===== Step 3 逐条采样 =====')
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
for k, i in enumerate(me_i):
    n = int(counts_day[k])
    if n == 0: continue
    prof = lam0_how[dow[i]] * scale_me[k]; prof = prof / prof.sum()
    hrs = rng.choice(24, size=n, p=prof)
    dp = sample_B3(B3e, n, rng)
    vv = np.clip(np.exp(rng.normal(mu0_t[hrs] + mu_shift_me[k][hrs], sig_ext, n)), 0.5, 99)  # mu_shift 为带符号效应(th_v<0 → 减速)
    ss = rng.choice(264, size=n, p=lam_ext_s)
    d_l.append(dp); t_l.append(hrs); s_l.append(ss); v_l.append(vv)
d_gen = np.concatenate(d_l); t_gen = np.concatenate(t_l)
s_gen = np.concatenate(s_l); v_gen = np.concatenate(v_l)
dur_gen = d_gen / v_gen
print(f'生成 {len(d_gen):,} 条 (真实中雨期望总量 {exp_days.sum():,.0f})')

# ================= Step 4 环带重力终点 + 容量约束(pass2b: 闸2红虚线回路) =================
print('===== Step 4 终点(环带重力 + 容量约束重采样) =====')
lam_det_samp = np.clip(rng.normal(B5_med, B5_std, len(d_gen)), 1.0, 3.0)
tgt = d_gen / lam_det_samp
band_w = np.maximum(tgt * (B5_std / B5_med), 0.8)
d_matrix = dist_km[1:264, 1:264] / 1.60934
logD = np.log(np.maximum(D_ext[1:264], 1e-9))
# 容量配额: 干日 DO 份额 × 总量 × 容许系数(闸2口径 1.3)
quota_share = do_dry[1:] / do_dry.sum()
quota = np.maximum(quota_share * len(d_gen) * 1.3, 30.0)   # 每zone配额条数(小zone至少30)
quota_zone = np.concatenate([[0.0], quota])                # 索引0占位
used = np.zeros(264)
do_gen = np.empty(len(d_gen), np.int32)
n_rej = 0
CH = 50_000
for st in range(0, len(d_gen), CH):
    en = min(st + CH, len(d_gen))
    dij = d_matrix[s_gen[st:en] - 1]
    hw = band_w[st:en][:, None]
    logw = logD[None, :] - ((dij - tgt[st:en][:, None]) ** 2) / (2 * hw ** 2)
    logw -= logw.max(1, keepdims=True)
    cump = np.cumsum(np.exp(logw), 1); cump /= cump[:, -1:]
    cur = (np.argmax(cump > rng.random(en - st)[:, None], 1) + 1).astype(np.int32)
    # 容量拒绝-重采样(最多5轮, 满额者改按剩余配额加权 D_ext 直采)
    for _ in range(5):
        over = used[cur] >= quota_zone[cur]
        if not over.any(): break
        idxs = np.where(over)[0]; n_rej += len(idxs)
        rem = np.maximum(quota - used[1:], 0)
        p_rem = rem * D_ext[1:264]; p_rem /= p_rem.sum()
        cur[idxs] = rng.choice(np.arange(1, 264), size=len(idxs), p=p_rem)
    used += np.bincount(cur, minlength=264)
    do_gen[st:en] = cur
print(f'终点完成 (容量拒绝重采样 {n_rej:,} 条 = {n_rej/len(d_gen)*100:.1f}%)')

# ================= Step 5 三道闸 =================
print('\n===== Step 5 三道闸 =====')
tri = np.abs(v_gen * dur_gen - d_gen) / d_gen
g_tri = float(np.percentile(tri, 99.9))
inflow_gen = np.bincount(do_gen, minlength=264)[1:].astype(float)
# 正确口径: 生成份额 vs 干日份额 的比值(此前误把逐日均量与全期总量混除)
dry_do_share = do_dry[1:] / do_dry.sum()
okz = dry_do_share > 1e-6
cap_ratio = (inflow_gen[okz] / len(d_gen)) / dry_do_share[okz]
g_cap = float(np.percentile(cap_ratio, 99))
worst = np.argsort(-cap_ratio)[:5] + 1
sig_gen = float(np.std(np.log(v_gen)))
print(f'闸1 三角一致: p99.9 {g_tri:.1e} {"✓" if g_tri < 1e-3 else "✗"}')
print(f'闸2 容量: 生成份额/干日份额 p99={g_cap:.2f} 最差zone {worst.tolist()} {"✓" if g_cap < 1.3 else "⚠"}')
print(f'闸3 sanity: σ(log v) {sig_gen:.3f} vs σ_ext {sig_ext:.3f} (比{sig_gen/sig_ext:.3f}) | 熵{H1:.3f} vs {H0:.3f} {"✓" if H1 < H0 else "✗"}')

# ================= V1 打分 =================
print('\n===== V1 中雨替身盲测 =====')
dist_me = dist[np.isin(dist_day, me_i)]
spd_me = spdV[np.isin(spdDay, me_i)]
hour_me = hour_cnt[me_i].sum(0).astype(float); hour_me /= hour_me.sum()
pu_me = pu_cnt[me_i].sum(0).astype(float); pu_me /= pu_me.sum()
do_me = do_cnt[me_i].sum(0).astype(float); do_me /= do_me.sum()

def js(p, q):
    p = np.maximum(p, 1e-12); p /= p.sum()
    q = np.maximum(q, 1e-12); q /= q.sum()
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
scores = dict(lam_t=js(hour_gen, hour_me), d=ks_sym(d_gen, dist_me),
              v=ks_sym(v_gen, spd_me), pu=js(pu_gen, pu_me), do=js(do_gs, do_me))

rng2 = np.random.default_rng(11)
d_ab = sample_B3(B3, NGEN, rng2)
hrs_ab = rng2.choice(24, size=NGEN, p=lam0_t)
v_ab = np.clip(np.exp(rng2.normal(mu0_t[hrs_ab], sigma0, NGEN)), 0.5, 99)
hour_ab = np.bincount(hrs_ab, minlength=24).astype(float); hour_ab /= hour_ab.sum()
scores_ab = dict(lam_t=js(hour_ab, hour_me), d=ks_sym(d_ab, dist_me),
                 v=ks_sym(v_ab, spd_me), pu=js(pu_share0, pu_me), do=js(do_share0, do_me))

print('得分(越低越好)      扰动管线  纯基线   pass1   pass2   改善vs基线')
P1 = json.load(open('D:/AI/条件转化/.flow_scores_pass1.json'))['scores']
P2 = json.load(open('D:/AI/条件转化/.flow_scores_pass2.json'))['scores']
lbl = dict(lam_t='JS λ(t)', d='KS P(d)', v='KS P(v)', pu='JS PU', do='JS DO')
for k in lbl:
    imp = (1 - scores[k] / scores_ab[k]) * 100
    print(f'  {lbl[k]:9s} {scores[k]:.4f}  {scores_ab[k]:.4f}  {P1[k]:.4f}  {P2[k]:.4f}  {imp:+.0f}%')

# 总量预测误差(第6项): 逐中雨日 Σreal vs Σpred
real_tot = hour_cnt[me_i].sum(1).astype(float)
pred_tot = exp_days
vol_mape = float(np.mean(np.abs(pred_tot / real_tot - 1)))
vol_mape_base = float(np.mean(np.abs(lam0_how[dow[me_i]].sum(1) / real_tot - 1)))
print(f'\n日总量预测: 扰动管线 MAPE={vol_mape*100:.1f}% vs 干日基线 MAPE={vol_mape_base*100:.1f}%')

# 日级外推质量
d_d_me, sh_me, rho_me = measure_day(me_i)
d_d_he, sh_he, rho_he = measure_day(he_i)
print('\n日级外推(轻雨标定→预测 vs 实测):')
print(f'  δ_d     中雨 pred{TH["d_d"]:+.4f} vs 实测{d_d_me:+.4f} | 强雨 pred{extrap_day(d_d_li, r_he):+.4f} vs 实测{d_d_he:+.4f}')
print(f'  短途pp  中雨 pred{TH["sh"]*100:+.2f} vs 实测{sh_me*100:+.2f} | 强雨 pred{extrap_day(sh_li, r_he)*100:+.2f} vs 实测{sh_he*100:+.2f}')
print(f'  ρ_ap    中雨 pred{TH["rho_ap"]:.3f} vs 实测{rho_me:.3f} | 强雨 pred{1+extrap_day(rho_li-1, r_he):.3f} vs 实测{rho_he:.3f}')

# 条件距离一致性(对生成器公平的 DO 指标): 真实 vs 生成 的 P(直线距离|PU=top zones)
sel_od = np.isin(A['od_day'].astype(int), me_i)
r_pu = (A['od_pair'][sel_od] // 264).astype(int); r_do = (A['od_pair'][sel_od] % 264).astype(int)
r_cnt = A['od_cnt'][sel_od].astype(float)
r_straight = dist_km[r_pu, r_do] / 1.60934
top_pu = np.argsort(-pu_cnt[me_i].sum(0))[1:31]     # top30 起点zone
g_straight = dist_km[s_gen, do_gen] / 1.60934
errs = []
for z in top_pu:
    mr_ = r_straight[r_pu == z]; cr_ = r_cnt[r_pu == z]
    mg_ = g_straight[s_gen == z]
    if len(mr_) < 50 or len(mg_) < 50: continue
    errs.append(abs(np.average(mr_, weights=cr_) - mg_.mean()) / np.average(mr_, weights=cr_))
cond_mape = float(np.mean(errs))
print(f'\nDO条件距离(直线mi, |真实-生成|/真实, top30 PU zone): MAPE={cond_mape*100:.1f}%')

np.savez_compressed('D:/AI/条件转化/.flow_gen.npz',
    d=d_gen[:300000].astype(np.float32), v=v_gen[:300000].astype(np.float32),
    t=t_gen[:300000].astype(np.int16), s=s_gen[:300000].astype(np.int16), do=do_gen[:300000].astype(np.int16),
    lam0_t=lam0_t, hour_me=hour_me, hour_gen=hour_gen, mu0_t=mu0_t,
    pu_share0=pu_share0, pu_me=pu_me, do_me=do_me, do_gs=do_gs, D_ext=D_ext, lam_ext_s=lam_ext_s,
    inten_me=inten[me_i], scale_me=scale_me)
json.dump(dict(pass_='2b', scores=scores, scores_ab=scores_ab, scores_pass1=P1, scores_pass2=P2,
               cond_dist_mape=cond_mape, n_rej=n_rej, B3=B3, B3e=B3e,
               TH=TH, theta=dict(vol_h=th_vol, v_h=(th_v if th_v is not None else 0.0), eta=eta, RC_H=RC_H, R_C=R_C),
               cal=dict(dvol_h=dvol_h_cal, d_v_h=(d_v_cal if sm.sum() > 200 else None), r_cal=r_cal_h, nh=int(cal_mask.sum())),
               measured_hour=measured,
               day_me=dict(d_d=d_d_me, sh=sh_me, rho=rho_me), day_he=dict(d_d=d_d_he, sh=sh_he, rho=rho_he),
               day_li=dict(d_d=d_d_li, sh=sh_li, rho=rho_li),
               vol=dict(mape=vol_mape, mape_base=vol_mape_base),
               B5=dict(med=B5_med, std=B5_std), B4=dict(sigma0=sigma0),
               gates=dict(tri=g_tri, cap_p99=g_cap, sig_ratio=sig_gen / sig_ext, H0=H0, H1=H1),
               r_li=r_li, r_med=r_med, r_he=r_he, NGEN=NGEN,
               n=dict(dry=len(dry_i), li=len(li_i), me=len(me_i), he=len(he_i))),
          open('D:/AI/条件转化/.flow_scores.json', 'w'), indent=1, ensure_ascii=False)
print('\nsaved (pass2)')
