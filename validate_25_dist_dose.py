# -*- coding: utf-8 -*-
"""距离三旋钮的剂量曲线: 分雨强档直读旋钮 → 拟合 θ·ln(1+r/RC) 饱和曲线 → 留出外推验证
旋钮: mu1_shift(短途均值移位) / w1_gain(短途权重增益) / ap_ratio(机场份额比值)
剂量轴 r = 日降水量(mm), 干燥日 r=0 为基线
输出: .flow_dist_dose.json + 距离剂量曲线.png
"""
import numpy as np, json
from scipy import optimize, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
B = json.load(open('D:/AI/条件转化/.flow_scores.json'))['B3']

dates = list(A['dates'])
dp = dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))
dp_arr = np.array([dp[d] for d in dates])
dist = A['dist'].astype(np.float64)
dist_day = A['dist_day'].astype(int)

BINS = [(0.5, 5, 'light', '小雨'), (5, 15, 'medium', '中雨'), (15, 30, 'h1', '大雨Ⅰ'),
        (30, 60, 'h2', '大雨Ⅱ'), (60, 1e9, 'extreme', '极端≥60mm')]
GRID = np.geomspace(0.05, 60, 450)

def mix_cdf(x, mu1s, w1g, apr):
    w1 = B['w1'] + w1g; wap = B['w_ap'] * apr
    s = w1 + B['w2'] + wap + B['w_pa']
    w1, w2, wap, wpa = w1/s, B['w2']/s, wap/s, B['w_pa']/s
    c = (w1 * stats.lognorm.cdf(x, B['sd1'], scale=np.exp(B['mu1'] + mu1s))
         + w2 * stats.lognorm.cdf(x, B['sd2'], scale=np.exp(B['mu2']))
         + wap * stats.lognorm.cdf(x, B['sd_ap'], scale=np.exp(B['mu_ap'])))
    pareto = np.where(x >= B['xmin'], 1 - (x / B['xmin']) ** (-B['alpha']), 0)
    return c + wpa * pareto

def fit_bin(d_sample):
    d_sample = d_sample[(d_sample > 0.05) & (d_sample < 100)]
    d_sample = np.sort(d_sample)
    F_emp = np.searchsorted(d_sample, GRID, side='right') / len(d_sample)
    def obj(p):
        return np.mean(np.abs(mix_cdf(GRID, p[0], p[1], p[2]) - F_emp))
    best = None
    for x0 in [(0, 0, 1), (-0.061, 0.025, 0.9)]:
        r = optimize.minimize(obj, x0, method='Nelder-Mead',
                              options=dict(maxiter=600, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun: best = r
    return best.x, best.fun

results = {}
print('=== 分雨强档直读旋钮 ===')
print(f"{'档':10s} {'r中位(mm)':>9s} {'天数':>4s} {'样本':>9s} {'mu1_shift':>10s} {'w1_gain':>8s} {'ap_ratio':>8s} {'L1误差':>8s}")
pts = []
for lo, hi, key, cname in BINS:
    m = (dp_arr >= lo) & (dp_arr < hi)
    d_sample = dist[np.isin(dist_day, np.where(m)[0])]
    r_med = float(np.median(dp_arr[m]))
    (mu1s, w1g, apr), fmin = fit_bin(d_sample)
    results[key] = dict(r=r_med, days=int(m.sum()), n=int(len(d_sample)),
                        cname=cname, mu1s=round(float(mu1s), 4), w1g=round(float(w1g), 4),
                        apr=round(float(apr), 4), l1=round(float(fmin), 5))
    pts.append((r_med, mu1s, w1g, apr))
    print(f"{cname:10s} {r_med:9.1f} {m.sum():4d} {len(d_sample):9d} {mu1s:10.4f} {w1g:8.4f} {apr:8.4f} {fmin:8.4f}")

# ---- 拟合剂量曲线 (对每条留出极端档做交叉验证) ----
rr = np.array([p[0] for p in pts])
def fit_curve(xs, ys, form):
    if form == 'dec':   # y = -A*ln(1+r/RC)
        f = lambda r, A, RC: -A * np.log1p(np.maximum(r, 0) / np.maximum(RC, 1.0))
        p0 = [0.05, 20]; bd = ([1e-4, 1.0], [0.5, 500])
    elif form == 'inc':  # y = +B*ln(1+r/RC)
        f = lambda r, B_, RC: B_ * np.log1p(np.maximum(r, 0) / np.maximum(RC, 1.0))
        p0 = [0.02, 20]; bd = ([1e-4, 1.0], [0.2, 500])
    else:                # apr: y = 1 - C*ln(1+r/RC)（形态天然锚定 r→0 时 =1）
        f = lambda r, C, RC: 1 - C * np.log1p(np.maximum(r, 0) / np.maximum(RC, 1.0))
        p0 = [0.05, 20]; bd = ([1e-4, 1.0], [0.2, 500])
    popt, _ = optimize.curve_fit(f, xs, ys, p0=p0, bounds=bd, maxfev=20000)
    return f, popt

mu1s_v = np.array([p[1] for p in pts]); w1g_v = np.array([p[2] for p in pts]); apr_v = np.array([p[3] for p in pts])
f_mu, p_mu = fit_curve(rr, mu1s_v, 'dec')
f_w1, p_w1 = fit_curve(rr, w1g_v, 'inc')
# ap_ratio: 小雨档未见明显效应(≈1, 属噪声)——从中雨档起拟合; 公式形态天然锚定 r→0 时 =1
f_ap, p_ap = fit_curve(rr[1:], apr_v[1:], 'aprf')
print('\n=== 剂量曲线拟合 ===')
print(f"mu1_shift(r) = -{p_mu[0]:.4f}·ln(1+r/{p_mu[1]:.1f})   R2={1-np.sum((mu1s_v-f_mu(rr,*p_mu))**2)/np.sum((mu1s_v-mu1s_v.mean())**2):.3f}  (5档)")
print(f"w1_gain(r)   = +{p_w1[0]:.4f}·ln(1+r/{p_w1[1]:.1f})   R2={1-np.sum((w1g_v-f_w1(rr,*p_w1))**2)/np.sum((w1g_v-w1g_v.mean())**2):.3f}  (5档)")
apr_y = apr_v[1:]
print(f"ap_ratio(r)  = 1-{p_ap[0]:.4f}·ln(1+r/{p_ap[1]:.1f})  R2={1-np.sum((apr_y-f_ap(rr[1:],*p_ap))**2)/np.sum((apr_y-apr_y.mean())**2):.3f}  (中雨起4档; 小雨≈1无效应)")

# ---- 留出验证: 前4档拟合 → 预测极端档(apr 为 中雨起前3档 → 预测极端) ----
hold = {}
f4, p4 = fit_curve(rr[:4], mu1s_v[:4], 'dec')
pe = float(f4(rr[4], *p4))
hold['mu1s'] = dict(pred_ex=round(pe, 4), actual=round(float(mu1s_v[4]), 4),
                    err=round(abs(pe - mu1s_v[4]) / abs(mu1s_v[4]) * 100, 1))
f4, p4 = fit_curve(rr[:4], w1g_v[:4], 'inc')
pe = float(f4(rr[4], *p4))
hold['w1g'] = dict(pred_ex=round(pe, 4), actual=round(float(w1g_v[4]), 4),
                   err=round(abs(pe - w1g_v[4]) / abs(w1g_v[4]) * 100, 1))
f4, p4 = fit_curve(rr[1:4], apr_v[1:4], 'aprf')
pe = float(f4(rr[4], *p4))
hold['apr'] = dict(pred_ex=round(pe, 4), actual=round(float(apr_v[4]), 4),
                   err=round(abs(pe - apr_v[4]) / abs(apr_v[4]) * 100, 1))
for k, v in hold.items():
    print(f"留出验证 {k}: 前4档拟合 → 预测极端档 {v['pred_ex']} (实际 {v['actual']}, 偏差 {v['err']}%)")

# ---- 外推物理性检查 ----
print('\n=== 外推物理性检查(曲线外推值) ===')
print(f"{'r(mm)':>6s} {'mu1中位变化%':>12s} {'w1':>7s} {'ap_ratio':>9s}")
for r in [15, 30, 60, 100, 148, 200]:
    ms, wg, ar = float(f_mu(r, *p_mu)), float(f_w1(r, *p_w1)), max(float(f_ap(r, *p_ap)), 0.55)
    med_now = np.exp(B['mu1'] + ms); med0 = np.exp(B['mu1'])
    w1 = B['w1'] + wg
    print(f"{r:6.0f} {(med_now/med0-1)*100:11.1f}% {w1:7.3f} {ar:9.3f}")
print('(ap_ratio 物理底线 0.55——更低进入崩溃档,由崩溃规则接管)')

results['fit'] = dict(mu1=dict(A=round(float(p_mu[0]), 4), RC=round(float(p_mu[1]), 1)),
                      w1=dict(B=round(float(p_w1[0]), 4), RC=round(float(p_w1[1]), 1)),
                      ap=dict(C=round(float(p_ap[0]), 4), RC=round(float(p_ap[1]), 1)),
                      holdout=hold)

# ---- 实用口径 KS 交叉验证(极端档): 基线不调 vs 曲线外推 vs 分档直读 ----
rng = np.random.default_rng(7)
def sample_mix(mu1s, w1g, apr, n):
    w1 = B['w1'] + w1g; wap = B['w_ap'] * apr
    s = w1 + B['w2'] + wap + B['w_pa']
    w1, w2, wap, wpa = w1/s, B['w2']/s, wap/s, B['w_pa']/s
    u = rng.random(n); x = np.empty(n)
    c1 = w1; c2 = c1 + w2; c3 = c2 + wap
    m1 = u < c1; m2 = (u >= c1) & (u < c2); m3 = (u >= c2) & (u < c3); m4 = ~(m1 | m2 | m3)
    x[m1] = rng.lognormal(B['mu1'] + mu1s, B['sd1'], m1.sum())
    x[m2] = rng.lognormal(B['mu2'], B['sd2'], m2.sum())
    x[m3] = rng.lognormal(B['mu_ap'], B['sd_ap'], m3.sum())
    x[m4] = B['xmin'] * (1 - rng.random(m4.sum())) ** (-1 / B['alpha'])
    return np.clip(x, 0.05, 80)

m = (dp_arr >= 60)
dd_ex = dist[np.isin(dist_day, np.where(m)[0])]
dd_ex = dd_ex[(dd_ex > 0.05) & (dd_ex < 80)]
n = min(200000, len(dd_ex))
emp = rng.choice(dd_ex, n, replace=False)
ks_base = stats.ks_2samp(sample_mix(0, 0, 1, n), emp).statistic
ks_curve = stats.ks_2samp(sample_mix(float(f_mu(98.6, *p_mu)), float(f_w1(98.6, *p_w1)),
                                     float(f_ap(98.6, *p_ap)), n), emp).statistic
ks_direct = stats.ks_2samp(sample_mix(mu1s_v[4], w1g_v[4], apr_v[4], n), emp).statistic
print(f'\n=== 极端档(≥60mm) KS 交叉验证 ===')
print(f'基线不调: KS={ks_base:.3f} | 曲线外推: KS={ks_curve:.3f} | 分档直读: KS={ks_direct:.3f}')
results['ks_extreme'] = dict(base=round(float(ks_base), 3), curve=round(float(ks_curve), 3),
                             direct=round(float(ks_direct), 3))
json.dump(results, open('D:/AI/条件转化/.flow_dist_dose.json', 'w'), ensure_ascii=False, indent=1)
print('saved .flow_dist_dose.json')

# ---- 图: 三条剂量曲线 ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle('距离三旋钮的剂量曲线：分雨强档直读 → 饱和曲线拟合 → 外推有据（2024-04~2025-10, 419 个有效日, 627 万样本）',
             fontsize=14, fontweight='bold', y=1.02)
r_max_obs = rr[-1]
panels = [
    (rr, mu1s_v, f_mu, p_mu, 'μ1 旋钮：短途均值移位 (mi)', 'dec', '全5档'),
    (rr, w1g_v, f_w1, p_w1, 'w1 旋钮：短途权重增益 (pp)', 'inc', '全5档'),
    (rr[1:], apr_v[1:], f_ap, p_ap, 'w_ap 旋钮：机场份额比值', 'aprf', '中雨起4档'),
]
for ax, (xs_p, ys, f, pp, title, form, scope) in zip(axes, panels):
    rr_fine = np.linspace(0, 220, 300)
    ax.plot(rr_fine, f(rr_fine, *pp), lw=2.4, color='#C00000', label='拟合曲线')
    ax.scatter(xs_p, ys, s=90, color='#4472C4', zorder=5, label=f'分档直读（{scope}）')
    for x_, y_ in zip(xs_p, ys):
        ax.annotate(f'{y_:.3f}', (x_, y_), textcoords='offset points', xytext=(4, 7), fontsize=8.5, color='#1F4E79')
    if form == 'aprf':
        ax.scatter([rr[0]], [apr_v[0]], s=60, facecolors='none', edgecolors='#7F7F7F', linewidths=1.4, label='小雨≈1（未见效应）')
    ax.axvspan(r_max_obs, 220, color='#C00000', alpha=0.06)
    ax.axvline(r_max_obs, color='#C00000', ls=':', lw=1.2)
    ax.text(r_max_obs + 4, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f'外推区\n(> {r_max_obs:.0f}mm)', fontsize=8.5, color='#C00000')
    ax.set_xlabel('日降水量 r (mm)'); ax.set_title(title, fontsize=11.5, loc='left')
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc='best')
axes[0].text(0.03, 0.06, '外推物理性：μ1 中位最多右移到 ~1.0mi；w1 上界 = w1+w2；ap_ratio ≥ 0.55 底线（更极端进崩溃档）',
             transform=axes[0].transAxes, fontsize=8, color='#666666')
plt.tight_layout()
plt.savefig('D:/AI/条件转化/距离剂量曲线.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 距离剂量曲线.png')
