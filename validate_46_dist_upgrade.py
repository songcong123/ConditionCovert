# -*- coding: utf-8 -*-
"""距离公式特征升级: 单变量 μ1(r) → 分层/多特征公式
   M0 现行: μ1 = -θ·ln(1+r/RC)                        (单变量, validate_25)
   M1 分层: 工作日/周末各一套 θ·RC                     (日型特征)
   M2 加项: μ1 = -θ·ln(1+r/RC) + β1·周末 + β2·滞后雨比   (线性特征项)
   M3 参考: RF(日降水+日型)                             (上界参照)
   评价: 内插(<30mm)拟合MAE + 外推(≥30)分箱偏差"""
import numpy as np, json
from scipy import optimize, stats as st
from sklearn.ensemble import RandomForestRegressor

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates']); n = len(dates)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
wd = np.array([((np.datetime64(d, 'D').view('int64') + 3) % 7 < 5).astype(int) for d in dates])
B3 = json.load(open(r'D:/AI/条件转化/.flow_scores.json'))['B3']
GRID = np.geomspace(0.3, 60, 450)

def mix_cdf(x, mu1s):
    s = B3['w1'] + B3['w2'] + B3['w_ap'] + B3['w_pa']
    c = ((B3['w1']/s) * st.lognorm.cdf(x, B3['sd1'], scale=np.exp(B3['mu1'] + mu1s))
         + (B3['w2']/s) * st.lognorm.cdf(x, B3['sd2'], scale=np.exp(B3['mu2']))
         + (B3['w_ap']/s) * st.lognorm.cdf(x, B3['sd_ap'], scale=np.exp(B3['mu_ap'])))
    return c + (B3['w_pa']/s) * np.where(x >= B3['xmin'], 1 - (x / B3['xmin']) ** (-B3['alpha']), 0)

dry_i = np.where(kl == 'dry')[0]
dd = dist[np.isin(dist_day, dry_i)]; dd = dd[(dd > 0.3) & (dd < 60)]
F0 = np.searchsorted(np.sort(dd), GRID, 'right') / len(dd)

# 逐日 μ1 + 特征(周末旗标、雨强方差代理: 当日逐时雨强的最大/均值比 → 集中度)
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd_, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd_) in didx: inten[didx[str(dd_)], hh] = mm

rows = []
for i in range(n):
    if kl[i] == 'dry': continue
    s = dist[dist_day == i]; s = s[(s > 0.3) & (s < 60)]
    if len(s) < 800: continue
    Fd = np.searchsorted(np.sort(s), GRID, 'right') / len(s)
    r_ = optimize.minimize(lambda m: np.abs(mix_cdf(GRID, m[0]) - Fd).mean(), [-0.02],
                           method='Nelder-Mead', options=dict(maxiter=200, xatol=1e-4, fatol=1e-6))
    mu1 = float(r_.x[0])
    if not (-0.15 < mu1 < 0.05): continue
    ih = inten[i]; ih = ih[ih > 0]
    conc = float(ih.max() / max(ih.mean(), 1e-6)) if len(ih) else 1.0   # 雨强集中度(大=短时暴雨)
    rows.append((dp[i], wd[i], conc, mu1))
S = np.array(rows)
rr_, wv, cc, ym = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
m_in, m_ex = rr_ < 30, rr_ >= 30
print(f'样本 {len(S)} 日 | 训练(<30) {m_in.sum()} | 外推(≥30) {m_ex.sum()}')
print(f'工作日 μ1 均值 {ym[wv==1].mean()*100:.1f}% vs 周末 {ym[wv==0].mean()*100:.1f}% | 集中度相关 r={np.corrcoef(cc, ym)[0,1]:.2f}')

def fit_single(m):
    def obj(p):
        th, RC = p
        return np.mean((th * np.log1p(rr_[m] / RC) - ym[m]) ** 2)
    best = None
    for RC0 in [1, 3, 5.4, 10, 20]:
        r_ = optimize.minimize(obj, [-0.02, RC0], bounds=[(-0.3, 0), (0.5, 50)], method='Nelder-Mead')
        if best is None or r_.fun < best.fun: best = r_
    return best.x

def fit_add(m):
    def obj(p):
        th, RC, b_we, b_cc = p
        base = th * np.log1p(rr_[m] / RC)
        return np.mean((base + b_we * (1 - wv[m]) + b_cc * cc[m] - ym[m]) ** 2)
    best = None
    for RC0 in [3, 5.4, 10]:
        r_ = optimize.minimize(obj, [-0.02, RC0, 0.0, 0.0], method='Nelder-Mead',
                               bounds=[(-0.3, 0), (0.5, 50), (-0.1, 0.1), (-0.05, 0.05)], options=dict(maxiter=5000, xatol=1e-5, fatol=1e-8))
        if best is None or r_.fun < best.fun: best = r_
    return best.x

# M0 单变量
p0 = fit_single(m_in)
f0 = lambda r: p0[0] * np.log1p(np.asarray(r, float) / p0[1])
# M1 分层(工作日/周末各一套)
p_wd = fit_single(m_in & (wv == 1)); p_we = fit_single(m_in & (wv == 0))
f1 = lambda r, w: (p_wd[0] * np.log1p(r / p_wd[1]) if w == 1 else p_we[0] * np.log1p(r / p_we[1]))
# M2 加项
p2 = fit_add(m_in)
f2 = lambda r, w, c: p2[0] * np.log1p(r / p2[1]) + p2[2] * (1 - w) + p2[3] * c
# M3 RF
rf = RandomForestRegressor(400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(np.column_stack([np.log1p(rr_[m_in]), wv[m_in]]), ym[m_in])
pred_rf = rf.predict(np.column_stack([np.log1p(rr_), wv]))

print(f'\nM0 现行单变量: θ={p0[0]:.4f} RC={p0[1]:.1f}')
print(f'M1 分层: 工作日 θ={p_wd[0]:.4f}/RC={p_wd[1]:.1f} | 周末 θ={p_we[0]:.4f}/RC={p_we[1]:.1f}')
print(f'M2 加项: θ={p2[0]:.4f} RC={p2[1]:.1f} β周末={p2[2]:.4f} β集中度={p2[3]:.5f}')

BINS = [(0.5, 5), (5, 15), (15, 30), (30, 60), (60, 200)]
NAMES = ['0.5-5', '5-15', '15-30', '30-60', '≥60']
preds = {
    'M0 现行': f0(rr_),
    'M1 分层': np.array([f1(r_, w_) for r_, w_ in zip(rr_, wv)]),
    'M2 加项': f2(rr_, wv, cc),
    'M3 RF': pred_rf,
}
print(f'\n{"模型":10s} {"内插MAE":>8s} | 分箱偏差(内插3箱 | 外推2箱)')
hdr = ' '.join([f'{nm:>8s}' for nm in NAMES])
print(f'{"":12s}{hdr}')
for nm, pr in preds.items():
    mae_in = np.mean(np.abs(pr[m_in] - ym[m_in])) * 100
    devs = []
    for lo, hi in BINS:
        m = (rr_ >= lo) & (rr_ < hi)
        devs.append(abs(np.mean(pr[m]) - np.mean(ym[m])) * 100)
    print(f'{nm:10s} {mae_in:7.2f}pp | ' + ' '.join([f'{d:7.2f}' for d in devs]) + '   (pp)')
