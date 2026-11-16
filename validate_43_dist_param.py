# -*- coding: utf-8 -*-
"""距离·参数口径: 逐日拟合 μ1 移位(固定其他参数, validate_25 同口径) → 日级样本
   RF(日降水log+日型, <30mm训练) vs 公式 μ1(r)=−0.0171·ln(1+r/5.4)
   内插箱(<30) / 外推箱(30-60, ≥60) 分箱对比"""
import numpy as np, json
from scipy import optimize
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
    w1 = B3['w1'] + B3['w2'] + B3['w_ap'] + B3['w_pa']
    w1, w2, wap, wpa = B3['w1']/w1, B3['w2']/w1, B3['w_ap']/w1, B3['w_pa']/w1
    from scipy import stats as st
    c = (w1 * st.lognorm.cdf(x, B3['sd1'], scale=np.exp(B3['mu1'] + mu1s))
         + w2 * st.lognorm.cdf(x, B3['sd2'], scale=np.exp(B3['mu2']))
         + wap * st.lognorm.cdf(x, B3['sd_ap'], scale=np.exp(B3['mu_ap'])))
    return c + wpa * np.where(x >= B3['xmin'], 1 - (x / B3['xmin']) ** (-B3['alpha']), 0)

dry_i = np.where(kl == 'dry')[0]
dd = dist[np.isin(dist_day, dry_i)]; dd = dd[(dd > 0.3) & (dd < 60)]
F0 = np.searchsorted(np.sort(dd), GRID, 'right') / len(dd)

# 逐日拟合 μ1
samples = []
for i in range(n):
    if kl[i] == 'dry': continue
    s = dist[dist_day == i]; s = s[(s > 0.3) & (s < 60)]
    if len(s) < 800: continue
    Fd = np.searchsorted(np.sort(s), GRID, 'right') / len(s)
    obj = lambda m: np.abs(mix_cdf(GRID, m[0]) - Fd).mean()
    r_ = optimize.minimize(obj, [-0.02], method='Nelder-Mead',
                           options=dict(maxiter=200, xatol=1e-4, fatol=1e-6))
    mu1 = float(r_.x[0])
    samples.append((dp[i], wd[i], mu1))
    # 限制范围
    if not (-0.15 < mu1 < 0.05): samples.pop()
S = np.array(samples)
print(f'逐日 μ1 样本: {len(S)} 个雨日 | μ1 范围 {S[:,2].min():.3f}~{S[:,2].max():.3f}')

X = np.column_stack([np.log1p(S[:, 0]), S[:, 1]])
ym = S[:, 2]
rr = S[:, 0]
f_mu = lambda r: -0.0171 * np.log1p(np.asarray(r, float) / 5.4)

m_tr = rr < 30
rf = RandomForestRegressor(400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X[m_tr], ym[m_tr])
pred = rf.predict(X)

BINS = [(0.5, 5, '0.5-5', '内插'), (5, 15, '5-15', '内插'), (15, 30, '15-30', '内插'),
        (30, 60, '30-60', '外推'), (60, 200, '≥60', '外推')]
print(f'\n{"箱":8s} {"域":4s} {"n":>3s} {"实测μ1":>8s} {"RF":>8s} {"公式":>8s} | {"RF差":>7s} {"公式差":>7s} 赢家')
for lo, hi, nm, dom in BINS:
    m = (rr >= lo) & (rr < hi)
    if m.sum() < 3: continue
    a = np.mean(ym[m]) * 100
    rp = np.mean(pred[m]) * 100
    fp = f_mu(np.mean(rr[m])) * 100
    print(f'{nm:8s} {dom:4s} {m.sum():3d} {a:7.1f}% {rp:7.1f}% {fp:7.1f}% | {abs(rp-a):6.1f}pp {abs(fp-a):6.1f}pp  {"RF" if abs(rp-a)<abs(fp-a) else "公式"}')
# 内插/外推整体 MAE
mi = rr < 30; me = rr >= 30
print(f'\n内插整体MAE(<30): RF {np.mean(np.abs(pred[mi]-ym[mi]))*100:.2f}pp vs 公式 {np.mean(np.abs(f_mu(rr[mi])-ym[mi]))*100:.2f}pp')
print(f'外推整体MAE(≥30): RF {np.mean(np.abs(pred[me]-ym[me]))*100:.2f}pp vs 公式 {np.mean(np.abs(f_mu(rr[me])-ym[me]))*100:.2f}pp')
