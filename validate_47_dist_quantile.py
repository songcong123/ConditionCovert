# -*- coding: utf-8 -*-
"""距离·分位数剖面对比: 混合分布不能只看中位 —— P10/P25/P50/P75/P90 逐分位对比
   实际(箱内行程池化) vs 三种合成: 公式单参数(μ1平移) / 公式三旋钮 / RF(日级预测平移)
   内插箱 5-15mm | 外推箱 30-60 / ≥60mm"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
NAVY = '#1F4E79'; RED = '#C00000'; GREY = '#999999'; MUTED = '#666666'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates']); n = len(dates)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
B3 = json.load(open(r'D:/AI/条件转化/.flow_scores.json'))['B3']

dry_i = np.where(kl == 'dry')[0]
db = dist[np.isin(dist_day, dry_i)]; db = db[(db > 0.3) & (db < 60)]
base_med = np.median(np.log(db))

# RF(日级, <30 训练): 标签=日 log 中位偏移
wd = np.array([((np.datetime64(d, 'D').view('int64') + 3) % 7 < 5).astype(int) for d in dates])
day_idx, dY = [], []
for i in range(n):
    if kl[i] == 'dry': continue
    s = dist[dist_day == i]; s = s[(s > 0.3) & (s < 60)]
    if len(s) < 800: continue
    day_idx.append(i); dY.append(np.median(np.log(s)) - base_med)
day_idx = np.array(day_idx); dY = np.array(dY)
dR = dp[day_idx]
X = np.column_stack([np.log1p(dR), wd[day_idx]])
rf = RandomForestRegressor(400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X[dR < 30], dY[dR < 30])

# 三旋钮合成器(validate_25 系数): 采样基线→施加三旋钮
DDOSE = json.load(open(r'D:/AI/条件转化/.flow_dist_dose.json'))['fit']
def knobs(Rday):
    mu1 = -DDOSE['mu1']['A'] * np.log1p(Rday / DDOSE['mu1']['RC'])
    w1g = DDOSE['w1']['B'] * np.log1p(Rday / DDOSE['w1']['RC'])
    apr = 1 - DDOSE['ap']['C'] * np.log1p(Rday / DDOSE['ap']['RC'])
    return mu1, w1g, apr
from scipy import stats as st
def sample_knob(rng, Rday, n_s):
    mu1, w1g, apr = knobs(Rday)
    w1 = B3['w1'] + w1g; wap = B3['w_ap'] * apr
    s = w1 + B3['w2'] + wap + B3['w_pa']
    w1, w2, wap, wpa = w1/s, B3['w2']/s, wap/s, B3['w_pa']/s
    u = rng.random(n_s)
    c1 = w1; c2 = c1 + w2; c3 = c2 + wap
    x = np.empty(n_s)
    m1 = u < c1; m2 = (u >= c1) & (u < c2); m3 = (u >= c2) & (u < c3); m4 = ~(m1 | m2 | m3)
    x[m1] = np.exp(rng.normal(B3['mu1'] + mu1, B3['sd1'], m1.sum()))
    x[m2] = np.exp(rng.normal(B3['mu2'], B3['sd2'], m2.sum()))
    x[m3] = np.exp(rng.normal(B3['mu_ap'], B3['sd_ap'], m3.sum()))
    x[m4] = B3['xmin'] * (1 - rng.random(m4.sum())) ** (-1 / B3['alpha'])
    return np.clip(x, 0.3, 60)
f_mu = lambda r: -DDOSE['mu1']['A'] * np.log1p(np.asarray(r, float) / DDOSE['mu1']['RC'])

Q = [10, 25, 50, 75, 90]
rng = np.random.default_rng(7)
BINS = [(5, 15, '5-15', '内插'), (30, 60, '30-60', '外推'), (60, 200, '≥60', '外推')]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
fig.suptitle('距离·分位数剖面：混合分布逐分位对比（实际 vs 三种合成）', fontsize=14, fontweight='bold', y=1.03)

for ax, (lo, hi, nm, dom) in zip(axes, BINS):
    m = np.where((dR >= lo) & (dR < hi))[0]
    act = np.log(dist[np.isin(dist_day, day_idx[m]) & (dist > 0.3) & (dist < 60)])
    syn_mu, syn_kn, syn_rf = [], [], []
    for k in m:
        i = day_idx[k]; ns = 6000
        sub = rng.choice(db, size=ns, replace=True)
        syn_mu.append(np.log(sub) + f_mu(dR[k]))
        syn_rf.append(np.log(sub) + rf.predict(X[k:k+1])[0])
        syn_kn.append(np.log(sample_knob(rng, dR[k], ns)))
    qa = np.percentile(act, Q)
    qm = np.percentile(np.concatenate(syn_mu), Q)
    qk = np.percentile(np.concatenate(syn_kn), Q)
    qr = np.percentile(np.concatenate(syn_rf), Q)
    xq = np.arange(len(Q))
    ax.plot(xq, (np.exp(qa) - np.exp(qa)) * 0, alpha=0)  # dummy
    e_mu = (np.exp(qm) / np.exp(qa) - 1) * 100
    e_kn = (np.exp(qk) / np.exp(qa) - 1) * 100
    e_rf = (np.exp(qr) / np.exp(qa) - 1) * 100
    ax.axhline(0, color='#999', lw=1, ls=':')
    ax.plot(xq, e_mu, 'o-', lw=2, ms=8, color=NAVY, label='公式单参数(μ1平移)')
    ax.plot(xq, e_kn, 's-', lw=2, ms=8, color='#7030A0', label='公式三旋钮(μ1+w1+机场)')
    ax.plot(xq, e_rf, '^-', lw=2, ms=8, color=RED, label='RF(日级平移)')
    for x_, v_ in zip(xq, e_mu): ax.annotate(f'{v_:+.0f}', (x_, v_), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8.5, color=NAVY)
    for x_, v_ in zip(xq, e_kn): ax.annotate(f'{v_:+.0f}', (x_, v_), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8.5, color='#7030A0')
    for x_, v_ in zip(xq, e_rf): ax.annotate(f'{v_:+.0f}', (x_, v_), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8.5, color=RED)
    ax.set_xticks(xq); ax.set_xticklabels([f'P{q}' for q in Q], fontsize=10.5)
    ax.set_xlabel('分位数'); ax.set_ylabel('合成 vs 实际 偏差 (%)')
    ax.set_title(f'{nm} mm/日（{dom}, n={len(m)}天）', fontsize=11.5, loc='left', color=NAVY)
    ax.legend(fontsize=8.5, loc='best'); ax.grid(alpha=0.3)
    print(f'{nm:6s} 偏差: μ1 {np.round(e_mu,1)} | 三旋钮 {np.round(e_kn,1)} | RF {np.round(e_rf,1)}')

fig.text(0.5, 0.005, '读法: 0%=完美。P10/P25=短途区, P50=中途, P75/P90=长尾(机场)。单参数(μ1平移)所有分位同样移动; 三旋钮让短途再密、长尾再稀 —— 看哪种合成贴合 0 线',
         ha='center', fontsize=10, color=DGREEN, fontweight='bold')
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/距离分位数剖面.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 距离分位数剖面.png')
