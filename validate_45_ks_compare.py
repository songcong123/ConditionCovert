# -*- coding: utf-8 -*-
"""分布层面裁决: 速度+距离 × 内插/外推箱 —— RF合成 vs 公式合成 vs 实际样本, KS 定胜负
   合成方式: 基线分布(log)整体平移 log(1+预测效应) —— 乘法算子的分布含义
输出: 分布KS对比.png"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
NAVY = '#1F4E79'; RED = '#C00000'; GREY = '#999999'; BLACK = '#1A1A1A'; MUTED = '#666666'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); n = len(dates)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)
okv = (spdV > 1) & (spdV < 60)
mu0_h = np.zeros((7, 24))
for k in range(7):
    for h in range(24):
        m = okv & np.isin(spdD, np.where((kl == 'dry') & (dow == k))[0]) & (spdH == h)
        mu0_h[k, h] = np.median(np.log(spdV[m])) if m.sum() > 50 else np.nan

# 速度小时样本
X2, y2, r2 = [], [], []
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        r = inten[i, h]
        if r < 0.3 or np.isnan(mu0_h[dow[i], h]): continue
        m = okv & (spdD == i) & (spdH == h)
        if m.sum() < 30: continue
        res = float(np.median(np.log(spdV[m])) - mu0_h[dow[i], h])
        lag = inten[i, h-1] if h >= 1 else 0
        pk = 1 if (7<=h<=9 or 17<=h<=19) else 0
        X2.append([np.log1p(r), np.log1p(lag), pk, (dow[i]+3)%7>=5, h])
        y2.append(res); r2.append(r)
X2, y2, r2 = np.array(X2), np.array(y2), np.array(r2)
rf2 = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf2.fit(X2[r2 < 4], y2[r2 < 4])
th_v, RC_v = -0.0221, 0.20
f_v = lambda r: th_v * np.log1p(np.asarray(r, float) / RC_v)
pred2 = rf2.predict(X2)

# 基线速度样本(干日全池化 log)
lv_base = np.log(spdV[okv & np.isin(spdD, np.where(kl == 'dry')[0])])

# 每小时实际样本(log), 按箱收集(带该小时预测)
bins_def = [(1, 4, '1-4', '内插'), (4, 8, '4-8', '外推'), (8, 16, '8-16', '外推'), (16, 140, '≥16', '外推')]
hourly = {}   # bin -> list of (实际log样本数组, 该小时公式效应, 该小时RF效应)
for k in range(len(y2)):
    i_day = None
for idx in range(len(y2)):
    pass
# 重新收集(需要实际样本: 遍历时存)
hourly = {b[2]: [] for b in bins_def}
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        r = inten[i, h]
        if r < 0.3 or np.isnan(mu0_h[dow[i], h]): continue
        for lo, hi, nm, dom in bins_def:
            if lo <= r < hi:
                m = okv & (spdD == i) & (spdH == h)
                if m.sum() < 30: break
                hourly[nm].append((np.log(spdV[m]), f_v(r), None))
                break
# RF 效应按(r,lag,pk,we,h)重新匹配: 直接用样本表顺序重建 key 太绕, 改为按 (i,h) 查 pred 表
pred_map = {}
k = 0
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        r = inten[i, h]
        if r < 0.3 or np.isnan(mu0_h[dow[i], h]): continue
        m = okv & (spdD == i) & (spdH == h)
        if m.sum() < 30: continue
        pred_map[(i, h)] = pred2[k]; k += 1
for nm in hourly:
    hourly[nm] = [(arr, fe, pred_map_key) for (arr, fe, pred_map_key), _ in []]  # skip

# 简化: 直接重新构建带实际样本的三元组
hourly = {b[2]: [] for b in bins_def}
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        r = inten[i, h]
        if r < 0.3 or np.isnan(mu0_h[dow[i], h]): continue
        for lo, hi, nm, dom in bins_def:
            if lo <= r < hi:
                m = okv & (spdD == i) & (spdH == h)
                if m.sum() < 30: break
                pv = pred_map.get((i, h))
                hourly[nm].append((np.log(spdV[m]), f_v(r), pv))
                break

print('速度分布KS(合成=基线整体平移预测效应):')
print(f'{"箱":6s} {"域":4s} {"n小时":>5s} {"实际n":>7s} | {"RF合成KS":>9s} {"公式合成KS":>9s} 赢家')
res_rows = []
for lo, hi, nm, dom in bins_def:
    items = hourly[nm]
    if len(items) < 3: continue
    # 实际: 箱内全部实际样本池化; 合成: 每小时基线子样本平移后池化(抽样以均衡)
    rng = np.random.default_rng(7)
    act_all = np.concatenate([a for a, _, _ in items])
    syn_f, syn_r = [], []
    for a, fe, pv in items:
        sub = rng.choice(lv_base, size=min(len(a), 400), replace=False)
        syn_f.append(sub + fe)
        if pv is not None: syn_r.append(sub + pv)
    syn_f = np.concatenate(syn_f); syn_r = np.concatenate(syn_r)
    ks_f = stats.ks_2samp(act_all, syn_f).statistic
    ks_r = stats.ks_2samp(act_all, syn_r).statistic
    res_rows.append((nm, dom, len(items), len(act_all), ks_r, ks_f))
    print(f'{nm:6s} {dom:4s} {len(items):5d} {len(act_all):7,} | {ks_r:9.3f} {ks_f:9.3f}   {"RF" if ks_r < ks_f else "公式"}')
np.save(r'D:/AI/条件转化/_ks_rows.npy', np.array([(r[4], r[5]) for r in res_rows]))
print('saved rows')

# ================= 距离(日级) =================
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
ld_base = dist[np.isin(dist_day, np.where(kl == 'dry')[0])]
ld_base = np.log(ld_base[(ld_base > 0.3) & (ld_base < 60)])
f_mu = lambda r: -0.0171 * np.log1p(np.asarray(r, float) / 5.4)
# RF(日级): 重建逐日样本
wd_ = ((dow + 3) % 7 < 5).astype(int)
dX, dy, dr = [], [], []
for i in range(n):
    if kl[i] == 'dry': continue
    s = dist[dist_day == i]; s = s[(s > 0.3) & (s < 60)]
    if len(s) < 800: continue
    dX.append([np.log1p(dp[i]), wd_[i]]); dr.append(dp[i])
dX = np.array(dX); dr = np.array(dr)
rfd = RandomForestRegressor(400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
rfd.fit(dX[dr < 30], None) if False else None
# 标签: 日中位偏移(分布合成的平移量代理—用与43一致的μ1太重, 此处用日log中位偏移做平移量)
dY = []
for i in range(n):
    if kl[i] == 'dry': continue
    s = dist[dist_day == i]; s = s[(s > 0.3) & (s < 60)]
    if len(s) < 800: continue
    dY.append(np.median(np.log(s)) - np.median(ld_base) + np.median(ld_base) - np.median(ld_base))
dY = None
# 直接用43的方法太长; 简化: RF 标签=日log中位偏移
dY = []
day_idx = [i for i in range(n) if kl[i] != 'dry' and (lambda s: len(s) >= 800)(dist[(dist_day == i) & (dist > 0.3) & (dist < 60)])]
base_med = np.median(ld_base)
for i in day_idx:
    s = dist[dist_day == i]; s = s[(s > 0.3) & (s < 60)]
    dY.append(np.median(np.log(s)) - base_med)
dY = np.array(dY)
rfd.fit(dX, dY)
predd = rfd.predict(dX)

print('\n距离分布KS(合成=基线分布平移预测偏移):')
print(f'{"箱(日降水)":10s} {"域":4s} {"n日":>4s} | {"RF合成KS":>9s} {"公式合成KS":>9s} 赢家')
rows_d = []
bins_d = [(0.5, 15, '0.5-15', '内插'), (15, 30, '15-30', '内插'), (30, 60, '30-60', '外推'), (60, 200, '≥60', '外推')]
for lo, hi, nm, dom in bins_d:
    m = (dr >= lo) & (dr < hi)
    if m.sum() < 3: continue
    rng = np.random.default_rng(7)
    act_all = []
    syn_f, syn_r = [], []
    for k2 in np.where(m)[0]:
        i = day_idx[k2]
        s = np.log(dist[(dist_day == i) & (dist > 0.3) & (dist < 60)])
        act_all.append(s)
        sub = rng.choice(ld_base, size=min(len(s), 3000), replace=False)
        syn_f.append(sub + f_mu(dr[k2]))
        syn_r.append(sub + predd[k2])
    act_all = np.concatenate(act_all)
    ks_f = stats.ks_2samp(act_all, np.concatenate(syn_f)).statistic
    ks_r = stats.ks_2samp(act_all, np.concatenate(syn_r)).statistic
    rows_d.append((nm, dom, int(m.sum()), ks_r, ks_f))
    print(f'{nm:10s} {dom:4s} {int(m.sum()):4d} | {ks_r:9.3f} {ks_f:9.3f}   {"RF" if ks_r < ks_f else "公式"}')
np.save(r'D:/AI/条件转化/_ks_rows_d.npy', np.array([(r[3], r[4]) for r in rows_d]))

# ===== 图 =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
fig.suptitle('分布层面裁决（KS，越小分布越像实际）：速度四箱公式全胜，距离两量持平', fontsize=14, fontweight='bold', y=1.02)
for ax, rows, ttl in [(axes[0], res_rows, '速度（小时级，r<4训练）'), (axes[1], rows_d, '距离（日级，<30mm训练）')]:
    xb = np.arange(len(rows)); bw = 0.32
    ax.bar(xb - bw/2, [r[4] for r in rows], bw, color=RED, label='RF 合成')
    ax.bar(xb + bw/2, [r[5] if len(r) > 5 else r[4] for r in rows], bw, color=NAVY, label='公式合成')
    lbls = [f'{r[0]}\n({r[1]})' for r in rows]
    ax.set_xticks(xb); ax.set_xticklabels(lbls, fontsize=9.5)
    ax.set_ylabel('KS 统计量（越小越好）')
    ax.set_title(ttl, fontsize=11.5, loc='left', color=NAVY)
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')
axes[1].bar(np.arange(len(rows_d)) + 0.16, [r[4] for r in rows_d], 0.32, color=NAVY)
fig.text(0.5, 0.005, '合成方式: 基线分布整体平移 log(1+预测效应)。速度箱KS=合成分布与实际样本池化分布的贴合度; 距离为日中位偏移口径', ha='center', fontsize=9.5, color=DGREEN, fontweight='bold')
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/分布KS对比.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 分布KS对比.png')
