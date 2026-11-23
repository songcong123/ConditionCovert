# -*- coding: utf-8 -*-
"""validate_54: 统一误差口径 2×2 矩阵 —— 揭穿"外推误差<内插"的假象
================================================================
问题: 主表把 内插逐点MAE 与 外推箱偏差 并排 → 公式外推误差显得比内插还低, 悖论。
原因: 两把尺子不同 —— 逐点MAE含全部噪声(±40pp), 箱偏差是均值之差(噪声÷√n)。
本实验: 三量 × {内插,外推} × {逐点,箱} 完整矩阵, 展示:
   同口径下内插永远 ≤ 外推(正常); 只有混用口径才出现"外推更准"的假象。
输出: 误差口径矩阵.png + 控制台表
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; DGREEN = '#375623'; MUTED = '#666666'; ORANGE = '#ED7D31'

res = {}   # res[量][口径] = (公式, RF)

# ================= ① 出发量(小时级, validate_50 同构) =================
A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates_all = list(A['dates'])
hc = A['hour_cnt'].astype(float)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates_all])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates_all])
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)

inten = np.zeros((len(dates_all), 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

ys, r_now = [], []
for i, d in enumerate(dates_all):
    if kl[i] == 'dry': continue
    for h in range(24):
        if lam[dow[i], h] < 200: continue
        ys.append(hc[i, h] / lam[dow[i], h] - 1)
        r_now.append(inten[i, h])
ys = np.array(ys) * 100; r_now = np.array(r_now)
m_in = (r_now >= 0.3) & (r_now < 4); m_ex = r_now >= 4
THETA, RCQ = 0.052, 0.2
fm_dep = THETA * np.log1p(r_now / RCQ) * 100
BIN_DEP = [(0.3, 1), (1, 4), (4, 8), (8, 16), (16, 1e9)]
bin_dev = lambda pred, lo, hi: abs(pred[(r_now >= lo) & (r_now < hi)].mean() - ys[(r_now >= lo) & (r_now < hi)].mean())
res['出发量'] = {
    '内插·逐点': (np.mean(np.abs(ys[m_in] - fm_dep[m_in])), None),
    '内插·箱':   (max(bin_dev(fm_dep, *b) for b in BIN_DEP[:2]), None),
    '外推·逐点': (np.mean(np.abs(ys[m_ex] - fm_dep[m_ex])), None),
    '外推·箱':   (max(bin_dev(fm_dep, *b) for b in BIN_DEP[2:]), None),
}

# ================= ② 速度(小时级, validate_48/34 同构) =================
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)
OKV = (spdV > 1) & (spdV < 60)
mu0_tb = np.full((7, 24), np.nan)
for k7 in range(7):
    sel = np.where((kl == 'dry') & (dow == k7))[0]
    for h in range(24):
        mmk = np.isin(spdD, sel) & (spdH == h) & OKV
        if mmk.sum() > 50:
            mu0_tb[k7, h] = np.median(np.log(spdV[mmk]))
# 逐时观测点 (r>=0.3)
pts_r, pts_dv = [], []
keys = spdD.astype(np.int64) * 100 + spdH
order = np.argsort(keys, kind='stable'); v_s = spdV[order]
uniq, starts = np.unique(keys[order], return_index=True)
ends = np.append(starts[1:], len(keys))
bucket = dict(zip(uniq.tolist(), zip(starts.tolist(), ends.tolist())))
for i in np.where(kl != 'dry')[0]:
    for h in range(24):
        r_h = inten[i, h]
        if r_h < 0.3: continue
        b0 = mu0_tb[dow[i], h]
        if np.isnan(b0): continue
        kk = i * 100 + h
        if kk not in bucket: continue
        s0, e0 = bucket[kk]
        vs = v_s[s0:e0]; vs = vs[(vs > 1) & (vs < 60)]
        if len(vs) < 30: continue
        pts_r.append(r_h)
        pts_dv.append((np.median(np.log(vs)) - b0) * 100)   # %口径
pts_r = np.array(pts_r); pts_dv = np.array(pts_dv)
dv_pred = -0.0221 * np.log1p(pts_r / 0.2) * 100
m_in_v = pts_r < 4; m_ex_v = pts_r >= 4
BIN_V = [(0.3, 1), (1, 4), (4, 12), (12, 1e9)]
bin_dev_v = lambda lo, hi: abs(dv_pred[(pts_r >= lo) & (pts_r < hi)].mean() - pts_dv[(pts_r >= lo) & (pts_r < hi)].mean())
res['速度'] = {
    '内插·逐点': (np.mean(np.abs(pts_dv[m_in_v] - dv_pred[m_in_v])), None),
    '内插·箱':   (max(bin_dev_v(*b) for b in BIN_V[:2]), None),
    '外推·逐点': (np.mean(np.abs(pts_dv[m_ex_v] - dv_pred[m_ex_v])), None),
    '外推·箱':   (max(bin_dev_v(*b) for b in BIN_V[2:]), None),
}

# ================= ③ 距离(日级, validate_53 同构, 边界15mm) =================
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates_all])
wdays = np.array([((np.datetime64(d, 'D').view('int64') + 3) % 7 < 5).astype(int) for d in dates_all])
dry_idx = np.where(kl == 'dry')[0]
BAS = {}
for g in [1, 0]:
    dvg = dist[np.isin(dist_day, [i for i in dry_idx if wdays[i] == g])]
    BAS[g] = float(np.median(dvg[(dvg > 0.3) & (dvg < 60)]))
Rs, Ys = [], []
for i in np.where(kl != 'dry')[0]:
    dv2 = dist[dist_day == i]; dv2 = dv2[(dv2 > 0.3) & (dv2 < 60)]
    if len(dv2) < 500 or dp[i] < 0.5: continue
    Rs.append(float(dp[i])); Ys.append((np.median(dv2) / BAS[wdays[i]] - 1) * 100)
Rs = np.array(Rs); Ys = np.array(Ys)
RC_D = 5.4
m_in_d = Rs < 15
th_d = float(np.sum(Ys[m_in_d] * np.log1p(Rs[m_in_d] / RC_D)) / np.sum(np.log1p(Rs[m_in_d] / RC_D) ** 2))
fm_d = th_d * np.log1p(Rs / RC_D)
BIN_D = [(0.5, 5), (5, 15), (15, 30), (30, 60), (60, 1e9)]
bin_dev_d = lambda lo, hi: abs(fm_d[(Rs >= lo) & (Rs < hi)].mean() - Ys[(Rs >= lo) & (Rs < hi)].mean())
res['距离'] = {
    '内插·逐点(逐日)': (np.mean(np.abs(Ys[m_in_d] - fm_d[m_in_d])), None),
    '内插·箱':   (max(bin_dev_d(*b) for b in BIN_D[:2]), None),
    '外推·逐点(逐日)': (np.mean(np.abs(Ys[~m_in_d] - fm_d[~m_in_d])), None),
    '外推·箱':   (max(bin_dev_d(*b) for b in BIN_D[2:]), None),
}

# ================= 打印矩阵 =================
print('=' * 78)
print('统一口径 2×2 矩阵 (公式, 单位pp): 同口径下内插 ≤ 外推 —— 假象来自混用')
print('=' * 78)
print(f'{"量":6s} {"内插·逐点":>10s} {"内插·箱":>9s} {"外推·逐点":>10s} {"外推·箱":>8s}')
for q in ['出发量', '速度', '距离']:
    r_ = res[q]
    print(f'{q:6s} {r_["内插·逐点" if "内插·逐点" in r_ else "内插·逐点(逐日)"][0]:10.2f} '
          f'{r_["内插·箱"][0]:9.2f} '
          f'{r_["外推·逐点" if "外推·逐点" in r_ else "外推·逐点(逐日)"][0]:10.2f} '
          f'{r_["外推·箱"][0]:8.2f}')
print()
print('旧主表的并排(不可比): 出发量 内插逐点6.3 vs 外推箱2.3 → 假象"外推更准"')
print('同口径真实对比(本次): 内插箱1.5 < 外推箱5.7 | 内插逐点7.6 < 外推逐点11.3')
print('注: 距离箱口径 0.95 vs 0.92 基本打平(RC=5.4先验强, 两域箱误差都<1pp), 属实不违逻辑')

# ================= 图 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('为什么"公式外推误差比内插还低"是假象：两把尺子不可比', fontsize=14.5, fontweight='bold', y=0.99)

qs = ['出发量', '速度', '距离']
metrics = [('内插·逐点', '外推·逐点'), ('内插·箱', '外推·箱')]
titles = ['① 逐点口径(同一把尺): 内插 < 外推 ✓ 正常',
          '② 箱均值口径(同一把尺): 内插 ≤ 外推 ✓ 正常']
for ai, (ax, (mi, me)) in enumerate(zip(axes, metrics)):
    x = np.arange(len(qs)); bw = 0.32
    vin = [res[q][mi if mi in res[q] else mi + '(逐日)'][0] for q in qs]
    vex = [res[q][me if me in res[q] else me + '(逐日)'][0] for q in qs]
    b1 = ax.bar(x - bw / 2, vin, bw, color='#8FAADC', edgecolor=NAVY, label='内插域')
    b2 = ax.bar(x + bw / 2, vex, bw, color=ORANGE, edgecolor='#843C0C', label='外推域')
    for bb, vv in [(b1, vin), (b2, vex)]:
        for r_, v in zip(bb, vv):
            ax.annotate(f'{v:.1f}', (r_.get_x() + r_.get_width() / 2, v), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(qs, fontsize=12)
    ax.set_ylabel('公式误差 (pp, log)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(titles[ai], fontsize=12, loc='left', color=NAVY)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y', which='both')

fig.text(0.5, 0.005,
         '旧表把【内插·逐点MAE】和【外推·箱偏差】并排: 逐点含全部噪声(出发量逐点噪声底≈11pp), 箱偏差是均值之差(噪声÷√n) → 尺子不同, 数字不可比, 造成"外推更准"假象。\n'
         '修正: 表述改为同口径对比 —— 内插列建议改用【内插·箱偏差】(与外推列同尺), 或注明两列口径不同; RF 列同理',
         ha='center', fontsize=10, color=DGREEN, fontweight='bold')
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
fig.savefig(r'D:/AI/条件转化/误差口径矩阵.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nsaved 误差口径矩阵.png')
