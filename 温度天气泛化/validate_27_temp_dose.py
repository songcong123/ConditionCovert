# -*- coding: utf-8 -*-
"""温度维度的剂量曲线标定(真实数据, 出发量/距离/速度):
   出发量乘子(T) = exp(−θ·ln(1+|T−T_opt|/RC))   热/冷两侧分开拟合
   混杂控制: 只用干日 + 基线按(同年×日型×同钟点×舒适温段16~24°C)取中位数 → 剥掉年份增长/时段/日型
   留出验证: 热侧只拟合 [24,32)°C, 预测 ≥32°C 热浪档
数据: .flow_agg.npz(hour_cnt/dist) + .flow_spd.npz(day,hour,v) + ISD 94728 TMP(2024/2025)
输出: .flow_temp_dose.json + 温度剂量曲线.png
"""
import numpy as np, json
from datetime import datetime, timedelta
import pandas as pd
from scipy import optimize, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 1) 逐时温度 (ISD FM-15, TMP 为十分之一°C, 9999=缺测; UTC→EDT=UTC-4) ----------
def load_temps(years=(2024, 2025)):
    out = {}
    for yr in years:
        df = pd.read_csv(f'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                         usecols=['DATE', 'REPORT_TYPE', 'TMP'], dtype=str)
        fm15 = df[df['REPORT_TYPE'] == 'FM-15']
        for s, tmp in zip(fm15['DATE'], fm15['TMP']):
            if not isinstance(tmp, str) or ',' not in tmp: continue
            v, q = tmp.split(',')
            try: v = float(v) / 10.0
            except: continue
            if v < -40 or v > 55: continue
            utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
            loc = utc - timedelta(hours=4)          # 窗口内全为 EDT
            key = (loc.strftime('%Y-%m-%d'), loc.hour)
            out.setdefault(key, []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}

temps = load_temps()
print(f'温度序列: {len(temps)} 个(日,时)点')

# ---------- 2) 对齐框架数据 ----------
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load('D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); hc = A['hour_cnt'].astype(np.float64)
klmap = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klmap[d] for d in dates])
n = len(dates)
dow = np.array([(np.datetime64(d, 'D').view('int64') + 3) % 7 for d in dates])  # 0=周一
wk = (dow < 5).astype(int)                       # 0=周末 1=工作日
yr = np.array([int(str(d)[:4]) for d in dates])
temp_h = np.full((n, 24), np.nan)
for i, d in enumerate(dates):
    for h in range(24):
        temp_h[i, h] = temps.get((d, h), np.nan)
dry = kl == 'dry'
T_OPT, COMFORT = 21.0, (16.0, 25.0)              # 舒适温段(基准), 最优温度网格搜索初值

dist = A['dist'].astype(np.float64); dist_day = A['dist_day'].astype(int)

# ---------- 3) 出发量: 逐时比值 vs 温度 (干日) ----------
base = {}   # (yr, wk, h) → 舒适段中位数
ratios = []  # (T, ratio)
for i in range(n):
    if not dry[i]: continue
    for h in range(24):
        t = temp_h[i, h]
        if np.isnan(t) or hc[i, h] <= 0: continue
        key = (yr[i], wk[i], h)
        if COMFORT[0] <= t <= COMFORT[1]:
            base.setdefault(key, []).append(hc[i, h])
        else:
            ratios.append((t, key, hc[i, h]))
basemap = {k: float(np.median(v)) for k, v in base.items() if len(v) >= 3}
ratios = [(t, hc / basemap[k]) for t, k, hc in ratios if k in basemap and basemap[k] > 0]
T = np.array([r[0] for r in ratios]); RT = np.array([r[1] for r in ratios])
hot = T > COMFORT[1]; cold = T < COMFORT[0]
x_hot = T[hot] - COMFORT[1]; x_cold = COMFORT[0] - T[cold]
print(f'样本: 热侧 {hot.sum()} | 冷侧 {cold.sum()} | 舒适段基准点 {sum(len(v) for v in base.values())}')

def dose(x, th, RC):
    return np.exp(-th * np.log1p(np.maximum(x, 0) / RC))

def fit_side(x, y, form):
    bins = [(0, 3), (3, 6), (6, 9), (9, 100)] if form == 'hot' else [(0, 3), (3, 6), (6, 12), (12, 100)]
    bx, by, bn = [], [], []
    for lo, hi in bins:
        m = (x >= lo) & (x < hi)
        if m.sum() >= 30:
            bx.append((lo + hi) / 2); by.append(float(np.median(y[m]))); bn.append(int(m.sum()))
    f = lambda xx, th, RC: dose(xx, th, RC)
    popt, _ = optimize.curve_fit(f, np.array(bx), np.array(by), p0=[0.1, 10],
                                 bounds=([0.0, 0.5], [0.6, 200]), maxfev=20000)
    return f, popt, np.array(bx), np.array(by), np.array(bn)

f_hot, p_hot, bx_h, by_h, bn_h = fit_side(x_hot, RT[hot], 'hot')
hot_all = dose(x_hot, *p_hot)
r2h = 1 - np.sum((RT[hot] - hot_all) ** 2) / np.sum((RT[hot] - RT[hot].mean()) ** 2)
# 留出: 只用 [0,8)°C 偏离 拟合, 预测 ≥8°C (即 ≥33°C)
m_hold = x_hot < 8
fh_popt, _ = optimize.curve_fit(lambda xx, th, RC: dose(xx, th, RC), x_hot[m_hold], RT[hot][m_hold],
                                p0=p_hot, bounds=([0.0, 0.5], [0.6, 200]), maxfev=20000)
ph = fh_popt
mask_e = x_hot >= 8
actual_e = float(np.median(RT[hot][mask_e])) if mask_e.any() else float('nan')
pred_e = float(np.median(dose(x_hot[mask_e], *ph))) if mask_e.any() else float('nan')
err_h = abs(pred_e - actual_e) / abs(1 - actual_e) * 100 if actual_e != 1 else 0
print('\n=== 出发量温度剂量(热侧) ===')
print(f'乘子(T) = exp(−{p_hot[0]:.4f}·ln(1+(T−25)/{p_hot[1]:.1f}))   R²={r2h:.3f}')
for xx, yy, nn in zip(bx_h, by_h, bn_h):
    print(f'  偏离档 {xx:4.1f}°C: 中位乘子 {yy:.3f} (n={nn})')
print(f'留出验证: 前2档拟合 → 热浪档(≥33°C) 预测 {pred_e:.3f} vs 实际 {actual_e:.3f} (对降幅的偏差 {err_h:.0f}%)')

f_cold, p_cold, bx_c, by_c, bn_c = (None, None, [], [], [])
if cold.sum() >= 60:
    f_cold, p_cold, bx_c, by_c, bn_c = fit_side(x_cold, RT[cold], 'cold')
    r2c = 1 - np.sum((RT[cold] - dose(x_cold, *p_cold)) ** 2) / np.sum((RT[cold] - RT[cold].mean()) ** 2)
    print(f'冷侧: 乘子 = exp(−{p_cold[0]:.4f}·ln(1+(25−T)/{p_cold[1]:.1f}))   R²={r2c:.3f}')
    for xx, yy, nn in zip(bx_c, by_c, bn_c):
        print(f'  偏离档 {xx:4.1f}°C: 中位乘子 {yy:.3f} (n={nn})')

# ---------- 4) 距离 (日级, 干日中位数) ----------
dd = np.zeros(n); 
for i in np.where(dry)[0]:
    s = dist[dist_day == i]
    if s.size > 500: dd[i] = float(np.median(s))
dmax = np.full(n, np.nan)
for i, d in enumerate(dates):
    tt = [temps.get((d, h)) for h in range(24) if not np.isnan(temp_h[i, h])]
    if tt: dmax[i] = max(tt)
good = dry & ~np.isnan(dmax) & (dd > 0)
# 距离基线: 温和日(dmax 20~28) 同年同日型中位
dist_ratio = []
dbase = {}
for i in np.where(good & (dmax >= 20) & (dmax <= 28))[0]:
    dbase.setdefault((yr[i], wk[i]), []).append(dd[i])
dmap = {k: float(np.median(v)) for k, v in dbase.items() if len(v) >= 5}
res_d = []
for i in np.where(good)[0]:
    k = (yr[i], wk[i])
    if k in dmap and dmap[k] > 0 and not (20 <= dmax[i] <= 28):
        res_d.append((dmax[i], dd[i] / dmap[k]))
res_d.sort()
res_d = np.array(res_d)
print('\n=== 距离热响应(日级, 中位里程比) ===')
for lo, hi in [(12, 20), (20, 28), (28, 32), (32, 50)]:
    m = (res_d[:, 0] >= lo) & (res_d[:, 0] < hi)
    if m.sum() >= 5:
        print(f'  日最高温 [{lo},{hi})°C: 中位比 {np.median(res_d[m, 1]):.3f} (n={m.sum()})')

# ---------- 5) 速度 (小时级, 干日中位) ----------
spd_med = {}
dh = S['day'].astype(int); hh = S['hour'].astype(int); vv = S['v']
vmin, vmax = 1.0, 60.0
ok_v = (vv > vmin) & (vv < vmax)
for i in np.where(dry)[0]:
    for h in range(24):
        m = ok_v & (dh == i) & (hh == h)
        if m.sum() >= 100:
            spd_med[(i, h)] = float(np.median(vv[m]))
spd_ratio = []
sbase = {}
for (i, h), v in spd_med.items():
    t = temp_h[i, h]
    if np.isnan(t): continue
    key = (yr[i], wk[i], h)
    if COMFORT[0] <= t <= COMFORT[1]:
        sbase.setdefault(key, []).append(v)
    else:
        spd_ratio.append((t, key, v))
smap = {k: float(np.median(v)) for k, v in sbase.items() if len(v) >= 3}
spd_ratio = [(t, v / smap[k]) for t, k, v in spd_ratio if k in smap and smap[k] > 0]
Ts = np.array([r[0] for r in spd_ratio]); RS = np.array([r[1] for r in spd_ratio])
hs = Ts > COMFORT[1]
print('\n=== 速度热响应(小时级, 中位比) ===')
for lo, hi in [(0, 3), (3, 6), (6, 100)]:
    m = hs & ((Ts - COMFORT[1]) >= lo) & ((Ts - COMFORT[1]) < hi)
    if m.sum() >= 30:
        print(f'  偏离档 [{lo},{hi})°C: 中位速度比 {np.median(RS[m]):.3f} (n={m.sum()})')

# ---------- 6) 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle('温度维度的剂量响应（干日, 2024-04~2025-10 暖季, 同年×日型×同钟点基线, 已排雨）',
             fontsize=14, fontweight='bold', y=1.02)
ax = axes[0]
ax.scatter(T[hot], RT[hot], s=10, alpha=0.25, color='#C00000', label='热侧样本')
ax.scatter(T[cold], RT[cold], s=10, alpha=0.25, color='#4472C4', label='冷侧样本')
tf = np.linspace(25, 40, 100)
ax.plot(tf, dose(tf - COMFORT[1], *p_hot), lw=2.4, color='#C00000', label='热侧拟合')
if p_cold is not None:
    tc = np.linspace(2, 16, 100)
    ax.plot(tc, dose(COMFORT[0] - tc, *p_cold), lw=2.4, color='#4472C4', label='冷侧拟合')
ax.axhline(1, color='#999', lw=1, ls=':')
ax.axvspan(COMFORT[0], COMFORT[1], color='#70AD47', alpha=0.10)
ax.set_xlabel('小时气温 (°C)'); ax.set_ylabel('出发量乘子（÷舒适段基线）')
ax.set_title(f'① 出发量：热侧 exp(−{p_hot[0]:.3f}·ln(1+x/{p_hot[1]:.0f})) R²={r2h:.2f}', fontsize=11.5, loc='left')
ax.set_ylim(0.3, 2.2); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc='upper right')

ax = axes[1]
if len(res_d):
    ax.scatter(res_d[:, 0], res_d[:, 1], s=16, alpha=0.5, color='#70AD47')
    for lo, hi in [(12, 20), (20, 28), (28, 32), (32, 50)]:
        m = (res_d[:, 0] >= lo) & (res_d[:, 0] < hi)
        if m.sum() >= 5:
            ax.scatter([(lo + hi) / 2], [np.median(res_d[m, 1])], s=130, marker='D', color='#C00000', zorder=5)
            ax.annotate(f'{np.median(res_d[m, 1]):.2f}', ((lo + hi) / 2, np.median(res_d[m, 1]) + 0.01),
                        ha='center', fontsize=9, color='#C00000')
ax.axhline(1, color='#999', lw=1, ls=':')
ax.set_xlabel('日最高温 (°C)'); ax.set_ylabel('日中位里程比（÷温和日基线）')
ax.set_title('② 距离：热浪日里程变化', fontsize=11.5, loc='left'); ax.grid(alpha=0.3)

ax = axes[2]
ax.scatter(Ts[hs], RS[hs], s=10, alpha=0.25, color='#ED7D31', label='热侧样本')
ax.axhline(1, color='#999', lw=1, ls=':')
for lo, hi in [(0, 3), (3, 6), (6, 100)]:
    m = hs & ((Ts - COMFORT[1]) >= lo) & ((Ts - COMFORT[1]) < hi)
    if m.sum() >= 30:
        ax.scatter([COMFORT[1] + (lo + hi) / 2], [np.median(RS[m])], s=130, marker='D', color='#C00000', zorder=5)
        ax.annotate(f'{np.median(RS[m]):.3f}', (COMFORT[1] + (lo + hi) / 2, np.median(RS[m]) + 0.005),
                    ha='center', fontsize=9, color='#C00000')
ax.set_xlabel('小时气温 (°C)'); ax.set_ylabel('速度中位比（÷舒适段基线）')
ax.set_title('③ 速度：热浪小时变化', fontsize=11.5, loc='left'); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('D:/AI/条件转化/温度剂量曲线.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 温度剂量曲线.png')

json.dump(dict(th_hot=round(float(p_hot[0]), 4), RC_hot=round(float(p_hot[1]), 1), r2_hot=round(float(r2h), 3),
               th_cold=(round(float(p_cold[0]), 4), round(float(p_cold[1]), 1)) if p_cold is not None else None,
               holdout=dict(pred=pred_e, actual=actual_e, err_pct=round(float(err_h), 1)),
               n_hot=int(hot.sum()), n_cold=int(cold.sum())),
          open('D:/AI/条件转化/.flow_temp_dose.json', 'w'), ensure_ascii=False, indent=1)
print('saved .flow_temp_dose.json')
