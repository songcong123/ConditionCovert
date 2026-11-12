# -*- coding: utf-8 -*-
"""温度剂量曲线（简化重写版）:
   y = 该(日,时)出发量 ÷ 同(年,日型,钟点)舒适段基线
   绘图: 散点(温度, 乘子) + 分位标记 + 阈值线
"""
import numpy as np, json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; OR = '#ED7D31'; MUTED = '#666666'
DGREEN = '#375623'; GRNB = '#70AD47'

# ---------- 1) 逐时温度 ----------
def load_temps():
    out = {}
    for yr in [2024, 2025]:
        df = pd.read_csv(rf'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                         usecols=['DATE', 'REPORT_TYPE', 'TMP'], dtype=str)
        fm15 = df[df['REPORT_TYPE'] == 'FM-15']
        for s, tmp in zip(fm15['DATE'], fm15['TMP']):
            if not isinstance(tmp, str) or ',' not in tmp: continue
            v = float(tmp.split(',')[0]) / 10.0
            if v < -40 or v > 55: continue
            utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
            loc = utc - timedelta(hours=4)
            out.setdefault((loc.strftime('%Y-%m-%d'), loc.hour), []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}

import pandas as pd
temps = load_temps()

# ---------- 2) 逐时出发量 ----------
A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates']); hc = A['hour_cnt'].astype(np.float64)
klmap = dict(zip(list(D['dates']), D['klass'].tolist()))
kl_w = np.array([klmap[d] for d in dates]); nw = len(dates)
yr_w = np.array([int(str(d)[:4]) for d in dates])
dow_w = np.array([(np.datetime64(d, 'D').view('int64') + 3) % 7 for d in dates])
dry = kl_w == 'dry'
temp_h = np.full((nw, 24), np.nan)
for i, d in enumerate(dates):
    for h in range(24):
        temp_h[i, h] = temps.get((d, h), np.nan)

# ---------- 3) 构建(温度, 乘子)样本 ----------
COMF = (16, 25)
# 舒适段基线: 按(年,日型,钟点)取中位
comfort_base = np.full((2, 2, 24), np.nan)
for yi in range(2):
    for w in range(2):
        for h in range(24):
            sel = np.where((yr_w == [2024,2025][yi]) & (dow_w == w) & dry &
                           (temp_h[:, h] >= COMF[0]) & (temp_h[:, h] <= COMF[1]))[0]
            vals = hour_cnt[sel, h]
            vals = vals[vals > 0]
            if len(vals) >= 3: comfort_base[yi, w, h] = np.median(vals)

# 每个干日干时: (温度, 乘子) 样本
samples = []
for i in range(nw):
    if not dry[i]: continue
    for h in range(24):
        t = temp_h[i, h]; c = hour_cnt[i, h]
        if np.isnan(t) or c <= 0: continue
        b = comfort_base[yi_map[i], wk_map[i], h]
        if np.isnan(b) or b <= 0: continue
        samples.append((t, c / b))

samples = np.array(samples)
T_arr = samples[:, 0]; M_arr = samples[:, 1]
print(f'温度×乘子样本: {len(T_arr)} 个')

# ---------- 4) 绘图 ----------
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.suptitle('温度剂量曲线（干日逐时，÷ 同钟点舒适段基线）', fontsize=13.5, fontweight='bold', y=0.98)

# 舒适段阴影
ax.axvspan(COMF[0], COMF[1], color='#E2F0D9', alpha=0.3, zorder=0)
ax.text(0.5 * (COMF[0] + COMF[1]), 1.35, '舒适段', ha='center', fontsize=10, color='#8FAADC', fontweight='bold')

# 冷侧散点
cold = (T_arr >= 0) & (T_arr < COMF[0])
ax.scatter(T_arr[cold], M_arr[cold], s=14, alpha=0.4, color='#4472C4', label=f'冷侧 <{COMF[0]:.0f}°C（n={cold.sum()}）', zorder=3)
# 舒适段散点
comf = (T_arr >= COMF[0]) & (T_arr <= COMF[1])
ax.scatter(T_arr[comf], M_arr[comf], s=14, alpha=0.4, color='#8FAADC', label=f'舒适段 {COMF[0]:.0f}~{COMF[1]:.0f}°C（n={comf.sum()}）', zorder=3)
# 偏热散点
hotm = (T_arr > COMF[1])
ax.scatter(T_arr[hotm], M_arr[hotm], s=14, alpha=0.4, color=OR, label=f'偏热 >{COMF[1]:.0f}°C（n={hotm.sum()}）', zorder=3)

# 参考线 y=1
ax.axhline(1, color='#999', lw=1.2, ls=':', zorder=1)
ax.text(0.02, 1.02, 'y = 1（无效应）', fontsize=9, color=MUTED, transform=ax.get_yaxis_transform())

# 均值趋势线（按温度分箱）
t_bins = np.arange(0, 42, 2)
t_centers, t_meds = [], []
for i in range(len(t_bins) - 1):
    m = (T_arr >= t_bins[i]) & (T_arr < t_bins[i + 1])
    if m.sum() >= 5:
        t_centers.append((t_bins[i] + t_bins[i + 1]) / 2)
        t_meds.append(np.median(M_arr[m]))
if t_centers:
    ax.plot(t_centers, t_meds, lw=2.5, color=NAVY, marker='o', ms=7, label='分箱中位（趋势）', zorder=5)

# 热浪阈值线
ax.axvline(33, color=RED, lw=1.5, ls='--', zorder=2)
ax.text(33.3, 1.32, '33°C', fontsize=10, color=RED, fontweight='bold')

ax.set_xlabel('小时气温 T (°C)', fontsize=12)
ax.set_ylabel('出发量乘子（÷ 同钟点舒适段基线）', fontsize=12)
ax.set_xlim(0, 42); ax.set_ylim(0.5, 1.6)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(alpha=0.3)

fig.text(0.5, 0.005,
         '读法：温度在舒适段内时乘子 ≈1.0（无效应）；偏离舒适段时乘子随偏离开头变化——如果冷侧/热侧的乘子都 <1，说明极端温度压制出行',
         ha='center', fontsize=10, color=DGREEN, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(r'D:/AI/条件转化/温度天气泛化/温度剂量曲线.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 温度剂量曲线.png')
