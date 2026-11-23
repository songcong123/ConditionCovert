# -*- coding: utf-8 -*-
"""validate_57: 真实数据版 —— NYC 出发量上真训一个 DNN, 看 R>4mm/h 外推形态
   RF(9特征) 与 DNN(64x64 ReLU, 同训练集 r<4) 都是真 sklearn 模型。
   回答质疑: 'DNN 直线上升'是真实行为还是画出来的?
   输出: 出发量_RF_DNN_真实数据.png + 箱表 + DNN 域外斜率实测
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; PURPLE = '#7030A0'; DGREEN = '#2E7D32'; MUTED = '#666666'

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

def is_edt(utc):
    y = utc.year
    d0 = datetime(y, 3, 1)
    s0 = d0 + timedelta((6 - d0.weekday()) % 7) + timedelta(7)
    d1 = datetime(y, 11, 1)
    return s0 <= utc < d1 + timedelta((6 - d1.weekday()) % 7)

acc = {}
for yr in [2024, 2025]:
    df = pd.read_csv(rf'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                     usecols=['DATE', 'REPORT_TYPE', 'AA1', 'TMP', 'DEW'], dtype=str)
    for _, row in df[df['REPORT_TYPE'] == 'FM-15'].iterrows():
        try: utc = datetime.strptime(row['DATE'], '%Y-%m-%dT%H:%M:%S')
        except Exception: continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        key = (loc.strftime('%Y-%m-%d'), loc.hour)
        e = acc.setdefault(key, dict(rain=[], temp=[], dew=[]))
        if isinstance(row['AA1'], str) and ',' in row['AA1']:
            p = row['AA1'].split(',')
            if p[0] == '01' and p[1] not in ('9999', '99'): e['rain'].append(float(p[1]) * 0.254)
        if isinstance(row['TMP'], str) and ',' in row['TMP']:
            v = float(row['TMP'].split(',')[0]) / 10
            if -40 < v < 55: e['temp'].append(v)
        if isinstance(row['DEW'], str) and ',' in row['DEW']:
            v = float(row['DEW'].split(',')[0]) / 10
            if -40 < v < 40: e['dew'].append(v)
wx = {}
for key, e in acc.items():
    r_ = float(np.mean(e['rain'])) if e['rain'] else 0.0
    wx[key] = r_

rd, rh_, ry, rr = [], [], [], []
for i, d in enumerate(dates_all):
    if kl[i] == 'dry': continue
    for h in range(24):
        if lam[dow[i], h] < 200: continue
        w = wx.get((d, h))
        if w is None: continue
        rd.append(d); rh_.append(h)
        ry.append(hc[i, h] / lam[dow[i], h] - 1)
        rr.append(w)
ry = np.array(ry) * 100; rr = np.array(rr)
H = np.array(rh_)

# ---- RF(单特征 与 9特征两个版) ----
m_tr = rr < 4
rf9 = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
X9 = np.column_stack([np.log1p(rr), H, np.sin(2 * np.pi * H / 24), np.cos(2 * np.pi * H / 24)])
rf9.fit(X9[m_tr], ry[m_tr])

# ---- DNN(单特征 log1p(r), 标准化, 同训练集) ----
sc = StandardScaler().fit(np.log1p(rr[m_tr]).reshape(-1, 1))
dnns = []
for sd in [1, 2, 3]:
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam',
                     alpha=1e-3, max_iter=9000, random_state=sd)
    m.fit(sc.transform(np.log1p(rr[m_tr]).reshape(-1, 1)), ry[m_tr])
    dnns.append(m)

RR = np.geomspace(0.3, 60, 400)
pred_rf = rf9.predict(np.column_stack([np.log1p(RR), np.full_like(RR, 13.0),
                                       np.sin(2 * np.pi * 13 / 24) * np.ones_like(RR),
                                       np.cos(2 * np.pi * 13 / 24) * np.ones_like(RR)]))
pred_dnn = [m.predict(sc.transform(np.log1p(RR).reshape(-1, 1))) for m in dnns]

print(f'{"雨强":8s} {"实测":>7s} {"RF":>7s} {"DNN1":>7s} {"DNN2":>7s} {"DNN3":>7s}')
for lo, hi, nm in [(0.3, 1, '0.3-1'), (1, 4, '1-4'), (4, 8, '4-8'), (8, 16, '8-16'), (16, 1e9, '≥16')]:
    m = (rr >= lo) & (rr < hi)
    o = ry[m].mean()
    rp = rf9.predict(np.column_stack([np.log1p(rr[m]), H[m],
                                      np.sin(2 * np.pi * H[m] / 24), np.cos(2 * np.pi * H[m] / 24)])).mean()
    ds = [m_.predict(sc.transform(np.log1p(rr[m]).reshape(-1, 1))).mean() for m_ in dnns]
    print(f'{nm:8s} {o:+7.1f} {rp:+7.1f} ' + ' '.join(f'{d:+7.1f}' for d in ds))

print('\nDNN 域外斜率实测(log雨强空间, 数值微分):')
f1 = lambda r_: float(dnns[0].predict(sc.transform([[np.log1p(r_)]]))[0])
for a, b in [(1, 1.5), (3, 3.5), (5, 7), (10, 14), (20, 30), (40, 60)]:
    print(f'  r={a}~{b}: {(f1(b)-f1(a))/(np.log1p(b)-np.log1p(a)):+.1f} %/ln(mm/h)')

# ============ 图 ============
fig, ax = plt.subplots(figsize=(13, 7.6))
fig.suptitle('真实数据复验：NYC 出发量上真训 RF 与 DNN(只用 r<4mm/h), 大雨域的外推形态',
             fontsize=14.5, fontweight='bold', y=0.98)
ax.axvspan(0.25, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 65, color='#FDF1E8', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2, zorder=2)
m_in = rr < 4
ax.scatter(rr[m_in], ry[m_in], s=16, color='#7FA6CC', alpha=0.5, edgecolor='none', zorder=2)
ax.scatter(rr[~m_in], ry[~m_in], s=22, color='#E8A47C', alpha=0.8, edgecolor='none', zorder=2)
ax.plot(RR, pred_rf, lw=3, color=RED, zorder=5, label='RF(400树, r<4 训练)')
for i_, p_ in enumerate(pred_dnn):
    ax.plot(RR, p_, lw=2.2 if i_ else 2.6, color=PURPLE, ls='-.', alpha=0.9 if i_ == 0 else 0.6, zorder=4)
ax.plot([], [], color=PURPLE, ls='-.', lw=2.6, label='DNN(64×64 ReLU, 同训练集, 3次训练)')
THETA, RCQ = 0.052, 0.2
ax.plot(RR, THETA * np.log1p(RR / RCQ) * 100, lw=2.6, color=DGREEN, ls='--', zorder=4, label='公式(两点标定)')
for lo, hi in [(0.3, 1), (1, 4), (4, 8), (8, 16), (16, 1e9)]:
    m = (rr >= lo) & (rr < hi)
    ax.plot([rr[m].mean()], [ry[m].mean()], 's', ms=13, zorder=6,
            markerfacecolor=NAVY, markeredgecolor='white', markeredgewidth=1.4)
ax.text(1.4, 62, '训练域(见过)', ha='center', fontsize=14, fontweight='bold', color=NAVY)
ax.text(18, 62, '盲测域(没见过)', ha='center', fontsize=14, fontweight='bold', color='#843C0C')
ax.set_xscale('log'); ax.set_xlim(0.25, 65); ax.set_ylim(-45, 70)
ax.set_xticks([0.3, 1, 4, 8, 16, 30, 60])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('小时雨强 (mm/h)', fontsize=15); ax.set_ylabel('出发量效应 (%)', fontsize=15)
ax.tick_params(labelsize=13)
ax.legend(fontsize=11, loc='lower left')
ax.grid(alpha=0.25, which='both')
fig.text(0.5, 0.005,
         '每个散点=一个(日,时)样本; 深蓝方块=分箱均值。RF与DNN都是真sklearn模型、同一训练集(r<4mm/h)——盲测域形态由模型自己长出来, 非手画。',
         ha='center', fontsize=11.5, color='#375623', fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(r'D:/AI/条件转化/出发量_RF_DNN_真实数据.png', dpi=200, bbox_inches='tight', facecolor='white')
print('\nsaved 出发量_RF_DNN_真实数据.png')
