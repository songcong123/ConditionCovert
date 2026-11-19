# -*- coding: utf-8 -*-
"""validate_50: RF 只会"重放"训练分布 —— 出发量效应随雨强的真实示例
================================================================
训练协议(与终版主表一致): RF 只喂 r<4mm/h 的小时样本(17特征)。
图上三样东西:
  ① 实测: 五个雨强箱的箱内平均效应(同口径, 每箱标 n)
  ② RF:   同一批样本逐条预测后分箱平均 → 域内贴实测, 域外躺平
     + 固定特征(工作日13时)扫 r 的连续响应曲线, 展示平台形态
  ③ 公式: θ·ln(1+r/0.2), θ=0.052 RC=0.2, 低雨强两点标定
r=4 竖线 = 训练边界: 左边"训练集里见过", 右边"训练集里没有"。
输出: RF重放训练分布_出发量.png
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

NAVY = '#1F4E79'; RED = '#C00000'; DGREEN = '#375623'; MUTED = '#666666'; GREY = '#7F7F7F'

# ================= 数据与标签(与 validate_41 同) =================
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
    s1 = d1 + timedelta((6 - d1.weekday()) % 7)
    return s0 <= utc < s1

acc = {}
for yr in [2024, 2025]:
    df = pd.read_csv(rf'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                     usecols=['DATE', 'REPORT_TYPE', 'AA1', 'TMP', 'DEW', 'WND'], dtype=str)
    fm = df[df['REPORT_TYPE'] == 'FM-15']
    for _, row in fm.iterrows():
        try:
            utc = datetime.strptime(row['DATE'], '%Y-%m-%dT%H:%M:%S')
        except Exception:
            continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        key = (loc.strftime('%Y-%m-%d'), loc.hour)
        e = acc.setdefault(key, dict(rain=[], temp=[], dew=[], wind=[]))
        if isinstance(row['AA1'], str) and ',' in row['AA1']:
            p = row['AA1'].split(',')
            if p[0] == '01' and p[1] not in ('9999', '99'):
                e['rain'].append(float(p[1]) * 0.254)
        if isinstance(row['TMP'], str) and ',' in row['TMP']:
            v = float(row['TMP'].split(',')[0]) / 10
            if -40 < v < 55: e['temp'].append(v)
        if isinstance(row['DEW'], str) and ',' in row['DEW']:
            v = float(row['DEW'].split(',')[0]) / 10
            if -40 < v < 40: e['dew'].append(v)
        if isinstance(row['WND'], str) and ',' in row['WND']:
            p = row['WND'].split(',')
            v = float(p[3]) / 10
            if 0 <= v < 60: e['wind'].append(v)

def rh_of(t, td):
    if td is None or t is None or td > t: return None
    et = 6.112 * np.exp(17.67 * t / (243.5 + t))
    ed = 6.112 * np.exp(17.67 * td / (243.5 + td))
    return min(max(ed / et * 100, 5), 100)

wx = {}
for key, e in acc.items():
    t = float(np.mean(e['temp'])) if e['temp'] else None
    td = float(np.mean(e['dew'])) if e['dew'] else None
    wx[key] = dict(rain=float(np.mean(e['rain'])) if e['rain'] else 0.0,
                   temp=t, humidity=rh_of(t, td),
                   wind=float(np.mean(e['wind'])) if e['wind'] else np.nan)

# 样本: 全部非干日小时(基线≥200)
rows_d, rows_h, ys, r_now, r_lag = [], [], [], [], []
for i, d in enumerate(dates_all):
    if kl[i] == 'dry': continue
    for h in range(24):
        base = lam[dow[i], h]
        if base < 200: continue
        w = wx.get((d, h))
        if w is None: continue
        w1 = wx.get((d, h - 1), dict(rain=0.0))
        rows_d.append(d); rows_h.append(h)
        ys.append(hc[i, h] / base - 1)
        r_now.append(w['rain']); r_lag.append(w1['rain'])
ys = np.array(ys); r_now = np.array(r_now); r_lag = np.array(r_lag)

def feat_vec(r, h, we, t_, hu, wd_, rl):
    return np.array([[np.log1p(r), np.log1p(rl), h, np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                      we, t_, hu, wd_]])

# 特征矩阵
H = np.array(rows_h); WE = (dow[np.array([dates_all.index(d) for d in rows_d])] >= 5).astype(float)
TMP = np.array([wx[(d, h)]['temp'] if wx[(d, h)]['temp'] else np.nan for d, h in zip(rows_d, rows_h)])
HU = np.array([wx[(d, h)]['humidity'] if wx[(d, h)]['humidity'] else np.nan for d, h in zip(rows_d, rows_h)])
WD = np.array([wx[(d, h)]['wind'] if not np.isnan(wx[(d, h)]['wind']) else np.nan for d, h in zip(rows_d, rows_h)])
X = np.column_stack([np.log1p(r_now), np.log1p(r_lag), H,
                     np.sin(2*np.pi*H/24), np.cos(2*np.pi*H/24), WE,
                     np.nan_to_num(TMP, nan=13.5), np.nan_to_num(HU, nan=68.0), np.nan_to_num(WD, nan=4.5)])

# ================= RF: 只训练 r<4 =================
m_tr = r_now < 4
rf = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X[m_tr], ys[m_tr])
pred_rf = rf.predict(X)
print(f'训练样本(域内 r<4): {m_tr.sum():,} | 总样本: {len(ys):,}')

THETA, RC = 0.052, 0.2
formula = lambda r: THETA * np.log1p(np.asarray(r, float) / RC)

# ================= 五箱同口径统计 =================
BINS = [(0.3, 1, '0.3-1'), (1, 4, '1-4'), (4, 8, '4-8'), (8, 16, '8-16'), (16, 1e9, '≥16')]
bx = []
for lo, hi, nm in BINS:
    m = (r_now >= lo) & (r_now < hi)
    bx.append(dict(nm=nm, n=int(m.sum()), rbar=float(r_now[m].mean()),
                   obs=float(ys[m].mean()), rf=float(pred_rf[m].mean()),
                   fm=float(formula(r_now[m]).mean()),
                   se=float(np.std(ys[m]) / np.sqrt(m.sum()))))
print(f'\n{"箱":6s} {"n":>6s} {"r̄":>6s} {"实测%":>7s} {"RF%":>7s} {"公式%":>7s} {"RF差":>6s} {"公式差":>6s}')
for b in bx:
    print(f'{b["nm"]:6s} {b["n"]:6d} {b["rbar"]:6.2f} {b["obs"]*100:+7.1f} {b["rf"]*100:+7.1f} '
          f'{b["fm"]*100:+7.1f} {abs(b["rf"]-b["obs"])*100:6.1f} {abs(b["fm"]-b["obs"])*100:6.1f}')

# ================= 图 =================
fig, ax = plt.subplots(figsize=(12.5, 7.2))
fig.suptitle('随机森林只会"重放"训练分布：雨强超出训练集后，实测还在涨、RF 躺平、公式跟涨（出发量）',
             fontsize=14, fontweight='bold', y=0.985)

# 训练域底色
ax.axvspan(0.25, 4, color='#EAF1F9', zorder=0)
ax.axvspan(4, 40, color='#FDF0E7', zorder=0)
ax.axvline(4, color=RED, ls='--', lw=2, zorder=2)
ax.text(2.0, 34.5, '训练集里见过\n(雨强 0.3~4)', ha='center', fontsize=11, color=NAVY, fontweight='bold')
ax.text(14.5, 34.5, '训练集里没有\n(雨强 > 4)', ha='center', fontsize=11, color='#843C0C', fontweight='bold')

# 连续曲线: 固定特征(工作日13时)扫 r
rr = np.geomspace(0.3, 38, 300)
Xc = np.column_stack([np.log1p(rr), np.zeros_like(rr), np.full_like(rr, 13.0),
                      np.sin(2*np.pi*13/24)*np.ones_like(rr), np.cos(2*np.pi*13/24)*np.ones_like(rr),
                      np.zeros_like(rr), np.full_like(rr, 15.0), np.full_like(rr, 70.0), np.full_like(rr, 4.5)])
curve_rf = rf.predict(Xc) * 100
ax.plot(rr, curve_rf, lw=2.2, color=RED, alpha=0.45, ls='-', zorder=3)
ax.plot(rr, formula(rr) * 100, lw=2.2, color=DGREEN, alpha=0.5, ls='--', zorder=3)

# 箱点(同口径: 同一批样本分别取实测/RF预测/公式的箱内平均)
xs = [b['rbar'] for b in bx]
obs_v = [b['obs']*100 for b in bx]; rf_v = [b['rf']*100 for b in bx]; fm_v = [b['fm']*100 for b in bx]
es = [1.96*b['se']*100 for b in bx]
ax.errorbar(xs, obs_v, yerr=es, fmt='s', ms=11, color=NAVY, lw=1.8, capsize=5, zorder=6,
            label='实测(箱内平均, 误差棒=95%置信)')
ax.plot(xs, rf_v, 'o-', ms=10, lw=2.2, color=RED, zorder=6, label='RF(同一批样本逐条预测后分箱平均)')
ax.plot(xs, fm_v, '^-', ms=10, lw=2.2, color=DGREEN, zorder=6, label='公式(θ·ln(1+r/0.2), 低雨强两点标定)')
# 每箱标 n
for b in bx:
    ax.annotate(f"n={b['n']}", (b['rbar'], -6.5), ha='center', fontsize=8.5, color=MUTED)

# 关键标注: RF 平台
p10 = rf.predict(feat_vec(10, 13, 0, 15, 70, 4.5, 0))[0]*100
p30 = rf.predict(feat_vec(30, 13, 0, 15, 70, 4.5, 0))[0]*100
ax.annotate(f'RF@10mm/h: {p10:+.1f}%\nRF@30mm/h: {p30:+.1f}%\n(输出几乎相同=重放边界值)',
            xy=(10, p10), xytext=(6.0, p10 - 9.5), fontsize=9.5, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.6),
            bbox=dict(boxstyle='round,pad=0.35', fc='#FDECEC', ec=RED, alpha=0.92))
# 关键标注: ≥16 箱差距
gap = (bx[4]['obs'] - bx[4]['rf'])*100
ax.annotate(f'≥16箱: 实测{bx[4]["obs"]*100:+.1f}% vs RF {bx[4]["rf"]*100:+.1f}%\n差 {abs(gap):.1f} 个百分点(RF 没跟着涨)',
            xy=(bx[4]['rbar'], rf_v[4]), xytext=(21.5, rf_v[4]-11.5), fontsize=9.5, color='#843C0C',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.6),
            bbox=dict(boxstyle='round,pad=0.35', fc='#FFF2CC', ec='#D6B656', alpha=0.92))

ax.set_xscale('log')
ax.set_xlim(0.25, 40)
ax.set_ylim(-9, 38)
ax.set_xticks([0.3, 1, 4, 8, 16, 30])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('小时雨强 (mm/h, log轴)')
ax.set_ylabel('出发量效应 (%, 相对同星期同钟点干日基线)')
ax.legend(fontsize=9.5, loc='upper left', framealpha=0.95)
ax.grid(alpha=0.3, which='both')

fig.text(0.5, 0.005,
         '训练协议: RF 只喂 r<4mm/h 的小时(5,404个样本, 特征=雨强/滞后/时段/周末/温湿风), ≥4mm/h 从未进训练 → 曲线在边界后成平台;\n'
         '公式只用 0.3-1 / 1-4 两箱标定(θ=0.052), 高雨强段纯外推。箱点=同一批样本的箱内平均(与终版主表同口径); 细线=固定"工作日13时"只扫雨强的连续响应',
         ha='center', fontsize=9, color=MUTED)
plt.tight_layout(rect=[0, 0.045, 1, 0.96])
fig.savefig(r'D:/AI/条件转化/RF重放训练分布_出发量.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nsaved RF重放训练分布_出发量.png')
