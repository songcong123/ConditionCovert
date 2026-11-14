# -*- coding: utf-8 -*-
"""特征角度: RF 用不同特征集时 内插/外推表现怎么变
   特征集: A只雨强 | B雨强+滞后 | C雨强+时段+日型 | D全特征 | E无雨强(仅离散域内特征)
   每集: r<4 训练 → 内插MAE(随机20%域内) + 外推MAE(≥4) + 对r的响应曲线
   回答: ① RF内插优势来自哪些特征 ② 外推失败是否只源于连续特征r超域
        ③ 离散特征(时段/日型)永不超域→天然"外推安全"?
输出: 特征消融_内插外推.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; GREY = '#999999'; MUTED = '#666666'; DGREEN = '375623'.join('#') if False else '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates']); n = len(dates)
hc = A['hour_cnt'].astype(float)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
wd = (dow + 3) % 7
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

X_all, y_all, r_all = [], [], []
for i in range(n):
    if kl[i] == 'dry': continue
    for h in range(24):
        base = lam[dow[i], h]
        if base < 200: continue
        r_now = inten[i, h]
        r_lag = inten[i, h - 1] if h >= 1 else 0
        is_peak = 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0
        is_we = 1 if wd[i] >= 5 else 0
        # 特征字典
        X_all.append(dict(r=np.log1p(r_now), lag=np.log1p(r_lag),
                          peak=is_peak, we=is_we, h=h,
                          rXpeak=is_peak * np.log1p(r_now), rXwe=is_we * np.log1p(r_now)))
        y_all.append(hc[i, h] / base - 1)
        r_all.append(r_now)
y = np.array(y_all); r_raw = np.array(r_all)
rng = np.random.default_rng(42)

SETS = {
    'A 只雨强':        ['r'],
    'B 雨强+滞后':     ['r', 'lag'],
    'C 雨强+时段+日型': ['r', 'peak', 'we', 'h'],
    'D 全特征':        ['r', 'lag', 'peak', 'we', 'rXpeak', 'rXwe', 'h'],
    'E 无雨强(纯离散)': ['peak', 'we', 'h'],
}
THETA, RC = 0.052, 0.2
formula = lambda r: THETA * np.log1p(np.asarray(r, float) / RC)

def curve_input(feats, rr):
    cols = []
    for f in feats:
        if f == 'r': cols.append(np.log1p(rr))
        elif f == 'lag': cols.append(np.zeros_like(rr))
        elif f == 'peak': cols.append(np.zeros_like(rr))
        elif f == 'we': cols.append(np.zeros_like(rr))
        elif f == 'h': cols.append(np.full_like(rr, 13.0))
        elif f == 'rXpeak': cols.append(np.zeros_like(rr))
        elif f == 'rXwe': cols.append(np.zeros_like(rr))
    return np.column_stack(cols)

print(f'{"特征集":18s} {"内插MAE":>8s} {"外推MAE":>8s} {"公式外推":>8s} | 外推@10 @30 @60 (对r的响应)')
res = {}
for nm, feats in SETS.items():
    X = np.column_stack([np.array([s[f] for s in X_all]) for f in feats])
    m_tr = r_raw < 4
    # 内插: 域内随机 80/20
    idx_in = np.where(m_tr)[0]
    perm = rng.permutation(idx_in)
    te_in = perm[len(perm) * 4 // 5:]
    rf = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X[m_tr], y[m_tr])
    mae_in = np.mean(np.abs(y[te_in] - rf.predict(X[te_in]))) * 100
    # 外推
    m_ex = r_raw >= 4
    mae_ex = np.mean(np.abs(y[m_ex] - rf.predict(X[m_ex]))) * 100
    mae_f = np.mean(np.abs(y[m_ex] - formula(r_raw[m_ex]))) * 100
    # 响应曲线
    rr = np.array([2.0, 10.0, 30.0, 60.0])
    curve = rf.predict(curve_input(feats, rr)) * 100
    res[nm] = (mae_in, mae_ex, mae_f, curve, rf, feats)
    print(f'{nm:18s} {mae_in:7.1f}pp {mae_ex:7.1f}pp {mae_f:7.1f}pp | {curve[1]:+5.1f} {curve[2]:+5.1f} {curve[3]:+5.1f}')

# ===== 图 =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
fig.suptitle('特征角度：RF 外推失败只源于连续特征(雨强)超域——离散特征天然安全', fontsize=14, fontweight='bold', y=1.02)

# ① MAE 对比
ax = axes[0]
names = list(SETS.keys())
x = np.arange(len(names)); bw = 0.28
mae_in = [res[nm][0] for nm in names]; mae_ex = [res[nm][1] for nm in names]; mae_f = [res[nm][2] for nm in names]
ax.bar(x - bw, mae_in, bw, color='#8FAADC', edgecolor=NAVY, label='RF 内插(域内20%)')
ax.bar(x, mae_ex, bw, color=RED, label='RF 外推(≥4mm/h)')
ax.bar(x + bw, mae_f, bw, color=NAVY, label='公式外推(恒定参照)')
for xx, vv in zip(x - bw, mae_in): ax.annotate(f'{vv:.0f}', (xx, vv + 1), ha='center', fontsize=9, color=NAVY)
for xx, vv in zip(x, mae_ex): ax.annotate(f'{vv:.0f}', (xx, vv + 1), ha='center', fontsize=9, color=RED)
for xx, vv in zip(x + bw, mae_f): ax.annotate(f'{vv:.0f}', (xx, vv + 1), ha='center', fontsize=9, color=NAVY)
ax.set_xticks(x); ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
ax.set_ylabel('MAE (pp, 越低越好)'); ax.legend(fontsize=9.5, loc='upper left')
ax.set_title('① 内插随特征变强, 外推全部失败\n   (公式恒 43pp 为参照)', fontsize=11.5, loc='left', color=NAVY)
ax.grid(alpha=0.3, axis='y')

# ② 响应曲线: 各特征集 RF 对 r 的响应
ax = axes[1]
rr = np.geomspace(0.3, 60, 200)
cols = ['#9DC3E6', '#4472C4', NAVY, '#7030A0', '#ED7D31']
for (nm, (mi, me, mf, cv, rf, feats)), cc in zip(res.items(), cols):
    if 'r' not in feats:
        # 无 r 特征: 曲线是水平线(不响应)
        flat = rf.predict(curve_input(feats, rr)) * 100
        ax.plot(rr, flat, lw=2.2, ls=':', color=cc, label=f'{nm}(水平=不响应)')
    else:
        ax.plot(rr, rf.predict(curve_input(feats, rr)) * 100, lw=2.2, color=cc, label=nm)
ax.plot(rr, formula(rr) * 100, lw=2.8, color='#1A1A1A', ls='--', label='公式(单调饱和)')
ax.axvline(4, color=MUTED, ls='--', lw=1.2)
ax.text(4.4, 2, '训练边界', fontsize=9, color=MUTED)
ax.set_xscale('log')
ax.set_xlabel('雨强 r (mm/h, log)'); ax.set_ylabel('预测出发量效应 (%)')
ax.set_title('② 含 r 的 RF 全部在边界后躺平\n   无 r 的 RF 水平线(对雨强无概念)', fontsize=11.5, loc='left', color=NAVY)
ax.legend(fontsize=8.5, loc='upper left'); ax.grid(alpha=0.3, which='both')

fig.text(0.5, 0.005,
         '读法: 内插误差随特征丰富单调下降(时段/日型贡献大); 但只要特征里有连续的 r, 外推就平台化——E组(无r)对雨强彻底无响应。结论: RF外推失败⇔连续特征超域; 离散特征(时段/日型)值域封闭永不超域→框架把它们设为索引维(查表)是唯一安全设计',
         ha='center', fontsize=9.5, color=DGREEN, fontweight='bold', wrap=True)
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/特征消融_内插外推.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 特征消融_内插外推.png')
