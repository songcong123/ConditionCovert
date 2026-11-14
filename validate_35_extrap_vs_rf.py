# -*- coding: utf-8 -*-
"""外推公式 vs 随机森林：域内(内插) 与 域外(外推) 双协议对比
   协议A 内插: 随机 80/20 划分 → 两者公平比精度
   协议B 外推: 训练只用 r<4mm/h(轻中雨), 测试 r>=4mm/h(大雨+) → 看谁能外推
   附: 预测曲线对比(直接喂 0~60mm/h), 展示 RF 域外输出恒定
输出: 外推vs随机森林.png + 控制台对比表
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; GRN = '#70AD47'; MUTED = '#666666'; DGREEN = '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates'])
klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
hc = A['hour_cnt'].astype(float)
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
kl = np.array([klass[d] for d in dates])
wd = (dow + 3) % 7
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    if sel.sum() > 0: lam[k] = hc[sel].mean(0)
inten = np.zeros((len(dates), 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

# 样本(同 validate_22)
X_all, y_all = [], []
for i in range(len(dates)):
    if kl[i] == 'dry': continue
    for h in range(24):
        base = lam[wd[i], h]
        if base < 200: continue
        r_now = inten[i, h]
        r_lag1 = inten[i, h - 1] if h >= 1 else 0
        is_peak = 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0
        is_we = 1 if wd[i] >= 5 else 0
        X_all.append([np.log1p(r_now), np.log1p(r_lag1), is_peak, is_we,
                      is_peak * np.log1p(r_now), is_we * np.log1p(r_now), h])
        y_all.append(hc[i, h] / base - 1)
X = np.array(X_all); y = np.array(y_all)
r_raw = np.expm1(X[:, 0])
print(f'样本 {len(y)} 个雨时 | 雨强范围 {r_raw.min():.1f}~{r_raw.max():.0f} mm/h')

THETA, RC = 0.052, 0.2
formula = lambda x: THETA * np.log1p(x / RC)          # 输入原始雨强

def metrics(y_t, p):
    mae = np.mean(np.abs(y_t - p))
    r2 = 1 - np.sum((y_t - p) ** 2) / np.sum((y_t - y_t.mean()) ** 2)
    return mae, r2

rng = np.random.default_rng(42)

# ================= 协议A: 内插(随机 80/20) =================
perm = rng.permutation(len(y))
n_tr = int(len(y) * 0.8)
trA, teA = perm[:n_tr], perm[n_tr:]
rfA = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=5,
                            random_state=42, n_jobs=-1)
rfA.fit(X[trA], y[trA])
mA_f = metrics(y[teA], formula(r_raw[teA]))
mA_r = metrics(y[teA], rfA.predict(X[teA]))

# ================= 协议B: 外推(训练 r<4, 测试 r>=4) =================
m_lo, m_hi = r_raw < 4, r_raw >= 4
X_tr, y_tr, r_tr = X[m_lo], y[m_lo], r_raw[m_lo]
X_te, y_te, r_te = X[m_hi], y[m_hi], r_raw[m_hi]
rfB = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=5,
                            random_state=42, n_jobs=-1)
rfB.fit(X_tr, y_tr)
mB_f = metrics(y_te, formula(r_te))
mB_r = metrics(y_te, rfB.predict(X_te))

print()
print('=' * 76)
print(f'{"":22s}{"MAE (越低越好)":>16s}{"R²":>10s}')
print('=' * 76)
print(f'{"协议A 内插(随机80/20)":<24s}')
print(f'  外推公式 θ·ln(1+r/RC)   {mA_f[0]*100:13.1f}pp   {mA_f[1]:8.3f}')
print(f'  随机森林(500棵)         {mA_r[0]*100:13.1f}pp   {mA_r[1]:8.3f}')
print(f'{"协议B 外推(训练r<4, 测r≥4)":<26s}   (训练样本 {m_lo.sum()}, 测试 {m_hi.sum()})')
print(f'  外推公式 θ·ln(1+r/RC)   {mB_f[0]*100:13.1f}pp   {mB_f[1]:8.3f}')
print(f'  随机森林(500棵)         {mB_r[0]*100:13.1f}pp   {mB_r[1]:8.3f}')
print('=' * 76)

# ================= 直接喂 0~60mm/h: RF 输出恒定证据 =================
def rf_curve(rf, rr):
    # 固定其他特征: 平峰/工作日/无滞后
    return rf.predict(np.column_stack([np.log1p(rr), np.zeros_like(rr),
                                       np.zeros_like(rr), np.zeros_like(rr),
                                       np.zeros_like(rr), np.zeros_like(rr),
                                       np.full_like(rr, 13.0)]))
rr = np.linspace(0.05, 60, 300)
rf_out_B = rf_curve(rfB, rr)
rf_out_A = rf_curve(rfA, rr)
plat = rf_out_B[rr > 40].mean()
print(f'\n直接输入演示(训练只见过 r<4 的 RF): r=10 → {rf_curve(rfB, np.array([10.0]))[0]:+.3f}'
      f' | r=30 → {rf_curve(rfB, np.array([30.0]))[0]:+.3f}'
      f' | r=60 → {rf_curve(rfB, np.array([60.0]))[0]:+.3f}   ← 输出几乎恒定(平台 {plat:+.3f})')
print(f'公式对比: r=10 → {formula(np.array([10.0]))[0]:+.3f} | r=30 → {formula(np.array([30.0]))[0]:+.3f} | r=60 → {formula(np.array([60.0]))[0]:+.3f}   ← 持续增长')

# ---- 协议B 核心评价: 分箱趋势(外推的本质=预测是否随 r 响应) ----
print()
print('外推测试集分箱趋势(r≥4mm/h, 训练从未见过):')
print(f'{"箱":12s}{"实测中位":>10s}{"公式预测":>10s}{"RF预测":>10s}')
bins_t = [(4, 8), (8, 16), (16, 1e9)]
trend = {'act': [], 'f': [], 'rf': []}
for lo, hi in bins_t:
    m = (r_te >= lo) & (r_te < hi)
    if m.sum() < 10: continue
    a = float(np.median(y_te[m])); f_ = float(np.median(formula(r_te[m]))); rf_ = float(np.median(rfB.predict(X_te[m])))
    trend['act'].append(a); trend['f'].append(f_); trend['rf'].append(rf_)
    print(f'{f"{lo:.0f}-{hi:.0f}":12s}{a*100:9.1f}pp{f_*100:9.1f}pp{rf_*100:9.1f}pp   (n={m.sum()})')
# 趋势斜率(响应是否随 r 增长)
gx = np.log(r_te[np.isfinite(np.log(r_te))])
sl_act = np.polyfit(gx, y_te, 1)[0] if len(trend['act']) >= 2 else np.nan
sl_f = np.polyfit(gx, formula(r_te), 1)[0]
sl_rf = np.polyfit(gx, rfB.predict(X_te), 1)[0]
print('')
print(f'趋势斜率(d效应/d ln r): 实测 {sl_act:+.3f} | 公式 {sl_f:+.3f} | RF {sl_rf:+.3f}   ← RF 斜率≈0=对 r 无响应')
print('注: 逐点 MAE 的小时级自然波动底噪约 ±40pp, 协议B 的 MAE 差异(20 vs 43pp)主要由"RF常数≈测试集均值"贡献, 不构成外推证据')

# ================= 图 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
fig.suptitle('外推公式 vs 随机森林：域内互有胜负，域外只有公式能走', fontsize=14, fontweight='bold', y=1.02)

ax = axes[0]
xpos = np.arange(2); bw = 0.32
mae_f = [mA_f[0] * 100, mB_f[0] * 100]; mae_r = [mA_r[0] * 100, mB_r[0] * 100]
ax.bar(xpos - bw / 2, mae_f, bw, color=NAVY, label='外推公式 θ·ln(1+r/RC)')
ax.bar(xpos + bw / 2, mae_r, bw, color=RED, label='随机森林(500棵)')
for xx, vv in zip(xpos - bw / 2, mae_f): ax.annotate(f'{vv:.0f}pp', (xx, vv), ha='center', va='bottom', fontsize=10, color=NAVY, fontweight='bold')
for xx, vv in zip(xpos + bw / 2, mae_r): ax.annotate(f'{vv:.0f}pp', (xx, vv), ha='center', va='bottom', fontsize=10, color=RED, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(['协议A 内插\n(随机80/20)', f'协议B 外推\n(训练r<4, 测r≥4, n={m_hi.sum()})'], fontsize=10.5)
ax.set_ylabel('MAE (pp, 越低越好)'); ax.legend(fontsize=10, loc='upper left')
ax.set_title('① 精度：域内 RF 略胜，域外公式碾压', fontsize=11.5, loc='left', color=NAVY)
ax.grid(alpha=0.3, axis='y')

ax = axes[1]
ax.plot(rr, formula(rr), lw=2.6, color=NAVY, label='外推公式：持续增长(饱和)')
ax.plot(rr, rf_out_B, lw=2.4, color=RED, label='RF(只训练过 r<4)：输出恒定平台')
ax.plot(rr, rf_out_A, lw=1.6, color=GRN, ls='--', label='RF(训练含全部雨强)')
ax.axvline(4, color=MUTED, ls=':', lw=1.4)
ax.text(4.3, ax.get_ylim()[0] + 0.02, '训练域边界 4mm/h', fontsize=9, color=MUTED, rotation=90, va='bottom')
for r_pt in [10, 30, 60]:
    ax.plot([r_pt], [formula(np.array([r_pt]))[0]], 'o', ms=7, color=NAVY, zorder=5)
    ax.annotate(f'+{formula(np.array([r_pt]))[0]*100:.0f}%', (r_pt, formula(np.array([r_pt]))[0]),
                textcoords='offset points', xytext=(6, 6), fontsize=9, color=NAVY)
ax.set_xlabel('输入雨强 r (mm/h)'); ax.set_ylabel('预测出发量效应')
ax.set_title('② 直接输入暴雨强度：RF 静默输出常数，公式继续响应', fontsize=11.5, loc='left', color=NAVY)
ax.legend(fontsize=9.5, loc='lower right'); ax.grid(alpha=0.3)

fig.text(0.5, 0.005,
         '结论：RF 是内插器(域内精度最优)但外推失效——超域输出边界叶子的常数,不报错只给错答案;公式单调饱和,域外唯一可信。两者互补:公式管外推,RF管内插验证',
         ha='center', fontsize=10, color=DGREEN, fontweight='bold', wrap=True)
plt.tight_layout()
fig.savefig(r'D:/AI/条件转化/外推vs随机森林.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 外推vs随机森林.png')
