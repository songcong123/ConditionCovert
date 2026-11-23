# -*- coding: utf-8 -*-
"""validate_52: 出行距离·中位数口径拟合(与出发量同构的建模方式)
================================================================
标签: y = 当日行程中位距离 ÷ 干日中位距离 − 1   (百分比, 同出发量的"+23.8%"风格)
公式: δd(R) = θd·ln(1+R/RCd)  —— 只用两个低雨量箱两点标定, 高雨量箱纯盲测
对照: ① 均值口径(被长尾拖着, 展示差异)  ② 参数口径 μ1=−0.0171·ln(1+R/5.4) 自洽性
      ③ RF(日级, <30mm 训练) 外推对照
分箱: 0.5-5 / 5-15 / 15-30 (标定域) | 30-60 / ≥60 (盲测外推)
输出: 距离中位数拟合.png + 控制台口径表
"""
import numpy as np
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; DGREEN = '#375623'; MUTED = '#666666'; ORANGE = '#ED7D31'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates = list(A['dates']); n = len(dates)
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates])
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates])
wd = np.array([((np.datetime64(d, 'D').view('int64') + 3) % 7 < 5).astype(int) for d in dates])

# ---- 干日基线: 分日型(工作日/周末)中位距离, 均值对照 ----
dry_i = np.where(kl == 'dry')[0]
dd_all = dist[np.isin(dist_day, dry_i)]; dd_all = dd_all[(dd_all > 0.3) & (dd_all < 60)]
day_of = {i: wd[i] for i in range(n)}
dd_wd = dist[np.isin(dist_day, [i for i in dry_i if wd[i] == 1])]; dd_wd = dd_wd[(dd_wd > 0.3) & (dd_wd < 60)]
dd_we = dist[np.isin(dist_day, [i for i in dry_i if wd[i] == 0])]; dd_we = dd_we[(dd_we > 0.3) & (dd_we < 60)]
med_dry_wd = float(np.median(dd_wd)); med_dry_we = float(np.median(dd_we))
med_dry = float(np.median(dd_all)); mean_dry = float(np.mean(dd_all))
print(f'干日基线(分日型): 工作日中位 {med_dry_wd:.2f} mi | 周末中位 {med_dry_we:.2f} mi')
print(f'全池化对照: 中位 {med_dry:.2f} mi | 均值 {mean_dry:.2f} mi (长尾拖高 {mean_dry/med_dry:.2f}×)')

# ---- 雨日样本: 中位口径 y_med(同日型基线) 与 均值口径 y_mean(对照) ----
rows = []
for i in np.where(kl != 'dry')[0]:
    dv = dist[dist_day == i]; dv = dv[(dv > 0.3) & (dv < 60)]
    if len(dv) < 500 or dp[i] < 0.5: continue
    base_med = med_dry_wd if wd[i] == 1 else med_dry_we
    rows.append(dict(R=float(dp[i]),
                     ymed=float(np.median(dv) / base_med - 1),
                     ymean=float(np.mean(dv) / mean_dry - 1),
                     we=int(wd[i])))
import pandas as pd
S = pd.DataFrame(rows)
print(f'雨日样本: {len(S)} 天 | 日降水 {S.R.min():.1f}~{S.R.max():.0f} mm')

# ---- 分箱 ----
BINS = [(0.5, 5, '0.5-5'), (5, 15, '5-15'), (15, 30, '15-30'), (30, 60, '30-60'), (60, 1e9, '≥60')]
bx = []
for lo, hi, nm in BINS:
    m = (S.R >= lo) & (S.R < hi)
    if m.sum() < 3: continue
    bx.append(dict(nm=nm, n=int(m.sum()), Rbar=float(S.R[m].mean()),
                   ymed=float(S.ymed[m].mean()), ymean=float(S.ymean[m].mean())))

# ---- 两点标定(只用前两箱, 中位口径) ----
r1, d1 = bx[0]['Rbar'], bx[0]['ymed']
r2, d2 = bx[1]['Rbar'], bx[1]['ymed']
f = lambda RC: np.log1p(r1 / RC) / np.log1p(r2 / RC) - d1 / d2
RC_d = brentq(f, 1e-2, 200)
th_d = d1 / np.log1p(r1 / RC_d)
dd_f = lambda R: th_d * np.log1p(np.asarray(R, float) / RC_d)
print(f'\n两点标定(中位口径): δd(R) = {th_d:+.5f}·ln(1+R/{RC_d:.2f})')
print(f'自洽对照: 参数口径 μ1 = −0.0171·ln(1+R/5.4) → 换算 θ={-0.0171:+.5f}, RC=5.4')

# ---- RF 对照(日级, 纯雨强单特征, <30mm 训练) ----
X_rf = np.log1p(S.R).values.reshape(-1, 1)
m_tr = (S.R < 30).values
rf = RandomForestRegressor(400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X_rf[m_tr], S.ymed[m_tr])
pred_rf = rf.predict(X_rf)
pred_fm = dd_f(S.R)

# ---- 口径表 ----
print(f'\n{"箱":8s} {"n":>4s} {"R̄":>6s} {"实测中位%":>9s} {"公式%":>8s} {"RF%":>8s} {"公式差pp":>8s} {"RF差pp":>7s}')
for k, b in enumerate(bx):
    m = (S.R >= BINS[k][0]) & (S.R < BINS[k][1])
    fm_b = float(pred_fm[m].mean()); rf_b = float(pred_rf[m].mean())
    b['fm'] = fm_b; b['rf'] = rf_b
    tag = '标定' if k < 2 else ('内插' if k == 2 else '盲测')
    print(f'{b["nm"]+tag:8s} {b["n"]:4d} {b["Rbar"]:6.1f} {b["ymed"]*100:+9.2f} {fm_b*100:+8.2f} '
          f'{rf_b*100:+8.2f} {abs(fm_b-b["ymed"])*1e4:8.0f}×10⁻⁴ {abs(rf_b-b["ymed"])*1e4:7.0f}×10⁻⁴')
# 距离效应幅度只有百分之几, 用 pp 打印
print(f'\n{"箱":8s} {"实测中位%":>9s} {"公式%":>8s} {"RF%":>8s} {"公式差pp":>8s} {"RF差pp":>7s}')
for k, b in enumerate(bx):
    print(f'{b["nm"]:8s} {b["ymed"]*100:+9.2f} {b["fm"]*100:+8.2f} {b["rf"]*100:+8.2f} '
          f'{abs(b["fm"]-b["ymed"])*100:8.2f} {abs(b["rf"]-b["ymed"])*100:7.2f}')
mae_in_fm = float(np.mean(np.abs(S.ymed[m_tr] - pred_fm[m_tr])) * 100)
mae_in_rf = float(np.mean(np.abs(S.ymed[m_tr] - pred_rf[m_tr])) * 100)
print(f'\n内插拟合MAE(<30mm, 逐日): 公式 {mae_in_fm:.2f}pp | RF {mae_in_rf:.2f}pp')
m_ex = S.R >= 30
mae_ex_fm = float(np.mean(np.abs(S.ymed[m_ex] - pred_fm[m_ex])) * 100)
mae_ex_rf = float(np.mean(np.abs(S.ymed[m_ex] - pred_rf[m_ex])) * 100)
print(f'外推域逐日(≥30mm): 公式 {mae_ex_fm:.2f}pp | RF {mae_ex_rf:.2f}pp (逐点含日间噪声, 胜负看箱)')

# ---- 均值口径对照(只展示, 不拟合: 长尾拖动导致两点常不成单调关系, 不可稳定标定) ----
print('\n均值口径对照(各箱效应, 不拟合):')
for k, b in enumerate(bx):
    print(f'  箱{b["nm"]:7s} 中位效应 {b["ymed"]*100:+.2f}% | 均值效应 {b["ymean"]*100:+.2f}%')

# ---- 三点标定版(加 15-30 箱, 最小二乘) —— 修"两点信息不足" ----
from scipy.optimize import curve_fit
Rs3 = np.array([b['Rbar'] for b in bx[:3]]); Ys3 = np.array([b['ymed'] for b in bx[:3]])
popt, _ = curve_fit(lambda R, th, RC: th * np.log1p(R / RC), Rs3, Ys3,
                    p0=[-0.02, 3.0], bounds=([-0.2, 0.05], [-1e-4, 100]), maxfev=20000)
th3, RC3 = popt
dd_f3 = lambda R: th3 * np.log1p(np.asarray(R, float) / RC3)
print(f'\n三点标定版(0.5-5/5-15/15-30 最小二乘): δd = {th3:+.5f}·ln(1+R/{RC3:.2f})')
for k, b in enumerate(bx):
    v3 = float(np.mean(dd_f3(S.R[(S.R >= BINS[k][0]) & (S.R < BINS[k][1])])))
    b['fm3'] = v3
    tag = '标定' if k < 3 else '盲测'
    print(f'  {b["nm"]}{tag}: 实测{b["ymed"]*100:+.2f}% | 三点公式{v3*100:+.2f}% (差{abs(v3-b["ymed"])*100:.2f}pp) | 两点公式{b["fm"]*100:+.2f}%')

# ================= 图 =================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
fig.suptitle('出行距离 · 中位数口径拟合（与出发量同构）：δd(R) = θ·ln(1+R/RC)，两点标定、高雨量盲测',
             fontsize=14, fontweight='bold', y=0.99)

ax = axes[0]
ax.scatter(S.R, S.ymed * 100, s=22, alpha=0.45, color='#B0B0B0', zorder=2, label=f'逐日观测(n={len(S)})')
RR = np.linspace(0.5, 175, 400)
ax.plot(RR, dd_f(RR) * 100, lw=2.8, color=RED, zorder=4, label=f'两点标定公式(只用两低箱)')
ax.plot(RR, dd_f3(RR) * 100, lw=2.2, color=DGREEN, ls=':', zorder=3.5, label='三点标定版(加15-30箱)')
ax.plot(RR, rf.predict(np.log1p(RR).reshape(-1, 1)) * 100,
        lw=2.2, color=NAVY, ls='--', zorder=3, label='RF(<30mm 训练)')
ax.axvline(30, color=MUTED, ls=':', lw=1.6)
ax.text(31, -3.3, '训练边界 30mm', fontsize=9.5, color=MUTED)
for k, b in enumerate(bx):
    cal = k < 2
    ax.plot([b['Rbar']], [b['ymed'] * 100], 's', ms=13, zorder=6,
            markerfacecolor='none' if cal else NAVY, markeredgecolor='#7F7F7F' if cal else NAVY, markeredgewidth=2)
    ax.annotate(f'{b["ymed"]*100:+.1f}', (b['Rbar'], b['ymed'] * 100), textcoords='offset points',
                xytext=(8, 6 if cal else -16), fontsize=9.5, color=MUTED if cal else NAVY)
ax.plot([], [], 's', ms=13, markerfacecolor='none', markeredgecolor='#7F7F7F', markeredgewidth=2, label='标定箱(0.5-15)')
ax.plot([], [], 's', ms=13, markerfacecolor=NAVY, markeredgecolor=NAVY, label='盲测箱(≥30, 未参与拟合)')
ax.set_xscale('log')
ax.set_xlabel('日降水 R (mm, log)'); ax.set_ylabel('中位距离效应 (%)')
ax.set_title('① 剂量曲线: 雨越大, 典型一趟越短\n   效应 30mm 后趋饱和 → RF 躺平值恰好命中; 单对数公式形状偏浅', fontsize=11.5, loc='left', color=NAVY)
ax.legend(fontsize=8.5, loc='upper right'); ax.grid(alpha=0.3, which='both')

ax = axes[1]
ax.plot(RR, med_dry * (1 + dd_f(RR)), lw=2.8, color=RED, label='公式预测中位距离')
meds = [med_dry * (1 + b['ymed']) for b in bx]
ax.scatter([b['Rbar'] for b in bx], meds, s=110, zorder=5,
           facecolor=['none', 'none', 'none', NAVY, NAVY],
           edgecolor=['#7F7F7F', '#7F7F7F', '#7F7F7F', NAVY, NAVY], linewidth=2)
for b, v in zip(bx, meds):
    ax.annotate(f'{v:.2f} mi', (b['Rbar'], v), textcoords='offset points', xytext=(8, 7), fontsize=9.5, color=NAVY)
ax.axhline(med_dry, color=MUTED, ls='--', lw=1.4)
ax.text(1.2, med_dry + 0.012, f'晴天基线 {med_dry:.2f} mi', fontsize=10, color=MUTED)
ax.set_xscale('log')
ax.set_xlabel('日降水 R (mm, log)'); ax.set_ylabel('中位距离 (mi)')
ax.set_title('② 翻译成绝对值: 中位距离随雨量收缩\n   (领导问"到底短多少"看这张)', fontsize=11.5, loc='left', color=NAVY)
ax.legend(fontsize=10, loc='upper right'); ax.grid(alpha=0.3, which='both')

fig.text(0.5, 0.005,
         f'口径: y=当日中位距离÷干日中位−1; 公式 δd={th_d:+.5f}·ln(1+R/{RC_d:.2f})(两点标定: 0.5-5与5-15两箱均值)。'
         f'选 median 不选 mean: 距离重尾(均值{mean_dry:.2f}=中位{med_dry:.2f}的{mean_dry/med_dry:.1f}倍, 被长途拖着日间波动大)。'
         f'与参数口径 μ1=−0.0171·ln(1+R/5.4) 同号同量级=自洽',
         ha='center', fontsize=9.5, color=DGREEN, fontweight='bold')
plt.tight_layout(rect=[0, 0.045, 1, 0.96])
fig.savefig(r'D:/AI/条件转化/距离中位数拟合.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nsaved 距离中位数拟合.png')
