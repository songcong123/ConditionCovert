# -*- coding: utf-8 -*-
"""过程图: 以 2024-04-03(周三暴雨) 为例, 展示"晴天基线 → 逐时调制 → 暴雨分布"全过程
三行(出发时间/速度/距离) × 三列(起点/调制/终点) + 底部空间注记
调整规则(两暖季实测剂量曲线):
  逐时乘子: 多维(r)=0.052*ln(1+r/0.2)   → λ(t)×(1+多维)
  逐时速度: 微v(r)=0.025*ln(1+r/0.3)    → 速度=基线中位×exp(-微v)
  距离三旋钮(右列用当日实测校准值演示): 峰左移/短途权重增/机场份额降"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats

A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load('D:/AI/条件转化/.flow_spd.npz')
J = json.load(open('D:/AI/条件转化/.flow_scores.json'))

DAY = '2024-04-03'
dates = list(A['dates']); idx = {d: i for i, d in enumerate(dates)}
klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
i0 = idx[DAY]
dow = int(np.datetime64(DAY, 'D').view('int64')) % 7
m_dry = np.array([klass[d] == 'dry' for d in dates])
dowv = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])
wed = np.where(m_dry & (dowv == dow))[0]          # 干周三
hour_cnt = A['hour_cnt'].astype(float)
base_h = hour_cnt[wed].mean(0) / 1000              # 千趟/时
act_h = hour_cnt[i0] / 1000

rain = {}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    rain[(str(dd), int(hh))] = float(mm)
r24 = np.array([rain.get((DAY, h), 0.0) for h in range(24)])

th_v, RC_v = 0.052, 0.2
th_s, RC_s = 0.025, 0.3
mult = 1 + th_v * np.log1p(r24 / RC_v)             # 逐时乘子
adj_h = base_h * mult
dvs = np.exp(-th_s * np.log1p(r24 / RC_s))         # 逐时速度乘子

spd_v = S['v'].astype(float); spd_d = S['day'].astype(int); spd_h = S['hour'].astype(int)
bm = np.isin(spd_d, wed)
mu0 = np.array([np.median(spd_v[bm & (spd_h == h)]) for h in range(24)])
am = spd_d == i0
act_v = np.array([np.median(spd_v[am & (spd_h == h)]) if (am & (spd_h == h)).sum() > 20 else np.nan
                  for h in range(24)])
adj_v = mu0 * dvs

dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
dd_base = dist[np.isin(dist_day, wed)]; dd_base = dd_base[(dd_base > .05) & (dd_base < 40)]
dd_act = dist[dist_day == i0]; dd_act = dd_act[(dd_act > .05) & (dd_act < 40)]

fig = plt.figure(figsize=(17, 13))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.34])
fig.suptitle('一个真实日子的全过程：2024-04-03（周三·暴雨 99mm）——晴天基线 怎样被调成 暴雨分布',
             fontsize=15, fontweight='bold', y=0.995)
hx = np.arange(24)

def colhead(c, txt):
    fig.text([0.055, 0.39, 0.72][c], 0.955, txt, fontsize=12.5, fontweight='bold',
             ha='left', color='#1a1a1a')

colhead(0, '① 起点：晴天基线（干周三平均）')
colhead(1, '② 调制：当天几点下多大雨 → 查表调参')
colhead(2, '③ 终点：调整后 vs 真实当天')

# ================= 行1: 出发时间 =================
ax = fig.add_subplot(gs[0, 0])
ax.bar(hx, base_h, color='#9DC3E6', edgecolor='#4472C4', lw=.5)
ax.set_title('基线 $\\lambda_0(t)$：干周三平均（61万趟/日）', fontsize=11)
ax.set_ylabel('千趟/时'); ax.set_xticks(range(0, 25, 4)); ax.grid(alpha=.3, axis='y')

ax = fig.add_subplot(gs[0, 1])
ax.bar(hx, r24, color='#4472C4', alpha=.85, label='当天雨强 (mm/h)')
ax.set_ylabel('雨强 mm/h')
ax2 = ax.twinx()
ax2.plot(hx, mult, 'o-', color='#C00000', lw=2, ms=4, label='查表所得乘子')
ax2.set_ylabel('多维乘子', color='#C00000'); ax2.set_ylim(1, mult.max() * 1.15)
ax.annotate('18时 13.5mm/h\n→ 乘 1.24', xy=(18, r24[18]), xytext=(10.5, r24.max() * .82),
            fontsize=10, color='#C00000', arrowprops=dict(arrowstyle='->', color='#C00000'))
ax.set_title('规则：$\\lambda(t)\\times(1+0.052\\ln(1+r_t/0.2))$ 逐时查表', fontsize=11)
ax.set_xticks(range(0, 25, 4)); ax.grid(alpha=.3, axis='y')

ax = fig.add_subplot(gs[0, 2])
ax.bar(hx, base_h, color='#D9D9D9', label='晴天基线')
ax.plot(hx, adj_h, 'o-', color='#ED7D31', lw=2, ms=3.5, label='调整后（预测）')
ax.plot(hx, act_h, 's-', color='#C00000', lw=1.6, ms=3, label='真实当天', alpha=.85)
tot_b, tot_a, tot_r = base_h.sum(), adj_h.sum(), act_h.sum()
ax.set_title(f'调整后 +{(tot_a/tot_b-1)*100:.0f}% vs 真实 +{(tot_r/tot_b-1)*100:.0f}%'
             f'（早高峰段贴合最好）', fontsize=11)
ax.set_ylabel('千趟/时'); ax.set_xticks(range(0, 25, 4))
ax.legend(fontsize=9); ax.grid(alpha=.3, axis='y')

# ================= 行2: 速度 =================
ax = fig.add_subplot(gs[1, 0])
ax.plot(hx, mu0, 'o-', color='#70AD47', lw=2.2, ms=4)
ax.annotate('晴天也分快慢：\n凌晨18.8 / 午后9.8 mph', xy=(16, mu0[16]), xytext=(3, mu0[16] - 3.2),
            fontsize=9.5, color='#375623', arrowprops=dict(arrowstyle='->', color='#70AD47'))
ax.set_title('基线中位速度 $\\mu_0(t)$（干周三）', fontsize=11)
ax.set_ylabel('mph'); ax.set_xticks(range(0, 25, 4)); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[1, 1])
ax.bar(hx, r24, color='#BFBFBF', label='当天雨强')
ax.set_ylabel('雨强 mm/h')
ax2 = ax.twinx()
ax2.plot(hx, adj_v, 'o-', color='#C00000', lw=2, ms=4)
ax2.set_ylabel('调整后中位速度 (mph)', color='#C00000')
ax2.set_ylim(6, 20)
ax.annotate('规则：基线 ×$e^{-0.025\\ln(1+r/0.3)}$\n雨大的小时减速多', xy=(16, adj_v[16]),
            xytext=(1.2, 7.2), fontsize=10, color='#C00000')
ax.set_title('逐时减速（乘法，不动形状）', fontsize=11)
ax.set_xticks(range(0, 25, 4)); ax.grid(alpha=.3, axis='y')

ax = fig.add_subplot(gs[1, 2])
ax.plot(hx, mu0, 'o-', color='#BFBFBF', lw=1.8, ms=3, label='晴天基线')
ax.plot(hx, adj_v, 'o-', color='#ED7D31', lw=2, ms=3.5, label='调整后（预测）')
ax.plot(hx, act_v, 's-', color='#C00000', lw=1.6, ms=3, label='真实当天', alpha=.85)
ok = ~np.isnan(act_v)
ax.set_title(f'全天中位：预测 {np.nanmedian(adj_v):.1f} vs 真实 {np.nanmedian(act_v):.1f} mph'
             f'（基线 {np.median(mu0):.1f}）', fontsize=11)
ax.set_ylabel('mph'); ax.set_xticks(range(0, 25, 4))
ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_ylim(6, 20)

# ================= 行3: 距离 =================
bins = np.geomspace(.05, 40, 60)
B3 = J['B3']
xs_d = np.geomspace(.05, 40, 500)
def mix_pdf4(x):
    B = B3
    p = (B['w1'] * stats.lognorm.pdf(x, B['sd1'], scale=np.exp(B['mu1']))
         + B['w2'] * stats.lognorm.pdf(x, B['sd2'], scale=np.exp(B['mu2']))
         + B['w_ap'] * stats.lognorm.pdf(x, B['sd_ap'], scale=np.exp(B['mu_ap'])))
    p = p + np.where(x >= B['xmin'], B['w_pa'] * B['alpha'] / B['xmin'] * (x / B['xmin']) ** (-(B['alpha'] + 1)), 0)
    return p
ax = fig.add_subplot(gs[2, 0])
ax.plot(xs_d, mix_pdf4(xs_d), lw=2.4, color='#ED7D31')
ax.fill_between(xs_d, 0, mix_pdf4(xs_d), color='#F8CBAD', alpha=.55)
ax.set_xscale('log')
ax.set_title('基线距离分布（短途大峰+机场鼓包，理论曲线）', fontsize=11)
ax.set_xlabel('距离 (mi, log)'); ax.set_ylabel('密度'); ax.grid(alpha=.3, which='both')

ax = fig.add_subplot(gs[2, 1])
B = J['B3']
xs = np.geomspace(.05, 40, 500)
pdf0 = B['w1'] * stats.lognorm.pdf(xs, B['sd1'], scale=np.exp(B['mu1'])) + \
       B['w_ap'] * stats.lognorm.pdf(xs, B['sd_ap'], scale=np.exp(B['mu_ap']))
ax.plot(xs, pdf0, color='#BFBFBF', lw=2)
ax.set_xscale('log'); ax.set_ylim(0, pdf0.max() * 1.3)
pk = np.exp(B['mu1'])
ax.annotate('', xy=(pk * .78, pdf0.max() * .62), xytext=(pk * 1.12, pdf0.max() * .62),
            arrowprops=dict(arrowstyle='->', lw=2.2, color='#C00000'))
ax.text(pk * .93, pdf0.max() * .66, '旋钮1：峰左移', fontsize=10, color='#C00000', ha='center')
ax.text(.07, pdf0.max() * .95, '旋钮2：短途占比↑\n(步行客改打车)', fontsize=10, color='#C00000')
ax.annotate('旋钮3：机场份额↓×0.9', xy=(13, B['w_ap'] * stats.lognorm.pdf(13, B['sd_ap'], scale=np.exp(B['mu_ap']))),
            xytext=(4.5, pdf0.max() * .30), fontsize=10, color='#C00000',
            arrowprops=dict(arrowstyle='->', color='#C00000'))
ax.set_title('三个旋钮（当日按实测校准演示）', fontsize=11)
ax.set_xlabel('距离 (mi, log)'); ax.grid(alpha=.3, which='both')

def logkde_pdf(dd, x):
    kde = stats.gaussian_kde(np.log(dd))
    return kde(np.log(x)) / x                      # log 空间密度 → 原空间密度
ax = fig.add_subplot(gs[2, 2])
ax.plot(xs_d, logkde_pdf(dd_base, xs_d), lw=2.2, color='#7F7F7F', label='晴天基线（实测平滑）')
ax.plot(xs_d, logkde_pdf(dd_act, xs_d), lw=2.2, color='#C00000', label='真实当天（实测平滑）')
ax.fill_between(xs_d, 0, logkde_pdf(dd_act, xs_d), color='#C00000', alpha=.10)
ax.axvline(np.median(dd_base), color='#7F7F7F', ls='--', lw=1.6)
ax.axvline(np.median(dd_act), color='#C00000', ls='--', lw=1.6)
ax.set_xscale('log')
ax.set_title(f'真实当天：中位 {np.median(dd_base):.1f}→{np.median(dd_act):.1f} mi'
             f'（{"" if np.median(dd_act)<np.median(dd_base) else "+"}{(np.median(dd_act)/np.median(dd_base)-1)*100:.0f}%，短途大增）',
             fontsize=11)
ax.set_xlabel('距离 (mi, log)'); ax.legend(fontsize=9); ax.grid(alpha=.3, which='both')

# ================= 底部: 空间注记 =================
ax = fig.add_subplot(gs[3, :]); ax.axis('off')
ax.text(0.5, 0.5,
        '第四个量·空间分布：日级【≈不动】（升消相抵，JS=0.0002）；雨时小时级有结构（机场 −27%、核心 +6%，γ 公式）；'
        '事件才大动空间（场馆×3.5 等，另一套查表算子）',
        ha='center', va='center', fontsize=12, family='Microsoft YaHei',
        bbox=dict(boxstyle='round', fc='#F3EEF7', ec='#9673A6', alpha=.95))

plt.tight_layout(rect=[0, 0, 1, 0.945])
plt.savefig('D:/AI/条件转化/调整过程示例.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 调整过程示例.png')
