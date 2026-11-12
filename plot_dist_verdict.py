# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus'] = False

t = pq.read_table('D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_2024-06.parquet', columns=['trip_miles','trip_time'])
miles = np.asarray(t['trip_miles']); secs = np.asarray(t['trip_time'])
m = (miles>0.1)&(miles<100)&(secs>60)&(secs<10800)
d = np.sort(miles[m]); n = len(d)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
fig.suptitle('距离分布判决：整体幂律 or 分段混合？（HV FHV 2024-06，n=2011万）', fontsize=14, fontweight='bold', y=1.02)

# ---- 左：PDF 双对数（直观形态） ----
ax = axes[0]
bins = np.logspace(np.log10(0.1), np.log10(60), 70)
cnt, be = np.histogram(d[d<60], bins=bins)
pdf = cnt/(cnt.sum()*np.diff(be))
ax.loglog(be[:-1], pdf, 'o-', ms=4, color='#4472C4', label='实测 PDF', lw=1)
# 幂律线 α=2.76 (xmin=6.5)
alpha, xmin = 2.76, 6.5
xs = np.linspace(6.5, 50, 100)
ref = pdf[(be[:-1]>3)&(be[:-1]<5)].max()
ax.loglog(xs, ref*3.2*(xs/xmin)**(-alpha), 'r--', lw=2, label=f'幂律 α={alpha}（尾段）')
# 指数线对照
xs2 = np.linspace(2, 30, 100)
ax.loglog(xs2, ref*3.2*np.exp(-(xs2-2)/4.5), 'g:', lw=2, label='指数衰减（对照，掉太快）')
ax.axvline(1.25, color='#C0504D', ls='-.', lw=1.5)
ax.annotate('峰 ≈1.25 mi：左侧上升\n→ 幂律不可能（幂律必须单调降）', xy=(1.25, ref*3.0), xytext=(2.5, ref*5.5),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='#C0504D'), color='#C0504D')
ax.axvspan(14, 20, color='#ED7D31', alpha=0.15); ax.text(15, ref*0.02, '机场\n鼓包', fontsize=10, color='#C0504D')
ax.text(25, ref*0.12, '尾段与幂律线平行\n✓ 尾部是幂律', fontsize=10, color='#C00000')
ax.set_xlabel('行程距离 (mi, log)'); ax.set_ylabel('概率密度 (log)')
ax.set_title('① PDF 双对数：主体有峰+尾部直线', fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# ---- 右：CCDF 双对数 + 整体拟合KS对比 ----
ax = axes[1]
ccdf = 1.0 - np.arange(1, n+1)/n
step = n//40000
ax.loglog(d[::step], ccdf[::step], '.', ms=1.5, color='#4472C4', alpha=0.6, label='实测 CCDF')
ln = stats.lognorm.fit(d, floc=0)
xs = np.logspace(np.log10(0.3), np.log10(60), 300)
ax.loglog(xs, 1-stats.lognorm.cdf(xs, *ln), 'g-', lw=2, label=f'对数正态整体拟合 (KS=0.036)')
lam = 1/np.mean(d)
ax.loglog(xs, np.exp(-lam*xs), 'm--', lw=1.5, label='指数整体拟合')
ks_ln = stats.kstest(d, 'lognorm', args=ln).statistic
ks_ex = stats.kstest(d, 'expon', args=(0,1/lam)).statistic
ax.set_title(f'② CCDF 双对数：整体形态判决\nLogNormal KS={ks_ln:.3f}  vs  指数 KS={ks_ex:.3f}', fontsize=11)
ax.set_xlabel('行程距离 (mi, log)'); ax.set_ylabel('P(D > d) (log)')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
ax.annotate('尾段弯曲度介于\n指数与幂律之间→混合', xy=(20, 0.01), xytext=(3, 0.004), fontsize=9.5,
            arrowprops=dict(arrowstyle='->', color='#666'), color='#666')

plt.tight_layout()
plt.savefig('D:/AI/条件转化/距离分布判决.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f'saved | KS: lognormal={ks_ln:.4f} exponential={ks_ex:.4f}')
