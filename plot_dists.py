# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

t = pq.read_table('D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_2024-06.parquet',
                  columns=['trip_miles','trip_time','pickup_datetime'])
miles = np.asarray(t['trip_miles']); secs = np.asarray(t['trip_time'])
dt = np.asarray(t['pickup_datetime'])
m = (miles>0.1)&(miles<100)&(secs>60)&(secs<10800)
d = miles[m]; v = d/(secs[m]/3600.0); vm = v[(v>0)&(v<100)]
hrs = (dt[m].astype('datetime64[s]').astype('datetime64[h]').astype(int))%24

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle('NYC 出行三大分布实测（HV FHV 2024-06，n=2011万）', fontsize=14, fontweight='bold', y=1.02)

# ---- ① 距离分布 ----
ax = axes[0]
bins = np.arange(0, 30.5, 0.5)
cnt, _ = np.histogram(d[(d<30)], bins=bins)
ax.bar(bins[:-1], cnt/1e6, width=0.5, color='#4472C4', alpha=0.8)
ax.set_yscale('log')
ax.axvspan(14, 20, color='#ED7D31', alpha=0.18)
mode = bins[np.argmax(cnt)]
ax.annotate(f'峰值 ≈ {mode:.1f} mi\n（主体：logNormal）', xy=(mode, cnt.max()/1e6), xytext=(4, cnt.max()/1e6*0.55),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='#333'))
ax.annotate('14–20 mi 机场鼓包 1.4×', xy=(17, cnt[28:40].max()/1e6), xytext=(19, cnt[28:40].max()/1e6*2.2),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='#C0504D'))
ax.text(21, cnt.max()/1e6*0.02, '尾部 Pareto\nα≈2.76', fontsize=10, color='#7030A0')
ax.set_xlabel('行程距离 (mi)', fontsize=10); ax.set_ylabel('频数 (百万, log)', fontsize=10)
ax.set_title('① 距离分布：混合形态\n= logNormal主体 + 机场分量 + Pareto尾', fontsize=11)
ax.grid(alpha=0.3, axis='y')

# ---- ② 出发时间分布 λ(t) ----
ax = axes[1]
share = np.bincount(hrs, minlength=24)/len(hrs)*100
colors = ['#A6A6A6']*24
for h in range(7,10): colors[h]='#ED7D31'
for h in range(16,20): colors[h]='#C00000'
for h in range(21,24): colors[h]='#7030A0'
ax.bar(range(24), share, color=colors)
peak = int(np.argmax(share))
ax.annotate(f'全天最大峰 {peak}:00 ({share[peak]:.1f}%)', xy=(peak, share[peak]), xytext=(peak-8.5, share[peak]*0.95),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='#333'))
ax.annotate('早高峰 7-9', xy=(8, share[8]), xytext=(2.5, share[8]*1.15), fontsize=9.5, color='#C0504D')
ax.annotate('夜间低谷 03-05\n（注意：速度反而最快）', xy=(4, share[4]), xytext=(0.2, 8.0), fontsize=9, color='#666')
ax.set_xlabel('出发时刻（小时）', fontsize=10); ax.set_ylabel('行程占比 (%)', fontsize=10)
ax.set_title('② 出发时间分布 λ(t)：晚峰最大\n橙=早高峰 红=晚高峰 紫=夜间活跃', fontsize=11)
ax.set_xticks(range(0,24,2)); ax.grid(alpha=0.3, axis='y')

# ---- ③ 速度分布 ----
ax = axes[2]
bins = np.linspace(0, 40, 81)
ax.hist(vm[vm<40], bins=bins, density=True, color='#70AD47', alpha=0.75, label='实测')
ln = stats.lognorm.fit(vm, floc=0)
xs = np.linspace(0.1, 40, 400)
ax.plot(xs, stats.lognorm.pdf(xs, *ln), 'r-', lw=2.2, label=f'LogNormal拟合 (σ={ln[0]:.2f})')
med = np.median(vm)
ax.axvline(med, color='#333', ls='--', lw=1.2)
ax.text(med+0.8, stats.lognorm.pdf(med,*ln)*0.92, f'中位 {med:.1f} mph\n≈{med*1.609:.0f} km/h', fontsize=10)
ax.text(24, 0.055, 'p99 = 37 mph\n（机场/高速段）', fontsize=9.5, color='#666')
ax.set_xlabel('平均速度 (mph)', fontsize=10); ax.set_ylabel('密度', fontsize=10)
ax.set_title('③ 速度分布：右偏 logNormal\n形状 σ 稳定，条件只平移 μ', fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('D:/AI/条件转化/出行分布图示.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved | 中位速度', round(med,1), 'σ(log v)', round(float(np.std(np.log(vm))),3), '峰值小时', peak, '距离峰mi', mode)
