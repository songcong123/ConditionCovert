# -*- coding: utf-8 -*-
import numpy as np, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

zone_name = {}
for r in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if r and r[0].isdigit():
        zone_name[int(r[0])] = f"{r[2][:16]}"

# ============ 图1：极端天气影响（雨日 vs 干燥日） ============
W = np.load('D:/AI/条件转化/.weather_agg.npz')
hr_r, hr_d = W['hr_r'], W['hr_d']; dr, dd = W['dr'], W['dd']
pu_r, pu_d = W['pu_r'], W['pu_d']; nr, nd = int(W['nr']), int(W['nd'])

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle(f'极端天气影响：雨日(日降水≥5mm, n={nr}) vs 干燥日(n={nd}) — NYC HV FHV, 2024.06–08',
             fontsize=13.5, fontweight='bold', y=1.03)

ax = axes[0]
sr = hr_r/nr; sd = hr_d/nd
x = np.arange(48)/2
ax.bar(x, sd, width=0.5, alpha=0.55, color='#8FAADC', label='干燥日')
ax.plot(x, sr, 'r-', lw=2.3, label='雨日')
ax.set_xlabel('出发时刻（小时）'); ax.set_ylabel('日均行程数')
ax.set_title(f'① 出发时间 λ(t)：雨日整体抬升 +{((hr_r.sum()/nr)/(hr_d.sum()/nd)-1)*100:.0f}%\n雨日 {hr_r.sum()/nr/1e3:.0f}k vs 干日 {hr_d.sum()/nd/1e3:.0f}k 趟/日', fontsize=11)
ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(range(0,25,4))

ax = axes[1]
bins = np.arange(0, 15.25, 0.25)
ax.hist(dd, bins=bins, density=True, alpha=0.55, color='#A9D18E', label=f'干燥日 (中位 {np.median(dd):.2f} mi)')
ax.hist(dr, bins=bins, density=True, alpha=0.55, color='#4472C4', label=f'雨日 (中位 {np.median(dr):.2f} mi)')
ax.axvline(np.median(dd), color='#548235', ls='--', lw=1.5); ax.axvline(np.median(dr), color='#C00000', ls='--', lw=1.5)
ax.set_xlabel('行程距离 (mi)'); ax.set_ylabel('密度')
ax.set_title(f'② 轨迹长度：分布左移（空间压缩）\n短途<2mi：雨 {(dr<2).mean()*100:.1f}% vs 干 {(dd<2).mean()*100:.1f}%（+{((dr<2).mean()-(dd<2).mean())*100:.1f}pp）', fontsize=11)
ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
shr_r = pu_r/pu_r.sum(); shr_d = pu_d/pu_d.sum()
diff = (shr_r - shr_d)*100
idx = np.argsort(np.abs(diff))[-12:]
idx = idx[np.argsort(diff[idx])]
colors = ['#C00000' if v > 0 else '#4472C4' for v in diff[idx]]
ax.barh([f'{zone_name.get(i,i)}' for i in idx], diff[idx], color=colors)
ax.axvline(0, color='#333', lw=1)
ax.set_xlabel('起点份额变化 (百分点, 雨日−干日)')
ax.set_title('③ 空间分布：雨日份额升降的区域\n红=更集中 蓝=更少', fontsize=11)
ax.grid(alpha=0.3, axis='x'); ax.tick_params(labelsize=8.5)

plt.tight_layout()
plt.savefig('D:/AI/条件转化/极端天气影响.png', dpi=150, bbox_inches='tight', facecolor='white')
print('图1 saved')

# ============ 图2：大型事件影响（MSG + Barclays 两行对比） ============
venues = [
    ('.event_agg.npz', 'MSG 麦迪逊广场花园（Zone 186，尼克斯/游骑兵）', 44, 76),
    ('.event_agg_189.npz', 'Barclays 巴克莱中心（Zone 189，篮网）', 27, 93),
]
fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
fig.suptitle('大型事件影响：球赛日 vs 非赛日 —— 场馆所在 Zone 出发行程, NYC HV FHV 2025.01–04（注意：两场馆均叠加轨道枢纽基线，脉冲被稀释）',
             fontsize=13, fontweight='bold', y=0.99)

for row, (f, title, ng, nn) in enumerate(venues):
    E = np.load(f'D:/AI/条件转化/{f}', allow_pickle=True)
    hg, hn = E['hr_g'], E['hr_n']; dg, dn = E['dg'], E['dn']; dog, don = E['do_g'], E['do_n']
    sg = hg/ng; sn = hn/nn
    x = np.arange(48)/2

    ax = axes[row][0]
    ax.bar(x, sn, width=0.5, alpha=0.55, color='#BDD7EE', label='非赛日')
    ax.plot(x, sg, 'r-', lw=2.3, label='赛日')
    ax.axvspan(21.5, 23.5, color='#C00000', alpha=0.12)
    r22 = sg[44:47].sum()/sn[44:47].sum()
    ax.annotate(f'散场窗 22:00–23:30\n赛日/非赛日 = {r22:.2f}', xy=(22.5, sg[44:47].max()),
                xytext=(14, sg[44:47].max()*0.85), fontsize=10, color='#C00000',
                arrowprops=dict(arrowstyle='->', color='#C00000'))
    ax.set_xlabel('出发时刻（小时）'); ax.set_ylabel('日均行程数')
    ax.set_title(f'① λ(t)：{title}\n散场窗占全日：赛日 {hg[44:47].sum()/hg.sum()*100:.1f}% vs 非赛日 {hn[44:47].sum()/hn.sum()*100:.1f}%', fontsize=10.5)
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(range(0,25,4))

    ax = axes[row][1]
    bins = np.arange(0, 15.25, 0.25)
    ax.hist(dn, bins=bins, density=True, alpha=0.55, color='#A9D18E', label=f'非赛日 (中位 {np.median(dn):.2f} mi)')
    ax.hist(dg, bins=bins, density=True, alpha=0.55, color='#ED7D31', label=f'赛日 (中位 {np.median(dg):.2f} mi)')
    ax.set_xlabel('行程距离 (mi)'); ax.set_ylabel('密度')
    ax.set_title('② 轨迹长度：赛日 vs 非赛日\n（枢纽基线主导，差异很小）', fontsize=10.5)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[row][2]
    sgo = dog/dog.sum(); sno = don/don.sum()
    idx = np.argsort(sgo)[-10:][::-1]
    yp = np.arange(10)
    ax.barh(yp-0.2, sgo[idx]*100, height=0.38, color='#ED7D31', label='赛日')
    ax.barh(yp+0.2, sno[idx]*100, height=0.38, color='#BDD7EE', label='非赛日')
    ax.set_yticks(yp); ax.set_yticklabels([f'{zone_name.get(i,i)}' for i in idx], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('目的地份额 (%)')
    ax.set_title('③ 赛后去向 DO top10', fontsize=10.5)
    ax.legend(); ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('D:/AI/条件转化/大型事件影响.png', dpi=150, bbox_inches='tight', facecolor='white')
print('图2 saved')
