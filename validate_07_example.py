# -*- coding: utf-8 -*-
"""直观真实例子: 2024-04-03 暴雨周三(全城) + 2024-12-31 跨年夜(时代广场)
左边讲一场雨, 右边讲一个夜。全程大白话注释, 无术语。"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
S = np.load('D:/AI/条件转化/.flow_spd.npz')

dates = list(A['dates'])
klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
idx = {d: i for i, d in enumerate(dates)}
hour_cnt = A['hour_cnt'].astype(float)
dist, dist_day = A['dist'].astype(float), A['dist_day'].astype(int)
spdV, spdDay = S['v'].astype(float), S['day'].astype(int)

# ---- 目标日: 2024-04-03 (周三, 99mm 暴雨) ----
tgt = '2024-04-03'
i0 = idx[tgt]
wd = int(np.datetime64(tgt, 'D').view('int64') % 7)   # 当天星期类(epoch基: 0=周四)
# 基线: 所有"干"的同一个星期几(与当天对齐)
wed_dry = [i for i, d in enumerate(dates)
           if klass[d] == 'dry' and int(np.datetime64(d, 'D').view('int64') % 7) == wd]
base_hr = hour_cnt[wed_dry].mean(0)
day_hr = hour_cnt[i0]
# 当天逐小时雨强
rain = {}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    rain[(str(dd), int(hh))] = float(mm)
rain_24 = np.array([rain.get((tgt, h), 0.0) for h in range(24)])

d_day = dist[dist_day == i0]
d_base = dist[np.isin(dist_day, wed_dry)]
v_day = spdV[spdDay == i0]
v_base = spdV[np.isin(spdDay, wed_dry)]

tot_day, tot_base = day_hr.sum(), base_hr.sum()
print(f'{tgt}: 总单量 {tot_day:,.0f} vs 普通干周三 {tot_base:,.0f} → {tot_day/tot_base-1:+.1%}')
print(f'雨: 日总量 {rain_24.sum():.0f}mm, 最大小时 {rain_24.max():.1f}mm/h @ {rain_24.argmax()}时')
print(f'中位距离: {np.median(d_day):.2f} vs {np.median(d_base):.2f} mi')
print(f'中位速度: {np.median(v_day):.1f} vs {np.median(v_base):.1f} mph')
print(f'干周三数 n={len(wed_dry)}')

# 早晨窗 6-9 时对比(雨最大的时段)
w = slice(6, 9)
print(f'早6-9时: 当天 {day_hr[w].sum():,.0f} vs 基线 {base_hr[w].sum():,.0f} → {day_hr[w].sum()/base_hr[w].sum()-1:+.1%}')

# ---- 跨年夜数据 ----
C = np.load('D:/AI/条件转化/.cases.npz')
dec230 = C['dec230'].astype(float) / 5      # 12月普通日子 均值(半小时)
nye230 = C['nye230'].astype(float)          # 12/31
jan1 = C['jan1_230'].astype(float)          # 1/1 凌晨
x48 = np.arange(48) / 2

# ================= 画图 =================
fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
fig.suptitle('两个真实日子：一场暴雨（2024-04-03 全城）和一个跨年夜（时代广场）',
             fontsize=15, fontweight='bold', y=0.99)

# (1) 暴雨天: 每小时订单 + 雨强
ax = fig.add_subplot(gs[0, 0])
ax.bar(np.arange(24), base_hr / 1000, alpha=0.5, color='#A9D18E', label='普通干周三（均值）')
ax.plot(np.arange(24), day_hr / 1000, 'r-o', ms=3.5, lw=2.2, label='2024-04-03 暴雨当天')
ax2 = ax.twinx()
ax2.bar(np.arange(24), rain_24, alpha=0.35, color='#4472C4', width=0.55)
ax2.set_ylabel('下雨强度 mm/h', color='#4472C4'); ax2.set_ylim(0, rain_24.max() * 3)
pk_h = int(rain_24.argmax())
morn_ratio = day_hr[6:9].sum() / base_hr[6:9].sum()
ax.annotate(f'早8点雨最急({rain_24[8]:.1f}mm/h)\n订单 = 平时 {day_hr[8]/base_hr[8]:.1f} 倍\n(早6-9时整体 +{(morn_ratio-1)*100:.0f}%)',
            xy=(8, day_hr[8] / 1000), xytext=(12, day_hr.max() / 1000 * 0.95),
            fontsize=10, color='#C00000', arrowprops=dict(arrowstyle='->', color='#C00000'))
ax.set_xlabel('时刻（时）'); ax.set_ylabel('订单量（千单/小时）')
ax.set_title(f'① 雨下得越猛，打车的人越多：全天 {tot_day/tot_base-1:+.0%}（多 {abs(tot_day-tot_base)/1e4:.0f} 千单）', fontsize=11.5)
ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3)
ax.set_xticks(range(0, 25, 4))

# (2) 暴雨天: 距离
ax = fig.add_subplot(gs[0, 1])
bins = np.arange(0, 10.25, 0.25)
ax.hist(d_base, bins=bins, density=True, alpha=0.5, color='#A9D18E', label=f'普通干周三（中位 {np.median(d_base):.1f} 英里）')
ax.hist(d_day, bins=bins, density=True, alpha=0.55, color='#4472C4', label=f'暴雨当天（中位 {np.median(d_day):.1f} 英里）')
ax.axvline(np.median(d_base), color='#548235', ls='--', lw=1.6)
ax.axvline(np.median(d_day), color='#C00000', ls='--', lw=1.6)
ax.set_xlabel('行程距离（英里）'); ax.set_ylabel('占比')
ax.set_title(f'② 但都是短途：中位距离 {np.median(d_base):.1f}→{np.median(d_day):.1f} 英里（{(np.median(d_day)/np.median(d_base)-1)*100:.0f}%）\n'
             f'步行能到的地方，下雨也要打车', fontsize=11.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (3) 暴雨天: 速度
ax = fig.add_subplot(gs[0, 2])
bins = np.arange(0, 36, 1)
ax.hist(v_base, bins=bins, density=True, alpha=0.5, color='#A9D18E', label=f'普通干周三（中位 {np.median(v_base):.1f} mph）')
ax.hist(v_day, bins=bins, density=True, alpha=0.55, color='#ED7D31', label=f'暴雨当天（中位 {np.median(v_day):.1f} mph）')
ax.axvline(np.median(v_base), color='#548235', ls='--', lw=1.6)
ax.axvline(np.median(v_day), color='#C00000', ls='--', lw=1.6)
ax.set_xlabel('全城平均车速（英里/小时）')
ax.set_ylabel('占比')
ax.set_title(f'③ 而且更堵：中位车速 {np.median(v_base):.1f}→{np.median(v_day):.1f} mph（{(np.median(v_day)/np.median(v_base)-1)*100:.0f}%）\n'
             '同样的路，多花几分钟在路上', fontsize=11.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (4) 跨年: 连续时间轴 (17:00→06:00)
ax = fig.add_subplot(gs[1, :2])
base_line = np.concatenate([dec230[34:48], dec230[0:12]])
d31 = np.concatenate([nye230[34:48], np.full(12, np.nan)])
j1 = np.concatenate([np.full(14, np.nan), jan1[:12]])
t = np.arange(26)
ax.plot(t, base_line, '-', color='#8FAADC', lw=2.2, label='12月普通夜晚')
ax.plot(t, d31, 'o-', color='#C00000', lw=2.2, ms=4, label='12/31 跨年夜（17:00–24:00）')
ax.plot(t, j1, 's-', color='#7030A0', lw=2.2, ms=4, label='1/1 凌晨（00:00–06:00）')
ax.axvline(14, color='#333', ls='--', lw=1.4)
ax.text(14.1, base_line.max() * 1.02, '午夜 0:00', fontsize=10)
ax.annotate('封街了 → 没车可打\n（平时约200单/半小时 → 0~3单）',
            xy=(8, 3), xytext=(2.0, 330), fontsize=10.5, color='#C00000',
            arrowprops=dict(arrowstyle='->', color='#C00000'))
pk = float(np.nanmax(j1))
pk_t = int(np.nanargmax(j1))
hh_hours = list(range(17, 24)) + list(range(0, 7))          # 26个半小时点对应的小时
pk_label = f'{hh_hours[pk_t // 2]}:{"30" if pk_t % 2 else "00"}'
base_night = float(np.mean(base_line[15:21]))               # 凌晨0:30-3:00基线均值
ax.annotate(f'散场了 → 爆发\n（凌晨{pk_label} 达平时 {pk/base_night:.1f} 倍）',
            xy=(pk_t, pk), xytext=(17, pk * 0.88), fontsize=10.5, color='#7030A0',
            arrowprops=dict(arrowstyle='->', color='#7030A0'))
ax.set_xticks(range(0, 26, 2))
ax.set_xticklabels([f'{["17","18","19","20","21","22","23","0","1","2","3","4","5","6"][k//2]}时' for k in range(0, 26, 2)], fontsize=9)
ax.set_xlabel('时间：12/31 傍晚 → 1/1 清晨')
ax.set_ylabel('时代广场(zone 230)出发订单（每半小时）')
ax.set_title('④ 跨年夜时代广场：先"冻结"（封街，订单归零），后"爆发"（午夜散场，冲到平时约3倍，high到凌晨3点半）', fontsize=11.5)
ax.legend(fontsize=9.5, loc='upper right'); ax.grid(alpha=0.3)

# (5) 数字卡片
ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
txt = f'''这天发生了什么（数据事实）
────────────────────
暴雨日 2024-04-03（周三）
· 一天下了 {rain_24.sum():.0f} 毫米雨
  ≈ 平时半个月的雨量
· 订单 +{(tot_day/tot_base-1)*100:.0f}%（多 {(tot_day-tot_base)/1e4:.0f} 千单）
· 中位距离 {np.median(d_base):.1f}→{np.median(d_day):.1f} 英里（{(np.median(d_day)/np.median(d_base)-1)*100:.0f}%）
· 中位车速 {np.median(v_base):.1f}→{np.median(v_day):.1f} mph（{(np.median(v_day)/np.median(v_base)-1)*100:.0f}%）

跨年夜 2024-12-31
· 17点封街后: 订单≈0
  （平时 ~200单/半小时）
· 凌晨{pk_label}: 冲到平时 {pk/base_night:.1f} 倍

一句话:
雨改变"打多少车、多远、多慢"
事件改变"什么时候、在哪里打车"'''
ax.text(0.02, 0.97, txt, fontsize=11.5, va='top', family='Microsoft YaHei',
        bbox=dict(boxstyle='round', facecolor='#FFF2CC', alpha=0.95))

plt.tight_layout()
plt.savefig('D:/AI/条件转化/真实例子_暴雨与跨年.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 真实例子_暴雨与跨年.png')
