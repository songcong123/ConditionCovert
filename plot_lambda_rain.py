# -*- coding: utf-8 -*-
"""出发时间 λ(t) 的极端天气逐时效应: NYC 真数据 vs DiDi 合成 对比图"""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

NYC = json.load(open('D:/AI/条件转化/.nyc_hourly_rain_effect.json'))
DD = json.load(open('D:/AI/条件转化/didi_analysis/06_extreme_weather_results.json'))

fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.6))
fig.suptitle('出发时间 λ(t) 的极端天气逐时效应（vs 同星期同时刻干净基线）——雨的形态效应不是均匀缩放',
             fontsize=14, fontweight='bold', y=1.02)

bins3 = [('r_0.3_1', '0.3-1 mm/h（轻）', '#8FAADC'), ('r_1_4', '1-4 mm/h（中）', '#ED7D31'), ('r_ge4', '≥4 mm/h（强）', '#C00000')]
ax = axes[0]
for k, lb, c in bins3:
    v = np.array(NYC[k]) * 100
    ax.plot(range(24), v, lw=2.2, color=c, label=f'{lb} 均值{np.mean(v):+.0f}%')
    pk = int(np.argmax(v))
    ax.annotate(f'{pk:02d}时 {v[pk]:+.0f}%', xy=(pk, v[pk]), xytext=(0, 8), textcoords='offset points',
                fontsize=9, color=c, ha='center')
ax.axhline(0, color='#666', lw=0.8)
ax.axvspan(7, 9, color='#BF9000', alpha=0.10); ax.axvspan(17, 19, color='#BF9000', alpha=0.10)
ax.text(8, ax.get_ylim()[0]*0.05-2, '早高峰', fontsize=9, color='#996300', ha='center')
ax.text(18, ax.get_ylim()[0]*0.05-2, '晚高峰', fontsize=9, color='#996300', ha='center')
ax.set_xticks(range(0, 24, 2)); ax.set_xlabel('时刻（NYC 当地）'); ax.set_ylabel('出发量效应 (%)')
ax.set_title('NYC 真数据（HV FHV 2024 暖季）\n轻雨抬早高峰 +19%；强雨压早高峰 −8%、抬正午/深夜', fontsize=11.5)
ax.legend(fontsize=9.5, loc='upper left'); ax.grid(alpha=0.3)

ax = axes[1]
cmap = {'wf(0,2] 轻度': '#8FAADC', 'wf(2,5] 中度': '#ED7D31', 'wf(5,10] 重度': '#E06666', 'wf>10 极端': '#C00000'}
for row in DD['lambda_profile']:
    c = cmap.get(row['dose'], '#666')
    prof = row['prof48']
    v24 = np.array([np.nanmean(prof[2*i:2*i+2]) for i in range(24)])  # 48槽→24时
    ax.plot(range(24), v24, lw=2.2, color=c, label=f"{row['dose']} 均值{row['mean']:+.0f}%")
    pk = int(np.nanargmax(v24))
    ax.annotate(f'{pk}时 {np.nanmax(v24):+.0f}%', xy=(pk, np.nanmax(v24)), xytext=(0, 8),
                textcoords='offset points', fontsize=9, color=c, ha='center')
ax.axhline(0, color='#666', lw=0.8)
ax.axvspan(17, 19, color='#BF9000', alpha=0.10)
ax.text(18, ax.get_ylim()[0]*0.05-2, '晚高峰', fontsize=9, color='#996300', ha='center')
ax.set_xticks(range(0, 24, 2)); ax.set_xlabel('时刻（模拟城）'); ax.set_ylabel('endo_2 效应 (%)')
ax.set_title('DiDi 合成城（预演场，数值≈真实 2~4 倍）\n剂量越大晚高峰放大越猛（+33%→+110%）', fontsize=11.5)
ax.legend(fontsize=9.5, loc='upper left'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('D:/AI/条件转化/出发时间_极端天气效应.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 出发时间_极端天气效应.png')
