# -*- coding: utf-8 -*-
"""validate_49: 天气条件分布摸底 —— 两个数据源对比
   滁州·业务数据(数值天气, 2023-03-31~04-08, 216小时): 天气类型/雨强/温湿风
   NYC·验证数据(ISD实测, 2024-2025 两年, 小时级): 干湿占比/雨强分箱/温湿
   输出: 天气分布摸底_两数据源.png + 控制台统计表
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; ORANGE = '#ED7D31'; GREY = '#7F7F7F'; MUTED = '#666666'; DGREEN = '#375623'
LBLUE = '#8FAADC'; RED = '#C00000'

# ================= 滁州(数值天气) =================
cz = pd.read_csv(r'D:/AI/条件转化/天气数值/chuzhou2_数值天气.csv')
cz_r = cz.rain.values
cz_t = cz.temperature_2m.values
cz_h = cz.relative_humidity_2m.values
cz_w = cz.wind_speed_10m.values
n_rain_cz = int((cz_r > 0).sum())
print('=' * 62)
print('滁州·业务数据(数值天气 2023-03-31~04-08, 216 小时)')
print('=' * 62)
print(f"天气类型(小时): {cz.text.value_counts().to_dict()}")
print(f"有雨小时: {n_rain_cz}/216 ({n_rain_cz/216*100:.0f}%) | 雨强范围 {cz_r[cz_r>0].min()}~{cz_r.max()} mm/h")
print(f"  主雨日 4月3日: {int(((cz.date=='2023-04-03')&(cz_r>0)).sum())} 个有雨小时")
print(f"温度: {cz_t.min():.1f}~{cz_t.max():.1f}°C (均值 {cz_t.mean():.1f})")
print(f"湿度: {cz_h.min():.0f}~{cz_h.max():.0f}% (均值 {cz_h.mean():.0f})")
print(f"风速: {cz_w.min():.1f}~{cz_w.max():.1f} km/h (均值 {cz_w.mean():.1f})")

# ================= NYC(ISD 实测, 复用 validate_41 解析) =================
from datetime import datetime, timedelta

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
                     usecols=['DATE', 'REPORT_TYPE', 'AA1', 'TMP', 'DEW'], dtype=str)
    fm = df[df['REPORT_TYPE'] == 'FM-15']
    for _, row in fm.iterrows():
        try:
            utc = datetime.strptime(row['DATE'], '%Y-%m-%dT%H:%M:%S')
        except Exception:
            continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        key = (loc.strftime('%Y-%m-%d'), loc.hour)
        e = acc.setdefault(key, dict(rain=[], temp=[], dew=[]))
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

ny_r, ny_t, ny_h = [], [], []
for key, e in acc.items():
    if e['rain']: ny_r.append(np.mean(e['rain']))
    if e['temp']:
        t = np.mean(e['temp'])
        ny_t.append(t)
        if e['dew']:
            td = np.mean(e['dew'])
            if td <= t:
                et = 6.112 * np.exp(17.67 * t / (243.5 + t))
                ed = 6.112 * np.exp(17.67 * td / (243.5 + td))
                ny_h.append(min(max(ed / et * 100, 5), 100))
ny_r = np.array(ny_r); ny_t = np.array(ny_t); ny_h = np.array(ny_h)

bins_r = [('0(无雨)', 0, 0.01), ('微量 0-0.3', 0.01, 0.3), ('0.3-1', 0.3, 1), ('1-4', 1, 4),
          ('4-12', 4, 12), ('≥12', 12, 1e9)]
print('\n' + '=' * 62)
print(f'NYC·验证数据(ISD 站72505394728 实测, 2024-2025, {len(ny_r):,} 小时)')
print('=' * 62)
tot = len(ny_r)
for nm, lo, hi in bins_r:
    cnt = int(((ny_r >= lo) & (ny_r < hi)).sum())
    print(f'  {nm:10s}: {cnt:6,} 小时 ({cnt/tot*100:5.1f}%)')
cal = ((ny_r >= 0.3) & (ny_r < 4)).sum() / tot * 100
ext = (ny_r >= 4).sum() / tot * 100
print(f'  → 公式标定域(0.3-4)占 {cal:.1f}% | 外推域(≥4)占 {ext:.1f}%')
print(f'温度: {ny_t.min():.1f}~{ny_t.max():.1f}°C | 湿度: {ny_h.min():.0f}~{ny_h.max():.0f}%')

# ================= 图 2行×4列 =================
fig, axes = plt.subplots(2, 4, figsize=(16, 8.8))
fig.suptitle('天气条件分布摸底：滁州·业务数据（9天数值天气） vs 纽约·验证数据（2年实测）',
             fontsize=14, fontweight='bold', y=0.99)

# ---- 行头标注 ----
fig.text(0.008, 0.955, '滁州·业务数据 2023.3.31-4.8', fontsize=11, fontweight='bold',
         color='#8C5A00', rotation=90, va='top', ha='center')
fig.text(0.008, 0.475, '纽约·验证数据 2024-2025', fontsize=11, fontweight='bold',
         color=NAVY, rotation=90, va='top', ha='center')

# ① 滁州天气类型
ax = axes[0, 0]
vc = cz.text.value_counts().sort_values()
bars = ax.barh(vc.index, vc.values, color=['#F2C9A0', '#E8B98A', '#D9995F', '#C9763B', '#A85A2A'][:len(vc)], edgecolor='black', linewidth=0.5)
for b, v in zip(bars, vc.values):
    ax.annotate(f'{v}h ({v/216*100:.0f}%)', (v + 1.5, b.get_y() + b.get_height()/2), va='center', fontsize=9)
ax.set_xlim(0, 118)
ax.set_xlabel('小时数')
ax.set_title('① 天气类型占比(共216小时)', fontsize=11, loc='left', color='#8C5A00')
ax.grid(alpha=0.3, axis='x')

# ② 滁州雨强
ax = axes[0, 1]
m = cz_r > 0
ax.hist(cz_r[m], bins=np.arange(0, 9.6, 0.6), color=ORANGE, edgecolor='black', linewidth=0.5)
ax.axvline(4, color=RED, ls='--', lw=1.4)
ax.text(4.15, ax.get_ylim()[1]*0.9, 'r=4\n(外推线)', fontsize=8.5, color=RED)
ax.set_xlabel('小时雨强 (mm/h)'); ax.set_ylabel('小时数')
ax.set_title(f'② 有雨小时的雨强分布(n={n_rain_cz})\n   主雨4月3日, 峰值8.6mm/h', fontsize=11, loc='left', color='#8C5A00')
ax.grid(alpha=0.3)

# ③ 滁州温度
ax = axes[0, 2]
ax.hist(cz_t, bins=14, color='#B7D7A8', edgecolor='black', linewidth=0.5)
ax.axvline(cz_t.mean(), color=DGREEN, ls='--', lw=1.4)
ax.annotate(f'均值{cz_t.mean():.1f}°C', (cz_t.mean()+0.4, ax.get_ylim()[1]*0.88), fontsize=9, color=DGREEN)
ax.set_xlabel('气温 (°C)')
ax.set_title(f'③ 气温分布: {cz_t.min():.1f}~{cz_t.max():.1f}°C\n   (春季一周, 无极端温度)', fontsize=11, loc='left', color='#8C5A00')
ax.grid(alpha=0.3)

# ④ 滁州湿度+风速(双子图内嵌太挤→湿度直方+风速文字)
ax = axes[0, 3]
ax.hist(cz_h, bins=12, color='#C7B8E0', edgecolor='black', linewidth=0.5)
ax.set_xlabel('相对湿度 (%)')
ax.set_title(f'④ 湿度分布: {cz_h.min():.0f}~{cz_h.max():.0f}%\n   (风速 {cz_w.min():.0f}~{cz_w.max():.0f} km/h)', fontsize=11, loc='left', color='#8C5A00')
ax.grid(alpha=0.3)

# ⑤ NYC 干湿占比+雨强分箱
ax = axes[1, 0]
names = [b[0] for b in bins_r][1:]
cnts = [int(((ny_r >= lo) & (ny_r < hi)).sum()) for _, lo, hi in bins_r][1:]
dry = tot - sum(cnts)
names = ['无雨'] + names; cnts = [dry] + cnts
cols = [GREY] + ['#CFE0F2', LBLUE, NAVY, ORANGE, RED]
bars = ax.bar(range(len(names)), cnts, color=cols, edgecolor='black', linewidth=0.5)
ax.set_yscale('log')
for i, (b, v) in enumerate(zip(bars, cnts)):
    ax.annotate(f'{v/tot*100:.1f}%', (i, v*1.25), ha='center', fontsize=8.5)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8.5, rotation=20)
ax.set_ylabel('小时数(log)')
ax.set_title('⑤ 各雨强档小时占比\n   蓝=标定域0.3-4 | 橙红=外推域≥4', fontsize=11, loc='left', color=NAVY)
ax.grid(alpha=0.3, axis='y', which='both')

# ⑥ NYC 雨强 log直方
ax = axes[1, 1]
m = ny_r > 0.01
ax.hist(ny_r[m], bins=np.logspace(np.log10(0.05), np.log10(140), 60), color=LBLUE, edgecolor=NAVY, linewidth=0.4)
ax.axvspan(0.3, 4, color=NAVY, alpha=0.10)
ax.axvspan(4, 140, color=RED, alpha=0.08)
ax.axvline(4, color=RED, ls='--', lw=1.4)
ax.text(0.36, ax.get_ylim()[1]*0.6, '标定域', fontsize=9, color=NAVY)
ax.text(5.5, ax.get_ylim()[1]*0.8, '外推域', fontsize=9, color=RED)
ax.set_xscale('log')
ax.set_xlabel('小时雨强 (mm/h, log)'); ax.set_ylabel('小时数')
ax.set_title(f'⑥ 有雨小时雨强分布(n={int(m.sum()):,})\n   峰值{ny_r.max():.0f}mm/h', fontsize=11, loc='left', color=NAVY)
ax.grid(alpha=0.3, which='both')

# ⑦ NYC 温度
ax = axes[1, 2]
ax.hist(ny_t, bins=48, color='#F2CB9B', edgecolor='#B98A4A', linewidth=0.4)
ax.axvline(0, color=MUTED, ls=':', lw=1.2)
ax.set_xlabel('气温 (°C)')
ax.set_title(f'⑦ 气温分布: {ny_t.min():.0f}~{ny_t.max():.0f}°C\n   (全年覆盖, 冬夏两坨)', fontsize=11, loc='left', color=NAVY)
ax.grid(alpha=0.3)

# ⑧ NYC 湿度
ax = axes[1, 3]
ax.hist(ny_h, bins=36, color='#C7B8E0', edgecolor='black', linewidth=0.4)
ax.set_xlabel('相对湿度 (%)')
ax.set_title(f'⑧ 湿度分布: {ny_h.min():.0f}~{ny_h.max():.0f}%\n   (右偏, 高湿常见)', fontsize=11, loc='left', color=NAVY)
ax.grid(alpha=0.3)

fig.text(0.5, 0.005,
         '口径: 滁州=ECMWF数值天气预报(小时步长, rain≈mm/h); 纽约=ISD地面实测(FM-15小时报告, AA1雨量×25.4mm)。\n'
         '要点: 业务数据(滁州)雨强覆盖0.1~8.6mm/h→以标定域为主; 验证数据(纽约)两年含4~134mm/h强雨→外推域有充足实测可盲测',
         ha='center', fontsize=9.5, color=DGREEN, fontweight='bold')
plt.tight_layout(rect=[0.015, 0.02, 1, 0.97])
fig.savefig(r'D:/AI/条件转化/天气分布摸底_两数据源.png', dpi=150, bbox_inches='tight', facecolor='white')
print('\nsaved 天气分布摸底_两数据源.png')
