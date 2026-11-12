# -*- coding: utf-8 -*-
"""标签生成可视化(柱状版): y ＝ 实测 ÷ 同类型晴天基线
   四个面板统一柱状对形式: 灰柱=基线, 红柱=实测, 顶部标比值 y
   ① 出发时间: 4/3 暴雨日 vs 干周三, 按时段(5档)聚合
   ② 空间分布: 24 强雨日 vs 干日, 按区域组(5组)
   ③ 距离分布: 52 大雨日 vs 干日, 按里程箱(6档)
   ④ 速度分布: 4/3 vs 干周三, 按时段(5档)
输出: 标签生成可视化.png
"""
import numpy as np, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
NAVY = '#1F4E79'
G, R_, O, GRN = '#7F7F7F', '#C00000', '#ED7D31', '#375623'

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
R = np.load(r'D:/AI/条件转化/.rain_spatial.npz')
Z = np.load(r'D:/AI/条件转化/.flow_zones.npz')
dates = list(A['dates']); n = len(dates)
hc = A['hour_cnt'].astype(np.float64)
dist = A['dist'].astype(np.float64); dist_day = A['dist_day'].astype(int)
kl = np.array([dict(zip(list(__import__('numpy').load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)['dates']),
                        __import__('numpy').load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)['klass'].tolist()))[d] for d in dates])
dow = np.array([(np.datetime64(d, 'D').view('int64') + 3) % 7 for d in dates])   # 0=周一
i43 = dates.index('2024-04-03')
wed_dry = np.where((dow == 2) & (kl == 'dry') & (np.arange(n) != i43))[0]
heavy_i = np.where(kl == 'heavy')[0]; dry_i = np.where(kl == 'dry')[0]

SEG = [('凌晨', 0, 6), ('早峰', 7, 9), ('白天', 10, 16), ('晚峰', 17, 20), ('夜间', 21, 23)]
def seg_sum(arr_day_hour, day_idx, lo, hi):
    return float(np.sum(arr_day_hour[day_idx, lo:hi + 1]))

def seg_sum_mean(arr_day_hour, day_list, lo, hi):
    return float(np.mean(np.sum(arr_day_hour[day_list, lo:hi + 1], axis=1)))

fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
fig.suptitle('标签生成可视化：y ＝ 实测 ÷ 同类型晴天基线（柱状对形式；全部真实数据）\n'
             '读法：每个量先分箱 → 灰柱（晴天基线）与红柱（雨天实测）并排 → 顶部红字就是该箱的标签 y',
             fontsize=13, fontweight='bold', y=0.985)

def bar_panel(ax, names, base_vals, act_vals, ylab, title, ylim=None):
    xs = np.arange(len(names)); wd = 0.36
    ax.bar(xs - wd / 2, base_vals, wd, color=G, label='基线（晴天）')
    ax.bar(xs + wd / 2, act_vals, wd, color=R_, label='实测（雨天）')
    for k in range(len(names)):
        yv = act_vals[k] / base_vals[k]
        ax.annotate(f'y={yv:.2f}', (xs[k] + wd / 2, act_vals[k] * 1.02), ha='center',
                    fontsize=10.5, fontweight='bold', color=R_)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11, loc='left')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    if ylim: ax.set_ylim(0, ylim)

# ---------- ① 出发时间（按时段） ----------
names1 = [s[0] for s in SEG]
base1 = [seg_sum_mean(hc, wed_dry, lo, hi) for _, lo, hi in SEG]
act1 = [seg_sum(hc, i43, lo, hi) for _, lo, hi in SEG]
bar_panel(axes[0, 0], names1, base1, act1, '出发量（趟）',
          '① 出发时间：y(时段) = λ实测 ÷ λ基线\n（按时段聚合 → 5 个标签值；24 小时粒度同理）',
          ylim=max(max(base1), max(act1)) * 1.22)

# ---------- ② 空间分布 ----------
pu = R['puB'].astype(np.float64); howB = R['howB']
dryz = R['pu_dry_how'].reshape(7 * 24, 264)
dsz = dryz / np.maximum(dryz.sum(1, keepdims=True), 1)
w_ = howB / max(howB.sum(), 1)
bB = (w_[:, None] * dsz).sum(0); bB /= bB.sum()
rB = pu / max(pu.sum(), 1)
bor = {}
for row in csv.reader(open(r'D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if row and row[0].isdigit(): bor[int(row[0])] = row[1]
d_cbd = np.zeros(264); d_cbd[1:264] = Z['dist_km'][161, 1:264]
AIR = [1, 132, 138]
CORE = [161, 162, 163, 164, 230, 231, 79, 24, 225]
MANH = [i for i in range(1, 264) if bor.get(i) == 'Manhattan' and i not in CORE + AIR]
INNER = [i for i in range(1, 264) if bor.get(i) != 'Manhattan' and i not in AIR and d_cbd[i - 1] < 5.0]
OUTER = [i for i in range(1, 264) if bor.get(i) != 'Manhattan' and i not in AIR and d_cbd[i - 1] >= 5.0]
CATS = [('机场', AIR), ('曼哈顿核心', CORE), ('曼哈顿其余', MANH), ('内环<5km', INNER), ('外环≥5km', OUTER)]
names2 = [c[0] for c in CATS]
base2 = [bB[zz].sum() * 100 for _, zz in CATS]
act2 = [rB[zz].sum() * 100 for _, zz in CATS]
bar_panel(axes[0, 1], names2, base2, act2, '出发份额 %',
          '② 空间分布：y(s) = λ实测(s) ÷ λ基线(s)\n（按区域组 → 每组一个比值；机场 y<1 核心 y>1）',
          ylim=max(max(base2), max(act2)) * 1.22)

# ---------- ③ 距离分布（按里程箱） ----------
dH = dist[np.isin(dist_day, heavy_i)]; dH = dH[(dH > 0.3) & (dH < 60)]
dD = dist[np.isin(dist_day, dry_i)]; dD = dD[(dD > 0.3) & (dD < 60)]
edges = np.array([0.3, 1, 2, 3, 5, 8, 12, 60])
names3 = ['<1', '1~2', '2~3', '3~5', '5~8', '8~12', '≥12']
hH, _ = np.histogram(dH, edges); hD, _ = np.histogram(dD, edges)
base3 = hD / hD.sum() * 100; act3 = hH / hH.sum() * 100
bar_panel(axes[1, 0], names3, base3, act3, '行程占比 %',
          '③ 距离分布：y(d) = p实测(d) ÷ p基线(d)\n（按里程箱 → 每箱一个比值；短途 y>1 长途 y<1）',
          ylim=max(max(base3), max(act3)) * 1.3)

# ---------- ④ 速度分布（按时段） ----------
spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(np.float64)
sel43 = spdD == i43
base4, act4 = [], []
for _, lo, hi in SEG:
    sb = spdV[np.isin(spdD, wed_dry) & (spdH >= lo) & (spdH <= hi)]
    s43 = spdV[sel43 & (spdH >= lo) & (spdH <= hi)]
    base4.append(float(np.median(sb)) if len(sb) >= 100 else np.nan)
    act4.append(float(np.median(s43)) if len(s43) >= 20 else np.nan)
bar_panel(axes[1, 1], names1, base4, act4, '中位速度 (mph)',
          '④ 速度：y(t) = μ实测(t) ÷ μ基线(t)\n（按时段聚合 → 全程 y<1：雨天变慢）',
          ylim=max(np.nanmax(base4), np.nanmax(act4)) * 1.22)

fig.text(0.5, 0.005, '标签 y 的用途：推理时 y(c) 乘回基线 → λ = λ基线 × y。每个标签都有出处：一根实测柱 ÷ 一根基线柱。',
         ha='center', fontsize=11, color=GRN, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 0.955])
fig.savefig(r'D:/AI/条件转化/标签生成可视化.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 标签生成可视化.png (bar version)')
