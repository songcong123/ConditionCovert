# -*- coding: utf-8 -*-
import numpy as np, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

bor = {}
for r in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if r and r[0].isdigit():
        bor[r[1]] = bor.get(r[1], 0) + 1

fig = plt.figure(figsize=(19, 13.5))
fig.suptitle('NYC TLC 行程数据集一页概览（研究窗口 2024-01 ~ 2025-12）',
             fontsize=20, fontweight='bold', y=0.98)
gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.30,
                      left=0.055, right=0.975, top=0.90, bottom=0.05)

# ① 四类数据月单量
ax = fig.add_subplot(gs[0, 0])
names = ['HV FHV\n(Uber/Lyft)', 'Yellow\n(黄出租车)', 'FHV\n(传统网约)', 'Green\n(绿出租车)']
vals = [2000, 330, 130, 5]
cols = ['#C00000', '#FFC000', '#8FAADC', '#70AD47']
b = ax.bar(names, vals, color=cols, width=0.6)
ax.set_yscale('log'); ax.set_ylim(1, 9000)
for r, v in zip(b, vals):
    ax.text(r.get_x()+r.get_width()/2, v*1.3, f'{v}万/月', ha='center', fontsize=11.5, fontweight='bold')
ax.set_title('① 四类数据集：月单量对比（log）\nHV FHV 为绝对主力，本项目全下 24 个月', fontsize=12.5)
ax.set_ylabel('单量/月（万，log）')
ax.grid(alpha=0.3, axis='y'); ax.tick_params(labelsize=10)

# ② 时间跨度与资产（条形替代环图，避免嵌套坐标轴重叠）
ax = fig.add_subplot(gs[0, 1])
ax.barh([1.55], [17.9], left=[2009.1], height=0.30, color='#D9D9D9')
ax.barh([1.55], [2.0], left=[2024], height=0.30, color='#C00000')
ax.text(2016.5, 1.95, '全部历史 2009–2026（约 40 亿行，60–80GB）', ha='center', fontsize=10.5, color='#595959')
ax.text(2024.4, 1.13, '本项目窗口 24 个月\n（HV FHV 11.2GB + Green 27MB）', ha='left', fontsize=10.5,
        color='#C00000', fontweight='bold')
# 资产构成：单根水平堆叠条
segs = [(11200, '#C00000', 'HV FHV 99%'), (77, '#ED7D31', '天气'), (27, '#FFC000', 'Green'), (2, '#4472C4', '其他')]
ax.set_xlim(2008.5, 2027); ax.set_ylim(0.1, 2.3); ax.axis('off')
ax.set_title('② 时间跨度与数据资产（单月实测 2011 万行）', fontsize=12.5, pad=18)
# 图例式说明代替堆叠条
ax.text(2009.1, 0.45, '已下载 11.3GB 构成：', fontsize=10.5, color='#333')
xx = 2013.4
for v, c, lab in segs:
    ax.barh([0.45], [v/11200*12.5], left=[xx], height=0.16, color=c)
    if v > 20:
        ax.text(xx + v/11200*12.5/2, 0.45, lab, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    else:
        ax.text(xx, 0.20, lab, fontsize=9, color=c)
    xx += v/11200*12.5 + 0.15

# ③ 字段结构
ax = fig.add_subplot(gs[0, 2])
gvals = [8, 9, 3, 5]
gcols = ['#C00000', '#ED7D31', '#8FAADC', '#70AD47']
ax.bar(range(4), gvals, color=gcols, width=0.58)
for i, v in enumerate(gvals):
    ax.text(i, v+0.3, str(v), ha='center', fontsize=12, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels(['时空核心\n时刻/Zone\n里程/时长', '费用与收费\n车费/小费/过路\nCBD费·机场费', '平台基地\nlicense\nbase编号', '服务标记\n拼车/轮椅\n无障碍'], fontsize=9.2)
ax.set_ylim(0, 11.5); ax.set_ylabel('列数')
ax.grid(alpha=0.3, axis='y')
ax.set_title('③ HV FHV 共 25 列（2025 实测 schema）\n★ cbd_congestion_fee = 逐单 CBD 进入标记', fontsize=12.5)

# ④ 空间粒度
ax = fig.add_subplot(gs[1, 0])
order = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR']
vals = [bor.get(o, 0) for o in order]
ax.barh(range(6), vals, color=['#C00000', '#ED7D31', '#FFC000', '#70AD47', '#4472C4', '#8FAADC'], height=0.6)
ax.set_yticks(range(6))
ax.set_yticklabels(['曼哈顿', '布鲁克林', '皇后区', '布朗克斯', '斯塔滕岛', '纽瓦克机场'], fontsize=10.5)
ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v+1, i, f'{v} 个', va='center', fontsize=10.5)
ax.set_xlim(0, 85)
ax.set_xlabel('Zone 数量（共 263）')
ax.grid(alpha=0.3, axis='x')
ax.set_title('④ 空间粒度：263 个 Taxi Zone（无经纬度）', fontsize=12.5)
ax.text(0.5, -0.30, '场馆锚点：时代广场230 · MSG186 · 巴克莱189 · 洋基69 · 花旗球场253 · 美网93',
        transform=ax.transAxes, ha='center', fontsize=10, color='#C0504D')

# ⑤ 外生变量四表
ax = fig.add_subplot(gs[1, 1:]); ax.axis('off')
tables = [
    ('NOAA 逐小时天气', '#4472C4', '中央公园基准站等 4 站\n2023–2025 共 12 个 CSV（77MB）\n⚠️ UTC 须转 NY 当地时间\njoin 键：(date, hour)'),
    ('事件日历 592 条', '#ED7D31', '球赛 568 场 + 年度活动 24 天\n六类型全覆盖\n含 Zone 锚点与到场/散场时间窗\njoin 键：(date, taxi_zone)'),
    ('联邦假日 60 天', '#70AD47', '2023–2026，NY 州口径\n含周末顺延补假\n中英双语列\njoin 键：date'),
    ('拥堵收费逐日', '#C00000', '2025-01-05 生效\n高峰 9 美元 / 非高峰 2.25 美元\nDiD 天然实验处理变量\njoin 键：date'),
]
for i, (t, c, desc) in enumerate(tables):
    x = 0.02 + i*0.245
    ax.add_patch(FancyBboxPatch((x, 0.04), 0.215, 0.92, boxstyle='round,pad=0.015',
                                fc='white', ec=c, lw=2.2))
    ax.text(x+0.1075, 0.855, t, ha='center', fontsize=12, fontweight='bold', color=c)
    ax.plot([x+0.03, x+0.185], [0.76]*2, color=c, lw=1, alpha=0.5)
    ax.text(x+0.1075, 0.40, desc, ha='center', va='center', fontsize=9.6, color='#333', linespacing=1.8)
ax.set_title('⑤ 配套外生变量表（已构建，含中文列）', fontsize=12.5, pad=14)
ax.text(0.5, -0.10, '关联方式：天气按 (date, hour) · 事件按 (date, zone) · 假日与政策按 date',
        transform=ax.transAxes, ha='center', fontsize=10.5, color='#C0504D')

# ⑥ 已测分布三条曲线
ax = fig.add_subplot(gs[2, 0])
d = np.linspace(0.2, 30, 300)
body = 0.75*np.exp(-((np.log(d)-np.log(1.2))**2)/(2*0.85**2))/d
air = 0.14*np.exp(-((np.log(d)-np.log(17))**2)/(2*0.25**2))/d
tail = 0.012*d**(-2.76); tail[d < 6.5] = 0
ax.plot(d, body+tail, '-', color='#4472C4', lw=2.2)
ax.fill_between(d, air, alpha=0.45, color='#ED7D31')
ax.plot(d, tail, '--', color='#7030A0', lw=1.8)
ax.set_xscale('log'); ax.set_yticklabels([])
ax.set_xlabel('距离 (mi, log)')
ax.text(0.45, 0.55, 'logNormal 主体\n峰≈1.2mi', fontsize=9.5, color='#2E4B8F', transform=ax.transAxes)
ax.text(0.60, 0.75, '机场鼓包 1.4×', fontsize=9.5, color='#C0504D', transform=ax.transAxes)
ax.text(0.62, 0.45, 'Pareto 尾\nα≈2.76', fontsize=9.5, color='#7030A0', transform=ax.transAxes)
ax.set_title('⑥a 距离分布：混合形态（实测 n=2011万）', fontsize=12)

ax = fig.add_subplot(gs[2, 1])
v = np.linspace(0.5, 45, 300)
mu, s = 2.48, 0.46
pdf = np.exp(-((np.log(v)-mu)**2)/(2*s**2))/(v*s*np.sqrt(2*np.pi))
ax.plot(v, pdf, color='#70AD47', lw=2.3)
ax.fill_between(v, pdf, alpha=0.25, color='#70AD47')
ax.axvline(11.5, color='#C00000', ls='--', lw=1.5)
ax.text(13, pdf.max()*0.86, '中位 11.5 mph ≈ 18.5 km/h', fontsize=9.8, color='#C00000')
ax.set_xlabel('速度 (mph)'); ax.set_yticklabels([])
ax.set_title('⑥b 速度：截断 LogNormal，σ(log v)≈0.46 稳定', fontsize=12)

ax = fig.add_subplot(gs[2, 2])
share24 = [4.0,2.8,2.0,1.6,1.6,1.8,2.7,3.9,4.8,4.6,4.3,4.3,4.4,4.5,4.9,5.0,5.1,5.5,5.7,5.5,5.1,5.4,5.5,5.0]
ax.fill_between(range(24), share24, color='#BDD7EE', alpha=0.75)
ax.plot(range(24), share24, color='#2E75B6', lw=2.2)
ax.annotate('晚峰 18 时 5.7%', xy=(18, 5.7), xytext=(8, 5.35), fontsize=9.8, color='#C00000',
            arrowprops=dict(arrowstyle='->', color='#C00000'))
ax.set_xlabel('小时'); ax.set_yticklabels([]); ax.set_xticks(range(0, 24, 4))
ax.set_ylim(0, 6.6)
ax.set_title('⑥c λ(t)：晚峰>早峰（午后是速度谷非需求谷）', fontsize=12)

plt.savefig('D:/AI/条件转化/NYC数据集介绍.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved')
