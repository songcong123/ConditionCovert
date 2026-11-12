# -*- coding: utf-8 -*-
"""空间分布真实例子图(全中文版):
① 美网: 观众从哪里打车去法拉盛(圆=人多, 色=翻倍)
② 马拉松: 封路时段哪些区域打车被冻住(红=受抑)
③ 对照: 暴雨天(无空间图案)
④ 中文卡片"""
import numpy as np, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import geopandas as gpd

SP = np.load('D:/AI/条件转化/.flow_spatial.npz')
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
Z = np.load('D:/AI/条件转化/.flow_zones.npz')

# ---- 地名英→中 ----
TRANS = [('Times Sq', '时代广场'), ('Penn Station', '宾州车站'), ('Midtown', '曼哈顿中城'),
         ('Upper East', '上东城'), ('Upper West', '上西城'), ('Central Park', '中央公园'),
         ('East Village', '东村'), ('West Village', '西村'), ('Harlem', '哈莱姆'),
         ('Soho', '苏豪区'), ('Wall St', '华尔街'), ('Financial', '金融区'),
         ('Battery', '炮台公园'), ('Lower East', '下东区'), ('Downtown', '下城'),
         ('Murray Hill', '默里山'), ('Chelsea', '切尔西'), ('Hell', '地狱厨房'),
         ('Flushing', '法拉盛'), ('LaGuardia', '拉瓜迪亚机场'), ('JFK', '肯尼迪机场'),
         ('Forest Hills', '森林山'), ('Astoria', '阿斯托里亚'), ('Long Island City', '长岛市'),
         ('Jamaica', '牙买加区'), ('Corona', '科罗纳'), ('Fresh Meadows', '新鲜草原'),
         ('Williamsburg', '威廉斯堡'), ('Greenpoint', '绿点区'), ('Boerum', '布伦姆山'),
         ('Park Slope', '公园坡'), ('Brooklyn Height', '布鲁克林高地'), ('DUMBO', '丹波区'),
         ('Downtown/Brooklyn', '布鲁克林下城'), ('Bedford', '贝德福德'), ('Bushwick', '布什威克'),
         ('Bay Ridge', '湾脊区'), ('Coney', '康尼岛'), ('Borough Park', '自治领公园'),
         ('Mott Haven', '莫特黑文'), ('Hunts Point', '亨茨角'), ('Highbridge', '高桥区'),
         ('South Bronx', '南布朗克斯'), ('Yankee', '洋基体育场'),
         ('Newark', '纽瓦克机场')]
BOR_ZH = {'Manhattan': '曼哈顿', 'Brooklyn': '布鲁克林', 'Queens': '皇后区',
          'Bronx': '布朗克斯', 'Staten Island': '斯塔滕岛', 'EWR': '新泽西·纽瓦克'}

zn, zb = {}, {}
for r in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv', encoding='utf-8-sig', newline='')):
    if r and r[0].isdigit(): zn[int(r[0])] = r[2]; zb[int(r[0])] = r[1]

def zh(i):
    s = zn.get(i, str(i))
    for k, v in TRANS:
        if k in s: return v
    return s

g = gpd.read_file('D:/AI/条件转化/data/tlc/zones/_shp/taxi_zones/taxi_zones.shp').to_crs(4326)
lat, lon = Z['lat'], Z['lon']
# 五大区中心(打大字用)
bor_pt, bor_name = [], []
for b, sub in g.groupby('borough'):
    if b in BOR_ZH:
        p = sub.geometry.unary_union.representative_point()
        bor_pt.append((p.x, p.y)); bor_name.append(BOR_ZH[b])

# ---- NYC 马拉松赛道路线(手绘近似, 依次跨: 韦拉扎诺桥/普拉斯基桥/皇后区桥/威利斯桥/麦迪逊桥) ----
ROUTE = np.array([
    (40.6066, -74.0248),  # 起点: 斯塔滕岛 Fort Wadsworth
    (40.6100, -74.0300), (40.6155, -74.0270),   # 韦拉扎诺海峡大桥(过海峡)
    (40.6190, -74.0210), (40.6260, -74.0170),   # 布鲁克林湾脊区
    (40.6380, -74.0130), (40.6500, -74.0110), (40.6620, -74.0100), (40.6740, -74.0090),  # 4大道北上
    (40.6850, -74.0100), (40.6890, -73.9950),   # 布伦姆山一带(受抑区)
    (40.6960, -73.9680), (40.7040, -73.9560), (40.7140, -73.9500),  # 南威廉斯堡/贝德福德
    (40.7250, -73.9530), (40.7330, -73.9520),   # 绿点区(受抑区)
    (40.7420, -73.9480),                        # 普拉斯基桥(进皇后区)
    (40.7450, -73.9430), (40.7500, -73.9480),   # 皇后区长岛市
    (40.7540, -73.9550), (40.7575, -73.9600),   # 皇后区大桥(进曼哈顿)
    (40.7650, -73.9400), (40.7750, -73.9385), (40.7850, -73.9370), (40.7970, -73.9340),  # 一大道北上
    (40.8035, -73.9290),                        # 威利斯桥(进布朗克斯)
    (40.8090, -73.9250), (40.8120, -73.9200),   # 布朗克斯138街小弯
    (40.8090, -73.9300),                        # 麦迪逊桥(回曼哈顿)
    (40.7980, -73.9500), (40.7900, -73.9580), (40.7800, -73.9630), (40.7720, -73.9690),  # 五大道南下
    (40.7690, -73.9750),                        # 终点: 中央公园
])

# ---------- 数据 ----------
is_evt = SP['is_evt']; uso = SP['uso_arr'].astype(float)
ev = uso[is_evt].sum(0) / is_evt.sum()
bs = uso[~is_evt].sum(0) / (~is_evt).sum()
mult = (ev + 2) / (bs + 2)
ev_tot, bs_tot = ev.sum(), bs.sum()
top5m = [i for i in np.argsort(-ev) if bs[i] >= 3][:5]

nov = SP['nov_pu'].astype(float)
mar = nov[0, 8:17].sum(axis=0)
bas = nov[1:, 8:17].sum(axis=1).mean(axis=0)
chg_m = np.where(bas >= 40, mar / np.maximum(bas, 1) - 1, np.nan)
mar_tot = mar.sum() / bas.sum() - 1
worst = np.argsort(np.nan_to_num(chg_m, nan=0))[:6]

dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klass[d] for d in dates])
pu = A['pu_cnt'].astype(float)
pu_he = pu[kl == 'heavy'].sum(0) / (kl == 'heavy').sum()
pu_dr = pu[kl == 'dry'].sum(0) / (kl == 'dry').sum()
city_ratio = pu_he.sum() / pu_dr.sum()
chg_r = np.where(pu_dr >= 300, (pu_he / pu_dr) / city_ratio - 1, np.nan)
ok = ~np.isnan(chg_r)

def gv(arr):
    v = np.full(264, np.nan); v[1:264] = arr[1:264]; return v

# ================= 画图 =================
fig = plt.figure(figsize=(18, 11.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
fig.suptitle('看地图识冲击：事件会在地图上"写字"，暴雨只会把整张图均匀调亮（全部真实数据）',
             fontsize=15, fontweight='bold', y=0.985)

def base_map(ax, title, sub, stats=None):
    g.plot(ax=ax, color='#F2F2F2', edgecolor='#BFBFBF', lw=0.4)
    for (x, y), nm in zip(bor_pt, bor_name):
        ax.text(x, y, nm, fontsize=15, color='#9E9E9E', ha='center', va='center',
                fontweight='bold', zorder=0, alpha=0.85)
    full = title + '\n' + sub + (f'\n统计周期：{stats}' if stats else '')
    ax.set_title(full, fontsize=12.5, loc='left')
    ax.set_xticks([]); ax.set_yticks([])

# ① 美网客源
ax = fig.add_subplot(gs[0, 0])
base_map(ax, '① 美网公开赛两周：观众从全城打车涌向法拉盛的球场（星标）',
         '怎么看：圆圈越大=人越多；颜色越红=翻倍越多；箭头=观众流向',
         '2024-08-26～09-08（14天） vs 前后各14天基线')
plot_z = [i for i in np.argsort(-ev) if ev[i] >= 25 and 1 <= i <= 263]
sc = ax.scatter([lon[i] for i in plot_z], [lat[i] for i in plot_z],
                s=[ev[i] * 1.1 for i in plot_z], c=[mult[i] for i in plot_z],
                cmap='YlOrRd', vmin=1, vmax=8, alpha=0.85, edgecolor='#7F4E0E', lw=0.5)
ax.scatter([lon[93]], [lat[93]], marker='*', s=700, c='#C00000', edgecolor='white', zorder=6)
ax.annotate('★ 法拉盛\n（美网球场，皇后区）', xy=(lon[93], lat[93]), xytext=(lon[93] - 0.24, lat[93] + 0.05),
            fontsize=11, color='#C00000', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.2))
# 客流箭头: 高倍客源地(机场/中城/威廉斯堡, 按名匹配) → 球场(弧线)
arrow_src = [i for i in top5m if any(k in zh(i) for k in ('拉瓜迪亚', '威廉斯堡', '中城'))]
for k, src in enumerate(arrow_src):
    rad = [0.25, -0.22, 0.18][k % 3]
    ax.annotate('', xy=(lon[93], lat[93]), xytext=(lon[src], lat[src]),
                arrowprops=dict(arrowstyle='-|>', color='#7F4E0E', lw=2.0, alpha=0.8,
                                connectionstyle=f'arc3,rad={rad}', shrinkA=14, shrinkB=26), zorder=5)
# 标签: 沿"远离球场"方向摆放(防重叠)
for i in top5m:
    dx, dy = lon[i] - lon[93], lat[i] - lat[93]
    d = np.hypot(dx, dy) + 1e-9
    r = max(d, 0.050)
    ax.annotate(f'{zh(i)}\n×{ev[i]/max(bs[i],0.1):.1f}', xy=(lon[i], lat[i]),
                xytext=(lon[93] + dx / d * r, lat[93] + dy / d * r + 0.012), fontsize=9.5, color='#222',
                ha='center', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#999', alpha=0.9))
plt.colorbar(sc, ax=ax, shrink=0.72, label='打车量是平时的几倍（红=高倍）', pad=0.01)

# ② 马拉松
ax = fig.add_subplot(gs[0, 1])
base_map(ax, '② 马拉松周日封路时段：红色=打车被"冻住"的区域',
         '怎么看：越红=打得越少；黑虚线=赛道路线（跑五区、过河走桥）；白框=沿线红区，橙框=非沿线红区（原因见框）',
         '2025-11-02（当天8–17时） vs 同月另外4个周日的同一时段')
gm = g.copy(); gm['chg'] = [gv(chg_m)[l] if 1 <= l <= 263 else np.nan for l in g['LocationID']]
gm.plot(ax=ax, column='chg', cmap='RdBu_r', vmin=-0.7, vmax=0.7,
        missing_kwds=dict(color='#D9D9D9'), edgecolor='#BFBFBF', lw=0.4, legend=True,
        legend_kwds={'label': '打车量变化（马拉松日 vs 普通周日的同一时段）', 'shrink': 0.72, 'pad': 0.01})
# 赛道路线: 白色描边 + 黑虚线
ax.plot(ROUTE[:, 1], ROUTE[:, 0], color='white', lw=5.5, solid_capstyle='round', zorder=4)
ax.plot(ROUTE[:, 1], ROUTE[:, 0], color='#1A1A1A', lw=2.6, ls=(0, (6, 3)), zorder=5)
ax.scatter([ROUTE[0, 1]], [ROUTE[0, 0]], marker='*', s=420, c='#006838', edgecolor='white', zorder=6)
ax.scatter([ROUTE[-1, 1]], [ROUTE[-1, 0]], marker='s', s=150, c='#1A1A1A', edgecolor='white', zorder=6)
ax.annotate('起点\n斯塔滕岛', xy=(ROUTE[0, 1], ROUTE[0, 0]), xytext=(-74.16, 40.565),
            fontsize=10, color='#006838', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#006838', lw=1.2), zorder=6)
ax.annotate('终点\n中央公园', xy=(ROUTE[-1, 1], ROUTE[-1, 0]), xytext=(-74.06, 40.835),
            fontsize=10, color='#1A1A1A', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=1.2), zorder=6)
for i in worst[::-1][:4]:
    if not np.isnan(chg_m[i]) and chg_m[i] < -0.40 and i != 12:
        ax.annotate(f'{zh(i)}\n{chg_m[i]*100:.0f}%', xy=(lon[i], lat[i]), fontsize=9.5,
                    color='#404040', ha='center',
                    xytext=(lon[i] - 0.02, lat[i] - 0.055),
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#999', alpha=0.85))
# 非沿线红区解释(橙框)
ax.annotate('炮台公园 −42%（非沿线）\n渡轮码头：早+71% 观众坐船去看起跑\n午后下城封控 −54%',
            xy=(lon[12], lat[12]), xytext=(lon[12] - 0.145, lat[12] + 0.006), fontsize=9,
            color='#8C4A0B', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FFF3E6', ec='#E36C0A', alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#E36C0A', lw=1.2), zorder=7)
ax.annotate('时代广场/中城 −36%（非沿线）\n终点安保圈 + 中城多街封路',
            xy=(lon[230], lat[230]), xytext=(lon[230] - 0.185, lat[230] - 0.075), fontsize=9,
            color='#8C4A0B', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FFF3E6', ec='#E36C0A', alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#E36C0A', lw=1.2), zorder=7)

# ③ 强雨日对照
ax = fig.add_subplot(gs[1, 0])
base_map(ax, '③ 对照组：暴雨天，地图上什么图案都没有',
         f'怎么看：已扣掉全市整体 +{(city_ratio-1)*100:.0f}% 的增幅，只剩 ±5% 左右的随机晃动（灰=样本太少不计）',
         '2024-04～10 中 24 个强雨日（日降水≥15mm）平均 vs 146 个晴天平均')
gr = g.copy(); gr['chg'] = [gv(chg_r)[l] if 1 <= l <= 263 else np.nan for l in g['LocationID']]
gr.plot(ax=ax, column='chg', cmap='RdBu_r', vmin=-0.7, vmax=0.7,
        missing_kwds=dict(color='#D9D9D9'), edgecolor='#BFBFBF', lw=0.4, legend=True,
        legend_kwds={'label': '打车量变化（强雨日 vs 晴天，扣整体增幅后）', 'shrink': 0.8, 'pad': 0.01})

# ④ 卡片
ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
names5 = '、'.join(f'{zh(i)}×{ev[i]/max(bs[i],.1):.1f}' for i in top5m[:3])
txt = f'''三张地图在说什么
──────────────────
① 美网（点状汇聚）
  去法拉盛球场的打车 ×{ev_tot/bs_tot:.1f}
  （{bs_tot:.0f}→{ev_tot:.0f} 单/日）
  涨得最凶的客源地：
  {names5}…
  → 机场、酒店区、网红街区

② 马拉松（线状冻结）
  全市 −{abs(mar_tot)*100:.0f}%，
  {int(np.nansum(chg_m<-0.3))} 个小区跌超 30%
  黑虚线=路线: 斯塔滕岛出发→
  布鲁克林→皇后区→曼哈顿→
  布朗克斯→终点中央公园(过河走桥)
  两个非沿线红区也有解释:
  · 炮台公园: 渡轮码头,早+71%
    (坐船看起跑)午后封控−54%
  · 时代广场: 终点安保圈+中城封路

③ 暴雨（均匀调亮）
  {np.mean(np.abs(chg_r[ok])<0.05)*100:.0f}% 的小区变化在 ±5% 内
  → 找不到任何图案

一句话:
事件改变"在哪里打车"（空间）
暴雨改变"打多少、多远、多慢"（幅度）'''
ax.text(0.02, 0.98, txt, fontsize=11.5, va='top', family='Microsoft YaHei',
        bbox=dict(boxstyle='round', facecolor='#FFF2CC', alpha=0.95))

plt.tight_layout()
plt.savefig('D:/AI/条件转化/空间分布真实例子.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 空间分布真实例子.png(全中文)')
