# -*- coding: utf-8 -*-
"""索引维 × 泛化维 示意图: 格子的作用与组合方式
   例A: 索引维=(工作日×小时)→24格, 泛化维=天气(每格一条响应曲线)
   例B: 索引维=(工作日)→1大格, 泛化维=小时+天气(格内两条曲线)
   合成示意结构, 非真实数据
输出: 索引泛化示意图.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; OR = '#ED7D31'; GREY = '#7F7F7F'
GRN = '#375623'; GRNB = '#70AD47'; LG = '#F2F2F2'
MUTED = '#666666'; TEXTC = '#1A1A1A'

fig = plt.figure(figsize=(16, 9), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def box(x, y, w, h, fc='white', ec=NAVY, lw=1.2, ls='-'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls))
def txt(x, y, s, size=10, color='black', bold=False, ha='left', va='center'):
    ax.text(x, y, s, fontsize=size, color=color,
            fontweight='bold' if bold else 'normal', ha=ha, va=va)

# ---------- 标题 ----------
txt(0.5, 0.962, '索引维 × 泛化维 示意图：格子的作用与组合方式（示意结构）', 17, NAVY, True, 'center')
txt(0.5, 0.928, '索引维 = 格子地址（值永远已知：查表定位）｜ 泛化维 = 格子里的曲线（取值可能没见过：评估/外推）｜ 每个格子 = 晴天基线值 ＋ 每条泛化曲线', 10.5, MUTED, False, 'center')

# ---------- 例 A ----------
txt(0.03, 0.892, '例 A：索引维 =（工作日 × 小时）→ 24 个格子；泛化维 = 天气', 11.5, NAVY, True)
txt(0.52, 0.892, '泛化维=天气：每格存一条响应曲线 ↓', 10, OR, True)
H = [6, 8, 10, 12, 14, 16, 18, 20]
BASE = [18000, 36000, 42000, 38000, 37000, 36000, 34000, 28000]
X0, CW, CY, CH = 0.03, 0.088, 0.755, 0.078
for k, (h, b) in enumerate(zip(H, BASE)):
    x = X0 + k * (CW + 0.004)
    hot = (h == 8)
    box(x, CY, CW, CH, 'white', RED if hot else NAVY, 2.0 if hot else 1.1)
    txt(x + CW / 2, CY + CH * 0.70, f'{b:,}', 10, NAVY, True, 'center')
    xs = np.linspace(x + 0.008, x + CW - 0.008, 12)
    rr = np.linspace(0, 1, 12)
    ys = CY + 0.008 + 0.030 * (1 - np.exp(-rr * 2.2))
    ax.plot(xs, ys, lw=1.6, color=RED if hot else GREY)
    txt(x + CW / 2, CY - 0.025, f'{h}时', 9.5, RED if hot else 'black', hot, 'center')
for xa in [0.12, 0.34, 0.56]:
    ax.annotate('', xy=(xa, CY + CH), xytext=(xa, 0.880),
                arrowprops=dict(arrowstyle='->', color=OR, lw=1.4))
txt(0.03, 0.700, '推理 c =（工作日, 8时, 暴雨30mm/h）：定位格子(工作日,8时) → 读晴天基线 36,000 → 天气曲线(30) = ×1.26 → 早峰出发 45,400 趟',
    10.5, TEXTC, False)

# ---------- 例 B ----------
txt(0.03, 0.648, '例 B：索引维 =（工作日）→ 1 个大格子；泛化维 = 小时 ＋ 天气（格内存两条曲线）', 11.5, NAVY, True)
box(0.03, 0.500, 0.41, 0.130, 'white', NAVY, 1.8)
txt(0.045, 0.615, '工作日格', 9.5, NAVY, True)
hv = np.linspace(0, 23, 24)
prof = 0.62 * np.exp(-((hv - 8) / 3.2) ** 2) + 0.38 * np.exp(-((hv - 18) / 3.2) ** 2)
xs = np.linspace(0.055, 0.225, 24); ys = 0.545 + 0.075 * prof / prof.max()
ax.plot(xs, ys, lw=2.0, color=GREY)
xs2 = np.linspace(0.265, 0.415, 20); rr2 = np.linspace(0, 50, 20)
ys2 = 0.545 + 0.075 * (1 - np.exp(-rr2 / 12))
ax.plot(xs2, ys2, lw=2.0, color=RED)
txt(0.055, 0.512, '泛化维=小时：λ0(t) 曲线', 8.5, GREY)
txt(0.262, 0.512, '泛化维=天气：δvol(r) 曲线', 8.5, RED)
txt(0.46, 0.590, '同一个"工作日出行量"模型：\nA = 24 格查表 ｜ B = 1 格两条曲线\n→ 答案一致（45,400 趟）', 10.5, TEXTC, False)
txt(0.46, 0.520, '选哪种？数据多→A（逐格精度高）；\n求平滑省参数→B（曲线可内插外推）', 10, MUTED, False)

txt(0.03, 0.452, '推理三步：① 定位格子（索引维永远已知）② 读晴天基线 ③ 泛化维逐个评估——域内插值（RF）、域外沿曲线外推', 11, NAVY, True)

# ---------- 底部三卡 ----------
cards = [
    (0.03, '格子的作用', ['地址 = 索引维组合', '内容 = 晴天基线 ＋ 每个泛化维一条曲线', '推理 = 定位格子 → 读基线 → 曲线评估']),
    (0.355, '适合当索引维', ['值域封闭 ＋ 推理时永远已知', '小时(24) / 日型(3) / 区域(263)', '→ 查表：精度高，需逐格数据（例A=24格）']),
    (0.68, '适合当泛化维', ['连续 ＋ 可能出现没见过的值', '雨强 / 温度 / 降雪', '→ 曲线：省数据、可外推（例B=2条曲线）']),
]
for x, t, lines in cards:
    box(x, 0.330, 0.29, 0.115, 'white', NAVY, 1.2)
    txt(x + 0.012, 0.420, t, 11, NAVY, True)
    for k, s in enumerate(lines):
        txt(x + 0.012, 0.390 - k * 0.026, s, 9.5, TEXTC)

# ---------- 换挡 + 绿条 ----------
txt(0.03, 0.275, '小时维可自由换挡：当索引（24 格·精度高）或 当泛化（1 条平滑曲线·省数据），可互换——例 A 与例 B 答案一致',
    10.5, GRN, True)
box(0.03, 0.185, 0.94, 0.062, '#E2F0D9', GRNB, 1.2)
txt(0.5, 0.216, '组合自由：任意维度可指定为索引或泛化——格子数量与曲线数量随之变化，路由与生成规则不变',
    11.5, GRN, True, 'center')

txt(0.5, 0.150, '已运行的实例：雨维（剂量曲线·盲测4%）｜ 温度维（观测域≈0·热浪+12%）｜ 元旦脉冲（×5.65）——每个都是"1 个泛化维 × 索引分片"的一次完整标定',
    10, MUTED, True, 'center')
fig.savefig(r'D:/AI/条件转化/索引泛化示意图.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 索引泛化示意图.png')
