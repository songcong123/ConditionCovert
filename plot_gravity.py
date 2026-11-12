# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16.5, 5.8))
fig.suptitle('重力模型图示：T_ij ∝ O_i · D_j · e^(−β·d_ij)   （流量 = 出发质量 × 吸引质量 × 距离衰减）', fontsize=14, fontweight='bold', y=1.02)

# ============ 面板1：机制概念 ============
ax1 = fig.add_subplot(131)
A = np.array([1.2, 5.0])
dests = {
    'B 商业区\n质量6000': (np.array([7.5, 8.2]), 6000),
    'C 枢纽\n质量3000':   (np.array([8.5, 5.0]), 3000),
    'D 居住\n质量5000':   (np.array([8.8, 1.6]), 5000),
    'E 公园\n质量1000':   (np.array([4.0, 1.0]), 1000),
}
beta = 0.25
for name, (pos, m) in dests.items():
    d = np.linalg.norm(pos - A)
    T = 8000 * m * np.exp(-beta * d)
    lw = 8 * T / max(8000 * mm[1] * np.exp(-beta * np.linalg.norm(mm[0]-A)) for mm in dests.values())
    ax1.annotate('', xy=pos, xytext=A,
                 arrowprops=dict(arrowstyle='-|>', lw=lw, color='#4472C4', alpha=0.75))
    r = 0.35 * np.sqrt(m/3000)
    circ = plt.Circle(pos, r, color='#ED7D31', alpha=0.85, zorder=3)
    ax1.add_patch(circ)
    ax1.text(pos[0], pos[1]-r-0.45, f'{name}\n距离{d:.1f} 流量T={T/1e5:.1f}', ha='center', fontsize=8.5)
circA = plt.Circle(A, 0.35*np.sqrt(8000/3000), color='#4472C4', alpha=0.9, zorder=3)
ax1.add_patch(circA)
ax1.text(A[0], A[1]+1.0, 'A 起点\n质量8000', ha='center', fontsize=10, fontweight='bold')
ax1.set_title('① 机制：质量越大流量越大\n箭头粗细 ∝ T_ij，越近越粗', fontsize=11)
ax1.set_xlim(-0.5, 11); ax1.set_ylim(-1.2, 10); ax1.set_aspect('equal'); ax1.axis('off')

# ============ 面板2：距离衰减曲线 ============
ax2 = fig.add_subplot(132)
d = np.linspace(0, 20, 200)
for b, c, lab in [(0.12, '#70AD47', 'β=0.12（晴天/平缓）'), (0.25, '#4472C4', 'β=0.25（雨天/更陡）'), (0.45, '#C00000', 'β=0.45（暴雨/极端压缩）')]:
    ax2.plot(d, np.exp(-b*d), color=c, lw=2.5, label=lab)
ax2.fill_between(d, np.exp(-0.12*d), np.exp(-0.45*d), color='#FFF2CC', alpha=0.6)
ax2.annotate('β 变大 → 远距离流量被压掉\n=「暴雨压缩空间分布」', xy=(8, 0.10), xytext=(8.5, 0.55),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='#666666'))
ax2.set_xlabel('距离 d_ij (km)', fontsize=10); ax2.set_ylabel('相对流量 e^(−β·d)', fontsize=10)
ax2.set_title('② 距离衰减：β 是核心参数\n（β(c) 可随外生条件变化）', fontsize=11)
ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1.05)

# ============ 面板3：环约束终点采样 ============
ax3 = fig.add_subplot(133)
O = np.array([0.0, 0.0])
R = 6.2
th = np.linspace(0, 2*np.pi, 200)
ax3.plot(O[0]+R*np.cos(th), O[1]+R*np.sin(th), color='#4472C4', lw=2, ls='--')
ax3.fill_betweenx(R*np.sin(th), R*np.cos(th)*0+ (R*np.cos(th))*0.86, R*np.cos(th)*0+R*np.cos(th)*1.1,
                  color='#4472C4', alpha=0.0)  # placeholder
band = plt.matplotlib.patches.Wedge((0,0), R*1.1, 0, 360, width=R*0.24, color='#DAE8FC', alpha=0.5, zorder=1)
ax3.add_patch(band)
cands = [
    ('商业区\nD=8000\n权重0.42', 40, 6.0, '#ED7D31', 0.42),
    ('枢纽\nD=5000\n权重0.31', 150, 6.5, '#70AD47', 0.31),
    ('居住区\nD=3000\n权重0.18', 262, 6.1, '#FFC000', 0.18),
    ('公园\n不在环上\n权重≈0', 95, 3.0, '#A6A6A6', 0.0),
]
for name, ang, dist, c, w in cands:
    p = np.array([dist*np.cos(np.radians(ang)), dist*np.sin(np.radians(ang))])
    r = 0.35 + 0.9*w
    ax3.add_patch(plt.Circle(p, r, color=c, alpha=0.9 if w>0 else 0.5, zorder=3))
    ax3.text(p[0], p[1]+r+0.35, name, ha='center', fontsize=8.5, color='#333333')
ax3.plot(*O, 's', color='#C00000', ms=10, zorder=4)
ax3.text(0.2, -0.9, '起点', fontsize=10, color='#C00000', fontweight='bold')
ax3.annotate('', xy=(R*np.cos(np.radians(20)), R*np.sin(np.radians(20))), xytext=(0.6*np.cos(np.radians(20)), 0.6*np.sin(np.radians(20))),
             arrowprops=dict(arrowstyle='<|-|>', color='#666666', lw=1.5))
ax3.text(R*0.55*np.cos(np.radians(28)), R*0.55*np.sin(np.radians(28)), 'd/λ≈6.2km', fontsize=9, color='#444444', rotation=25)
ax3.set_title('③ 应用：起点+距离 → 终点采样\n圆环上按 D_j·K(|d_ij−d/λ|) 加权抽样', fontsize=11)
ax3.set_xlim(-8.5, 8.5); ax3.set_ylim(-8.0, 8.5); ax3.set_aspect('equal'); ax3.axis('off')

plt.tight_layout()
plt.savefig('D:/AI/条件转化/重力模型图示.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved')
