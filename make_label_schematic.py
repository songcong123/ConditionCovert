# -*- coding: utf-8 -*-
"""标签生成示意图(合成曲线, 非真实数据):
   固定三段式: 灰线=晴天基线, 红线=雨天实测, 橙线=比值标签 y —— 三个量各一例
   每面板带 ÷ 算例标注
输出: 标签生成示意图.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

G, R_, O, GRN = '#7F7F7F', '#C00000', '#ED7D31', '#375623'
NAVY = '#1F4E79'
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
fig.suptitle('标签生成示意：y ＝ 实测 ÷ 同类型晴天基线（合成示意曲线，非真实数据）\n'
             '读法：灰线（晴天基线）与红线（雨天实测）逐点相除 → 橙线就是标签 y（围绕 1.0 浮动的乘子）',
             fontsize=13.5, fontweight='bold', y=0.99)

# ---------- ① 出发时间 ----------
t = np.arange(24)
base_t = 60 + 45 * np.exp(-((t - 8) / 2.2) ** 2) + 38 * np.exp(-((t - 18) / 2.5) ** 2)
y_t = 1 + 0.45 * np.exp(-((t - 8.5) / 3.0) ** 2) - 0.12 * np.exp(-((t - 16) / 5.0) ** 2)
act_t = base_t * y_t
ax = axes[0, 0]
ax.plot(t, base_t, lw=2.4, color=G, label='基线 λ基线(t)（晴天）')
ax.plot(t, act_t, lw=2.4, color=R_, label='实测 λ实测(t)（雨日）')
ax.annotate('', xy=(8, act_t[8]), xytext=(8, base_t[8]),
            arrowprops=dict(arrowstyle='<->', color=NAVY, lw=1.6))
ax.annotate(f'{act_t[8]:.0f} ÷ {base_t[8]:.0f} = {y_t[8]:.2f}', (8, (act_t[8] + base_t[8]) / 2),
            xytext=(10.5, 120), fontsize=10, color=NAVY, arrowprops=dict(arrowstyle='->', color=NAVY))
ax2 = ax.twinx()
ax2.plot(t, y_t, lw=2.6, color=O, label='标签 y(t)')
ax2.axhline(1, color=O, lw=1, ls=':'); ax2.set_ylim(0.4, 1.8)
ax2.set_ylabel('标签 y（比值）', color=O)
ax.set_title('① 出发时间：y(t) = λ实测(t) ÷ λ基线(t)\n（24 小时 → 24 个标签值；早峰 y≈1.4）', fontsize=11, loc='left')
ax.set_ylabel('出发量（示意）'); ax.set_xlabel('小时')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper left')

# ---------- ② 空间分布 ----------
grp = ['住宅区', '商务区', '机场', '交通枢纽', '文商圈']
base_s = np.array([30, 25, 18, 15, 12], float)
y_s = np.array([1.30, 0.72, 0.85, 1.15, 1.05])
act_s = base_s * y_s
ax = axes[0, 1]
xs = np.arange(5); wd = 0.36
ax.bar(xs - wd / 2, base_s, wd, color=G, label='基线份额（晴天）')
ax.bar(xs + wd / 2, act_s, wd, color=R_, label='实测份额（雨日）')
for k in range(5):
    ax.annotate(f'y={y_s[k]:.2f}', (xs[k] + wd / 2, act_s[k] + 0.6), ha='center',
                fontsize=10.5, fontweight='bold', color=R_)
ax.set_xticks(xs); ax.set_xticklabels(grp, fontsize=10.5)
ax.set_ylabel('出发份额（示意）')
ax.set_title('② 空间分布：y(s) = λ实测(s) ÷ λ基线(s)\n（每个区域一个比值：住宅 y>1 商务 y<1 → 重分配）', fontsize=11, loc='left')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

# ---------- ③ 距离分布 ----------
d = np.geomspace(0.3, 25, 80)
pdf = lambda mu, sd: np.exp(-((np.log(d) - mu) ** 2) / (2 * sd ** 2)) / (d * sd * np.sqrt(2 * np.pi))
base_d = pdf(0.95, 0.55); base_d /= base_d.max()
act_d = pdf(0.72, 0.55); act_d /= act_d.max()
y_d = act_d / base_d
ax = axes[1, 0]
ax.plot(d, base_d, lw=2.4, color=G, label='基线 p基线(d)（晴天）')
ax.plot(d, act_d, lw=2.4, color=R_, label='实测 p实测(d)（雨日，左移）')
ax3 = ax.twinx()
ax3.plot(d, y_d, lw=2.6, color=O, label='标签 y(d) = 密度比')
ax3.axhline(1, color=O, lw=1, ls=':'); ax3.set_ylim(0, 3.2)
ax3.set_ylabel('标签 y(d)', color=O)
ax.annotate('短途 y>1\n（雨天更近）', (0.45, 0.75), fontsize=9.5, color=NAVY)
ax.annotate('长途 y<1\n（雨天更少）', (6.5, 0.62), fontsize=9.5, color=NAVY)
ax.set_xscale('log')
ax.set_title('③ 距离分布：y(d) = p实测(d) ÷ p基线(d)\n（按里程分箱 → 一条比值曲线：短途>1 长途<1）', fontsize=11, loc='left')
ax.set_xlabel('里程 d (mi, log)'); ax.set_ylabel('密度（示意）')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper right')

# ---------- ④ 速度分布 ----------
y_v = 0.88 - 0.08 * np.exp(-((t - 8.5) / 2.5) ** 2)
base_v = 15 - 3.5 * np.exp(-((t - 8) / 2.2) ** 2) - 2.5 * np.exp(-((t - 18) / 2.5) ** 2)
act_v = base_v * y_v
ax = axes[1, 1]
ax.plot(t, base_v, lw=2.4, color=G, label='基线 μ基线(t)（晴天）')
ax.plot(t, act_v, lw=2.4, color=R_, label='实测 μ实测(t)（雨日）')
ax2 = ax.twinx()
ax2.plot(t, y_v, lw=2.6, color=O, label='标签 y(t)')
ax2.axhline(1, color=O, lw=1, ls=':'); ax2.set_ylim(0.5, 1.5)
ax2.set_ylabel('标签 y（比值）', color=O)
ax.annotate(f'8 时：{act_v[8]:.1f} ÷ {base_v[8]:.1f} = {y_v[8]:.2f}（慢 {abs(1-y_v[8])*100:.0f}%）', (8, act_v[8]),
            xytext=(14.2, 10.3), fontsize=10, color=NAVY, arrowprops=dict(arrowstyle='->', color=NAVY))
ax.set_title('④ 速度：y(t) = μ实测(t) ÷ μ基线(t)\n（24 小时 → 全程 y<1：雨天变慢）', fontsize=11, loc='left')
ax.set_ylabel('中位速度（示意）'); ax.set_xlabel('小时')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper right')

fig.text(0.5, 0.005,
         '标签 y 的用途：推理时 y(c) 乘回基线 → λ = λ基线 × y。每个标签都有出处：一条实测曲线 ÷ 一条同类型基线曲线。',
         ha='center', fontsize=11, color=GRN, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 0.955])
fig.savefig(r'D:/AI/条件转化/标签生成示意图.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 标签生成示意图.png')
