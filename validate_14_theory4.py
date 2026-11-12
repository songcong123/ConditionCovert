# -*- coding: utf-8 -*-
"""纯理论版四部件图(严格模式): 不加载任何实测数据文件
① λ₀(t): 周期活动节律 → 高斯混合(通勤行为理论, 示意参数)
② λ₀(s): 三种经典空间理论族——Zipf重力(幂律)/Clark指数衰减/辐射模型(Simini 2012)
③ P₀(d): 截断幂律族(González 2008) vs LogNormal族(乘性过程理论), 参数族示意
④ P₀(v|t): 速度=距离/时长(两正支撑量之比)→log v 近正态(生成机制理论), σ 族示意
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(17, 11))
fig.suptitle('纯理论版（严格模式）：四个基线部件 = 理论形式 + 文献参数，不含任何本项目实测数据',
             fontsize=15, fontweight='bold', y=0.99)

# ---------- ① 出发时间: 周期高斯混合(时间轴) ----------
ax = axes[0][0]
t = np.linspace(0, 24, 481)
def G(mu, s):
    dt_ = np.minimum(np.abs(t - mu), 24 - np.abs(t - mu))   # 周期化(绕回)
    return np.exp(-dt_ ** 2 / (2 * s ** 2))
base = np.full_like(t, 0.6)
c1, c2 = 3.5 * G(8, 1.3), 5.0 * G(17.8, 1.9)                 # 通勤双峰(早晚), 示意
lam = base + c1 + c2
lam /= np.trapz(lam, t)
ax.plot(t, lam, color='#4472C4', lw=3, label='理论节律（双通勤峰 + 底噪）')
ax.fill_between(t, 0, (c1) / np.trapz(base + c1 + c2, t), alpha=0.22, color='#4472C4')
ax.fill_between(t, (c1) / np.trapz(base + c1 + c2, t), (c1 + c2) / np.trapz(base + c1 + c2, t), alpha=0.22, color='#4472C4')
ax.text(8, 0.015, '早通勤峰\nμ≈8时', ha='center', fontsize=10, color='#1F4E79')
ax.text(17.8, 0.015, '晚通勤峰\nμ≈18时(最大)', ha='center', fontsize=10, color='#1F4E79')
ax.text(0.985, 0.965,
        r'$\lambda_0(t)=c+\sum_i a_i\,\mathcal{G}((t-\mu_i)\ \mathrm{mod}\ 24;\ \sigma_i)$' + '\n'
        r'时间轴·周期多峰 → 高斯混合(或 von Mises 族)' + '\n'
        r'参数=通勤行为理论示意(早8/晚18), 非本地拟合',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='#F4F8FC', ec='#6C8EBF', alpha=0.95))
ax.set_xlabel('时刻（时）'); ax.set_ylabel('份额'); ax.set_xticks(range(0, 25, 2))
ax.set_ylim(0, None); ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3)
ax.set_title('① 出发时间 $\\lambda_0(t)$：周期混合分布', fontsize=12.5, loc='left', color='#1F4E79')

# ---------- ② 空间: 三种理论衰减族(距离轴) ----------
ax = axes[0][1]
d = np.linspace(0.1, 40, 400)      # km
f_zipf = d ** -1.5; f_zipf /= np.trapz(f_zipf, d)
f_zipf2 = d ** -0.8; f_zipf2 /= np.trapz(f_zipf2, d)
f_clark = np.exp(-0.2 * d); f_clark /= np.trapz(f_clark, d)
m = 1.0
f_rad = 1.0 / ((d + m) * (d + 2 * m)); f_rad /= np.trapz(f_rad, d)   # 辐射模型一维示意
ax.plot(d, f_clark * 100, lw=2.6, color='#5B2C6F', label='Clark(1951) 指数衰减 $e^{-\\beta d}$, β≈0.2/km')
ax.plot(d, f_zipf * 100, lw=2.6, color='#9673A6', label='Zipf(1946) 重力幂律 $d^{-\\gamma}$, γ=1.5')
ax.plot(d, f_zipf2 * 100, lw=1.8, color='#D9C2E9', ls='--', label='重力幂律 γ=0.8（更平）')
ax.plot(d, f_rad * 100, lw=2.6, color='#C00000', ls='-.', label='辐射模型 Simini et al.(2012)：无参数')
ax.set_yscale('log'); ax.set_ylim(1e-3, None)
ax.set_xlabel('到市中心的距离 d (km)'); ax.set_ylabel('密度份额 (%, log轴)')
ax.legend(fontsize=9.5, loc='upper right'); ax.grid(alpha=0.3, which='both')
ax.text(0.03, 0.40,
        r'空间轴·单调衰减 → 三种理论族:' + '\n'
        r'重力: $\lambda_0(s)\propto d^{-\gamma}$ 或 $e^{-\beta d}$' + '\n'
        r'辐射: $T_{ij}\propto \frac{m_i m_j}{(m_i+s_{ij})(m_i+m_j+s_{ij})}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='#F7F2FA', ec='#9673A6', alpha=0.95))
ax.set_title('② 空间分布 $\\lambda_0(s)$：距离衰减族（与①不同族！）', fontsize=12.5, loc='left', color='#5B2C6F')

# ---------- ③ 距离: 两大理论族 ----------
ax = axes[1][0]
x = np.geomspace(0.05, 60, 700)
for beta, c in [(1.0, '#9DC3E6'), (1.5, '#2E75B6'), (2.0, '#1F4E79')]:
    f = (x + 0.2) ** (-beta) * np.exp(-x / 20.0); f /= np.trapz(f, x)
    ax.plot(x, f, lw=2.4, color=c, label=f'截断幂律 β={beta}（González et al. 2008 形式）')
for sd, c in [(0.5, '#F4B183'), (0.8, '#ED7D31'), (1.2, '#C55A11')]:
    f = stats.lognorm.pdf(x, sd, scale=np.exp(0.5)); f /= np.trapz(f, x)
    ax.plot(x, f, lw=2.0, ls='--', color=c, label=f'LogNormal σ={sd}（乘性过程理论）')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('距离 (mi, log轴)'); ax.set_ylabel('密度 (log轴)')
ax.legend(fontsize=9, loc='lower left'); ax.grid(alpha=0.3, which='both')
ax.text(0.98, 0.97,
        r'截断幂律: $P(r)=(r+r_0)^{-\beta}\,e^{-r/\kappa}$' + '\n'
        r'LogNormal: $P(r)=\frac{1}{r\sigma\sqrt{2\pi}}e^{-\frac{(\ln r-\mu)^2}{2\sigma^2}}$' + '\n'
        r'参数为文献量级示意, 非本项目拟合',
        transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
        bbox=dict(boxstyle='round', fc='#FBF4EE', ec='#ED7D31', alpha=0.95))
ax.set_title('③ 距离分布 $P_0(d)$：两大理论族（个体域→幂律族, 网约车域→对数正态+尾）', fontsize=12.5, loc='left', color='#833C0B')

# ---------- ④ 速度: 比值→log正态的生成机制 ----------
ax = axes[1][1]
vv = np.linspace(0.5, 45, 500)
for sd, c in [(0.3, '#A9D18E'), (0.45, '#70AD47'), (0.7, '#375623')]:
    ax.plot(vv, stats.lognorm.pdf(vv, sd, scale=12), lw=2.4, color=c,
            label=f'σ={sd}（中位固定 12 mph）')
ax.set_xlabel('速度 (mph)'); ax.set_ylabel('密度'); ax.set_xlim(0, 45)
ax.legend(fontsize=10, title='形状参数族', title_fontsize=10, loc='upper right'); ax.grid(alpha=0.3)
ax.text(0.02, 0.96,
        r'$v=\frac{d}{\Delta t}$：两个正支撑量之比' + '\n'
        r'$\Rightarrow \log v=\log d-\log\Delta t$ 近正态' + '\n'
        r'$\Rightarrow \log v=\mu_0(t)+\varepsilon,\ \ \varepsilon\sim\mathcal{N}(0,\sigma_0^2)$' + '\n'
        r'生成机制理论, 非拟合',
        transform=ax.transAxes, ha='left', va='top', fontsize=10.5,
        bbox=dict(boxstyle='round', fc='#F1F7EC', ec='#70AD47', alpha=0.95))
ax.set_title('④ 速度分布 $P_0(v\\,|\\,t)$：比值机制 → LogNormal', fontsize=12.5, loc='left', color='#375623')

plt.tight_layout()
plt.savefig('D:/AI/条件转化/基线四分布_理论版.png', dpi=150, bbox_inches='tight', facecolor='white')
print('saved 基线四分布_理论版.png (严格纯理论, 无任何实测数据)')
