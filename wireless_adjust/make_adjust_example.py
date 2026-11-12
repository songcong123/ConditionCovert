# -*- coding: utf-8 -*-
"""
第二步：读统计 JSON 出"调整过程示例"图（任何机器可跑，不依赖原始数据）
========================================================================
输入：
  stats_json   extract_trip_stats.py 的输出（wireless_stats_*.json）
  --day        演示的条件日（默认自动选：雨时数最多的日期）
  --out        输出 PNG 路径

逻辑（与 NYC 调整过程示例同构，适配一周无线行人数据）：
  1) 基线 = 窗口内晴日（按日型分组：工作日晴日均 / 周末晴日均）
  2) 条件日（雨天）逐时乘子 y(t) = n_trips(雨,时) ÷ 同型基线(时)   ← 标签实测
  3) 合成 = 基线 × [1+δvol(r)]（NYC 公式，可迁移假设）+ 距离三旋钮 + 速度 ×exp(−δv)
  4) 三行四列：A 基线 / B 雨强→乘子 / C 调整合成 / D 合成 vs 实测对照

输出：
  一张 PNG（3 行 × 4 列）+ 控制台打印（基线日清单、条件日、逐段乘子）
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1F4E79'; RED = '#C00000'; OR = '#ED7D31'; GREY = '#7F7F7F'
MUTED = '#666666'; DGREEN = '#375623'

# NYC 标定系数（可迁移假设 —— 行人方向可能相反，需重标）
THETA_V, RC_V = 0.052, 0.2      # 出发量 δvol
THETA_S, RC_S = 0.025, 0.3      # 速度 δv
D3_KNOBS = dict(mu1=-0.061, w1=0.025, long=0.9)   # 距离三旋钮


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stats_json')
    ap.add_argument('--day', default=None, help='条件日 YYYY-MM-DD（默认=雨时最多的日子）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    S = json.load(open(a.stats_json, encoding='utf-8'))
    dates = S['dates']
    H = np.arange(24)

    # ---- 日型（周一=0）----
    dow = {d: (np.datetime64(d, 'D').view('int64') + 3) % 7 for d in dates}
    wk = {d: 'weekend' if dow[d] >= 5 else 'workday' for d in dates}
    rain_h = {d: sum(1 for x in S['rain_mmh'][d] if x >= 1) for d in dates}
    dry_days = [d for d in dates if rain_h[d] == 0]
    if a.day: cond_day = a.day
    else: cond_day = max(dates, key=lambda d: rain_h[d])
    cond_type = wk[cond_day]
    base_days = [d for d in dry_days if wk[d] == cond_type] or dry_days
    print(f'条件日: {cond_day}（{cond_type}, 雨时 {rain_h[cond_day]}）')
    print(f'基线日（同型晴日）: {base_days}')

    # ---- 基线 ----
    base_n = np.median([S['n_trips'][d] for d in base_days], axis=0)
    sp = [[x for x in S['speed_median'][d]] for d in base_days]
    base_v = np.array([np.nanmedian([r[h] for r in sp if r[h] is not None]) if any(r[h] is not None for r in sp) else np.nan
                       for h in range(24)], dtype=float)
    bs = np.concatenate([S['dist_samples'][d] for d in base_days])
    if bs.size == 0: bs = np.array([800.0])
    dgrid = np.linspace(50, 5000, 200)
    pdf0 = np.exp(-((np.log(dgrid) - np.log(np.median(bs))) ** 2) / (2 * 0.8 ** 2)) / dgrid
    pdf0 /= pdf0.max()

    rain = np.array(S['rain_mmh'][cond_day], dtype=float)
    mult = 1 + THETA_V * np.log1p(rain / RC_V)
    dv = np.exp(-THETA_S * np.log1p(rain / RC_S))
    act_n = np.array(S['n_trips'][cond_day], dtype=float)
    act_v = np.array([np.nan if S['speed_median'][cond_day][h] is None
                      else S['speed_median'][cond_day][h] for h in range(24)], dtype=float)
    cs = S['dist_samples'][cond_day] or []
    if not cs: cs = list(bs)
    # 三旋钮合成距离曲线
    pdf1 = np.exp(-((np.log(dgrid) - np.log(np.median(bs) * np.exp(D3_KNOBS['mu1']))) ** 2) / (2 * 0.8 ** 2)) / dgrid
    pdf1 = pdf1 / pdf1.max() * (1 + 0.10 * np.exp(-((np.log(dgrid) - np.log(0.6 * np.median(bs))) ** 2)))
    pdf1 = np.where(dgrid > 2.8 * np.median(bs), pdf1 * D3_KNOBS['long'], pdf1)
    pdf1 /= pdf1.max()

    # ---- 图 ----
    out = a.out or f'无线行人_调整过程示例_{cond_day}.png'
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    fig.suptitle(f'无线行人 · 调整过程示例（条件日 {cond_day}｜基线 = 晴日 {",".join(base_days)}）\n'
                 f'基线 × 扰动算子（NYC 标定系数，可迁移假设）→ 合成 vs 实测',
                 fontsize=14.5, fontweight='bold', y=0.985)

    def style(ax, t, ylab='', xlab='小时'):
        ax.set_title(t, fontsize=11, loc='left', color=NAVY)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.grid(alpha=0.3)

    # 行1 出发时间
    ax = axes[0, 0]
    ax.bar(H, base_n, color='#8FAADC', edgecolor=NAVY, lw=0.6)
    style(ax, 'A. 晴日基线（逐时出发量）', '趟/时')
    ax.text(0.97, 0.95, f'基线 n={len(base_days)} 天（{cond_type}）', transform=ax.transAxes,
            fontsize=9, color=RED, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF2CC', ec='#D6B656'))

    ax = axes[0, 1]
    ax.bar(H, rain, color='#4472C4', alpha=0.8)
    ax2 = ax.twinx(); ax2.plot(H, mult, lw=2.4, color=RED, marker='o', ms=4)
    ax2.set_ylabel('乘子 1+δvol', color=RED)
    style(ax, 'B. 条件日雨强 → 乘子', '雨强 (mm/h)')

    ax = axes[0, 2]
    ax.bar(H, base_n, color=GREY, alpha=0.35, label='基线')
    ax.plot(H, base_n * mult, lw=2.4, color=RED, marker='o', ms=4, label='合成')
    style(ax, 'C. 基线 × 逐时乘子', '趟/时'); ax.legend(fontsize=9)

    ax = axes[0, 3]
    ok = act_n > 0
    ax.plot(H[ok], act_n[ok], lw=2.2, color='#4472C4', marker='s', ms=4, label='实测')
    ax.plot(H, base_n * mult, lw=2.2, color=RED, ls='--', label='合成')
    err = np.abs(act_n[ok] - (base_n * mult)[ok]) / np.maximum(act_n[ok], 1)
    style(ax, f'D. 对照（偏差中位 {np.median(err)*100:.0f}%）', '趟/时'); ax.legend(fontsize=9)

    # 行2 距离
    ax = axes[1, 0]
    ax.plot(dgrid / 1000, pdf0, lw=2.4, color='#8FAADC')
    ax.fill_between(dgrid / 1000, 0, pdf0, alpha=0.25, color='#8FAADC')
    style(ax, 'A. 基线距离分布', '密度', '距离 (km)')
    ax.text(0.97, 0.55, f'中位 {np.median(bs):.0f} m', transform=ax.transAxes,
            fontsize=9.5, color=MUTED, ha='right')

    ax = axes[1, 1]
    ax.plot(dgrid / 1000, pdf0, lw=1.2, color=GREY, ls='--', label='基线')
    ax.plot(dgrid / 1000, pdf1, lw=2.4, color=OR, label='三旋钮后')
    style(ax, 'B. 三旋钮（μ1/w1/长途）', '密度', '距离 (km)'); ax.legend(fontsize=9)

    ax = axes[1, 2]
    ax.plot(dgrid / 1000, pdf1, lw=2.4, color=OR)
    ax.fill_between(dgrid / 1000, 0, pdf1, alpha=0.2, color=OR)
    style(ax, 'C. 合成距离分布', '密度', '距离 (km)')

    ax = axes[1, 3]
    ax.hist(np.array(cs) / 1000, bins=np.linspace(0, 4, 25), density=True, alpha=0.35, color='#4472C4', label='实测')
    scale = np.median(np.array(cs) / 1000) / 0.4
    ax.plot(dgrid / 1000, pdf1 / (pdf1.max() * 4.0), lw=2.2, color=OR, ls='--', label='合成(峰值对齐)')
    ax.set_xlim(0, 4)
    style(ax, f'D. 对照（中位 {np.median(cs):.0f} m vs {np.median(bs)*np.exp(D3_KNOBS["mu1"]):.0f} m）',
          '密度', '距离 (km)'); ax.legend(fontsize=9)

    # 行3 速度
    ax = axes[2, 0]
    okb = ~np.isnan(base_v)
    ax.plot(H[okb], base_v[okb], lw=2.4, color='#8FAADC', marker='o', ms=4)
    style(ax, 'A. 基线逐时中位速度', 'km/h')

    ax = axes[2, 1]
    ax.bar(H, rain, color='#4472C4', alpha=0.8)
    ax2 = ax.twinx(); ax2.plot(H, dv, lw=2.4, color=RED, marker='o', ms=4)
    ax2.set_ylabel('exp(−δv)', color=RED)
    style(ax, 'B. 雨强 → 速度乘子', '雨强 (mm/h)')

    ax = axes[2, 2]
    ax.plot(H[okb], base_v[okb], lw=1.2, color=GREY, ls='--', label='基线')
    ax.plot(H[okb], base_v[okb] * dv[okb], lw=2.4, color=OR, label='合成')
    style(ax, 'C. 基线 × exp(−δv)', 'km/h'); ax.legend(fontsize=9)

    ax = axes[2, 3]
    oka = ~np.isnan(act_v)
    ax.plot(H[oka], act_v[oka], lw=2.2, color='#4472C4', marker='s', ms=4, label='实测')
    ax.plot(H[oka], (base_v * dv)[oka], lw=2.2, color=OR, ls='--', label='合成')
    style(ax, 'D. 对照', 'km/h'); ax.legend(fontsize=9)

    fig.text(0.5, 0.004,
             '系数出处：δvol=0.052·ln(1+r/0.2)、δv=0.025·ln(1+r/0.3)、距离三旋钮 —— 均为 NYC 出租车标定值（可迁移假设）;'
             '行人方向可能相反，须用自有数据重标。天气由 v3 逐时天气表 text→雨强映射。',
             ha='center', fontsize=10, color=DGREEN, fontweight='bold', wrap=True)
    plt.tight_layout(rect=[0, 0.035, 1, 0.945])
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
