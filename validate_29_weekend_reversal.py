# -*- coding: utf-8 -*-
"""反转一(周末×雨方向反转)证据计算——采用标定三旋钮同款口径(validate_06):
   出发量: 逐时 hour_cnt ÷ 同(年,日型,钟点)干日中位 → 类内体积加权比值 Σcnt/Σλ0
   速度:   log(v) − 同(年,日型,钟点)干日中位log(v) 的类内中位 → exp()
   距离:   该日 log(中位里程) − 同(年,日型)干日中位log 的类内均值 → exp()
   雨类小时: 时雨强 ≥0.3 mm/h(捕获雨响应); 晴类: 干日全天
输出: 反转一_周末方向反转/ 文件夹(3 面板图+单图+CSV+JSON)
"""
import numpy as np, json, csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

FOLD = r'D:/AI/条件转化/反转一_周末方向反转'
os.makedirs(FOLD, exist_ok=True)

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
dates = list(A['dates']); n = len(dates)
hc = A['hour_cnt'].astype(np.float64)
dist = A['dist'].astype(np.float64); dist_day = A['dist_day'].astype(int)
klmap = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
kl = np.array([klmap[d] for d in dates])
yr = np.array([int(str(d)[:4]) for d in dates])
dow = np.array([(np.datetime64(d, 'D').view('int64') + 3) % 7 for d in dates])   # 0=周一
wk = (dow < 5).astype(int)                                                        # 1=工作日 0=周末
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd_, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd_) in didx: inten[didx[str(dd_)], hh] = mm

# ---- 基线: 干日 ----
dry_i = np.where(kl == 'dry')[0]
# 距离: 每日 log 中位 + 长途份额(≥5mi, ≥10mi)
lmd = np.full(n, np.nan); sh5 = np.full(n, np.nan); sh10 = np.full(n, np.nan)
for i in range(n):
    s = dist[dist_day == i]
    s = s[(s > 0.1) & (s < 80)]
    if s.size >= 500:
        lmd[i] = np.median(np.log(s)); sh5[i] = (s >= 5).mean(); sh10[i] = (s >= 10).mean()
# 速度基线: log(v) 中位 per (yr, wk, hour)
lv = np.log(S['v'].astype(np.float64))
sd, sh = S['day'].astype(int), S['hour'].astype(int)
okv = (np.exp(lv) > 1) & (np.exp(lv) < 60)
mu0 = np.full((2, 2, 24), np.nan)      # [yr_idx, wk, hour]
yrs = [2024, 2025]; yidx = np.searchsorted(yrs, yr)
for yi in range(2):
    for w in range(2):
        for h in range(24):
            m = dry_i[y] if False else None
            sel = dry_i[(yr[dry_i] == yrs[yi]) & (wk[dry_i] == w)]
            m = okv & np.isin(sd, sel) & (sh == h)
            if m.sum() >= 300: mu0[yi, w, h] = np.median(lv[m])
# 出发量基线 lam0: (yr, wk, hour) 干日中位
lam0 = np.full((2, 2, 24), np.nan)
for yi in range(2):
    for w in range(2):
        for h in range(24):
            sel = dry_i[(yr[dry_i] == yrs[yi]) & (wk[dry_i] == w)]
            vals = hc[sel, h]
            if len(vals) >= 3: lam0[yi, w, h] = np.mean(vals)
# 距离干日基线 per (yr, wk): log 中位 + 长途份额
lmd_base = {}; sh5_base = {}; sh10_base = {}
for yi in range(2):
    for w in range(2):
        sel = dry_i[(yr[dry_i] == yrs[yi]) & (wk[dry_i] == w)]
        v1 = lmd[sel]; v1 = v1[~np.isnan(v1)]
        v2 = sh5[sel]; v2 = v2[~np.isnan(v2)]
        v3 = sh10[sel]; v3 = v3[~np.isnan(v3)]
        if len(v1) >= 5:
            lmd_base[(yi, w)] = float(np.median(v1)); sh5_base[(yi, w)] = float(np.mean(v2)); sh10_base[(yi, w)] = float(np.mean(v3))

# ---- 类统计 ----
BINS = [('晴', None), ('小雨', (0.3, 2.5)), ('中雨', (2.5, 8)), ('大雨≥8', (8, 1e9))]
res = {}
print(f"{'日型':6s} {'档':8s} {'天数':>4s} {'雨时数':>6s} | {'出发量比':>8s} {'里程比':>8s} {'速度比':>8s}")
plot = {}
for dt_name, w in [('工作日', 1), ('周末', 0)]:
    widx = np.where(wk == w)[0]
    for bname, rng in BINS:
        if rng is None:
            day_sel = widx[kl[widx] == 'dry']
            hrs = None
        else:
            rainy = widx[kl[widx] != 'dry']
            lo, hi = rng
            bmask = (inten >= lo) & (inten < hi)
            day_sel = rainy[np.any(bmask[rainy], axis=1)]
            hrs = bmask
        if len(day_sel) < 3:
            plot.setdefault(dt_name, {})[bname] = None
            continue
        yi = int(np.mode(yr[day_sel])) if False else int(np.bincount(yidx[day_sel]).argmax())
        # 出发量: 体积加权比值(仅雨时小时; 晴类全天)
        num = den = 0.0
        for i in day_sel:
            for h in range(24):
                L = lam0[yidx[i], w, h]
                if np.isnan(L) or L < 1: continue
                if hrs is None or (hrs[i, h] and inten[i, h] >= 0.3):
                    num += hc[i, h]; den += L
        vr = num / den if den > 0 else np.nan
        # 速度: 雨时 log 残差中位
        res_v = []
        for i in day_sel:
            for h in range(24):
                m = okv & (sd == i) & (sh == h)
                if hrs is not None and not (hrs[i, h] and inten[i, h] >= 0.3): continue
                mm = float(mu0[yidx[i], w, h])
                if np.isnan(mm) or m.sum() < 30: continue
                res_v.append(float(np.median(lv[m] - mm)))
        sr = float(np.exp(np.median(res_v))) if len(res_v) >= 20 else np.nan
        # 距离: 日 log 中位偏移均值 + 长途份额变化(pp)
        shifts = [lmd[i] - lmd_base[(yidx[i], w)] for i in day_sel if not np.isnan(lmd[i])]
        dr = float(np.exp(np.mean(shifts))) if len(shifts) >= 3 else np.nan
        s5 = [sh5[i] - sh5_base[(yidx[i], w)] for i in day_sel if not np.isnan(sh5[i])]
        s10 = [sh10[i] - sh10_base[(yidx[i], w)] for i in day_sel if not np.isnan(sh10[i])]
        d5 = float(np.mean(s5) * 100) if len(s5) >= 3 else np.nan      # pp
        d10 = float(np.mean(s10) * 100) if len(s10) >= 3 else np.nan  # pp
        res[(dt_name, bname)] = dict(days=len(day_sel), vr=round(vr, 3), dr=round(dr, 3), sr=round(sr, 3),
                                     long5=round(d5, 2), long10=round(d10, 2))
        plot.setdefault(dt_name, {})[bname] = dict(vol=round(vr, 3), dist=round(dr, 3), spd=round(sr, 3),
                                                   long5=d5, long10=d10, days=len(day_sel))
        print(f'{dt_name:6s} {bname:8s} {len(day_sel):4d} | {vr:8.3f} {dr:8.3f} {sr:8.3f} | 长途5mi {d5:+5.1f}pp 长途10mi {d10:+5.1f}pp')

# CSV + JSON
with open(os.path.join(FOLD, '反转1_分档统计.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w_ = csv.writer(f)
    w_.writerow(['日型', '雨强档', '天数', '出发量比(体积加权)', '中位里程比(对数)', '中位速度比(对数)'])
    for (dt_name, bname), r in res.items():
        w_.writerow([dt_name, bname, r['days'], r['vr'], r['dr'], r['sr']])
json.dump({f'{k[0]}|{k[1]}': v for k, v in res.items()},
          open(os.path.join(FOLD, '反转1_统计.json'), 'w'), ensure_ascii=False, indent=1)
print('saved CSV/JSON')

# ---------- 图 ----------
BORD = ['晴', '小雨', '中雨', '大雨≥8']
def draw_panel(ax, key, title, ylab, show_labels=True):
    xs = np.arange(4)
    for dt_name, cc in [('工作日', '#1F4E79'), ('周末', '#ED7D31')]:
        xs_p, ys_p = [], []
        for x_, bn in enumerate(BORD):
            v = plot.get(dt_name, {}).get(bn, {}).get(key)
            if v is not None: xs_p.append(x_); ys_p.append(v)
        ax.plot(xs_p, ys_p, lw=2.4, marker='o', ms=9, color=cc, label=dt_name)
        if show_labels:
            for x_, y_ in zip(xs_p, ys_p):
                off = (-16, 9) if cc == '#1F4E79' else ((-14, -17) if x_ >= 3 else (16, -17))
                ax.annotate(f'{y_:.3f}', (x_, y_), textcoords='offset points', xytext=off,
                            ha='center', fontsize=9.5, color=cc, fontweight='bold', annotation_clip=False)
    ax.axhline(1, color='#999', lw=1, ls=':')
    ax.set_xticks(xs); ax.set_xticklabels(BORD, fontsize=11)
    ax.set_ylim(0.82, 1.44)
    ax.set_title(title, fontsize=12, loc='left', color='#1F4E79')
    ax.set_ylabel(ylab, fontsize=10.5); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc='upper left')

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
fig.suptitle('反转一证据：同一个雨，工作日与周末的响应不同（÷同年×日型×同钟点干日基线，雨时小时≥0.3mm/h，2024-04~2025-10）',
             fontsize=13.5, fontweight='bold', y=1.02)
draw_panel(axes[0], 'vol', '① 出发量乘子：工作日递增 / 周末台阶（数值见左下卡）', '出发量比', show_labels=False)
draw_panel(axes[1], 'dist', '② 中位里程比：两类日型均缩短', '里程比')
draw_panel(axes[2], 'spd', '③ 中位速度比：两类日型均变慢', '速度比')
plt.tight_layout()
fig.savefig(os.path.join(FOLD, '反转1_证据总览.png'), dpi=150, bbox_inches='tight', facecolor='white')
for k, nm, tt in [(0, '反转1_总量剂量', '出发量比：工作日递增，周末恒定'), (1, '反转1_距离响应', '中位里程比'), (2, '反转1_速度响应', '中位速度比')]:
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.6))
    draw_panel(ax2, ['vol', 'dist', 'spd'][k], tt, ylab='比值（÷干日基线）')
    plt.tight_layout()
    fig2.savefig(os.path.join(FOLD, nm + '.png'), dpi=150, bbox_inches='tight', facecolor='white')
print('saved 4 png into', FOLD)
