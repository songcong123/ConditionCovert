# -*- coding: utf-8 -*-
"""流程验证 大型活动档: 跨场馆盲测 (MSG 标定 → 巴克莱盲预测)
与雨天档同哲学: 行程数据不参与标定参数的"另一场馆侧"估计——
  标定侧 = MSG(186, 尼克斯/游骑兵) 2025-01~04 赛日 vs 非赛日
  盲测侧 = 巴克莱(189, 篮网) 2025-01~04 赛日 (全程不用其赛日数据标定任何参数)
扰动参数(事件版五件套, 全部从 MSG 学):
  ① 总量倍率 vol_mult  ② 到场窗脉冲  ③ 散场窗脉冲(相对赛毕对齐)
  ④ 距离 log 偏移 δ_d  ⑤ 散场窗 DO 集中度(top10 份额)与平均直线距离偏移
对照 = 纯基线(巴克莱非赛日档案直接当预测)
输出 .flow_scores_event.json"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, csv, json
from scipy import stats

rng = np.random.default_rng(7)
MONTHS = ['2025-01', '2025-02', '2025-03', '2025-04']
CAL = 'D:/AI/条件转化/data/exogenous/events_calendar.csv'

# ---- 1) 日历: 两场馆赛日 + 散场窗起点(本地十进制小时) ----
def dec_hour(s):
    return int(s[11:13]) + int(s[14:16]) / 60.0
games = {186: {}, 189: {}}   # zone -> {date: dict(disp_start, arr_start)}
for e in csv.DictReader(open(CAL, encoding='utf-8')):
    z = int(e['taxi_zone'])
    if z in games and e['event_category'] == 'sports_game' and e['date'][:7] in MONTHS:
        games[z][e['date']] = dict(
            disp_start=dec_hour(e['dispersal_start_local']),
            arr_start=dec_hour(e['window_start_local']))
print(f"MSG 赛日 {len(games[186])} | 巴克莱赛日 {len(games[189])}")

# ---- 2) 扫描 parquet: 两个场馆 zone 的逐日档案 (带缓存, 重跑免扫) ----
import os
CACHE = 'D:/AI/条件转化/.event_days_cache.npz'
recs = {}   # (zone,date) -> dict(hr48, total, do264, dist)
if os.path.exists(CACHE):
    C = np.load(CACHE, allow_pickle=True)
    for z, d, hr48, do264, dist, tot in zip(C['z'], C['d'], C['hr48'], C['do264'], C['dist'], C['tot']):
        recs[(int(z), str(d))] = dict(hr48=hr48.astype(int), total=int(tot),
                                      do264=do264.astype(int), dist=np.asarray(dist, float))
    print(f'cache loaded: {len(recs)} zone-days')
else:
    for mo in MONTHS:
        t = pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                          columns=['pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_miles'])
        dt = np.asarray(t['pickup_datetime']); pu = np.asarray(t['PULocationID'])
        doo = np.asarray(t['DOLocationID']); mi = np.asarray(t['trip_miles'])
        dts = dt.astype('datetime64[s]')
        dates = dts.astype('datetime64[D]').astype(str)
        half = (dts.astype('datetime64[m]').astype(int) // 30) % 48
        for z in (186, 189):
            sel = (pu == z) & (mi > 0.1) & (mi < 100) & (doo > 0) & (doo <= 263)
            for d in np.unique(dates[sel]):
                m = sel & (dates == d)
                key = (z, str(d))
                r = recs.get(key)
                if r is None:
                    r = recs[key] = dict(hr48=np.zeros(48, int), total=0, do264=np.zeros(264, int), dist=[])
                np.add.at(r['hr48'], half[m], 1)
                r['do264'] += np.bincount(doo[m], minlength=264)
                s = mi[m]; r['dist'].append(s[s < 40][:60000])
                r['total'] += int(m.sum())
        print(f'{mo} done')
    for r in recs.values():
        r['dist'] = np.concatenate(r['dist']) if r['dist'] else np.array([])
    ks_ = list(recs.keys())
    np.savez_compressed(CACHE,
        z=np.array([k[0] for k in ks_]), d=np.array([k[1] for k in ks_]),
        hr48=np.array([recs[k]['hr48'] for k in ks_]),
        do264=np.array([recs[k]['do264'] for k in ks_]),
        dist=np.array([recs[k]['dist'][:30000] for k in ks_], dtype=object),
        tot=np.array([recs[k]['total'] for k in ks_]))
    print(f'cache saved: {len(recs)} zone-days')

dow = {d: (np.datetime64(d, 'D').view('int64') % 7) for (z, d) in recs}

def zone_days(z, game: bool):
    return sorted(d for (zz, d) in recs if zz == z and ((d in games[z]) == game))

def base_profile(z):
    """非赛日按 dow 平均的 48bin 档案 + 总量 + 距离 + DO"""
    gd = set(games[z]); ds = [d for (zz, d) in recs if zz == z and d not in gd]
    hr = np.zeros((7, 48)); nn = np.zeros(7)
    for d in ds:
        hr[dow[d]] += recs[(z, d)]['hr48']; nn[dow[d]] += 1
    prof = hr / np.maximum(nn, 1)[:, None]
    tot = {k: prof[k].sum() for k in range(7)}
    dist = np.concatenate([recs[(z, d)]['dist'] for d in ds])
    do = np.sum([recs[(z, d)]['do264'] for d in ds], 0)
    return prof, tot, dist, do, len(ds)

Z = np.load('D:/AI/条件转化/.flow_zones.npz')
dist_km = Z['dist_km'].astype(float)
straight_mi = dist_km / 1.60934
# zone0/缺质心行列可能为 nan → 用矩阵中位数填补, 防止 DO 指标被传染
_filler = np.nanmedian(straight_mi)
straight_mi = np.where(np.isfinite(straight_mi), straight_mi, _filler)

def do_mean_straight(do_vec):
    """DO 分布的平均直线距离(以全城各 zone 近似, 跨场馆可比)"""
    s = do_vec.astype(float)
    tot = s.sum()
    if tot == 0: return np.nan
    mean_per_do = straight_mi.mean(0)
    return float((s * mean_per_do).sum() / tot)

# ---- 3) MSG 标定 ----
zC = 186
profC, totC, distC, doC, nC = base_profile(zC)
gdsC = zone_days(zC, True)
vol_mult = float(np.mean([recs[(zC, d)]['total'] / totC[dow[d]] for d in gdsC]))
# 事件相对时间脉冲: rel = (bin中心 − 赛毕), 聚合所有赛日的 赛日份额 − 非赛日份额
# ⚠️ 赛毕时刻是 21:24 这类分数对齐, 直接切 0.5h 网格会有一半 bin 落空记零 →
#    改为 rel 量化到 0.5h 后按值聚合(每天每 rel 至多一个 bin), 预测时插值
REL_MIN, REL_MAX = -3.0, 3.0
prof_shareC = profC / profC.sum(1, keepdims=True).clip(1e-9)
pulse_acc = {}
for d in gdsC:
    r = recs[(zC, d)]
    share_g = r['hr48'] / max(r['total'], 1)
    share_b = prof_shareC[dow[d]]
    t_center = np.arange(48) / 2 + 0.25
    rel_q = np.round((t_center - games[zC][d]['disp_start']) * 2) / 2
    diff = share_g - share_b
    for rq, dv in zip(rel_q, diff):
        if REL_MIN <= rq < REL_MAX:
            pulse_acc.setdefault(float(rq), []).append(float(dv))
rel_grid = np.array(sorted(pulse_acc))
pulse_raw = np.clip(np.array([np.mean(pulse_acc[g]) for g in rel_grid]), 0, None)
pulse_s = np.convolve(np.pad(pulse_raw, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
def pulse_at(rels):
    """任意 rel 处的脉冲份额增量(线性插值, 域外为 0)"""
    return np.interp(rels, rel_grid, pulse_s, left=0.0, right=0.0)
med_g = np.median(np.log(np.concatenate([recs[(zC, d)]['dist'] for d in gdsC])))
delta_d = float(med_g - np.median(np.log(distC)))
# 散场窗 DO: 赛日全日 DO 与非赛日 DO 的 top10 份额差(近似散场贡献); 平均直线距离偏移
def top10_share(v):
    s = np.sort(v)[::-1]
    return float(s[:10].sum() / max(v.sum(), 1))
do_gC = np.sum([recs[(zC, d)]['do264'] for d in gdsC], 0)
ms_g = do_mean_straight(do_gC); ms_b = do_mean_straight(doC)
delta_ms = float(np.log(ms_g) - np.log(ms_b))
print(f'MSG 标定: vol×{vol_mult:.2f} | 峰值脉冲份额 +{pulse_s.max()*100:.1f}pp@rel{rel_grid[pulse_s.argmax()]:+.1f}h | '
      f'δ_d={delta_d:+.3f} | DO top10 {top10_share(doC)*100:.0f}%→{top10_share(do_gC)*100:.0f}% | DO均直距×{np.exp(delta_ms):.2f}')

# ---- 4) 盲预测巴克莱 ----
zT = 189
profT, totT, distT, doT, nT = base_profile(zT)
prof_shareT = profT / profT.sum(1, keepdims=True).clip(1e-9)
gdsT = zone_days(zT, True)
pred_dates = gdsT
NG = 800_000
day_tot_pred = {d: totT[dow[d]] * vol_mult for d in pred_dates}

def pred_share_day(d):
    """预测的 48bin 份额 = 非赛日 dow 份额 + MSG 脉冲(按该场赛毕时间对齐, 插值)"""
    share_b = prof_shareT[dow[d]]
    tc = np.arange(48) / 2 + 0.25
    rel = tc - games[zT][d]['disp_start']
    return np.clip(share_b + pulse_at(rel), 1e-6, None)

gen_hours, gen_dist = [], []
for d in pred_dates:
    p = pred_share_day(d); p = p / p.sum()
    n = int(round(day_tot_pred[d] / np.mean([recs[(zT, dd)]['total'] for dd in pred_dates]) * NG / len(pred_dates)))
    bins_ = rng.choice(48, size=n, p=p)
    gen_hours.append(bins_ * 0.5 + 0.25)
    # 距离: 经验分布有放回抽样 × 整体平移 e^δ_d —— 只平移不改形状(与雨天管线同哲学)
    gen_dist.append(rng.choice(distT, size=n, replace=True) * np.exp(delta_d))
gen_h = np.concatenate(gen_hours); gen_d = np.concatenate(gen_dist)

real_h = np.concatenate([np.repeat(np.arange(48) / 2 + 0.25, recs[(zT, d)]['hr48']) for d in pred_dates])
real_d = np.concatenate([recs[(zT, d)]['dist'] for d in pred_dates])

def js(p, q):
    p = np.maximum(p, 1e-12); p /= p.sum(); q = np.maximum(q, 1e-12); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))
def hist48(arr):
    h, _ = np.histogram(arr, bins=np.arange(0, 24.5, 0.5))
    return h.astype(float)
hg, hr_, hb = hist48(gen_h), hist48(real_h), profT.sum(0)
S = dict(lam_t=js(hg, hr_), lam_t_base=js(hb, hr_))
# 散场窗份额: 赛毕~+2h
def window_share(share48, d):
    tc = np.arange(48) / 2 + 0.25
    w = (tc >= games[zT][d]['disp_start']) & (tc < games[zT][d]['disp_start'] + 2)
    return float(share48[w].sum())
sh_real = float(np.mean([recs[(zT, d)]['hr48'][w].sum() / max(recs[(zT, d)]['total'], 1)
                         for d in pred_dates
                         for w in [ (np.arange(48)/2+0.25 >= games[zT][d]['disp_start']) & (np.arange(48)/2+0.25 < games[zT][d]['disp_start']+2) ]]))
sh_pred = float(np.mean([window_share(pred_share_day(d) / pred_share_day(d).sum(), d) for d in pred_dates]))
sh_base = float(np.mean([window_share(prof_shareT[dow[d]], d) for d in pred_dates]))
# 日总量
vol_mape = float(np.mean([abs(day_tot_pred[d] / recs[(zT, d)]['total'] - 1) for d in pred_dates]))
vol_mape_base = float(np.mean([abs(totT[dow[d]] / recs[(zT, d)]['total'] - 1) for d in pred_dates]))
# KS 距离
def ks(a, b):
    n = min(len(a), 1_000_000)
    return float(stats.ks_2samp(rng.choice(a, n, replace=len(a) < n), rng.choice(b, n, replace=len(b) < n)).statistic)
S['d'] = ks(gen_d, real_d); S['d_base'] = ks(distT, real_d)
# DO: 赛日真实平均直线距离 vs 预测(基线×e^delta_ms)
do_gT = np.sum([recs[(zT, d)]['do264'] for d in pred_dates], 0)
ms_real = do_mean_straight(do_gT); ms_baseT = do_mean_straight(doT)
ms_pred = ms_baseT * np.exp(delta_ms)
do_err = abs(ms_pred / ms_real - 1); do_err_b = abs(ms_baseT / ms_real - 1)
top10_change_pred = top10_share(do_gC) - top10_share(doC)
top10_change_real = top10_share(do_gT) - top10_share(doT)

print('\n巴克莱盲测 (MSG 标定参数迁移, 越低越好):')
print(f'  JS λ(t)    扰动 {S["lam_t"]:.4f} vs 基线 {S["lam_t_base"]:.4f}')
print(f'  KS P(d)    扰动 {S["d"]:.4f} vs 基线 {S["d_base"]:.4f}')
print(f'  散场窗份额 real {sh_real*100:.1f}% | pred {sh_pred*100:.1f}% | 基线 {sh_base*100:.1f}%')
print(f'  日总量MAPE 扰动 {vol_mape*100:.1f}% vs 基线 {vol_mape_base*100:.1f}%')
print(f'  赛日DO均直距 real {ms_real:.2f}mi | pred {ms_pred:.2f} (err {do_err*100:.0f}%) | 基线 err {do_err_b*100:.0f}%')
print(f'  DO集中度变化 pred {top10_change_pred*100:+.1f}pp vs 巴克莱实测 {top10_change_real*100:+.1f}pp')

json.dump(dict(design='MSG→Barclays cross-venue blind', n_cal=len(gdsC), n_target=len(gdsT),
               cal=dict(vol_mult=vol_mult, pulse_peak_pp=float(pulse_s.max() * 100),
                        pulse_peak_rel_h=float(rel_grid[pulse_s.argmax()]), delta_d=delta_d,
                        do_top10_base=top10_share(doC), do_top10_game=top10_share(do_gC),
                        delta_ms=delta_ms),
               scores=dict(lam_t=S['lam_t'], lam_t_base=S['lam_t_base'], d=S['d'], d_base=S['d_base'],
                           vol_mape=vol_mape, vol_mape_base=vol_mape_base,
                           window_share=dict(real=sh_real, pred=sh_pred, base=sh_base),
                           do_mean_straight=dict(real=ms_real, pred=float(ms_pred), base=ms_baseT,
                                                 err=do_err, err_base=do_err_b),
                           do_top10_change=dict(pred=float(top10_change_pred), real=float(top10_change_real))),
               msg_baseline_days=nC, bkn_baseline_days=nT),
          open('D:/AI/条件转化/.flow_scores_event.json', 'w'), indent=1, ensure_ascii=False)
print('saved .flow_scores_event.json')
