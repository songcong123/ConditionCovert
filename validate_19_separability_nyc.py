# -*- coding: utf-8 -*-
"""NYC 复刻 P(s|t) 可分性检验(与 didi_analysis/03_separability.py 同款三口径)
数据: HV FHV 2024-04~10, PU 计数 (day, hour, zone) —— 扫描一次缓存 .nyc_hh_zone.npz
口径: 干日基线 U(S|T) + 零假设 + 留出日 OOS + 逐时剖面 + 雨日/强雨日/雨时 U
输出 .nyc_separability.json"""
import numpy as np, json, os
import pyarrow.parquet as pq

rng = np.random.default_rng(7)
MONTHS = [f'2024-{m:02d}' for m in range(4, 11)]
CACHE = 'D:/AI/条件转化/.nyc_hh_zone.npz'

if os.path.exists(CACHE):
    C = np.load(CACHE, allow_pickle=True)
    dates = [str(x) for x in C['dates']]
    cnt = C['cnt']          # (n_days, 24, 264)
    print(f'cache loaded: {len(dates)} days')
else:
    dd = {}
    for mo in MONTHS:
        t = pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                          columns=['pickup_datetime', 'PULocationID'])
        dt = np.asarray(t['pickup_datetime']); pu = np.asarray(t['PULocationID'])
        del t
        dts = dt.astype('datetime64[s]')
        d = dts.astype('datetime64[D]').astype(str)
        h = (dts.astype('datetime64[h]').astype(int)) % 24
        m = (pu >= 1) & (pu <= 263)
        d, h, pu = d[m], h[m], pu[m]
        key = np.char.add(d, np.char.add('\x01', pu.astype('U3')))  # 不必: 直接组 (date_idx, h, pu)
        udates, di = np.unique(d, return_inverse=True)
        flat = (di.astype(np.int64) * 24 + h) * 264 + (pu - 1)
        bc = np.bincount(flat, minlength=len(udates) * 24 * 264).reshape(len(udates), 24, 264)
        for i, u in enumerate(udates):
            dd[u] = bc[i]
        print(f'{mo} done ({len(udates)} days)')
    dates = sorted(dd)
    cnt = np.stack([dd[u] for u in dates])
    np.savez_compressed(CACHE, dates=np.array(dates), cnt=cnt)
    print(f'cache saved: {len(dates)} days, total {cnt.sum():,}')

D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
dprec = dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))
day_idx = {d: i for i, d in enumerate(dates)}
kl = np.array([klass.get(d, 'na') for d in dates])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates])

def js_vec(p, q):
    p = np.maximum(p / p.sum(), 1e-12); q = np.maximum(q / q.sum(), 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

def mi_stats(J):
    J = J.astype(float); J = J / J.sum()
    Pt = J.sum(1, keepdims=True); Ps = J.sum(0, keepdims=True)
    S_ = Pt @ Ps
    mask = J > 0
    I = float((J[mask] * np.log(J[mask] / S_[mask])).sum())
    Hs = float(-(Ps[Ps > 0] * np.log(Ps[Ps > 0])).sum())
    return I, Hs, I / Hs

res = {}
m_dry = kl == 'dry'
dry_cnt = cnt[m_dry]                       # (n_dry, 24, 264)

# ---- 检验一: 干日 U(S|T), 两种粒度 ----
print('\n===== 检验一 干日 I(T;S) / U(S|T) (计数即需求) =====')
J_hh = dry_cnt.sum(0)                      # (24, 264)
I, Hs, U = mi_stats(J_hh)
N = int(J_hh.sum())
Pt = J_hh.sum(1) / J_hh.sum(); Ps0 = J_hh.sum(0) / J_hh.sum()
Sep = np.outer(Pt, Ps0).ravel()
draws = rng.multinomial(N, Sep / Sep.sum(), size=20).reshape(20, 24, 264)
I_null = np.array([mi_stats(d)[0] for d in draws])
z = (I - I_null.mean()) / max(I_null.std(), 1e-12)
res['U_hour'] = dict(I=I, H_S=Hs, U=U, I_null_mean=float(I_null.mean()), z=float(z), n_days=int(m_dry.sum()))
print(f'  T=小时(24): I={I:.4f} H(S)={Hs:.3f} U(S|T)={U*100:.2f}% | 零假设Ī={I_null.mean():.4f} z={z:.0f}')

J_how = np.zeros((7 * 24, 264))
for k in range(7):
    J_how[k*24:(k+1)*24] = dry_cnt[dow[m_dry] == k].sum(0)
I2, Hs2, U2 = mi_stats(J_how)
res['U_how'] = dict(I=I2, H_S=Hs2, U=U2)
print(f'  T=hour-of-week(168): I={I2:.4f} U(S|T)={U2*100:.2f}%')

# ---- 检验二: 留出日 OOS (干日奇偶分) ----
print('\n===== 检验二 留出日预测(奇偶日, JS 越低越好) =====')
di_dry = np.where(m_dry)[0]
tr = di_dry[di_dry % 2 == 1]; te = di_dry[di_dry % 2 == 0]
Ps_tr = cnt[tr].sum((0, 1)).astype(float); Ps_tr /= Ps_tr.sum()
Pst_tr = cnt[tr].sum(0).astype(float); Pst_tr /= Pst_tr.sum(1, keepdims=True)
jsA, jsB = [], []
for i in te:
    for h in range(24):
        v = cnt[i, h].astype(float)
        if v.sum() < 500: continue
        jsA.append(js_vec(Ps_tr, v))
        jsB.append(js_vec(Pst_tr[h], v))
jsA, jsB = np.array(jsA), np.array(jsB)
gain = (1 - jsB.mean() / jsA.mean()) * 100
res['oos'] = dict(js_global=float(jsA.mean()), js_cond=float(jsB.mean()), gain_pct=float(gain), n=int(len(jsA)))
print(f'  模型A P(s)全局: JS={jsA.mean():.5f}\n  模型B P(s|t)条件: JS={jsB.mean():.5f}  → 增益 {gain:+.1f}%')

# ---- 逐时剖面 ----
Ps_g = J_hh.sum(0) / J_hh.sum()
prof = np.array([js_vec(Ps_g, J_hh[h]) for h in range(24)])
res['hourly_js'] = [float(x) for x in prof]
hi = np.argsort(prof)[-5:][::-1]; lo = np.argsort(prof)[:3]
print('  P(s|t) 偏离最大: ' + ', '.join(f'{h:02d}时 JS={prof[h]:.4f}' for h in hi))
print('  偏离最小: ' + ', '.join(f'{h:02d}时 JS={prof[h]:.4f}' for h in lo))

# ---- 外生条件下 U ----
print('\n===== 外生条件下的 U(S|T) =====')
inten = {}
for dd_, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    inten[(str(dd_), hh)] = mm
rain_hours = np.zeros(cnt.shape[:2], dtype=bool)   # (n_days, 24)
for i, d in enumerate(dates):
    for h in range(24):
        if inten.get((d, h), 0) >= 0.3:
            rain_hours[i, h] = True
for name, mask_days, hh_mask in [
        ('干日', m_dry, None),
        ('雨日(全时段)', kl != 'dry', None),
        ('强雨日', kl == 'heavy', None),
        ('雨时(r≥0.3mm/h)', None, rain_hours)]:
    if hh_mask is not None:
        Jx = (cnt * hh_mask[:, :, None]).sum(0)
    else:
        Jx = cnt[mask_days].sum(0)
    Ix, _, Ux = mi_stats(Jx)
    res[f'U_{name}'] = dict(U=Ux, n=int(Jx.sum()))
    print(f'  {name:14s}: U(S|T)={Ux*100:.2f}%  (N={int(Jx.sum()):,})')

json.dump(res, open('D:/AI/条件转化/.nyc_separability.json', 'w'), indent=1, ensure_ascii=False)
print('\nsaved .nyc_separability.json')
