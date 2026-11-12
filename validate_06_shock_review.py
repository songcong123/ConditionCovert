# -*- coding: utf-8 -*-
"""暴雨(修正口径) + 事件 影响结果统一复核
暴雨: .flow_agg/.flow_spd/.flow_days (FM-15 去重后, 2024-04~10)
     - 24 强雨日四量效应(干日 hour-of-week 基线)
     - top5 单日 δ_vol (核对旧图 8/6 "+21%" 口径)
     - λ(t) 形态差(逐时份额), PU 热力图 JS, σ(log v)
事件: 直接读既有 npz 资产重算头条数字(MSG/Barclays/美网/跨年/马拉松/暴雪)
输出: 控制台账本 + .shock_review.json"""
import numpy as np, json
from scipy import stats

rng = np.random.default_rng(5)
D = np.load('D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
A = np.load('D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
S = np.load('D:/AI/条件转化/.flow_spd.npz')

dates = list(A['dates']); klass = dict(zip(D['dates'].tolist(), D['klass'].tolist()))
dprec = dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))
kl = np.array([klass[d] for d in dates])
dow = np.array([np.datetime64(d, 'D').view('int64') % 7 for d in dates])
hour_cnt = A['hour_cnt'].astype(np.int64)
pu_cnt = A['pu_cnt'].astype(np.int64); do_cnt = A['do_cnt'].astype(np.int64)
dist = A['dist'].astype(np.float64); dist_day = A['dist_day'].astype(np.int64)
spdV = S['v'].astype(np.float64); spdDay = S['day'].astype(np.int64); spdHr = S['hour'].astype(np.int64)

m_dry = kl == 'dry'; m_he = kl == 'heavy'; m_me = kl == 'medium'
dry_i, he_i, me_i = np.where(m_dry)[0], np.where(m_he)[0], np.where(m_me)[0]

# 干日 hour-of-week 基线
how = np.zeros((7, 24)); hown = np.zeros(7)
for k in range(7):
    sel = m_dry & (dow == k)
    how[k] = hour_cnt[sel].sum(0); hown[k] = sel.sum()
lam0_how = how / hown[:, None]
def exp_dry(i): return lam0_how[dow[i]].sum()

dd = dist[np.isin(dist_day, dry_i)]; dd = dd[(dd > 0.1) & (dd < 80)]
med_dry, sh_dry = np.median(np.log(dd)), (dd < 2).mean()
pu_dry = pu_cnt[m_dry].sum(0).astype(float); pu_share0 = pu_dry / pu_dry.sum()
lsv_dry = np.log(spdV[np.isin(spdDay, dry_i)])
mu0 = np.zeros(24)
hsel = spdHr[np.isin(spdDay, dry_i)]
for h in range(24): mu0[h] = lsv_dry[hsel == h].mean()
sig_dry = np.std(lsv_dry - mu0[hsel])

def js(p, q):
    p = np.maximum(p, 1e-12); p /= p.sum(); q = np.maximum(q, 1e-12); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

def day_fx(idx, tag):
    dvol = float(np.mean([hour_cnt[i].sum() / exp_dry(i) - 1 for i in idx]))
    d_d = float(np.mean([np.median(np.log(dist[dist_day == i])) - med_dry for i in idx]))
    sh = float(np.mean([(dist[dist_day == i] < 2).mean() - sh_dry for i in idx]))
    res = [np.mean(np.log(spdV[spdDay == i]) - mu0[spdHr[spdDay == i]]) for i in idx]
    d_v = float(np.mean(res)); sig = float(np.mean([np.std(np.log(spdV[spdDay == i]) - mu0[spdHr[spdDay == i]]) for i in idx]))
    pu_r = pu_cnt[idx].sum(0).astype(float)
    js_pu = js(pu_r / pu_r.sum(), pu_share0)
    prof = hour_cnt[idx].sum(0).astype(float); prof /= prof.sum()
    lam0_t = how.sum(0) / how.sum()
    gt = prof / lam0_t
    print(f'{tag}: δ_vol={dvol*100:+.1f}% δ_d(log中位)={d_d:+.3f} 短途={sh*100:+.1f}pp '
          f'δ_v={d_v:+.3f} σ={sig:.3f} PU_JS={js_pu:.4f}')
    return dict(dvol=dvol, d_d=d_d, sh=sh, d_v=d_v, sig=sig, js_pu=js_pu, g_t=gt.tolist())

print('===== 暴雨(修正口径) 四量效应 =====')
print(f'[参照] 干日 σ(log v)={sig_dry:.3f}')
he_mm = float(np.mean([dprec[d] for d in np.array(dates)[he_i]]))
R_he = day_fx(he_i, f'强雨日 n={len(he_i)} (均值{he_mm:.0f}mm)')
R_me = day_fx(me_i, f'中雨日 n={len(me_i)}')

# λ(t) 时段弹性(强雨): 早晚/日间/晚间 份额比
gt = np.array(R_he['g_t'])
print(f'λ(t) 形态: 早峰7-9时 {gt[7:9].mean():.2f}x | 日间12-16 {gt[12:16].mean():.2f}x | 晚间20-23 {gt[20:23].mean():.2f}x')

# top5 单日 δ_vol (核对旧 "8/6 +21%")
dp_all = np.array([dprec[d] for d in dates])
top5 = he_i[np.argsort(-dp_all[he_i])][:5]
print('\ntop5 单日(修正降水):')
single = {}
for i in top5:
    dv = hour_cnt[i].sum() / exp_dry(i) - 1
    dmed = np.median(np.log(dist[dist_day == i])) - med_dry
    single[dates[i]] = dict(mm=float(dp_all[i]), dvol=float(dv), d_d=float(dmed))
    print(f'  {dates[i]} ({dp_all[i]:.0f}mm, 周{"一二三四五六日"[dow[i]]}): δ_vol={dv*100:+.1f}%  δ_d={dmed:+.3f}')

# ===== 事件(既有npz重算) =====
print('\n===== 事件(重算头条) =====')
ev = {}
E = np.load('D:/AI/条件转化/.event_agg.npz')       # MSG zone186 2025.01-04
g, n = E['hr_g'].astype(float), E['hr_n'].astype(float)
ng, nn = 44, 76
# 散场窗 22:00-23:30 = 半小时桶 44:47
ev['MSG'] = dict(全天倍数=g.sum()/ng/(n.sum()/nn), 散场窗倍数=g[44:47].sum()/ng/(n[44:47].sum()/nn),
                 中位距离=float(np.median(E['dg'])), 基线中位=float(np.median(E['dn'])))
print(f"MSG(186): 全天 x{ev['MSG']['全天倍数']:.3f} | 散场窗22:00-23:30 x{ev['MSG']['散场窗倍数']:.2f} | 中位 {ev['MSG']['基线中位']:.2f}→{ev['MSG']['中位距离']:.2f}mi")
E9 = np.load('D:/AI/条件转化/.event_agg_189.npz')
g9, n9 = E9['hr_g'].astype(float), E9['hr_n'].astype(float)
ev['Barclays'] = dict(全天倍数=g9.sum()/27/(n9.sum()/93), 散场窗倍数=g9[44:47].sum()/27/(n9[44:47].sum()/93),
                      中位距离=float(np.median(E9['dg'])), 基线中位=float(np.median(E9['dn'])))
print(f"Barclays(189): 全天 x{ev['Barclays']['全天倍数']:.3f} | 散场窗 x{ev['Barclays']['散场窗倍数']:.2f} | 中位 {ev['Barclays']['基线中位']:.2f}→{ev['Barclays']['中位距离']:.2f}mi")
U = np.load('D:/AI/条件转化/.uso.npz')
hu, hb = U['hr_u'].astype(float), U['hr_b'].astype(float)
ev['美网'] = dict(全天倍数=hu.sum()/hb.sum(), 中位距离=float(np.median(U['du'])), 基线中位=float(np.median(U['db'])),
                夜场散场倍数=float(hu[42:48].sum()/14/(hb[42:48].sum()/14)))
print(f"美网(93, 14天): 全天 x{ev['美网']['全天倍数']:.1f} | 21-24时 x{ev['美网']['夜场散场倍数']:.1f} | 中位 {ev['美网']['基线中位']:.1f}→{ev['美网']['中位距离']:.1f}mi ({(np.median(U['du'])/np.median(U['db'])-1)*100:+.0f}%)")
C = np.load('D:/AI/条件转化/.cases.npz')
dec230 = C['dec230'].astype(float) / 5; nye230 = C['nye230'].astype(float); jan1 = C['jan1_230'].astype(float)
# 冻结窗 17-24 时(桶34:48) 与 凌晨释放 0-6(桶0:12)
ev['跨年230'] = dict(冻结期比率=float(nye230[34:48].mean()/dec230[34:48].mean()),
                    午夜脉冲倍数=float(jan1[0:12].max()/dec230[0:12].mean()),
                    凌晨3时半倍数=float(jan1[7]/dec230[7]) if dec230[7] > 0 else None)
print(f"跨年Zone230: 17-24时冻结 均值比 x{ev['跨年230']['冻结期比率']:.3f} | 凌晨峰值/基线 x{ev['跨年230']['午夜脉冲倍数']:.1f}")
M = np.load('D:/AI/条件转化/.marathon.npz')
m_day = M['2025_11_02'].astype(float); m_base = (M['2025_11_09'].astype(float)+M['2025_11_16'].astype(float)+M['2025_11_23'].astype(float))/3
ev['马拉松'] = dict(总量=m_day.sum()/m_base.sum()-1, 封路窗8_17=m_day[16:34].sum()/m_base[16:34].sum()-1,
                  傍晚17_21=m_day[34:42].sum()/m_base[34:42].sum()-1)
print(f"马拉松(全市): 总量 {ev['马拉松']['总量']*100:+.1f}% | 封路窗8-17时 {ev['马拉松']['封路窗8_17']*100:+.1f}% | 傍晚17-21时 {ev['马拉松']['傍晚17_21']*100:+.1f}%")
C2 = np.load('D:/AI/条件转化/.cases2.npz')
sb = C2['febSat_base_hr'].astype(float)/2; sn = C2['snowSat_extreme_hr'].astype(float)
ev['暴雪2/8'] = dict(总量=sn.sum()/sb.sum()-1, 中位距离=float(np.median(C2['snowSat_extreme_d'])), 基线中位=float(np.median(C2['febSat_base_d'])))
print(f"暴雪周六2/8(全市): 总量 {ev['暴雪2/8']['总量']*100:+.1f}% | 中位 {ev['暴雪2/8']['基线中位']:.2f}→{ev['暴雪2/8']['中位距离']:.2f}mi")

json.dump(dict(heavy=R_he, medium=R_me, single=single, events=ev, sig_dry=float(sig_dry)),
          open('D:/AI/条件转化/.shock_review.json', 'w'), indent=1, ensure_ascii=False)
print('\nsaved .shock_review.json')
