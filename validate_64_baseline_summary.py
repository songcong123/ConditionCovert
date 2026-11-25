# -*- coding: utf-8 -*-
"""validate_64: 基线四分布汇总表指标(核心特征 + 与真实数据偏差)
   口径: 按日期前后切分 —— 前70%日期=估计期, 后30%日期=留出期
         偏差 = 用估计期统计的分布, 去对留出期的真实数据, 看差多少(基线自身的留出误差)
   指标: 出发时间 KL(分工作日/周末) | 距离 分位数MAE(km) | 速度 KL(2km/h分箱) | 空间 份额相关系数
   输出: 控制台四行汇总表
"""
import numpy as np

MI2KM = 1.609344
A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
Z = np.load(r'D:/AI/条件转化/.flow_zones.npz')
dates = list(A['dates']); n = len(dates)
hc = A['hour_cnt'].astype(float)            # (日, 24) 小时级出发量
pu = A['pu_cnt'].astype(float)              # (日, 264) 各栅格出发量
dist = A['dist'].astype(float) * MI2KM; dday = A['dist_day'].astype(int)
v = S['v'].astype(float) * MI2KM; vday = S['day'].astype(int); vh = S['hour'].astype(int)
ok = ~np.isnan(Z['lat'].astype(float))
wd = np.array([(np.datetime64(d, 'D').view('int64') + 3) % 7 < 5 for d in dates])    # 工作日

cut = int(n * 0.7)
tr_d = np.arange(n) < cut; te_d = ~tr_d
has = hc.sum(1) > 0                                   # 剔除无数据的日期
tr_d &= has; te_d &= has
print(f'日期 {dates[0]} ~ {dates[-1]}, 共 {n} 天(有效 {has.sum()} 天) | 估计期 {tr_d.sum()} 天(~{dates[cut-1]}) | 留出期 {te_d.sum()} 天({dates[cut]}~)')
print(f'样本: 出发量 {int(hc.sum()):,} | 距离 {len(dist):,} | 速度 {len(v):,}')

def kl(q, p):
    q = np.asarray(q, float); p = np.asarray(p, float)
    q = q / q.sum(); p = p / p.sum()
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / p[m])))

# ================= ① 出发时间 =================
print('\n' + '=' * 70 + '\n① 出发时间')
for nm, mk in [('工作日', wd), ('周末', ~wd)]:
    h = hc[mk].sum(0); h = h / h.sum() * 100
    am = int(np.argmax(h[5:12]) + 5); pm = int(np.argmax(h[15:22]) + 15)
    peak = h[7:10].sum() + h[17:20].sum()
    print(f'  {nm}: 早峰 {am} 时({h[am]:.1f}%) 晚峰 {pm} 时({h[pm]:.1f}%) | 7-9+17-19 时合计 {peak:.1f}%')
    k = kl(hc[te_d & mk].sum(0), hc[tr_d & mk].sum(0))
    print(f'        留出 KL = {k:.4f}')
# 不分日型
h = hc.sum(0) / hc.sum() * 100
print(f'  全部: 7-9+17-19 时合计 {h[7:10].sum() + h[17:20].sum():.1f}% | 留出 KL = {kl(hc[te_d].sum(0), hc[tr_d].sum(0)):.4f}')
# 逐日: 估计期各小时占比 vs 留出期每天实际占比
for nm, mk in [('工作日', wd), ('周末', ~wd)]:
    p = hc[tr_d & mk].sum(0); p = p / p.sum() * 100
    day = hc[te_d & mk] / hc[te_d & mk].sum(1, keepdims=True) * 100
    kls = [kl(d_, p) for d_ in day]
    print(f'        {nm} 逐日: 占比MAE {np.abs(day - p).mean():.2f} pp | KL 均值 {np.mean(kls):.4f} (最大 {np.max(kls):.4f})')

# ================= ② 行程距离 =================
print('\n' + '=' * 70 + '\n② 行程距离')
qs = [10, 25, 50, 75, 90]
print(f'  全样本: 中位 {np.median(dist):.2f} km | 均值 {dist.mean():.2f} km | <=5km {(dist <= 5).mean()*100:.1f}% | '
      f'>=16km {(dist >= 16).mean()*100:.1f}% | 偏度 {((dist - dist.mean())**3).mean() / dist.std()**3:+.2f}')
dtr = dist[np.isin(dday, np.where(tr_d)[0])]; dte = dist[np.isin(dday, np.where(te_d)[0])]
q_tr = np.percentile(dtr, qs); q_te = np.percentile(dte, qs)
print('  分位数  估计期 ' + ' '.join(f'{x:6.2f}' for x in q_tr))
print('          留出期 ' + ' '.join(f'{x:6.2f}' for x in q_te))
print(f'  留出 分位数MAE = {np.abs(q_tr - q_te).mean():.3f} km | 中位差 = {abs(q_tr[2] - q_te[2]):.3f} km')
# 日中位 MAE(同日型基线)
med_tr = {k: np.median(dtr[np.isin(dday[np.isin(dday, np.where(tr_d)[0])], np.where(wd == k)[0])]) for k in [True, False]}
errs = []
for i in np.where(te_d)[0]:
    x = dist[dday == i]
    if len(x) > 500: errs.append(abs(np.median(x) - med_tr[wd[i]]))
print(f'  留出 日中位MAE(同日型基线) = {np.mean(errs):.3f} km')
q_err = []
for i in np.where(te_d)[0]:
    x = dist[dday == i]
    if len(x) > 500: q_err.append(np.abs(np.percentile(x, qs) - q_tr).mean())
print(f'  留出 逐日分位数MAE(P10~P90) = {np.mean(q_err):.3f} km (最大 {np.max(q_err):.3f})')
# 留出KS
from scipy import stats
ks = stats.ks_2samp(dtr[::20], dte[::5]).statistic
print(f'  留出 KS(估计期 vs 留出期) = {ks:.4f}')

# ================= ③ 速度 =================
print('\n' + '=' * 70 + '\n③ 速度')
print(f'  全样本: 均值 {v.mean():.2f} km/h | 标准差 {v.std():.2f} | 中位 {np.median(v):.2f} | P90 {np.percentile(v, 90):.1f}')
for nm, hrs in [('凌晨 3-5 时', range(3, 6)), ('早高峰 7-9 时', range(7, 10)), ('晚高峰 17-19 时', range(17, 20))]:
    print(f'  {nm} 中位 {np.median(v[np.isin(vh, list(hrs))]):.1f} km/h')
edges = np.arange(0, 72, 2)
vtr = v[np.isin(vday, np.where(tr_d)[0])]; vte = v[np.isin(vday, np.where(te_d)[0])]
ptr, _ = np.histogram(vtr, edges); pte, _ = np.histogram(vte, edges)
print(f'  留出 KL(2 km/h 分箱, 0~70) = {kl(pte, ptr):.4f} | KS = {stats.ks_2samp(vtr[::20], vte[::5]).statistic:.4f}')
p0 = ptr / ptr.sum() * 100
kls, maes = [], []
for i in np.where(te_d)[0]:
    c, _ = np.histogram(v[vday == i], edges)
    if c.sum() < 1000: continue
    kls.append(kl(c, ptr)); maes.append(np.abs(c / c.sum() * 100 - p0).mean())
print(f'  留出 逐日: 分箱占比MAE {np.mean(maes):.3f} pp | KL 均值 {np.mean(kls):.4f} (最大 {np.max(kls):.4f})')

# ================= ④ 空间网格 =================
print('\n' + '=' * 70 + '\n④ 空间网格')
g = pu.sum(0); g[~ok] = 0; gs = np.sort(g)[::-1]; cum = np.cumsum(gs) / gs.sum()
print(f'  有效栅格 {int(ok.sum())} | Top10 占 {cum[9]*100:.1f}% | Top20 占 {cum[19]*100:.1f}% | 前20%栅格({int(round(ok.sum()*0.2))}个) 占 {cum[int(round(ok.sum()*0.2))-1]*100:.1f}%')
p = pu[tr_d].sum(0)[ok]; q = pu[te_d].sum(0)[ok]
print(f'  留出 份额相关系数(汇总) = {np.corrcoef(p, q)[0,1]:.4f}')
ps = p / p.sum() * 100
rs, maes, kls = [], [], []
for i in np.where(te_d)[0]:
    x = pu[i, ok]; x = x / x.sum()
    rs.append(np.corrcoef(ps, x * 100)[0, 1]); maes.append(np.abs(ps - x * 100).mean()); kls.append(kl(x, p))
print(f'  留出 逐日: 份额相关系数 均值 {np.mean(rs):.4f} (最小 {np.min(rs):.4f}) | 份额MAE {np.mean(maes):.3f} pp | KL 均值 {np.mean(kls):.4f}')
print(f'  留出 空间 KL = {kl(q, p):.4f}')
