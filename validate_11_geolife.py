# -*- coding: utf-8 -*-
"""GeoLife 验证: 个体移动域 vs NYC 网约车域 三个量对照
1) 距离分布: LogNormal vs 截断幂律(González形式) 数值CDF网格拟合 + KS; 尾α对照NYC
2) 速度分布: σ(log v) 全体与分小时(检验"σ稳定"是否跨域成立)
3) 时间节律: 移动点频 hour-of-week 形态 vs NYC λ₀(t) 三峰
输出 .geolife_val.npz + 控制台报告"""
import numpy as np, os, glob
from datetime import datetime, timedelta
from scipy import stats

RAW = 'D:/AI/条件转化/data/geolife_raw'
plt_files = glob.glob(os.path.join(RAW, '**/*.plt'), recursive=True)
print(f'PLT 轨迹文件数: {len(plt_files)}', flush=True)
users = set()
for f in plt_files: users.add(f.replace('\\', '/').split('/')[-3])   # .../Data/{user}/Trajectory/x.plt
print(f'用户数: {len(users)}')
# 北京范围框(剔除异地/全球旅行段): 纬39.40-40.20, 经115.90-117.10
BJ = (39.40, 40.20, 115.90, 117.10)

R = 6371.0
def hav(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

sk_l, st_l, sh_l, jump_l = [], [], [], []   # 步长km, 步时s, 步小时, 30min窗位移km
how_cnt = np.zeros((7, 24)); n_pts = 0
for f in plt_files:
    try:
        arr = np.loadtxt(f, delimiter=',', skiprows=6, usecols=(0, 1, 5, 6), dtype=str)
    except Exception:
        continue
    if len(arr) == 0: continue
    lat = arr[:, 0].astype(float); lon = arr[:, 1].astype(float)
    ts = np.array([datetime.strptime(d + ' ' + t, '%Y-%m-%d %H:%M:%S') for d, t in zip(arr[:, 2], arr[:, 3])])
    ts = np.array([t + timedelta(hours=8) for t in ts])   # PLT 时间戳为 UTC → 北京当地时间
    n_pts += len(ts)
    if len(ts) < 2: continue
    dt = np.diff(ts).astype('timedelta64[s]').astype(float)
    keep = (dt >= 10) & (dt <= 1800)
    if not keep.any(): continue
    la1, lo1, la2, lo2 = lat[:-1][keep], lon[:-1][keep], lat[1:][keep], lon[1:][keep]
    d = hav(la1, lo1, la2, lo2)
    ok = (d > 0.005) & (d < 200)
    if not ok.any(): continue
    idx = np.where(keep)[0][ok]
    # 北京范围内且移动的是有效城市步
    la_m, lo_m = lat[1:][idx], lon[1:][idx]
    inbj = (la_m >= BJ[0]) & (la_m <= BJ[1]) & (lo_m >= BJ[2]) & (lo_m <= BJ[3])
    idx = idx[inbj]
    if len(idx) == 0: continue
    dstep = hav(lat[:-1][idx], lon[:-1][idx], lat[1:][idx], lon[1:][idx])
    sk_l.append(dstep); st_l.append(dt[idx])
    t2 = ts[1:][idx]
    how = np.array([t.weekday()*24 + t.hour for t in t2])
    sh_l.append(how)
    np.add.at(how_cnt, (how // 24, how % 24), dstep)   # 距离加权: GPS室内抖动≈0贡献
    # 30min 窗位移(González 可比口径): 取每30分钟桶内最后一点
    tt = np.array([(t - ts[0]).total_seconds() for t in ts])
    b = (tt // 1800).astype(int)
    ub = np.unique(b)
    last_idx = [np.where(b == u)[0][-1] for u in ub]
    li = np.array(last_idx)
    dtt = np.diff(ts[li]).astype('timedelta64[s]').astype(float)
    dj = hav(lat[li][:-1], lon[li][:-1], lat[li][1:], lon[li][1:])
    okj = (dtt > 600) & (dtt <= 3900) & (dj > 0.05) & (dj < 100)
    if okj.sum() > 0:
        bjok = (lat[li][1:][okj] >= BJ[0]) & (lat[li][1:][okj] <= BJ[1]) & \
               (lon[li][1:][okj] >= BJ[2]) & (lon[li][1:][okj] <= BJ[3])
        jump_l.append(dj[okj][bjok])
print(f'总 GPS 点 {n_pts:,}')
sk = np.concatenate(sk_l); st = np.concatenate(st_l); sh = np.concatenate(sh_l)
v_kmh = sk / (st / 3600.0)
mv = (v_kmh > 0.5) & (v_kmh < 150)
sk, st, v_kmh, sh = sk[mv], st[mv], v_kmh[mv], sh[mv]
print(f'移动步(有效速度) {len(sk):,}')

# ---------- 1) 距离分布: 两种口径 ----------
jmp = np.concatenate(jump_l) if jump_l else np.array([])
print(f'30min窗位移样本 {len(jmp):,} | 短步样本 {len(sk):,}')
d_mi = jmp / 1.60934                          # 主口径: 窗位移(González可比)
D_ = d_mi[(d_mi > 0.03) & (d_mi < 62)]        # >50m
rng = np.random.default_rng(1)
sub = np.sort(rng.choice(D_, min(len(D_), 300000), replace=False))
emp = np.arange(1, len(sub)+1) / len(sub)
ln_all = np.log(D_)
mu_ln, sd_ln = stats.norm.fit(ln_all)
ks_ln = stats.kstest(ln_all, 'norm', args=(mu_ln, sd_ln)).statistic
# 截断幂积 P(r)∝(r+r0)^-β·exp(-r/κ), 数值CDF, 粗网格+细化两轮
def tpl_ks(x, emp_srt, r0, beta, kap):
    g = np.geomspace(x[0], max(x[-1], kap*10), 3000)
    pdf = (g + r0) ** (-beta) * np.exp(-g / kap)
    c = np.cumsum(pdf); c /= c[-1]
    return np.max(np.abs(np.interp(x, g, c) - emp_srt)), g, c
best = None
for r0 in (0.2, 0.5, 1.0, 2.0, 5.0):
    for beta in np.linspace(0.4, 3.2, 29):
        for kap in np.geomspace(2, 500, 20):
            k, _, _ = tpl_ks(sub, emp, r0, beta, kap)
            if best is None or k < best[0]: best = (k, r0, beta, kap)
ks0, r0a, ba, ka = best
for r0 in np.linspace(max(0.05, r0a-0.5), r0a+0.5, 11):
    for beta in np.linspace(max(0.1, ba-0.25), ba+0.25, 21):
        for kap in np.geomspace(ka/2, ka*2, 21):
            k, _, _ = tpl_ks(sub, emp, r0, beta, kap)
            if k < best[0]: best = (k, r0, beta, kap)
ks_tpl, r0f, betaf, kapf = best
tail = D_[D_ > 1.0]
alpha_geo = 1 + len(tail) / np.log(tail).sum()
print('\n===== 1) 距离分布(个体域·30min窗位移口径) =====')
print(f'LogNormal : KS={ks_ln:.4f}  μ={mu_ln:.2f} σ={sd_ln:.2f} 中位={np.exp(mu_ln):.3f} mi')
print(f'截断幂律  : KS={ks_tpl:.4f}  r0={r0f:.2f}mi β={betaf:.2f} κ={kapf:.1f}mi')
print(f'Pareto尾(>1mi) α={alpha_geo:.2f}   [NYC网约车: α=2.86(>6.5mi), 主体logNormal峰1.76mi]')
print(f'{"截断幂律更优" if ks_tpl < ks_ln else "LogNormal更优"} (ΔKS={abs(ks_ln-ks_tpl):.4f})')

# ---------- 2) 速度分布 ----------
lv = np.log(v_kmh)
sig_all = float(np.std(lv))
sig_h = np.zeros(24); cnt_h = np.zeros(24, int)
for h in range(24):
    s_ = lv[sh % 24 == h]
    if len(s_) > 1000: sig_h[h] = np.std(s_); cnt_h[h] = len(s_)
ok = cnt_h > 0
print('\n===== 2) 速度分布(log km/h) =====')
print(f'全体 σ={sig_all:.3f}（个体混合模式 walk/bike/car/bus）')
print(f'分小时 σ 范围 {sig_h[ok].min():.3f}~{sig_h[ok].max():.3f}（24时剖面）')
print(f'  [NYC网约车门到门: σ₀=0.44(log mph) 跨雨强/时段稳定]')

# ---------- 3) 时间节律(距离加权, 剔除室内GPS抖动) ----------
how_share = how_cnt / how_cnt.sum()
wd_mean = how_share.reshape(7, 24).mean(0); wd_mean /= wd_mean.sum()
print('\n===== 3) 时间节律(移动距离加权, hour-of-week 平均) =====')
peaks = np.argsort(-wd_mean)[:5]
print('Top5 小时: ' + ', '.join(f'{h}时({wd_mean[h]*100:.1f}%)' for h in sorted(peaks)))
print(f'早峰7-9时 {wd_mean[7:9].sum()*100:.1f}% | 午间11-13 {wd_mean[11:13].sum()*100:.1f}% | 晚峰17-19 {wd_mean[17:19].sum()*100:.1f}% | 夜间0-5 {wd_mean[0:5].sum()*100:.1f}%')
print('  [NYC HV FHV: 晚峰17-18时5.7%最大, 三峰]')

np.savez_compressed('D:/AI/条件转化/.geolife_val.npz',
    d_mi=sub.astype(np.float32), wd_mean=wd_mean, how_cnt=how_cnt,
    sig_h=sig_h, cnt_h=cnt_h,
    tpl=np.array([ks_tpl, r0f, betaf, kapf]), ln=np.array([ks_ln, mu_ln, sd_ln]),
    alpha=alpha_geo, sig_all=sig_all, v_kmh=np.sort(v_kmh)[:300000].astype(np.float32))
print('\nsaved .geolife_val.npz')
