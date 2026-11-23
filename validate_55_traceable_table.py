# -*- coding: utf-8 -*-
"""validate_55: 终版结论表 · 逐箱明细可追溯版
================================================
生成三层可追溯链: 终版结论表(6列) → 逐箱明细(箱/n/R̄/实测/RF/公式/差) → 脚本复现
口径: 箱内均值之差(实测均值−预测均值); 终版表每域每模型取"自己的最差箱"(不挑箱)
协议: 出发量/速度=小时级 r<4 训练; 距离=日级 <15mm 训练(与滁州业务同构)
输出: 控制台逐箱表 + Excel v5 新增 sheet[逐箱明细-终版口径] + 更新 sheet[终版结论表]
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

A = np.load(r'D:/AI/条件转化/.flow_agg.npz', allow_pickle=True)
D = np.load(r'D:/AI/条件转化/.flow_days.npz', allow_pickle=True)
dates_all = list(A['dates']); n = len(dates_all)
hc = A['hour_cnt'].astype(float)
kl = np.array([dict(zip(D['dates'].tolist(), D['klass'].tolist()))[d] for d in dates_all])
dow = np.array([int(np.datetime64(d, 'D').view('int64')) % 7 for d in dates_all])
lam = np.zeros((7, 24))
for k in range(7):
    sel = (kl == 'dry') & (dow == k)
    lam[k] = hc[sel].mean(0)
inten = np.zeros((n, 24))
didx = {d: i for i, d in enumerate(D['dates'])}
for dd, hh, mm in zip(D['rh_date'], D['rh_hour'].astype(int), D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)], hh] = mm

detail = {}   # detail[量] = list[dict(箱,n,R̄,obs,rf,fm,d_rf,d_fm, 域)]

# ================= ① 出发量(小时级, 9特征 RF, validate_50 同构) =================
def is_edt(utc):
    y = utc.year
    d0 = datetime(y, 3, 1)
    s0 = d0 + timedelta((6 - d0.weekday()) % 7) + timedelta(7)
    d1 = datetime(y, 11, 1)
    return s0 <= utc < d1 + timedelta((6 - d1.weekday()) % 7)

acc = {}
for yr in [2024, 2025]:
    df = pd.read_csv(rf'D:/AI/条件转化/data/weather/isd/72505394728_{yr}.csv',
                     usecols=['DATE', 'REPORT_TYPE', 'AA1', 'TMP', 'DEW', 'WND'], dtype=str)
    for _, row in df[df['REPORT_TYPE'] == 'FM-15'].iterrows():
        try: utc = datetime.strptime(row['DATE'], '%Y-%m-%dT%H:%M:%S')
        except Exception: continue
        loc = utc - timedelta(hours=4 if is_edt(utc) else 5)
        key = (loc.strftime('%Y-%m-%d'), loc.hour)
        e = acc.setdefault(key, dict(rain=[], temp=[], dew=[], wind=[]))
        if isinstance(row['AA1'], str) and ',' in row['AA1']:
            p = row['AA1'].split(',')
            if p[0] == '01' and p[1] not in ('9999', '99'): e['rain'].append(float(p[1]) * 0.254)
        if isinstance(row['TMP'], str) and ',' in row['TMP']:
            v = float(row['TMP'].split(',')[0]) / 10
            if -40 < v < 55: e['temp'].append(v)
        if isinstance(row['DEW'], str) and ',' in row['DEW']:
            v = float(row['DEW'].split(',')[0]) / 10
            if -40 < v < 40: e['dew'].append(v)
        if isinstance(row['WND'], str) and ',' in row['WND']:
            p = row['WND'].split(',')
            v = float(p[3]) / 10
            if 0 <= v < 60: e['wind'].append(v)
wx = {}
for key, e in acc.items():
    t = float(np.mean(e['temp'])) if e['temp'] else None
    td = float(np.mean(e['dew'])) if e['dew'] else None
    hu = None
    if t is not None and td is not None and td <= t:
        et = 6.112 * np.exp(17.67 * t / (243.5 + t)); ed = 6.112 * np.exp(17.67 * td / (243.5 + td))
        hu = min(max(ed / et * 100, 5), 100)
    wx[key] = dict(rain=float(np.mean(e['rain'])) if e['rain'] else 0.0, temp=t, hu=hu,
                   wind=float(np.mean(e['wind'])) if e['wind'] else np.nan)

rd, rh_, ry, rr, rlag, feat_rows = [], [], [], [], [], []
for i, d in enumerate(dates_all):
    if kl[i] == 'dry': continue
    for h in range(24):
        if lam[dow[i], h] < 200: continue
        w = wx.get((d, h))
        if w is None: continue
        w1 = wx.get((d, h - 1), dict(rain=0.0))
        rd.append(d); rh_.append(h); ry.append(hc[i, h] / lam[dow[i], h] - 1)
        rr.append(w['rain']); rlag.append(w1['rain'])
ry = np.array(ry) * 100; rr = np.array(rr)
H = np.array(rh_); WE = (dow[[dates_all.index(d) for d in rd]] >= 5).astype(float)
TMP = np.nan_to_num([wx[(d, h)]['temp'] if wx[(d, h)]['temp'] else 13.5 for d, h in zip(rd, rh_)], nan=13.5)
HU = np.nan_to_num([wx[(d, h)]['hu'] if wx[(d, h)]['hu'] else 68.0 for d, h in zip(rd, rh_)], nan=68.0)
WD = np.nan_to_num([wx[(d, h)]['wind'] if not np.isnan(wx[(d, h)]['wind']) else 4.5 for d, h in zip(rd, rh_)], nan=4.5)
X = np.column_stack([np.log1p(rr), np.log1p(np.array(rlag)), H,
                     np.sin(2 * np.pi * H / 24), np.cos(2 * np.pi * H / 24), WE, TMP, HU, WD])
m_tr = rr < 4
rf_dep = RandomForestRegressor(400, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_dep.fit(X[m_tr], ry[m_tr])
pr_dep = rf_dep.predict(X)
fm_dep = 0.052 * np.log1p(rr / 0.2) * 100
BINS_DEP = [(0.3, 1, '内插'), (1, 4, '内插'), (4, 8, '外推'), (8, 16, '外推'), (16, 1e9, '外推')]
detail['出发量'] = []
for lo, hi, dom in BINS_DEP:
    m = (rr >= lo) & (rr < hi)
    detail['出发量'].append(dict(box=f'{lo:g}-{hi if hi < 1e8 else "∞"}', n=int(m.sum()),
                         rbar=float(rr[m].mean()), obs=float(ry[m].mean()),
                         rf=float(pr_dep[m].mean()), fm=float(fm_dep[m].mean()), dom=dom))

# ================= ② 速度(小时级, RF 单特征, validate_54 补算同构) =================
S = np.load(r'D:/AI/条件转化/.flow_spd.npz')
spdD = S['day'].astype(int); spdH = S['hour'].astype(int); spdV = S['v'].astype(float)
OK = (spdV > 1) & (spdV < 60)
mu0 = np.full((7, 24), np.nan)
for k7 in range(7):
    sel = np.where((kl == 'dry') & (dow == k7))[0]
    for h in range(24):
        mk = np.isin(spdD, sel) & (spdH == h) & OK
        if mk.sum() > 50: mu0[k7, h] = np.median(np.log(spdV[mk]))
keys = spdD.astype(np.int64) * 100 + spdH
order = np.argsort(keys, kind='stable'); v_s = spdV[order]
uniq, st = np.unique(keys[order], return_index=True); en = np.append(st[1:], len(keys))
bk = dict(zip(uniq.tolist(), zip(st.tolist(), en.tolist())))
pts_r, pts_dv = [], []
for i in np.where(kl != 'dry')[0]:
    for h in range(24):
        r_h = inten[i, h]
        if r_h < 0.3: continue
        b0 = mu0[dow[i], h]
        if np.isnan(b0): continue
        kk = i * 100 + h
        if kk not in bk: continue
        s0, e0 = bk[kk]
        vs = v_s[s0:e0]; vs = vs[(vs > 1) & (vs < 60)]
        if len(vs) < 30: continue
        pts_r.append(r_h); pts_dv.append((np.median(np.log(vs)) - b0) * 100)
pts_r = np.array(pts_r); pts_dv = np.array(pts_dv)
rf_spd = RandomForestRegressor(400, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
mt = pts_r < 4
rf_spd.fit(np.log1p(pts_r[mt]).reshape(-1, 1), pts_dv[mt])
pr_spd = rf_spd.predict(np.log1p(pts_r).reshape(-1, 1))
fm_spd = -0.0221 * np.log1p(pts_r / 0.2) * 100
BINS_V = [(0.3, 1, '内插'), (1, 4, '内插'), (4, 12, '外推'), (12, 1e9, '外推')]
detail['速度'] = []
for lo, hi, dom in BINS_V:
    m = (pts_r >= lo) & (pts_r < hi)
    detail['速度'].append(dict(box=f'{lo:g}-{hi if hi < 1e8 else "∞"}', n=int(m.sum()),
                        rbar=float(pts_r[m].mean()), obs=float(pts_dv[m].mean()),
                        rf=float(pr_spd[m].mean()), fm=float(fm_spd[m].mean()), dom=dom))

# ================= ③ 距离(日级, <15mm, validate_53 同构) =================
dist = A['dist'].astype(float); dist_day = A['dist_day'].astype(int)
dp = np.array([dict(zip(D['dates'].tolist(), D['dprecip'].tolist()))[d] for d in dates_all])
wdays = np.array([((np.datetime64(d, 'D').view('int64') + 3) % 7 < 5).astype(int) for d in dates_all])
dry_idx = np.where(kl == 'dry')[0]
BAS = {}
for g in [1, 0]:
    dvg = dist[np.isin(dist_day, [i for i in dry_idx if wdays[i] == g])]
    BAS[g] = float(np.median(dvg[(dvg > 0.3) & (dvg < 60)]))
Rs, Ys = [], []
for i in np.where(kl != 'dry')[0]:
    dv2 = dist[dist_day == i]; dv2 = dv2[(dv2 > 0.3) & (dv2 < 60)]
    if len(dv2) < 500 or dp[i] < 0.5: continue
    Rs.append(float(dp[i])); Ys.append((np.median(dv2) / BAS[wdays[i]] - 1) * 100)
Rs = np.array(Rs); Ys = np.array(Ys)
RC_D = 5.4
mt_d = Rs < 15
th_d = float(np.sum(Ys[mt_d] * np.log1p(Rs[mt_d] / RC_D)) / np.sum(np.log1p(Rs[mt_d] / RC_D) ** 2))
fm_d = th_d * np.log1p(Rs / RC_D)
rf_d = RandomForestRegressor(400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf_d.fit(np.log1p(Rs[mt_d]).reshape(-1, 1), Ys[mt_d])
pr_d = rf_d.predict(np.log1p(Rs).reshape(-1, 1))
BINS_D = [(0.5, 5, '内插'), (5, 15, '内插'), (15, 30, '外推'), (30, 60, '外推'), (60, 1e9, '外推')]
detail['距离'] = []
for lo, hi, dom in BINS_D:
    m = (Rs >= lo) & (Rs < hi)
    detail['距离'].append(dict(box=f'{lo:g}-{hi if hi < 1e8 else "∞"}', n=int(m.sum()),
                        rbar=float(Rs[m].mean()), obs=float(Ys[m].mean()),
                        rf=float(pr_d[m].mean()), fm=float(fm_d[m].mean()), dom=dom))

# ================= 打印逐箱明细 + 汇总终版数字 =================
final = {}
print('=' * 96)
print('逐箱明细 (单位 %, 差=|预测均值−实测均值|, pp)')
print('=' * 96)
for q in ['出发量', '速度', '距离']:
    print(f'\n【{q}】')
    print(f'{"箱":10s} {"域":4s} {"n":>5s} {"R̄":>7s} {"实测":>8s} {"RF":>8s} {"公式":>8s} {"RF差":>7s} {"公式差":>7s}')
    best = dict(内插={'rf': 0, 'fm': 0}, 外推={'rf': 0, 'fm': 0})
    for r_ in detail[q]:
        d_rf = abs(r_['rf'] - r_['obs']); d_fm = abs(r_['fm'] - r_['obs'])
        r_['d_rf'] = d_rf; r_['d_fm'] = d_fm
        best[r_['dom']]['rf'] = max(best[r_['dom']]['rf'], d_rf)
        best[r_['dom']]['fm'] = max(best[r_['dom']]['fm'], d_fm)
        print(f'{r_["box"]:10s} {r_["dom"]:4s} {r_["n"]:5d} {r_["rbar"]:7.2f} {r_["obs"]:+8.2f} '
              f'{r_["rf"]:+8.2f} {r_["fm"]:+8.2f} {d_rf:7.2f} {d_fm:7.2f}')
    final[q] = best
    print(f'  → 最差箱: 内插 RF {best["内插"]["rf"]:.2f} / 公式 {best["内插"]["fm"]:.2f} | '
          f'外推 RF {best["外推"]["rf"]:.2f} / 公式 {best["外推"]["fm"]:.2f}')
print('\n(出发量 ≥16 最重箱同箱对比参照: RF 13.0 vs 公式 2.3 —— 若按"同箱"口径)')

# ================= 写入 Excel v5 =================
wb = openpyxl.load_workbook(r'D:/AI/条件转化/四量内外终版_v4.xlsx')
F = '微软雅黑'
thin = Border(*[Side(style='thin', color='FFD9D9D9')]*4)
if '逐箱明细-终版口径' in wb.sheetnames: del wb['逐箱明细-终版口径']
ws = wb.create_sheet('逐箱明细-终版口径', 1)
ws['B2'] = '逐箱明细(终版口径): 终版结论表每个数的来源 —— 差=|箱内预测均值−实测均值| (pp), 每域取最差箱'
ws['B2'].font = Font(name=F, size=12, bold=True, color='FF1F4E79')
row = 4
for q in ['出发量', '速度', '距离']:
    ws.cell(row, 2, f'【{q}】').font = Font(name=F, size=11, bold=True, color='FF1F4E79')
    row += 1
    heads = ['箱(mm)', '域', 'n', 'R̄', '实测%', 'RF%', '公式%', '|RF差|pp', '|公式差|pp']
    for c, h in enumerate(heads, 2):
        cell = ws.cell(row, c); cell.value = h
        cell.font = Font(name=F, size=10, bold=True, color='FFFFFFFF')
        cell.fill = PatternFill('solid', fgColor='FF1F4E79'); cell.border = thin
        cell.alignment = Alignment('center', 'center')
    row += 1
    for r_ in detail[q]:
        vals = [r_['box'], r_['dom'], r_['n'], round(r_['rbar'], 2), round(r_['obs'], 2),
                round(r_['rf'], 2), round(r_['fm'], 2), round(r_['d_rf'], 2), round(r_['d_fm'], 2)]
        for c, v in enumerate(vals, 2):
            cell = ws.cell(row, c); cell.value = v
            cell.font = Font(name=F, size=10,
                             bold=(c in (9, 10)) and r_['dom'] == '外推')
            cell.border = thin; cell.alignment = Alignment('center', 'center')
        row += 1
    ws.cell(row, 2, f"最差箱 → 内插 RF {final[q]['内插']['rf']:.2f} / 公式 {final[q]['内插']['fm']:.2f} | "
                    f"外推 RF {final[q]['外推']['rf']:.2f} / 公式 {final[q]['外推']['fm']:.2f}"
            ).font = Font(name=F, size=10, bold=True, color='FF375623')
    row += 3
ws.cell(row, 2, '复现: validate_55_traceable_table.py | 协议: 出发量/速度=小时级r<4训练(9特征RF/单特征RF); '
                '距离=日级<15mm(RC=5.4先验,θ域内标定)').font = Font(name=F, size=9.5, color='FF666666')
for col, w in zip('BCDEFGHIJ', [11, 7, 6, 7, 9, 9, 9, 10, 11]):
    ws.column_dimensions[col].width = w

# 更新终版结论表(出发量外推改最差箱口径 13.6/5.7)
ws2 = wb['终版结论表']
upd = {5: (final['出发量']['内插']['rf'], final['出发量']['内插']['fm'], final['出发量']['外推']['rf'], final['出发量']['外推']['fm']),
       6: (final['速度']['内插']['rf'], final['速度']['内插']['fm'], final['速度']['外推']['rf'], final['速度']['外推']['fm']),
       7: (final['距离']['内插']['rf'], final['距离']['内插']['fm'], final['距离']['外推']['rf'], final['距离']['外推']['fm'])}
for r_, (a, b, c_, d_) in upd.items():
    for c, v in zip((4, 5, 6, 7), (round(a, 2), round(b, 2), round(c_, 1), round(d_, 1))):
        ws2.cell(r_, c).value = v
ws2.cell(10, 2).value = ('参照口径: 出发量若按"≥16最重箱同箱对比"则为 RF 13.0 vs 公式 2.3; 本表统一用"各域各模型自己的最差箱"(不挑箱)。')
ws2.cell(10, 2).font = Font(name=F, size=9.5, color='FF666666')
wb.save(r'D:/AI/条件转化/四量内外终版_v5.xlsx')
print('\nsaved 四量内外终版_v5.xlsx (新增 sheet[逐箱明细-终版口径] + 终版结论表同步更新)')
