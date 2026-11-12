# -*- coding: utf-8 -*-
"""分层模型验证: 全局公式 × 直读修正系数 γ
第一层: 全局 θ·ln(1+r/RC)（全数据拟合,最稳）
第二层: γ(时段,日型) = 各格实测效应均值 ÷ 全局公式在该格的预测值（直读比值,无拟合）
测试: 留出大雨+暴雨段验证外推
"""
import numpy as np, pandas as pd, json
from scipy.optimize import curve_fit

A=np.load('.flow_agg.npz',allow_pickle=True)
D=np.load('.flow_days.npz',allow_pickle=True)
dates=list(A['dates'])
klass=dict(zip(D['dates'].tolist(),D['klass'].tolist()))
hc=A['hour_cnt'].astype(float)
dow=np.array([int(np.datetime64(d,'D').view('int64'))%7 for d in dates])
kl=np.array([klass[d] for d in dates])
m_dry=kl=='dry'; wd=(dow+3)%7
is_we=wd>=5
lam=np.zeros((7,24))
for k in range(7):
    sel=m_dry&(wd==k)
    if sel.sum()>0: lam[k]=hc[sel].mean(0)
inten=np.zeros((len(dates),24))
didx={d:i for i,d in enumerate(dates)}
for dd,hh,mm in zip(D['rh_date'],D['rh_hour'].astype(int),D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)],hh]=mm

def period(h):
    if h<=6: return 0
    if h<=9: return 1
    if h<=16: return 2
    if h<=19: return 3
    return 4
PN=['凌晨','早峰','白天','晚峰','夜间']

# 全部雨时样本
rows=[]
for i in range(len(dates)):
    if kl[i]=='dry': continue
    dtype='周末' if is_we[i] else '工作日'
    for h in range(24):
        r=inten[i,h]
        if r<0.3: continue
        base=lam[wd[i],h]
        if base<200: continue
        eff=hc[i,h]/base-1
        rows.append((r,eff,period(h),dtype))
df=pd.DataFrame(rows,columns=['r','effect','period','daytype'])
df['is_we']=(df.daytype=='周末').astype(int)

# ===== 分割: 训练(轻中雨) / 测试(大雨+) =====
tr=df[df.r<4].copy(); te=df[df.r>=4].copy()
print(f'训练(轻中雨): {len(tr)} | 测试(大雨+): {len(te)}')

# ===== 第一层: 全局公式（训练段拟合）=====
tr_rain=tr[tr.r>=0.3]
def gmodel(r,theta,RC): return theta*np.log1p(r/RC)
popt_g,_=curve_fit(gmodel,tr_rain.r.values,tr_rain.effect.values,p0=[0.05,0.5])
th_g,rc_g=popt_g
print(f'全局公式: 效应 = {th_g:.3f}·ln(1+r/{rc_g:.2f})')

# 全局公式在测试段的预测
pred_g_te=gmodel(te.r.values,th_g,rc_g)
mae_g=np.mean(np.abs(te.effect.values-pred_g_te))

# ===== 第二层: 训练段直读修正系数 γ =====
# γ(时段,日型) = 训练段该格实测效应均值 ÷ 全局公式在该格 r 均值处的预测
# 用 r<4 的训练数据, 每格足够样本
gamma_tab={}
for dt in ['工作日','周末']:
    for s in range(5):
        sub=tr[(tr.daytype==dt)&(tr.period==s)]
        if len(sub)<10: continue
        actual=sub.effect.mean()
        # 全局公式在该格平均 r 处的预测
        r_bar=sub.r.mean()
        pred_g=gmodel(r_bar,th_g,rc_g)
        if pred_g>0.001:
            gamma_tab[(dt,s)]=actual/pred_g
print('\n训练段直读 γ(时段,日型) = 实测均值/全局预测:')
for (dt,s),g in sorted(gamma_tab.items()):
    print(f'  {dt}·{PN[s]:4s}: γ={g:.2f}')
# 平滑: 周末格样本少 → 用同时段工作日γ的 weekend/workday 全局比值兜底
g_we_global=np.nan
sub_we=tr[(tr.is_we==1)&(tr.r>=0.3)]
sub_wd=tr[(tr.is_we==0)&(tr.r>=0.3)]
if len(sub_we)>30 and len(sub_wd)>30:
    g_we_global=sub_we.effect.mean()/max(sub_wd.effect.mean(),1e-6)
print(f'  周末/工作日 全局比值兜底 = {g_we_global:.2f}')

def get_gamma(dt,s):
    key=(dt,s)
    if key in gamma_tab: return gamma_tab[key]
    # 兜底1: 同日型全局
    same_dt=[v for (d,_),v in gamma_tab.items() if d==dt]
    if same_dt: return np.mean(same_dt)
    # 兜底2: 同时段跨日型×全局日型比
    cross=[v for (d,_),v in gamma_tab.items() if _==s]
    if cross and g_we_global>0: return np.mean(cross)*(g_we_global if dt=='周末' else 1/g_we_global)
    return 1.0

# ===== 测试段预测 =====
pred_h=np.array([
    gmodel(r,th_g,rc_g)*get_gamma('周末' if w else '工作日',s)
    for r,s,w in zip(te.r.values,te.period.values,te.is_we.values)
])
mae_h=np.mean(np.abs(te.effect.values-pred_h))

# ===== 三方对比 =====
# 单变量（全局公式，无γ）
mae_single=mae_g
# 多维回归（上轮结果 14.9pp 从文件读）
try:
    prev=json.load(open('.multidim_validation.json'))
    mae_multi_reg=prev['mae_multi']
except: mae_multi_reg=14.9

print()
print('='*70)
print(f'{"方法":30s} {"外推MAE":>10s} {"自然波动":>10s}')
print('-'*70)
print(f'{"全局公式（单变量）":30s} {mae_single*100:9.1f}pp {te.effect.std()*100:9.1f}pp')
print(f'{"多维回归（上轮,分时段θ）":30s} {mae_multi_reg*100:9.1f}pp')
print(f'{"分层: 全局×直读γ(时段,日型)":30s} {mae_h*100:9.1f}pp {te.effect.std()*100:9.1f}pp')
print('='*70)
best=min([(1-mae_single),(1-mae_h)])
print()

# 分格对比
print('外推段分格（实际 / 全局 / 分层）:')
for dt in ['工作日','周末']:
    for s in range(5):
        m=(te.daytype==dt)&(te.period==s)
        if m.sum()<3: continue
        a=te.effect[m].mean()
        pg=pred_g_te[m.values].mean()
        ph=pred_h[m.values].mean()
        mark_h='✓' if abs(ph-a)<abs(pg-a) else '✗'
        print(f'  {dt}·{PN[s]:4s}: 实际{a*100:+6.1f}% | 全局{pg*100:+6.1f}% | 分层{ph*100:+6.1f}% {mark_h}  (n={m.sum()})')

mae_by_cell_h=[];mae_by_cell_g=[]
for dt in ['工作日','周末']:
    for s in range(5):
        m=(te.daytype==dt)&(te.period==s)
        if m.sum()<3: continue
        a=te.effect[m].values
        mae_by_cell_g.append(np.mean(np.abs(a-pred_g_te[m.values])))
        mae_by_cell_h.append(np.mean(np.abs(a-pred_h[m.values])))
print(f'\n分格MAE均值: 全局 {np.mean(mae_by_cell_g)*100:.1f}pp | 分层 {np.mean(mae_by_cell_h)*100:.1f}pp')

# 保存
import json as _j
_j.dump(dict(mae_global=mae_single, mae_hybrid=mae_h,
             mae_multireg_prev=mae_multi_reg,
             theta_g=th_g, rc_g=rc_g,
             gamma_tab={f'{d}_{PN[s]}':v for (d,s),v in gamma_tab.items()}),
        open('.multidim_validation.json','w'), indent=1, ensure_ascii=False)
print('saved .multidim_validation.json')
