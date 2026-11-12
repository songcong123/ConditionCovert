# -*- coding: utf-8 -*-
"""多维联合公式 vs 单变量公式 对比验证
多维: 效应(r,时段,日型) = θ_s·ln(1+r/RC) × γ_d + β_{s,d}
  θ_s: 各时段雨响应斜率（从数据拟合）
  γ_d: 日型缩放（工作日=1, 周末=拟合）
  β:   无雨基线偏移（本质=λ₀(日型,时段)）
单变量: 效应(r) = θ·ln(1+r/RC)（现行）
评测: 留出大雨+暴雨箱做外推测试"""
import numpy as np, pandas as pd
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

# 时段划分
def period(h):
    if h<=6: return 0   # 凌晨
    if h<=9: return 1   # 早峰
    if h<=16: return 2  # 白天
    if h<=19: return 3  # 晚峰
    return 4            # 夜间
PNAMES=['凌晨','早峰','白天','晚峰','夜间']

# ===== 构建雨时样本 =====
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
        rows.append((r,eff,period(h),dtype,wd[i],h))
df=pd.DataFrame(rows,columns=['r','effect','period','daytype','wd','hour'])
df['is_we']=(df.daytype=='周末').astype(int)
print(f'雨时样本: {len(df)}')
print(f'  训练(轻中雨 r<4): {len(df[df.r<4])} | 测试(大雨+ r>=4): {len(df[df.r>=4])}')

tr=df[df.r<4].copy(); te=df[df.r>=4].copy()

# ===== 单变量公式（现行基线方法）=====
trA=tr[tr.r>=0.3]
def model1(r,theta,RC): return theta*np.log1p(r/RC)
popt1,_=curve_fit(model1,trA.r.values,trA.effect.values,p0=[0.05,0.5])
pred_single=model1(te.r.values,*popt1)
mae1=np.mean(np.abs(te.effect.values-pred_single))

# ===== 多维公式 =====
# 效应(r,s,d) = β_s + θ_s·ln(1+r/RC)   （β_s 为时段固定效应，各时段独立拟合 θ_s）
# 训练段（r<4）分时段拟合 θ_s
print()
print('分时段拟合 θ_s（训练段 r<4）:')
theta_s={}
for s in range(5):
    sub=tr[(tr.period==s)&(tr.r>=0.3)]
    if len(sub)<20:
        theta_s[s]=np.nan; print(f'  {PNAMES[s]}: 样本不足(n={len(sub)})'); continue
    ln_r=np.log1p(sub.r.values)
    # 线性回归: eff = a + θ·ln(1+r/RC), RC固定=0.2（先固定，后续再优化）
    X=np.column_stack([np.ones(len(sub)),ln_r])
    coef,res,_,_=np.linalg.lstsq(X,sub.effect.values,rcond=None)
    theta_s[s]=coef[1]
    print(f'  {PNAMES[s]}: 截距={coef[0]:+.3f} θ={coef[1]:+.3f} (n={len(sub)})')

# 日型缩放 γ: 周末/工作日的 θ 比
gamma={}
for dt in ['工作日','周末']:
    sub=tr[(tr.daytype==dt)&(tr.r>=0.3)&(tr.period==2)]  # 白天段（样本最多）
    if len(sub)>=20:
        ln_r=np.log1p(sub.r.values)
        X=np.column_stack([np.ones(len(sub)),ln_r])
        coef,_,_,_=np.linalg.lstsq(X,sub.effect.values,rcond=None)
        gamma[dt]=coef[1]
        print(f'  {dt}白天 θ={coef[1]:+.3f} (n={len(sub)})')
if '周末' in gamma and '工作日' in gamma:
    g_we=gamma['周末']/max(gamma['工作日'],1e-6)
else:
    g_we=0.6
print(f'  日型缩放 γ_周末/工作日 = {g_we:.2f}')

# 多维预测
# β=0（因变量已是对基线比值，截距归属基线本身）；只用 θ_s×γ_d 调制
def pred_multi(r,period_idx,is_we):
    th=theta_s[period_idx] if not np.isnan(theta_s[period_idx]) else theta_s[2]
    # θ_s 太小（<0.02）说明该时段训练段信噪比不足 → 用全局 θ 替代
    th_all=None
    if th<0.02: th=0.09  # 全局回退值（训练段总拟合斜率量级）
    g=g_we if is_we else 1.0
    return th*g*np.log1p(r/0.2)

pred_multi_all=np.array([pred_multi(row.r,int(row.period),int(row.is_we)) for _,row in te.iterrows()])
mae2=np.mean(np.abs(te.effect.values-pred_multi_all))

# 也算训练段（内插）的精度
te_in=tr[tr.r>=0.3]
pred_multi_in=np.array([pred_multi(row.r,int(row.period),int(row.is_we)) for _,row in te_in.iterrows()])
mae2_in=np.mean(np.abs(te_in.effect.values-pred_multi_in))

# ===== 汇总对比 =====
print()
print('='*70)
print('          单变量公式      多维联合公式')
print('='*70)
print(f'外推(大雨+暴雨)  {mae1*100:5.1f}pp         {mae2*100:5.1f}pp')
print(f'内插(轻中雨段)   —             {mae2_in*100:5.1f}pp')
print(f'自然波动(测试集)  {te.effect.std()*100:5.1f}pp         {te.effect.std()*100:5.1f}pp')
print('='*70)
impr=(1-mae2/mae1)*100
print(f'多维 vs 单变量: 改善 {impr:+.0f}%')
print()

# 分时段×日型细看
print('外推段分解（实际 / 单变量 / 多维）:')
for dt in ['工作日','周末']:
    sub=te[te.daytype==dt]
    sub_single=pred_single[te.daytype==dt]
    for s in range(5):
        m=sub.period==s
        if m.sum()<3: continue
        pm=pred_multi_all[np.where(te.daytype==dt)[0]][m.values] if False else None
        # 重新算多维
        vals=[pred_multi(row.r,int(row.period),int(row.is_we)) for _,row in sub[m].iterrows()]
        print(f'  {dt}·{PNAMES[s]:4s}: 实际{sub[m].effect.mean()*100:+6.1f}% | 单变量{sub_single[m.values].mean()*100:+6.1f}% | 多维{np.mean(vals)*100:+6.1f}%  (n={m.sum()})')

# ===== 保存 =====
import json as _j
json_data=dict(
    theta_s={PNAMES[s]:float(theta_s[s]) for s in range(5)},
    gamma_weekend=float(g_we),
    mae_single=mae1, mae_multi=mae2, mae_multi_interp=mae2_in,
    improvement_pct=float(impr)
)
open('D:/AI/条件转化/.multidim_validation.json','w').write(_j.dumps(json_data,indent=1))
print('\nsaved .multidim_validation.json')
