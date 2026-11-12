# -*- coding: utf-8 -*-
"""比较四种拟合方法的预测精度：
1. 现行：三箱拟合 θ·ln(1+r/RC)（基线）
2. 多元回归：加高峰/周末/滞后交互项
3. 随机森林：非线性+自动交互
4. 分层模型：日级随机效应+小时级剂量响应
输出：各自 MAE / R² / 预测区间覆盖率"""
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

A=np.load('D:/AI/条件转化/.flow_agg.npz',allow_pickle=True)
D=np.load('D:/AI/条件转化/.flow_days.npz',allow_pickle=True)
dates=list(A['dates'])
klass=dict(zip(D['dates'].tolist(),D['klass'].tolist()))
hc=A['hour_cnt'].astype(float)
dow=np.array([int(np.datetime64(d,'D').view('int64'))%7 for d in dates])
kl=np.array([klass[d] for d in dates])
m_dry=kl=='dry'
wd=(dow+3)%7
lam=np.zeros((7,24))
for k in range(7):
    sel=m_dry&(dow==k)
    if sel.sum()>0: lam[k]=hc[sel].mean(0)

inten=np.zeros((len(dates),24))
didx={d:i for i,d in enumerate(dates)}
for dd,hh,mm in zip(D['rh_date'],D['rh_hour'].astype(int),D['rh_mm']):
    if str(dd) in didx: inten[didx[str(dd)],hh]=mm

# 构建样本：所有雨时(>=0.3mm/h) + 无雨时（作为对照，效应=0附近）
X_all=[]; y_all=[]; meta=[]  # meta=(day_idx, hour)
for i in range(len(dates)):
    if kl[i]=='dry': continue
    for h in range(24):
        base=lam[wd[i],h]
        if base<200: continue
        eff=hc[i,h]/base-1
        r_now=inten[i,h]
        r_lag1=inten[i,h-1] if h>=1 else 0
        r_lag2=inten[i,h-2] if h>=2 else 0
        is_peak=1 if (7<=h<=9 or 17<=h<=19) else 0
        is_we=1 if wd[i]>=5 else 0
        hour_cos=np.cos(2*np.pi*h/24); hour_sin=np.sin(2*np.pi*h/24)
        X_all.append([np.log1p(r_now), np.log1p(r_lag1), np.log1p(r_lag2),
                       is_peak, is_we, is_peak*np.log1p(r_now),
                       is_we*np.log1p(r_now), hour_cos, hour_sin, h])
        y_all.append(eff)
        meta.append((i,h))

X=np.array(X_all); y=np.array(y_all); meta=np.array(meta)
print(f'样本量: {len(y)} 个小时级观测')

# 交叉验证：留出法（随机 80/20）
rng=np.random.default_rng(42)
perm=rng.permutation(len(y))
n_train=int(len(y)*0.8)
tr,te=perm[:n_train],perm[n_train:]
X_tr,X_te,y_tr,y_te=X[tr],X[te],y[tr],y[te]

# ========== 方法1：现行三箱拟合 ==========
theta,RC=0.052,0.2
pred1=theta*np.log1p(X_te[:,0])  # X[:,0]=ln(1+r_now)
mae1=np.mean(np.abs(y_te-pred1))
r2_1=1-np.sum((y_te-pred1)**2)/np.sum((y_te-y_te.mean())**2)
# 预测区间
resid1=y_tr-theta*np.log1p(X_tr[:,0])
ci1_95=np.percentile(np.abs(resid1),95)

# ========== 方法2：多元线性回归（Ridge） ==========
lr=Ridge(alpha=1.0)
lr.fit(X_tr,y_tr)
pred2=lr.predict(X_te)
mae2=np.mean(np.abs(y_te-pred2))
r2_2=1-np.sum((y_te-pred2)**2)/np.sum((y_te-y_te.mean())**2)
resid2=y_tr-lr.predict(X_tr)
ci2_95=np.percentile(np.abs(resid2),95)

# 打印系数
names=['ln(1+r_现在)','ln(1+r_前1h)','ln(1+r_前2h)','是否高峰','是否周末',
       '高峰×雨强','周末×雨强','小时cos','小时sin','小时']
print('\n方法2 系数:')
for n,c in zip(names,lr.coef_):
    print(f'  {n:12s} {c:+.4f}')
print(f'  截距         {lr.intercept_:+.4f}')

# ========== 方法3：随机森林 ==========
rf=RandomForestRegressor(n_estimators=500,max_depth=10,min_samples_leaf=5,random_state=42,n_jobs=-1)
rf.fit(X_tr,y_tr)
pred3=rf.predict(X_te)
mae3=np.mean(np.abs(y_te-pred3))
r2_3=1-np.sum((y_te-pred3)**2)/np.sum((y_te-y_te.mean())**2)
resid3=y_tr-rf.predict(X_tr)
ci3_95=np.percentile(np.abs(resid3),95)
# 特征重要性
print('\n方法3 特征重要性:')
imp_idx=np.argsort(-rf.feature_importances_)
for j in imp_idx[:6]:
    print(f'  {names[j]:12s} {rf.feature_importances_[j]:.3f}')

# ========== 方法4：分层（日级中位数 + 小时级回归） ==========
# 先算每天的中位效应（日级偏移），再看小时级残差能否被雨强解释
day_ids=meta[:,0]
unique_days=np.unique(day_ids)
# 训练集上标定日级偏移
day_shift={}
for d in unique_days:
    m_tr=tr[day_ids[tr]==d]
    if len(m_tr)>0:
        day_shift[d]=np.median(y[m_tr])
# 测试集预测：日级偏移 + 小时级回归残差
lr_res=Ridge(alpha=1.0)
resid_tr=y_tr-np.array([day_shift.get(day_ids[j],0) for j in tr])
lr_res.fit(X_tr,resid_tr)
pred4=lr_res.predict(X_te)+np.array([day_shift.get(day_ids[j],0) for j in te])
# 对未见过的天，日级偏移用训练集全局中位数
global_shift=np.median(list(day_shift.values()))
pred4=np.where(np.isin(day_ids[te],list(day_shift.keys())),pred4,
               lr_res.predict(X_te)+global_shift)
mae4=np.mean(np.abs(y_te-pred4))
r2_4=1-np.sum((y_te-pred4)**2)/np.sum((y_te-y_te.mean())**2)

# ========== 汇总 ==========
print()
print('='*70)
print('                    MAE(越低越好)   R²(越高越好)   95%区间宽度')
print('='*70)
print(f'1.三箱拟合(现行)     {mae1*100:6.1f}pp       {r2_1:.3f}         ±{ci1_95*100:.0f}pp')
print(f'2.多元回归(10特征)   {mae2*100:6.1f}pp       {r2_2:.3f}         ±{ci2_95*100:.0f}pp')
print(f'3.随机森林(500棵)    {mae3*100:6.1f}pp       {r2_3:.3f}         ±{ci3_95*100:.0f}pp')
print(f'4.分层(日级+小时级)  {mae4*100:6.1f}pp       {r2_4:.3f}         —')
print('='*70)
print()
# 方法间差异的显著性（简化：看MAE差 vs MAE 的标准误)
for nm,mae in [('多元回归',mae2),('随机森林',mae3),('分层',mae4)]:
    diff=mae1-mae
    print(f'  {nm} vs 现行: MAE 改善 {diff*100:+.1f}pp')
print()
# 4/3 单日测试
print('===== 4/3 暴雨日逐时预测对比 =====')
DAY='2024-04-03'; i0=didx[DAY]
day_mask_te=day_ids[te]==i0
if day_mask_te.sum()>0:
    X_43=X_te[day_mask_te]; y_43=y_te[day_mask_te]
    p1=theta*np.log1p(X_43[:,0])
    p2=lr.predict(X_43)
    p3=rf.predict(X_43)
    print(f'  小时数: {len(y_43)}')
    print(f'  三箱拟合  MAE: {np.mean(np.abs(y_43-p1))*100:.1f}pp')
    print(f'  多元回归  MAE: {np.mean(np.abs(y_43-p2))*100:.1f}pp')
    print(f'  随机森林  MAE: {np.mean(np.abs(y_43-p3))*100:.1f}pp')
