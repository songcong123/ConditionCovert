# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, pyarrow.parquet as pq
from datetime import datetime, timedelta
def nth(y,m,n):
    d=datetime(y,m,1); c=0
    while True:
        if d.weekday()==6:
            c+=1
            if c==n: return d
        d+=timedelta(days=1)

# 修正口径: 逐小时取最大（去重 FM-15/FM-16），再按日汇总
recs=[]
for y in [2024,2025]:
    df=pd.read_csv(f'D:/AI/条件转化/data/weather/isd/72505394728_{y}.csv',usecols=['DATE','TMP','AA1'],dtype=str)
    for _,r in df.iterrows():
        try: utc=datetime.strptime(r['DATE'],'%Y-%m-%dT%H:%M:%S')
        except: continue
        loc=utc-timedelta(hours=4 if nth(utc.year,3,2)<=utc<nth(utc.year,11,1) else 5)
        aa=r['AA1']
        if not isinstance(aa,str) or ',' not in aa: continue
        p=aa.split(',')
        if p[0]!='01' or p[1] in ('9999','99'): continue
        t=float(r['TMP'].split(',')[0])/10 if ',' in r['TMP'] else np.nan
        recs.append((loc.strftime('%Y-%m-%d'),loc.hour,float(p[1])*0.254,t))
w=pd.DataFrame(recs,columns=['date','h','mm','t'])
hmax=w.groupby(['date','h'])['mm'].max().reset_index()
g=hmax.groupby('date')['mm'].sum().to_frame('mm')
g=g.join(w.groupby('date')['t'].min().rename('tmin'))
print('== 修正后降水极值日 top6 =='); print(g.sort_values('mm',ascending=False).head(6).round(1).to_string())
snow=g[(g['mm']>=8)&(g['tmin']<-0.5)].sort_values('mm',ascending=False)
print('== 修正后降雪日 =='); print(snow.head(5).round(1).to_string())

# ============ 案例聚合 ============
def scan(months, want):
    '''want: dict of date->label; 返回 {label: (half_hour_counts[48], dist_sample, dow)} + zone230 hourly'''
    out={}; z230={}
    for mo in months:
        t=pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                        columns=['pickup_datetime','trip_miles','PULocationID'])
        dt=np.asarray(t['pickup_datetime']); mi=np.asarray(t['trip_miles']); pu=np.asarray(t['PULocationID'])
        dts=dt.astype('datetime64[s]')
        dates=dts.astype('datetime64[D]').astype(str)
        half=(dts.astype('datetime64[m]').astype(int)//30)%48
        m=(mi>0.1)&(mi<100)
        for d_,lab in want.items():
            if not d_.startswith(mo) and not any(d2.startswith(mo) for d2 in ([d_] if isinstance(d_,str) else d_)):
                pass
        for d_ in set(sum([[k] if isinstance(k,str) else k for k in want],[])):
            if d_.startswith(mo):
                lab=want[d_]
                sel=dates==d_
                e=out.setdefault(lab,[np.zeros(48,int),[],[]])
                np.add.at(e[0],half[sel&m],1)
                e[1].append(mi[sel&m][:400000])
                try: e[2].append(datetime.strptime(d_,'%Y-%m-%d').weekday())
                except: e[2].append(0)
                z=sel&m&(pu==230)
                if z.sum()>0:
                    z230.setdefault(lab,np.zeros(48,int)); np.add.at(z230[lab],half[z],1)
        print(f'{mo} scanned')
    for lab in out: out[lab][1]=np.concatenate(out[lab][1]) if out[lab][1] else np.array([])
    return out,z230

# 案例1: 暴雨极值日 2024-09-29 (基线: 9月干日)
# 案例2: 暴雪日 2024-01-06 (基线: 1月干日工作日)
# 案例3: NYE 2024-12-31 + 2025-01-01 (zone230)
# 案例4: 马拉松 2025-11-02 (基线: 11月其他周日)
sep_base=['2024-09-0%d'%i for i in [3,4,10,11,17,18,24,25]]+['2024-09-'+str(i) for i in [1,2,5,8,9,12,15,16,23,26,29]]
sep_base=[d for d in set(['2024-09-02','2024-09-03','2024-09-09','2024-09-10','2024-09-11','2024-09-12','2024-09-16','2024-09-17','2024-09-18','2024-09-19','2024-09-23','2024-09-24','2024-09-25','2024-09-26'])]
jan_base=['2024-01-0%d'%i for i in [2,3,4,5,9,10,11,12,16,17,18,19,23,24,25,26,30,31]]+['2024-01-%d'%i for i in [1,2,8,15,22,29]]
jan_base=['2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-09','2024-01-10','2024-01-11','2024-01-12','2024-01-16','2024-01-17','2024-01-18','2024-01-19','2024-01-23','2024-01-24','2024-01-25','2024-01-26']
nov_sundays=['2025-11-'+str(d).zfill(2) for d in [2,9,16,23,30]]
want={}
for d_ in sep_base: want[d_]='sep_dry'
want['2024-09-29']='rain_extreme'
for d_ in jan_base: want[d_]='jan_base'
want['2024-01-06']='snow'
want['2024-12-29']='nye_pre'; want['2024-12-30']='nye_pre'
want['2024-12-31']='nye_day'; want['2025-01-01']='nye_pulse'
for d_ in ['2024-12-27','2024-12-28','2024-12-02','2024-12-03','2024-12-04']: want[d_]='dec_base'
want['2025-01-02']='jan2_base'; want['2025-01-03']='jan2_base'
for d_ in nov_sundays: want[d_]='nov_sun'
out,z230=scan(['2024-09','2024-01','2024-12','2025-01','2025-11'], want)
np.savez('D:/AI/条件转化/.cases.npz',
    **{f'{lab}_hr':v[0] for lab,v in out.items()},
    **{f'{lab}_d':v[1][:800000] for lab,v in out.items()},
    **{f'{lab}_n':np.array([len(v[2])]) for lab,v in out.items()},
    nye230=z230.get('nye_day',np.zeros(48)), jan1_230=z230.get('nye_pulse',np.zeros(48)),
    pre230=z230.get('nye_pre',np.zeros(48)), dec230=z230.get('dec_base',np.zeros(48)))
for lab,v in sorted(out.items()):
    n=len(v[2]); med=np.median(v[1]) if len(v[1]) else 0
    print(f'{lab}: {n}天, 日均{v[0].sum()/max(n,1)/1e3:.0f}k趟, 距离中位{med:.2f}mi')
