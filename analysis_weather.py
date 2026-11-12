# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, pyarrow.parquet as pq, csv
from datetime import datetime, timedelta

def nth_sunday(y,m,n):
    d=datetime(y,m,1); c=0
    while True:
        if d.weekday()==6:
            c+=1
            if c==n: return d
        d+=timedelta(days=1)
def is_edt(dt):
    y=dt.year; return nth_sunday(y,3,2)<=dt<nth_sunday(y,11,1)

# ---- 1) 解码中央公园站 2024-06/07/08 逐小时降水 (UTC→NY本地) ----
# ⚠️ 2026-11-06 修复: 只用 FM-15 常规报 + 同(日,时)取 max ——
#    此前未过滤 REPORT_TYPE 且逐报求和, FM-16(SPECI)与 FM-15 同小时重复
#    报 AA1 会被重复累加, 日降水虚高 2~4 倍(与 validate_01_weather.py 对齐)
rows=[]
for mo in ['2024-06','2024-07','2024-08']:
    df=pd.read_csv(f'D:/AI/条件转化/data/weather/isd/72505394728_{mo[:4]}.csv',
                   usecols=['DATE','REPORT_TYPE','AA1'],dtype=str)
    df=df[df['DATE'].str.startswith(mo)]
    df=df[df['REPORT_TYPE'].str.strip()=='FM-15']
    for _,r in df.iterrows():
        try: utc=datetime.strptime(r['DATE'],'%Y-%m-%dT%H:%M:%S')
        except: continue
        loc=utc-timedelta(hours=4 if is_edt(utc) else 5)
        aa=r['AA1']
        if not isinstance(aa,str) or ',' not in aa: continue
        p=aa.split(',')
        if p[0]!='01' or p[1] in ('9999','99'): continue
        mm=float(p[1])*0.254
        rows.append((loc, mm))
w=pd.DataFrame(rows,columns=['dt','precip_mm'])
w['date']=w['dt'].dt.strftime('%Y-%m-%d'); w['hour']=w['dt'].dt.hour
hourly=w.groupby(['date','hour'])['precip_mm'].max()   # 同(日,时)多报取 max 防重复
daily=hourly.groupby(level=0).sum()
rain_hours=hourly
dry_days=set(daily[daily<0.5].index)
rain_days=set(daily[daily>=5.0].index)
print(f'2024-06~08: 干燥日 {len(dry_days)} | 雨日(≥5mm) {len(rain_days)} | 样例雨日 {sorted(rain_days)[:6]}')

# ---- 2) 扫描三个月行程，按雨/干日聚合 ----
hr_r=np.zeros(48,int); hr_d=np.zeros(48,int)     # 半小时桶
dist_r=[]; dist_d=[]
pu_r=np.zeros(264,int); pu_d=np.zeros(264,int)
do_r=np.zeros(264,int); do_d=np.zeros(264,int)
rain_hour_set=set()  # (date,hour) 下雨小时
for (d_,h_),mm in rain_hours.items():
    if d_ in rain_days and mm>=0.25: rain_hour_set.add((d_,int(h_)))
for mo in ['2024-06','2024-07','2024-08']:
    t=pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                    columns=['pickup_datetime','PULocationID','DOLocationID','trip_miles'])
    dt=np.asarray(t['pickup_datetime']); pu=np.asarray(t['PULocationID']); doo=np.asarray(t['DOLocationID']); mi=np.asarray(t['trip_miles'])
    dts=dt.astype('datetime64[s]')
    dates=(dts.astype('datetime64[D]').astype(str))
    hours=(dts.astype('datetime64[h]').astype(int))%24
    half=(dts.astype('datetime64[m]').astype(int)//30)%48
    rain_day_mask=np.isin(dates,list(rain_days))
    dry_day_mask=np.isin(dates,list(dry_days))
    m=(mi>0.1)&(mi<100)&(pu>0)&(pu<=263)&(doo>0)&(doo<=263)
    for mask,tag in [(rain_day_mask&m,'r'),(dry_day_mask&m,'d')]:
        if tag=='r':
            np.add.at(hr_r,half[mask],1); pu_r+=np.bincount(pu[mask],minlength=264); do_r+=np.bincount(doo[mask],minlength=264)
            s=mi[mask]; dist_r.append(s[s<40][:700000])
        else:
            np.add.at(hr_d,half[mask],1); pu_d+=np.bincount(pu[mask],minlength=264); do_d+=np.bincount(doo[mask],minlength=264)
            s=mi[mask]; dist_d.append(s[s<40][:700000])
    print(f'{mo} done')
dr=np.concatenate(dist_r); dd=np.concatenate(dist_d)
np.savez('D:/AI/条件转化/.weather_agg.npz', hr_r=hr_r,hr_d=hr_d,dr=dr[:2000000],dd=dd[:2000000],
         pu_r=pu_r,pu_d=pu_d,do_r=do_r,do_d=do_d,nd=int(len(dry_days)),nr=int(len(rain_days)))
print(f'雨日行程 {hr_r.sum():,} | 干日行程 {hr_d.sum():,}')
print(f'距离中位: 雨 {np.median(dr):.2f} vs 干 {np.median(dd):.2f} mi | 短途<2mi占比: 雨 {(dr<2).mean()*100:.1f}% vs 干 {(dd<2).mean()*100:.1f}%')
