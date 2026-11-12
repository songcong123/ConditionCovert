# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, pyarrow.parquet as pq, csv

ev=list(csv.DictReader(open('D:/AI/条件转化/data/exogenous/events_calendar.csv',encoding='utf-8')))
game_days=set(e['date'] for e in ev if e['taxi_zone']=='186' and e['event_category']=='sports_game')
months=['2025-01','2025-02','2025-03','2025-04']
gd=[d for d in game_days if any(d.startswith(m) for m in months)]
print(f'2025-01~04 MSG(Knicks/Rangers) 主场日: {len(gd)} 天')

hr_g=np.zeros(48,int); hr_n=np.zeros(48,int)
dg=[]; dn=[]; do_g=np.zeros(264,int); do_n=np.zeros(264,int)
for mo in months:
    t=pq.read_table(f'D:/AI/条件转化/data/tlc/fhvhv/fhvhv_tripdata_{mo}.parquet',
                    columns=['pickup_datetime','PULocationID','DOLocationID','trip_miles'])
    dt=np.asarray(t['pickup_datetime']); pu=np.asarray(t['PULocationID']); doo=np.asarray(t['DOLocationID']); mi=np.asarray(t['trip_miles'])
    dts=dt.astype('datetime64[s]'); dates=dts.astype('datetime64[D]').astype(str)
    half=(dts.astype('datetime64[m]').astype(int)//30)%48
    sel=(pu==186)&(mi>0.1)&(mi<100)&(doo>0)&(doo<=263)
    gm=sel&np.isin(dates,gd); nm=sel&~np.isin(dates,gd)
    np.add.at(hr_g,half[gm],1); np.add.at(hr_n,half[nm],1)
    do_g+=np.bincount(doo[gm],minlength=264); do_n+=np.bincount(doo[nm],minlength=264)
    dg.append(mi[gm]); dn.append(mi[nm])
    print(f'{mo} done (赛日Zone186: {gm.sum():,} | 非赛日: {nm.sum():,})')
dg=np.concatenate(dg); dn=np.concatenate(dn)
np.savez('D:/AI/条件转化/.event_agg.npz', hr_g=hr_g,hr_n=hr_n,dg=dg[:900000],dn=dn[:900000],
         do_g=do_g,do_n=do_n,ng=len(gd))
disp=hr_g[42:47].sum()/hr_g.sum()*100; disp_n=hr_n[42:47].sum()/hr_n.sum()*100
print(f'散场窗21-23:30占全日: 赛日 {disp:.1f}% vs 非赛日 {disp_n:.1f}%')
print(f'Zone186出发距离中位: 赛日 {np.median(dg):.2f} vs 非赛日 {np.median(dn):.2f} mi')
