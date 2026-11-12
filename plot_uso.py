# -*- coding: utf-8 -*-
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus']=False
U=np.load('D:/AI/条件转化/.uso.npz')
hu,hb=U['hr_u']/14,U['hr_b']/14; du,db=U['du'],U['db']
x=np.arange(48)/2
fig,axes=plt.subplots(1,2,figsize=(14.5,5.4))
fig.suptitle('典型事件案例：美网公开赛（Zone 93 法拉盛，2024-08-26~09-08，14天）vs 前后基线（各7天）',
             fontsize=13.5,fontweight='bold',y=1.03)
ax=axes[0]
ax.bar(x,hb,width=0.5,alpha=0.55,color='#A9D18E',label='非赛期基线（743趟/日）')
ax.plot(x,hu,'r-',lw=2.4,label='美网赛期（2373趟/日, 3.2x）')
ax.axvspan(11,17,color='#ED7D31',alpha=0.10); ax.axvspan(19,24,color='#7030A0',alpha=0.10)
ax.text(12,hu.max()*0.86,'日场',fontsize=10,color='#C0504D'); ax.text(20.2,hu.max()*0.95,'夜场散场\n21-24时 3.1x',fontsize=10,color='#7030A0')
ax.set_ylabel('Zone 93 日均出发量')
ax.set_title('① λ(t)：白天+夜间双脉冲，全天 3.2 倍\n（对照：MSG 球赛仅 +2.3% —— 非枢纽场馆不被稀释）',fontsize=11)
ax.legend(fontsize=9.5); ax.grid(alpha=0.3); ax.set_xticks(range(0,25,4))
ax=axes[1]
bins=np.arange(0,30.5,0.5)
ax.hist(db,bins=bins,density=True,alpha=0.55,color='#A9D18E',label=f'非赛期（中位 {np.median(db):.1f} mi）')
ax.hist(du,bins=bins,density=True,alpha=0.55,color='#ED7D31',label=f'美网期（中位 {np.median(du):.1f} mi）')
ax.axvline(np.median(db),color='#548235',ls='--',lw=1.5); ax.axvline(np.median(du),color='#C00000',ls='--',lw=1.5)
ax.set_xlabel('行程距离 (mi)'); ax.set_ylabel('密度')
ax.set_title('② 轨迹长度：观众来自远方\n中位 7.6→10.7 mi（+41%），长距离分量显著抬升',fontsize=11)
ax.legend(fontsize=9.5); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('D:/AI/条件转化/美网影响.png',dpi=150,bbox_inches='tight',facecolor='white')
print('saved')
