# -*- coding: utf-8 -*-
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus']=False
C=np.load('D:/AI/条件转化/.cases.npz')
# 连续时间轴: 17:00 → 06:00, 共13小时26个半小时点
base=np.concatenate([C['dec230'][34:48]/5, C['dec230'][:12]/5])   # 12月基线(同窗)
d31=np.concatenate([C['nye230'][34:48], np.full(12,np.nan)])      # 12/31 17:00-24:00
j1=np.concatenate([np.full(14,np.nan), C['jan1_230'][:12]])       # 1/1 00:00-06:00
labels=[]
for h in list(range(17,24))+list(range(0,7)):
    labels += [f'{h:02d}:00',f'{h:02d}:30']
t=np.arange(len(base))
xt=pos=list(range(0,len(base),2))
fig,ax=plt.subplots(figsize=(14,5.5))
ax.plot(t,base,'-',color='#8FAADC',lw=2.4,marker='.',ms=0,label='12月常态基线（同窗口均值）')
ax.plot(t,d31,'o-',color='#C00000',lw=2.4,ms=5,label='12/31 跨年夜（17:00–24:00）')
ax.plot(t,j1,'s-',color='#7030A0',lw=2.4,ms=5,label='1/1 元旦（00:00–06:00）')
ax.axvline(14,color='#333',ls='--',lw=1.5)
ax.text(14,470,'  午夜 0:00（跨天）',fontsize=10,color='#333')
ax.axvspan(0,14,color='#C00000',alpha=0.06)
ax.axvspan(19,22,color='#7030A0',alpha=0.12)
ax.annotate('① 封街冻结（12/31 全程）\n基线191–359/半小时 → 实测0–3趟',xy=(8,320),xytext=(0.3,430),fontsize=10.5,color='#C00000',arrowprops=dict(arrowstyle='->',color='#C00000'))
ax.annotate('② 延迟释放（1/1 凌晨2:30起）\n03:30达基线4.2倍',xy=(20.5,219),xytext=(13.5,400),fontsize=10.5,color='#7030A0',arrowprops=dict(arrowstyle='->',color='#7030A6'))
ax.set_xticks(xt); ax.set_xticklabels([labels[i][:5] for i in xt],fontsize=9)
ax.set_xlabel('时间（12/31 17:00 → 次日 1/1 06:00）',fontsize=11)
ax.set_ylabel('Zone 230 出发量（每半小时）',fontsize=11)
ax.set_title('跨年夜时代广场波形解剖：封街冻结 → 深夜延迟释放（横轴为连续时钟）',fontsize=13,fontweight='bold')
ax.legend(fontsize=10,loc='upper right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('D:/AI/条件转化/跨年夜解剖.png',dpi=150,bbox_inches='tight',facecolor='white')
print('fig saved')
