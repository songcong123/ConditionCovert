# -*- coding: utf-8 -*-
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus']=False
C=np.load('D:/AI/条件转化/.cases.npz')
base=np.concatenate([C['dec230'][34:48]/5, C['dec230'][:12]/5])
d31=np.concatenate([C['nye230'][34:48], np.zeros(12)])
j1=np.concatenate([np.full(14,np.nan), C['jan1_230'][:12]])
t=np.arange(-7,6,0.5)
fig,axes=plt.subplots(1,2,figsize=(15,5.2))
fig.suptitle('跨年夜 Zone 230（时报广场/时代广场）波形解剖：封街冻结 → 深夜延迟释放',fontsize=13.5,fontweight='bold',y=1.03)
ax=axes[0]
ax.plot(t,base,'-',color='#8FAADC',lw=2.2,label='12月常态（日均）')
ax.plot(t,d31,'o-',color='#C00000',lw=2.2,ms=4,label='12/31 跨年夜')
ax.plot(t,j1,'s-',color='#7030A0',lw=2.2,ms=4,label='1/1 元旦凌晨')
ax.axvspan(-7,0,color='#C00000',alpha=0.08)
ax.axvspan(2.5,4,color='#7030A0',alpha=0.12)
ax.annotate('① 封街冻结：18:00–24:00\n常规 191–359/半小时 → 仅 0–3 趟',xy=(-4,300),xytext=(-6.8,430),fontsize=10,color='#C00000',arrowprops=dict(arrowstyle='->',color='#C00000'))
ax.annotate('② 延迟释放：02:30 起爬升\n03:30 达基线 4.2 倍',xy=(3.5,219),xytext=(0.5,420),fontsize=10,color='#7030A0',arrowprops=dict(arrowstyle='->',color='#7030A0'))
ax.set_xlabel('时间（负值=12/31，正值=1/1）'); ax.set_ylabel('Zone230 半小时出发量')
ax.set_title('① 波形：不是“晚间聚集+午夜爆发”\n而是“冻结 → 凌晨延迟释放”',fontsize=11)
ax.legend(fontsize=9.5); ax.grid(alpha=0.3); ax.set_xticks(range(-7,7,2))

names=['Clinton East\n(地狱厨房,紧邻)','Penn Station\n(186)','Midtown South','Midtown Center','Flatiron','Clinton Hill\n(布鲁克林!)','Murray Hill','Clinton West']
bl=[4321,2434,3231,1593,2119,1270,1250,1294]; nw=[2362,1166,1879,1160,1533,1994,1879,1725]
xp=np.arange(8); w=0.38
ax=axes[1]
ax.bar(xp-w/2,bl,w,color='#8FAADC',label='12月凌晨00-04基线')
ax.bar(xp+w/2,nw,w,color='#ED7D31',label='1/1凌晨00-04')
ax.set_xticks(xp); ax.set_xticklabels(names,fontsize=7.6)
ratios=[f'{n/b:.1f}x' for n,b in zip(nw,bl)]
for i,(n,b) in enumerate(zip(nw,bl)):
    ax.text(i,max(n,b)+90,f'{n/b:.1f}x',ha='center',fontsize=9,fontweight='bold',color='#C00000' if n>b else '#4472C4')
ax.set_ylabel('出发行程数')
ax.set_title('② 环带位移：紧邻区 <1x（还在封锁区）\n外围宴会区 >1x（布鲁克林/穆雷山 1.5-1.6x）',fontsize=11)
ax.legend(fontsize=9.5); ax.grid(alpha=0.3,axis='y')
plt.tight_layout()
plt.savefig('D:/AI/条件转化/跨年夜解剖.png',dpi=150,bbox_inches='tight',facecolor='white')
print('saved')
