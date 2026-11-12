# -*- coding: utf-8 -*-
import numpy as np, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus']=False
zn={}
for r in csv.reader(open('D:/AI/条件转化/data/tlc/zones/taxi_zone_lookup.csv',encoding='utf-8-sig',newline='')):
    if r and r[0].isdigit(): zn[int(r[0])]=r[2][:14]

C2=np.load('D:/AI/条件转化/.cases2.npz'); C1=np.load('D:/AI/条件转化/.cases.npz'); M=np.load('D:/AI/条件转化/.marathon.npz')
x=np.arange(48)/2
fig,axes=plt.subplots(2,3,figsize=(18,10.5))
fig.suptitle('实际案例：极端天气与大型事件的真实影响（NYC HV FHV 实测）',fontsize=15,fontweight='bold',y=0.995)

# ① 极端暴雨 λ(t)
ax=axes[0][0]
base=C2['augTue_base_hr']/3; ext=C2['rainTue_extreme_hr']
ax.bar(x,base,width=0.5,alpha=0.5,color='#A9D18E',label='干周二基线(n=3, 526k/日)')
ax.plot(x,ext,'r-',lw=2.3,label='极端暴雨 8/6 周二 140mm (638k/日)')
ax.set_title(f'① 极端暴雨 λ(t)：总量 +{(638/526-1)*100:.0f}%\n距离中位 3.08→2.67mi, 短途占比 +5.5pp',fontsize=11)
ax.set_xlabel('时刻(小时)'); ax.set_ylabel('行程数'); ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(range(0,25,4))

# ② 剂量-响应
ax=axes[0][1]
labels=['干日\n基线','轻中雨\n(26天)','极端暴雨\n(140mm)','暴雪周六\n(2/8)']
vol=[0,7,21,-3]; short=[0,2.2,5.5,1.0]; med=[0,-0.16,-0.41,-0.08]
xp=np.arange(4); w=0.27
ax.bar(xp-w,vol,w,label='总量变化 %',color='#4472C4')
ax.bar(xp,short,w,label='短途占比 pp',color='#ED7D31')
ax.bar(xp+w,[m*100 for m in med],w,label='中位距离变化 (×0.01mi)',color='#70AD47')
ax.axhline(0,color='#333',lw=1)
ax.set_xticks(xp); ax.set_xticklabels(labels,fontsize=9.5)
ax.set_title('② 剂量-响应：雨越强效应越大\n（δ_vol/δ_d 参数标定的实测锚点）',fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3,axis='y')

# ③ 暴雪 λ(t)
ax=axes[0][2]
base=C2['febSat_base_hr']/2; ext=C2['snowSat_extreme_hr']
ax.bar(x,base,width=0.5,alpha=0.5,color='#BDD7EE',label='干周六基线(n=2, 800k/日)')
ax.plot(x,ext,'b-',lw=2.3,color='#2E75B6',label='暴雪周六 2/8 (778k/日)')
ax.set_title('③ 暴雪 λ(t)：总量仅 -3%（未崩塌）\n但日内形态改变（看上午/下午对比）',fontsize=11)
ax.set_xlabel('时刻(小时)'); ax.set_ylabel('行程数'); ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(range(0,25,4))

# ④ NYE 时代广场
ax=axes[1][0]
pre=C1['dec230']/5; d31=C1['nye230']; j1=C1['jan1_230']
series=np.concatenate([d31,np.zeros(0)])
ax.plot(x,pre,'-',color='#8FAADC',lw=2,label='12月基线 Zone230(日均)')
ax.plot(x,d31,'o-',color='#C00000',lw=2.2,ms=3.5,label='12/31 跨年夜')
ax.plot(x,j1,'s-',color='#7030A0',lw=2.2,ms=3.5,label='1/1 元旦凌晨')
ax.axvspan(21,24,color='#C00000',alpha=0.1); ax.axvspan(0,3,color='#7030A0',alpha=0.1)
pk=max(d31.max(),j1.max())
ax.annotate('午夜散场脉冲\n(0:00-3:00)',xy=(1.5,j1[:8].max()),xytext=(6,pk*0.8),fontsize=10,color='#7030A0',arrowprops=dict(arrowstyle='->',color='#7030A0'))
ax.set_title('④ 跨年夜 Zone230（时代广场）：\n双重脉冲（晚间聚集+午夜爆发）',fontsize=11)
ax.set_xlabel('时刻(小时)'); ax.set_ylabel('Zone230 出发行程数'); ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(range(0,25,4))

# ⑤ 马拉松
ax=axes[1][1]
others=(M['2025_11_09']+M['2025_11_16']+M['2025_11_23'])/3
ax.bar(x,others,width=0.5,alpha=0.5,color='#A9D18E',label='普通周日(n=3, 703k/日)')
ax.plot(x,M['2025_11_02'],'r-',lw=2.3,label='马拉松周日 11/2 (712k/日)')
ax.axvspan(8,17,color='#C00000',alpha=0.08)
ax.annotate('封路时段 8-17 时\n(全市大面积交通管制)',xy=(12,M['2025_11_02'][24:34].max()),xytext=(1.2,M['2025_11_02'].max()*0.92),fontsize=9.5,arrowprops=dict(arrowstyle='->',color='#666'))
ax.set_title('⑤ NYC 马拉松周日：总量 +1%\n但日内重排（封路时段受抑/傍晚反弹）',fontsize=11)
ax.set_ylabel('行程数'); ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(range(0,25,4))

# ⑥ 结论文字
ax=axes[1][2]; ax.axis('off')
txt='''结论速览
────────────────
暴雨(极端): +21% 总量, 中位 -0.41mi, 短途 +5.5pp
轻中雨:     +7% 总量,  中位 -0.16mi, 短途 +2.2pp
→ 剂量-响应成立, 可标定 δ_vol(r), δ_d(r)

暴雪: 总量未崩(-3%), 打破"先升后崩"直觉
→ 周六+替代效应(地铁受阻改打车)对冲

跨年: Zone230 午夜脉冲(最尖锐的事件信号)
马拉松: 日总量+1% 但日内重排(空间抑制)
→ 事件效应要"对的窗口+对的zone"才可见

时段弹性: 雨日早高峰仅+3%, 晚间+10%
→ 通勤刚需无弹性, 娱乐出行高弹性'''
ax.text(0.02,0.97,txt,fontsize=11,va='top',family='Microsoft YaHei',
        bbox=dict(boxstyle='round',facecolor='#FFF2CC',alpha=0.9))

plt.tight_layout()
plt.savefig('D:/AI/条件转化/极端案例实测.png',dpi=150,bbox_inches='tight',facecolor='white')
print('saved')
