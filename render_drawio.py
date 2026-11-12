# -*- coding: utf-8 -*-
import re, xml.etree.ElementTree as ET
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
plt.rcParams['font.family']=['Microsoft YaHei','SimHei','DejaVu Sans']  # 直接列表才触发逐字形回退；经 sans-serif 别名不回退
plt.rcParams['axes.unicode_minus']=False

XML='D:/AI/条件转化/极端天气轨迹生成流程.drawio'
tree=ET.parse(XML); root=tree.getroot()
cells={}
nodes={}; edges=[]
def clean(v):
    v=re.sub(r'<br\s*/?>','\n',v or '')
    v=re.sub(r'<[^>]+>','',v)
    return v
for c in root.iter('mxCell'):
    cid=c.get('id'); cells[cid]=c
    if c.get('vertex')=='1':
        g=c.find('mxGeometry')
        if g is None: continue
        st=c.get('style') or ''
        nodes[cid]=dict(x=float(g.get('x',0)),y=float(g.get('y',0)),
            w=float(g.get('width',80)),h=float(g.get('height',40)),
            label=clean(c.get('value')),style=st)
    elif c.get('edge')=='1':
        st=c.get('style') or ''
        pts=[]
        geo=c.find('mxGeometry')
        if geo is not None:
            arr=geo.find("Array")
            if arr is not None:
                for p in arr.findall('mxPoint'):
                    pts.append((float(p.get('x')),float(p.get('y')))) if p.get('x') and p.get('y')!='None' else None
        def styleval(k,d=None):
            m=re.search(k+r'=([^;]+)',st); return m.group(1) if m else d
        edges.append(dict(src=c.get('source'),dst=c.get('target'),pts=pts,
            label=clean(c.get('value')),dashed='dashed=1' in st,
            color=styleval('strokeColor','#595959'),fontcolor=styleval('fontColor',None),
            ex=styleval('exitX'),ey=styleval('exitY'),nx=styleval('entryX'),ny=styleval('entryY')))
X0,Y0=0,1550; SCALE=0.0135
fig,ax=plt.subplots(figsize=(18.5,21))
ax.set_xlim(-10,1320); ax.set_ylim(1560,-30); ax.axis('off'); ax.set_aspect('equal')
for cid,n in nodes.items():
    st=n['style']; x,y,w,h=n['x'],n['y'],n['w'],n['h']
    fill=re.search(r'fillColor=(#[0-9A-Fa-f]+)',st); fc=fill.group(1) if fill else 'white'
    edgec=re.search(r'strokeColor=(#[0-9A-Fa-f]+)',st); ec=edgec.group(1) if edgec else '#666666'
    dashed='dashed=1' in st
    fs=re.search(r'fontSize=(\d+)',st); fsz=int(fs.group(1))*1.15 if fs else 11
    bold='fontStyle=1' in st or n['label'].startswith(('Step','输出','①','②','③'))
    if 'rhombus' in st:
        cx,cy=x+w/2,y+h/2
        ax.add_patch(Polygon([(cx,y),(x+w,cy),(cx,y+h),(x,cy)],closed=True,fc=fc,ec=ec,lw=1.8))
    else:
        ax.add_patch(FancyBboxPatch((x+3,y+3),w-6,h-6,boxstyle='round,pad=4,rounding_size=6',fc=fc,ec=ec,lw=1.8,ls='--' if dashed else '-'))
    ax.text(x+w/2,y+h/2,n['label'],ha='center',va='center',fontsize=fsz*0.82,
            fontweight='bold' if bold else 'normal',linespacing=1.55,color='#1a1a1a')
def border(n,fx,fy):
    fx=float(fx) if fx else 0.5; fy=float(fy) if fy else 0.5
    return (n['x']+fx*n['w'], n['y']+fy*n['h'])
for e in edges:
    s,t=nodes.get(e['src']),nodes.get(e['dst'])
    if not s or not t: continue
    p0=border(s,e['ex'],e['ey']); p1=border(t,e['nx'],e['ny'])
    path=[p0]+[pt for pt in e['pts'] if pt[0] is not None]+[p1]
    xs=[q[0] for q in path]; ys=[q[1] for q in path]
    ax.plot(xs,ys,color=e['color'],lw=1.8,ls='--' if e['dashed'] else '-',zorder=1,solid_capstyle='round')
    arr=FancyArrowPatch(path[-2],path[-1],arrowstyle='-|>',mutation_scale=16,color=e['color'],lw=1.8,
                        ls='--' if e['dashed'] else '-',zorder=2,shrinkA=0,shrinkB=0)
    ax.add_patch(arr)
    if e['label']:
        segs=[np.hypot(xs[i+1]-xs[i],ys[i+1]-ys[i]) for i in range(len(xs)-1)]
        L=sum(segs); acc=0; mi=segs[0]/2
        for i,sg in enumerate(segs):
            if acc+sg>=L/2: mi=(L/2-acc); k=i; break
            acc+=sg
        frac=mi/max(segs[k],1e-9)
        mx=xs[k]+(xs[k+1]-xs[k])*frac; my=ys[k]+(ys[k+1]-ys[k])*frac
        ax.text(mx,my,e['label'],fontsize=8.6,ha='center',va='center',color=e['fontcolor'] or '#333',
                bbox=dict(boxstyle='round,pad=0.22',fc='white',ec='none',alpha=0.92),zorder=3)
plt.savefig('D:/AI/条件转化/极端天气轨迹生成流程.png',dpi=140,bbox_inches='tight',facecolor='white')
print('saved | nodes:',len(nodes),'edges:',len(edges))
