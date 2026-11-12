# -*- coding: utf-8 -*-
"""从三篇已下载的 arXiv 论文提取: 关键句(按关键词定位+页码) + 关键图(最大嵌入图导出/页面渲染兜底)
输出: quotes_*.json + figures/<paper>_fig*.png"""
import pymupdf, json, os, re

os.makedirs('D:/AI/条件转化/papers/figures', exist_ok=True)

PAPERS = {
    'Gonzalez2008': dict(file='Gonzalez2008_human_mobility_arXiv0806.1256.pdf',
        kws=['truncated power law', 'jump size', 'radius of gyration', 'regularity', 'Lévy', 'return to',
             'high degree', 'probability distribution']),
    'Wang2019': dict(file='Wang2019_ridehailing_demand_arXiv1911.03441.pdf',
        kws=['Chengdu', 'GAIA', 'order', 'November', 'demand', 'convolutional', 'million']),
    'MoveGPT': dict(file='MoveGPT_arXiv2505.18670.pdf',
        kws=['mixture of experts', 'location', 'retrieval', 'fine-tun', 'pre-train', 'POI', 'cross-city',
             'zero-shot', 'scaling']),
}

def sentences(text):
    text = re.sub(r'\s+', ' ', text)
    return re.split(r'(?<=[.!?]) +', text)

for name, cfg in PAPERS.items():
    doc = pymupdf.open(f'D:/AI/条件转化/papers/{cfg["file"]}')
    hits = []
    for pno in range(len(doc)):
        txt = doc[pno].get_text()
        for s in sentences(txt):
            if len(s) < 40 or len(s) > 400: continue
            low = s.lower()
            for kw in cfg['kws']:
                if kw.lower() in low:
                    hits.append(dict(page=pno + 1, kw=kw, text=s.strip()))
                    break
    # 每个关键词取最靠前的2条, 去重文本
    seen, picked = set(), []
    by_kw = {}
    for h in hits:
        k = h['kw']
        if by_kw.get(k, 0) >= 2: continue
        if h['text'][:60] in seen: continue
        seen.add(h['text'][:60]); picked.append(h)
        by_kw[k] = by_kw.get(k, 0) + 1
    json.dump(picked, open(f'D:/AI/条件转化/papers/quotes_{name}.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'== {name}: {len(picked)} 条关键句 ==')
    # 图: 每页找最大嵌入图, 取全文档前3大
    imgs = []
    for pno in range(len(doc)):
        for img in doc[pno].get_images(full=True):
            xref = img[0]
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.width * pix.height < 40000: continue   # 太小的logo跳过
                imgs.append((pix.width * pix.height, pno + 1, xref))
            except Exception:
                pass
    imgs.sort(reverse=True)
    saved = 0
    used_pages = set()
    for area, pno, xref in imgs[:8]:
        if saved >= 3: break
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        out = f'D:/AI/条件转化/papers/figures/{name}_p{pno}_fig{saved+1}.png'
        pix.save(out); saved += 1; used_pages.add(pno)
    # 兜底: 若嵌入图<2张, 渲染含关键词最密集的页面
    if saved < 2:
        cnt = {}
        for h in picked: cnt[h['page']] = cnt.get(h['page'], 0) + 1
        for pno in sorted(cnt, key=cnt.get, reverse=True)[:2 - saved]:
            pix = doc[pno - 1].get_pixmap(dpi=140)
            pix.save(f'D:/AI/条件转化/papers/figures/{name}_page{pno}.png')
    print(f'   figures saved: {saved}(嵌入) 页面兜底: {used_pages}')
