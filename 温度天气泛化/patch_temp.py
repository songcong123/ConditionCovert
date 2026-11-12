# -*- coding: utf-8 -*-
"""补丁: 雨维标注错位修正 + 温度面板外推区注释内收"""
p = r'D:/AI/条件转化/温度天气泛化/make_dual_dose.py'
lines = open(p, encoding='utf-8').read().split('\n')

# 1) 雨维循环块整体替换（定位: 以 'for rr, lab in' 开头的行）
i = next(k for k, l in enumerate(lines) if l.startswith('for rr, lab in'))
new_block = [
    "for rr, lab, off in [(10, '+20%', (-30, 10)), (30, '+26%', (5, 10)), (148, '+34%', (-16, 10))]:",
    "    m = 1 + 0.052 * np.log1p(rr / 0.2)",
    "    ax.plot([rr], [m], 'o', ms=8, color='#C00000', zorder=5)",
    "    ax.annotate(f'{rr}mm/h\\n{lab}', (rr, m), textcoords='offset points', xytext=off, fontsize=9, color='#C00000')",
]
lines[i:i + 4] = new_block

# 2) 温度面板: 外推区注释改为面板内右对齐单行
j = next(k for k, l in enumerate(lines) if "外推区(>40°C)" in l)
lines[j] = "ax.annotate('外推区（挂低置信）', (43.5, 1.17), fontsize=8.5, color='#666', ha='right')"

src = '\n'.join(lines)
open(p, 'w', encoding='utf-8').write(src)
print('patched OK')
print([l for l in lines if '外推区' in l or 'for rr' in l])
