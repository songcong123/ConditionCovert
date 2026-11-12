const pptxgen = require('pptxgenjs');
const W = 13.33, H = 7.5, M = 0.5;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE';
p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666';

let s = p.addSlide();
s.background = { color: 'FFFFFF' };

s.addText('外推示意：可控范围内标定公式 → 超出范围沿公式延伸', {
  x: M, y: 0.2, w: W - 2 * M, h: 0.4, fontSize: 20, fontFace: F, bold: true, color: NAVY, margin: 0 });

const chartW = 7.2, chartH = chartW / (7.5 / 4.5);
s.addImage({ path: '_extrapolation_chart.png', x: M, y: 0.85, w: chartW, h: chartH });

const rx = M + chartW + 0.4, rw = W - rx - M;
s.addShape(p.shapes.LINE, { x: rx - 0.2, y: 0.85, w: 0, h: chartH, line: { color: 'BFBFBF', width: 1 } });
const pts = [
  ['① 标定（可控范围）', '用轻雨/中雨的实测三个点，拟合出公式 θ·ln(1+r/RC) 的两个系数'],
  ['② 外推（超范围）', '要预测暴雨 30mm/h？代入公式 → +26%。数据没见过，但公式形式有物理约束（饱和），不会爆炸'],
  ['③ 验证（盲测）', '拿真实大雨时段对比：公式预测 +24% vs 实际 +23%（偏差 4%）→ 外推可信'],
];
let py = 0.95;
pts.forEach((pt, j) => {
  s.addShape(p.shapes.OVAL, { x: rx, y: py + 0.05, w: 0.26, h: 0.26, fill: { color: NAVY }, line: { type: 'none' } });
  s.addText(String(j + 1), { x: rx, y: py + 0.05, w: 0.26, h: 0.26, fontSize: 11, fontFace: F, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
  s.addText([{ text: pt[0] + '\n', options: { fontSize: 13, bold: true, color: NAVY } }, { text: pt[1], options: { fontSize: 11, color: '333333' } }],
    { x: rx + 0.4, y: py, w: rw - 0.4, h: 1.5, fontFace: F, align: 'left', valign: 'top', margin: 0, lineSpacingMultiple: 1.3 });
  py += 1.65;
});

s.addText('为什么公式能外推（三个保证）', { x: rx, y: py + 0.1, w: rw, h: 0.3, fontSize: 13, fontFace: F, bold: true, color: NAVY, margin: 0 });
const guarantees = [
  ['① 形式有物理约束', '单调递增+增量饱和→外推不爆炸；6 种饱和公式互验差异<3pp'],
  ['② 系数数据标定', '只需轻中雨数据（365 样本估 2 系数）；盲测偏差仅 4%'],
  ['③ 形状参数锁死', 'σ0/α 不随条件变（实测恒定）；跨年独立拟合全参数重合'],
];
let gy = py + 0.5;
guarantees.forEach((g) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: rx, y: gy, w: rw, h: 0.52, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1 }, rectRadius: 0.04 });
  s.addText([{ text: g[0] + '  ', options: { fontSize: 9.5, bold: true, color: NAVY } }, { text: g[1], options: { fontSize: 8.5, color: '444444' } }],
    { x: rx + 0.08, y: gy, w: rw - 0.16, h: 0.52, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.1 });
  gy += 0.62;
});

p.writeFile({ fileName: 'D:/AI/条件转化/外推示意.pptx' }).then(() => console.log('saved 外推示意.pptx'));
