const pptxgen = require('pptxgenjs');
const W = 13.33, H = 7.5, M = 0.5;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE';
p.author = '条件转化';
p.title = '条件转化·方案与示例';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', BG = 'D6E4F0', LIGHT = 'EAF1F8', RED = 'C00000', GRAY = '808080', TEXT = '1A1A1A';

// ============ Slide 1: 流程（Step2 直接带公式）+ 暴雨效果卡 ============
{
  let s = p.addSlide(); s.background = { color: 'FFFFFF' };
  s.addText('方案总流程：常态基线 P0 × 条件调整 f(c)', {
    x: M, y: 0.2, w: W - 2 * M, h: 0.5, fontSize: 24, fontFace: F, bold: true, color: NAVY, margin: 0 });
  s.addText('任意外生条件通用 —— 只改参数，不改分布形状；Step 2 每行就是对应量的转换公式', {
    x: M, y: 0.66, w: W - 2 * M, h: 0.26, fontSize: 12, fontFace: F, color: GRAY, margin: 0 });

  const y0 = 1.02, rH = 0.56, rGap = 0.06;

  // 输入框
  const inX = M, inW = 1.35;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: inX, y: y0 + 0.4, w: inW, h: 0.5, fill: { color: LIGHT }, line: { color: NAVY, width: 1.2 }, rectRadius: 0.05 });
  s.addText('常态行程数据', { x: inX, y: y0 + 0.4, w: inW, h: 0.5, fontSize: 10, fontFace: F, bold: true, color: TEXT, align: 'center', valign: 'middle', margin: 0 });
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: inX, y: y0 + 1.3, w: inW, h: 0.5, fill: { color: LIGHT }, line: { color: NAVY, width: 1.2 }, rectRadius: 0.05 });
  s.addText('外生条件 c', { x: inX, y: y0 + 1.3, w: inW, h: 0.5, fontSize: 10, fontFace: F, bold: true, color: TEXT, align: 'center', valign: 'middle', margin: 0 });
  s.addShape(p.shapes.LINE, { x: inX + inW, y: y0 + 0.65, w: 0.25, h: 0, line: { color: NAVY, width: 2, endArrowType: 'triangle' } });

  // Step 1：四个量的基线
  const s1X = inX + inW + 0.32, s1W = 1.95, s1TH = 0.3;
  const s1H = s1TH + 4 * rH + 3 * rGap + 0.1;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: s1X, y: y0, w: s1W, h: s1H, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1.8 }, rectRadius: 0.06 });
  s.addText('Step 1  基线\n（晴天直读）', { x: s1X, y: y0 + 0.02, w: s1W, h: s1TH, fontSize: 10, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0, lineSpacingMultiple: 1.05 });
  const labels1 = ['出发时间', '空间分布', '行程距离', '速度'];
  labels1.forEach((t, i) => {
    const ry = y0 + s1TH + 0.05 + i * (rH + rGap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: s1X + 0.08, y: ry, w: s1W - 0.16, h: rH, fill: { color: BG }, line: { color: NAVY, width: 0.8 }, rectRadius: 0.03 });
    s.addText(t, { x: s1X + 0.15, y: ry, w: s1W - 0.3, h: rH, fontSize: 11, fontFace: F, bold: true, color: TEXT, align: 'left', valign: 'middle', margin: 0 });
  });

  // 平行箭头
  const s2X = s1X + s1W + 0.45, s2W = 3.3;
  labels1.forEach((_, i) => {
    const yM = y0 + s1TH + 0.05 + i * (rH + rGap) + rH / 2;
    s.addShape(p.shapes.LINE, { x: s1X + s1W, y: yM, w: s2X - s1X - s1W - 0.05, h: 0, line: { color: NAVY, width: 2.2, endArrowType: 'triangle' } });
  });

  // Step 2：每行直接写公式
  const s2TH = 0.3;
  const s2H = s2TH + 4 * rH + 3 * rGap + 0.1;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: s2X, y: y0, w: s2W, h: s2H, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1.8 }, rectRadius: 0.06 });
  s.addText('Step 2  条件调整（乘系数 / 调参数）', { x: s2X, y: y0 + 0.02, w: s2W, h: s2TH, fontSize: 10.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
  const labels2 = [
    ['① 出发量 ×(1+δvol)', 'δvol = 0.052·ln(1+r/0.2)'],
    ['② 空间 ×γ(s,c)', '雨：核心×1.07 / 锚点×0.7；事件查表'],
    ['③ 距离调 3 参数', 'μ1−0.06 · w1+0.025 · w3×0.9'],
    ['④ 速度 ×exp(−δv)', 'δv = −0.025·ln(1+r/0.3)，σ0=0.44 锁死'],
  ];
  labels2.forEach((t, i) => {
    const ry = y0 + s2TH + 0.05 + i * (rH + rGap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: s2X + 0.08, y: ry, w: s2W - 0.16, h: rH, fill: { color: BG }, line: { color: NAVY, width: 0.8 }, rectRadius: 0.03 });
    s.addShape(p.shapes.OVAL, { x: s2X + 0.13, y: ry + 0.2, w: 0.17, h: 0.17, fill: { color: NAVY }, line: { type: 'none' } });
    s.addText('①②③④'[i], { x: s2X + 0.13, y: ry + 0.2, w: 0.17, h: 0.17, fontSize: 8, fontFace: F, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
    s.addText([{ text: t[0], options: { fontSize: 9.5, bold: true, color: NAVY, breakLine: true } },
               { text: t[1], options: { fontSize: 7.5, color: '444444' } }],
      { x: s2X + 0.37, y: ry, w: s2W - 0.5, h: rH, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });
  });

  // 链条 Step3-5
  const s3X = s2X + s2W + 0.45;
  labels1.forEach((_, i) => {
    const yM = y0 + s2TH + 0.05 + i * (rH + rGap) + rH / 2;
    s.addShape(p.shapes.LINE, { x: s2X + s2W, y: yM, w: s3X - s2X - s2W - 0.05, h: 0, line: { color: NAVY, width: 2, endArrowType: 'triangle' } });
  });
  const cW = 3.0, cH = 0.62, cGap = 0.16;
  [['Step 3  采样合成', '时刻·起点·距离·速度'], ['Step 4  终点配对', '按起点+距离选终点'], ['Step 5  校验 → 输出', '三角一致·容量·sanity']].forEach((c, i) => {
    const cy = y0 + i * (cH + cGap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: s3X, y: cy, w: cW, h: cH, fill: { color: i === 2 ? LIGHT : 'FFFFFF' }, line: { color: NAVY, width: i === 2 ? 2 : 1.2 }, rectRadius: 0.05 });
    s.addText([{ text: c[0], options: { fontSize: 11, bold: true, breakLine: true } }, { text: c[1], options: { fontSize: 8.5, color: '555555' } }],
      { x: s3X + 0.1, y: cy, w: cW - 0.2, h: cH, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.1 });
    if (i < 2) s.addShape(p.shapes.LINE, { x: s3X + cW / 2, y: cy + cH, w: 0, h: cGap, line: { color: NAVY, width: 1.5, endArrowType: 'triangle' } });
  });

  // 底部：四个量的暴雨效果卡
  const fY = y0 + s1H + 0.28;
  s.addShape(p.shapes.LINE, { x: M, y: fY, w: W - 2 * M, h: 0, line: { color: 'BFBFBF', width: 1 } });
  s.addText('暴雨时的实测效果（红色数字来自 52 个真实暴雨日）', { x: M, y: fY + 0.04, w: W - 2 * M, h: 0.28, fontSize: 12, fontFace: F, bold: true, color: NAVY, margin: 0 });
  const eff = [
    ['① 出发时间', '+9~22%', '峰形不动'],
    ['② 空间分布', 'JS=0.0009', '几乎不动'],
    ['③ 行程距离', '中位 −6%', '短途 +2.5pp'],
    ['④ 速度', '慢 4~10%', '形状不变'],
  ];
  const cGapX = 0.15, cW2 = (W - 2 * M - 3 * cGapX) / 4;
  eff.forEach((c, i) => {
    const cx = M + i * (cW2 + cGapX);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: cx, y: fY + 0.38, w: cW2, h: 1.25, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1.2 }, rectRadius: 0.05,
      shadow: { type: 'outer', blur: 5, offset: 2, angle: 45, opacity: 0.1 } });
    s.addText(c[0], { x: cx + 0.1, y: fY + 0.42, w: cW2 - 0.2, h: 0.26, fontSize: 10, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
    s.addText(c[1], { x: cx + 0.1, y: fY + 0.66, w: cW2 - 0.2, h: 0.55, fontSize: 20, fontFace: F, bold: true, color: RED, align: 'left', valign: 'middle', margin: 0 });
    s.addText(c[2], { x: cx + 0.1, y: fY + 1.24, w: cW2 - 0.2, h: 0.3, fontSize: 8.5, fontFace: F, color: GRAY, align: 'left', margin: 0 });
  });

  s.addText('校验不通过 → 回 Step 2 修正；用可观测的真实雨天做盲测，通过才放行外推到极端', {
    x: M, y: fY + 1.78, w: W - 2 * M, h: 0.26, fontSize: 10, fontFace: F, color: GRAY, align: 'center', margin: 0 });
  s.addText('来源：NYC HV FHV 2024+2025 两暖季实测（2.61 亿行）', {
    x: M, y: H - 0.4, w: W - 2 * M, h: 0.25, fontSize: 9, fontFace: F, color: '999999', align: 'right', margin: 0 });
}

// ============ Slide 2: 调整过程示例 ============
{
  let s = p.addSlide(); s.background = { color: 'FFFFFF' };
  s.addText('调整过程示例：2024-04-03 周三暴雨 99mm —— 晴天基线怎样被调成暴雨分布', {
    x: M, y: 0.25, w: W - 2 * M, h: 0.55, fontSize: 20, fontFace: F, bold: true, color: NAVY, margin: 0 });
  s.addImage({ path: '调整过程示例.png', x: M, y: 1.0, w: 7.6, h: 5.94 });
  const rx = 8.35, rw = W - M - rx;
  s.addShape(p.shapes.LINE, { x: rx - 0.2, y: 1.0, w: 0, h: 5.6, line: { color: 'BFBFBF', width: 1 } });
  const pts = [
    ['① 出发时间', '总量 +14%（预测）vs +25%（真实）；峰形不动、早高峰段贴合最好'],
    ['② 速度', '逐时 ×exp(−δv)：雨大的小时减速多；形状不动（σ0=0.44）'],
    ['③ 距离', '三旋钮：短途峰左移 1.74→1.64 mi、短途占比 +2.5pp、长途 ×0.9'],
    ['④ 空间', '暴雨天【不动】：起点热力图与晴天 JS=0.0002（事件才动空间）'],
  ];
  let py = 1.1;
  pts.forEach((pt, j) => {
    s.addShape(p.shapes.OVAL, { x: rx, y: py + 0.05, w: 0.24, h: 0.24, fill: { color: NAVY }, line: { type: 'none' } });
    s.addText('①②③④'[j], { x: rx, y: py + 0.05, w: 0.24, h: 0.24, fontSize: 10, fontFace: F, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
    s.addText([{ text: pt[0] + '\n', options: { fontSize: 12.5, bold: true, color: NAVY } }, { text: pt[1], options: { fontSize: 10, color: '333333' } }],
      { x: rx + 0.36, y: py, w: rw - 0.36, h: 1.15, fontFace: F, align: 'left', valign: 'top', margin: 0, lineSpacingMultiple: 1.25 });
    py += 1.35;
  });
  s.addText('来源：NYC HV FHV 2024+2025 两暖季实测', { x: M, y: H - 0.42, w: W - 2 * M, h: 0.25, fontSize: 9, fontFace: F, color: '999999', align: 'right', margin: 0 });
}

// ============ 图页 ============
const aspect = { '基线四分布.png': 1.55, '距离调整三旋钮.png': 1.9, '空间分布真实例子.png': 1.57,
                 '真实例子_暴雨与跨年.png': 1.79, '暴雨起点分布分析.png': 1.74, 'GeoLife验证.png': 3.12 };
const figs = [
  ['基线四分布.png', '基线四分布', '四个基线部件的真实长相：时间三峰 / 空间地图 / 距离四分量 / 速度 μ0(t)+σ0 稳定'],
  ['距离调整三旋钮.png', '距离调整：三个旋钮', '短途更短（μ1 左移）、短途更多（w1 增）、长途更少（w3 降）→ 合成 vs 实测 KS=0.024'],
  ['空间分布真实例子.png', '空间分布：三种冲击', '美网点汇聚 ×3.5 / 马拉松沿线冻结 −48% / 暴雨全城近似不动'],
  ['真实例子_暴雨与跨年.png', '两个真实日子', '2024-04-03 暴雨：订单 +27%、中位距离 3.1→2.4 mi；跨年夜：封街 ×0.004 → 午夜释放 ×3'],
  ['暴雨起点分布分析.png', '暴雨对起点分布的影响', '雨时级实测：机场 −27%、密集核心 +7%（向心收缩）'],
  ['GeoLife验证.png', '跨域验证（GeoLife 北京个体）', '个体域=截断幂律 ≠ 网约车域=LN 混合；形状可迁移、参数需本地重建'],
];
figs.forEach((f) => {
  let s = p.addSlide(); s.background = { color: 'FFFFFF' };
  s.addText(f[1], { x: M, y: 0.25, w: W - 2 * M, h: 0.55, fontSize: 24, fontFace: F, bold: true, color: NAVY, margin: 0 });
  s.addText(f[2], { x: M, y: 0.8, w: W - 2 * M, h: 0.35, fontSize: 13, fontFace: F, color: GRAY, margin: 0 });
  const ar = aspect[f[0]];
  let dispW = W - 2 * M, dispH = dispW / ar;
  if (dispH > 5.7) { dispH = 5.7; dispW = dispH * ar; }
  const x0 = (W - dispW) / 2;
  s.addImage({ path: f[0], x: x0, y: 1.35, w: dispW, h: dispH });
  s.addText('来源：NYC HV FHV 实测 / GeoLife 官方数据集', { x: M, y: H - 0.42, w: W - 2 * M, h: 0.25, fontSize: 10, fontFace: F, color: '999999', align: 'right', margin: 0 });
});

p.writeFile({ fileName: 'D:/AI/条件转化/条件转化_方案与示例.pptx' }).then(() => console.log('saved'));
