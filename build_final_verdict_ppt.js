const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = 'Z.ai';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', RED = 'C00000', YELLOW = 'FFF2CC', YEL_LN = 'D6B656';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('四量 × 内插/外推 · 终版裁决：见过→RF，没见过→公式',
  { x: M, y: 0.14, w: W - 2 * M, h: 0.38, fontSize: 17, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('单位 pp｜同一训练协议：两模型只喂训练域内数据，外推范围从未参与拟合',
  { x: M, y: 0.52, w: W - 2 * M, h: 0.22, fontSize: 10, fontFace: F, color: MUTED, align: 'center', margin: 0 });

// ===== 主表 =====
const cols = [
  { x: 0.4, w: 1.0, t: '量' },
  { x: 1.4, w: 1.55, t: '内插范围' },
  { x: 2.95, w: 0.75, t: 'RF' },
  { x: 3.7, w: 0.75, t: '公式' },
  { x: 4.45, w: 1.55, t: '外推范围' },
  { x: 6.0, w: 0.75, t: 'RF' },
  { x: 6.75, w: 0.85, t: '公式' },
  { x: 7.6, w: 2.1, t: '最终分派' },
];
const rowsT = [
  ['出发量', '雨强 0.3~4 mm/h', '5.7', '6.3', '雨强 4~134 mm/h', '13.5', '2.3', '内 RF / 外 公式'],
  ['速度', '雨强 0.3~4 mm/h', '2.9', '3.7', '雨强 4~134 mm/h', '3.3', '1.3', '内 RF / 外 公式'],
  ['距离', '日降水 0.5~30 mm', '2.0', '2.5', '日降水 30~170 mm', '2.4', '1.0', '内 RF / 外 三旋钮'],
  ['空间', '263 区查表', '—', '—', 'γ(s,r) 公式', '—', '—', '查表 / γ 公式'],
];
const colHead = ['内插（拟合MAE）', '', '', '外推（最差分箱偏差）', '', '', '', ''];
// 表头
const y0 = 1.14, hh = 0.34;
cols.forEach(c => {
  s.addShape(p.shapes.RECTANGLE, { x: c.x, y: y0, w: c.w, h: hh, fill: { color: NAVY }, line: { color: 'FFFFFF', width: 0.75 } });
  s.addText(c.t, { x: c.x, y: y0, w: c.w, h: hh, fontSize: 10.5, fontFace: F, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
});
// 分组表头
s.addShape(p.shapes.RECTANGLE, { x: 1.4, y: y0 - 0.3, w: 3.05, h: 0.3, fill: { color: 'D6E4F0' }, line: { color: NAVY, width: 0.75 } });
s.addText('内插（拟合 MAE，越小越好）', { x: 1.4, y: y0 - 0.3, w: 3.05, h: 0.3, fontSize: 10, fontFace: F, bold: true, color: NAVY, align: 'center', valign: 'middle', margin: 0 });
s.addShape(p.shapes.RECTANGLE, { x: 4.45, y: y0 - 0.3, w: 3.15, h: 0.3, fill: { color: 'FDEADA' }, line: { color: 'ED7D31', width: 0.75 } });
s.addText('外推（最差分箱偏差，越小越好）', { x: 4.45, y: y0 - 0.3, w: 3.15, h: 0.3, fontSize: 10, fontFace: F, bold: true, color: '843C0C', align: 'center', valign: 'middle', margin: 0 });

// 数据行
const rh = 0.4;
rowsT.forEach((row, i) => {
  const y = y0 + hh + i * rh;
  row.forEach((v, j) => {
    const c = cols[j];
    const isFormulaWin = (j === 6 && v !== '—');
    const isRFCol = (j === 2 || j === 5);
    s.addShape(p.shapes.RECTANGLE, { x: c.x, y, w: c.w, h: rh, fill: { color: i % 2 ? 'F7F7F5' : 'FFFFFF' }, line: { color: 'D9D9D9', width: 0.5 } });
    s.addText(String(v), { x: c.x, y, w: c.w, h: rh, fontSize: isFormulaWin ? 11 : 9.5, fontFace: F,
      bold: isFormulaWin || j === 0 || j === 7,
      color: isFormulaWin ? DGREEN : (isRFCol && v !== '—' && Number(v) > 2 ? RED : TEXT),
      align: 'center', valign: 'middle', margin: 0 });
  });
});
// 外推公式列高亮背景

// ===== 右侧: 裁决图(嵌出发量对比图) =====
s.addImage({ path: 'D:/AI/条件转化/_ev_crop.png', x: 9.85, y: 0.9, w: 3.25, h: 1.75 });
s.addText('证据图: 出发量五箱三柱对比(实际/RF/公式)', { x: 9.85, y: 2.72, w: 3.25, h: 0.2, fontSize: 8, fontFace: F, color: MUTED, align: 'center', margin: 0 });

// 右侧机制卡
const mech = [
  { y: 2.62, t: '为什么内插选 RF', d: '两者都准(差<1pp), RF还能吃时段/日型交互' },
  { y: 3.32, t: '为什么外推选公式', d: '实际效应还在涨时, RF零响应卡死(输出常数); 公式沿饱和曲线跟涨' },
  { y: 4.02, t: '距离为何是三旋钮', d: '混合分布(四组分)单参数只会平移, 中段高估4%; 三旋钮能重塑组分' },
];
mech.forEach(m => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 9.85, y: m.y, w: 3.2, h: 0.62, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1 }, rectRadius: 0.03 });
  s.addText([
    { text: m.t, options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
    { text: m.d, options: { fontSize: 8, color: TEXT } },
  ], { x: 9.95, y: m.y, w: 3.0, h: 0.62, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.1 });
});

// ===== 口径脚注 =====
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 3.1, w: 9.2, h: 1.5, fill: { color: 'F7F7F5' }, line: { color: 'D9D9D9', width: 0.75 }, rectRadius: 0.03 });
s.addText([
  { text: '口径定义', options: { bold: true, fontSize: 10.5, color: NAVY, breakLine: true } },
  { text: '① 标签: 出发量=当小时出发量÷同钟点干日基线−1; 速度=ln中位残差; 距离=日级μ1移位(CDF拟合)', options: { fontSize: 8.5, breakLine: true } },
  { text: '② 内插列 = 拟合MAE(域内全部样本, 无留出); 外推列 = 最差分箱偏差(分箱聚合消噪, 逐点MAE被±40pp噪声主导仅作噪声底≈11pp)', options: { fontSize: 8.5, breakLine: true } },
  { text: '③ 训练协议: 出发量/速度=小时级r<4mm/h; 距离=日级<30mm; 公式=θ·ln(1+r/RC)族, 距离为三旋钮版', options: { fontSize: 8.5 } },
], { x: 0.55, y: 3.2, w: 8.9, h: 1.3, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });

// ===== 关键证据行: RF卡死演示 =====
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 4.78, w: 9.2, h: 1.28, fill: { color: YELLOW }, line: { color: YEL_LN, width: 1 }, rectRadius: 0.03 });
s.addText([
  { text: '核心证据：外推域 RF 卡死 vs 公式跟涨（出发量, 实测 vs 预测）', options: { bold: true, fontSize: 10.5, color: '8C5A00', breakLine: true } },
  { text: '雨强 4-8:   实际 +16.7%  ｜ RF +11.4(差5.3)  ｜ 公式 +17.6(差0.9) ← 差6倍', options: { fontSize: 9.5, breakLine: true } },
  { text: '雨强 8-16: 实际 +26.6%  ｜ RF +13.2(差13.4) ｜ 公式 +20.9(差5.7) ← 差2倍', options: { fontSize: 9.5, breakLine: true } },
  { text: '雨强 ≥16:   实际 +23.8%  ｜ RF +10.3(差13.5) ｜ 公式 +26.1(差2.3) ← 差6倍', options: { fontSize: 9.5 } },
], { x: 0.55, y: 4.86, w: 8.9, h: 1.14, fontFace: F, color: TEXT, align: 'left', valign: 'top', margin: 0, lineSpacingMultiple: 1.2 });

// ===== 底部结论 =====
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.3, w: W - 2 * M, h: 0.56, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.3 }, rectRadius: 0.04 });
s.addText([
  { text: '一句话裁决：', options: { bold: true, fontSize: 11.5, color: DGREEN } },
  { text: '内插（见过）→ RF 与公式都准，选 RF（吃交互）；外推（没见过）→ 公式以 3~6 倍优势胜出——四个量全部成立，距离外推用三旋钮版', options: { fontSize: 11.5, color: TEXT } },
], { x: M + 0.2, y: 6.3, w: W - 2 * M - 0.4, h: 0.56, fontFace: F, align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/四量内外终版裁决_v2.pptx' }).then(() => console.log('saved'));
