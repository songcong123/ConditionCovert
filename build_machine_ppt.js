const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = 'Z.ai';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const BLUE_BG = 'D6E4F0', PURPLE_TX = '7030A0', PURPLE_BG = 'E1D5E7',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', OR_BG = 'FCE4D6', OR_TX = '843C0C', GREY = '999999';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('Excel 数值生成机器：12 个数怎么算出来的',
  { x: M, y: 0.14, w: W - 2 * M, h: 0.38, fontSize: 17, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('三行 = 三个量的流水线；每行四步：原始数据 → 标签 y → 预测 ŷ → 汇总成表里的数',
  { x: M, y: 0.52, w: W - 2 * M, h: 0.22, fontSize: 10.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

const boxR = (x, y, w, h, fc, ec, lw) => s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: lw || 1.3 }, rectRadius: 0.04 });
const rich = (x, y, w, h, runs, align = 'left') => s.addText(runs, { x: x + 0.1, y, w: w - 0.2, h, fontFace: F, color: TEXT, align, valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });
const arr = (x, y, w, h, o = {}) => s.addShape(p.shapes.LINE, { x, y, w, h, flipV: !!o.flipV, line: { color: o.color || GREY, width: o.width || 2, endArrowType: 'triangle' } });

// 列标题带
const colY = 0.86;
const colHeads = [
  { x: 0.4, w: 2.35, t: '① 原始数据' },
  { x: 3.15, w: 3.3, t: '② 标签 y（实际值）' },
  { x: 6.85, w: 3.3, t: '③ 预测 ŷ（两个引擎）' },
  { x: 10.55, w: 2.4, t: '④ 汇总 → Excel' },
];
colHeads.forEach(c => {
  s.addShape(p.shapes.RECTANGLE, { x: c.x, y: colY, w: c.w, h: 0.3, fill: { color: NAVY }, line: { color: 'FFFFFF', width: 0.75 } });
  s.addText(c.t, { x: c.x, y: colY, w: c.w, h: 0.3, fontSize: 11, fontFace: F, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
});

// ============ 行1 出发量 ============
const y1 = 1.28, h1 = 1.22;
// 数据
boxR(0.4, y1, 2.35, h1, 'FFFFFF', NAVY);
rich(0.4, y1, 2.35, h1, [
  { text: '出发量计数', options: { bold: true, fontSize: 11, color: NAVY, breakLine: true } },
  { text: '每小时出发趟数\n（NYC 2.61亿行聚合）', options: { fontSize: 9, breakLine: true } },
  { text: '基线 = 同星期×钟点\n干日平均出发量', options: { fontSize: 9, color: MUTED } },
]);
arr(2.75, y1 + h1 / 2, 0.4, 0);
// 标签
boxR(3.15, y1, 3.3, h1, YELLOW = 'FFF2CC', 'D6B656');
rich(3.15, y1, 3.3, h1, [
  { text: 'y = 出发量 ÷ 基线 − 1', options: { bold: true, fontSize: 11, color: '8C5A00', breakLine: true } },
  { text: '例：11300÷10000−1 = +0.13', options: { fontSize: 9, breakLine: true } },
  { text: '（该小时效应 +13%）', options: { fontSize: 9, color: MUTED } },
]);
arr(6.45, y1 + h1 / 2, 0.4, 0);
// 预测
boxR(6.85, y1, 3.3, h1, BLUE_BG, NAVY);
rich(6.85, y1, 3.3, h1, [
  { text: 'RF：17特征 → 效应值', options: { bold: true, fontSize: 10, color: NAVY, breakLine: true } },
  { text: '公式：0.052·ln(1+r/0.2)', options: { bold: true, fontSize: 10, color: NAVY, breakLine: true } },
  { text: '例 r=2: RF +11% / 公式 +12.4%', options: { fontSize: 9, color: MUTED } },
]);
arr(10.15, y1 + h1 / 2, 0.4, 0);
// 汇总
boxR(10.55, y1, 2.4, h1, GREEN_BG, GREEN_LN);
rich(10.55, y1, 2.4, h1, [
  { text: '内插：|y−ŷ| 平均', options: { fontSize: 9.5, breakLine: true } },
  { text: '→ 5.7 / 6.3 pp', options: { bold: true, fontSize: 11, color: DGREEN, breakLine: true } },
  { text: '外推：箱内均值之差', options: { fontSize: 9.5, breakLine: true } },
  { text: '→ 13.5 / 2.3 pp', options: { bold: true, fontSize: 11, color: DGREEN } },
]);

// ============ 行2 速度 ============
const y2 = 2.62, h2 = 1.18;
boxR(0.4, y2, 2.35, h2, 'FFFFFF', NAVY);
rich(0.4, y2, 2.35, h2, [
  { text: '速度行程', options: { bold: true, fontSize: 11, color: NAVY, breakLine: true } },
  { text: '每条行程的速度\n（627万条样本）', options: { fontSize: 9, breakLine: true } },
  { text: '基线 = 同钟点干日\n中位 log 速度', options: { fontSize: 9, color: MUTED } },
]);
arr(2.75, y2 + h2 / 2, 0.4, 0);
boxR(3.15, y2, 3.3, h2, 'FFF2CC', 'D6B656');
rich(3.15, y2, 3.3, h2, [
  { text: 'y = ln(实际中位速) − ln(基线中位速)', options: { bold: true, fontSize: 10, color: '8C5A00', breakLine: true } },
  { text: '例：ln(9.5)−ln(11.2) = −0.165', options: { fontSize: 9, breakLine: true } },
  { text: '（减速 ≈15%）', options: { fontSize: 9, color: MUTED } },
]);
arr(6.45, y2 + h2 / 2, 0.4, 0);
boxR(6.85, y2, 3.3, h2, BLUE_BG, NAVY);
rich(6.85, y2, 3.3, h2, [
  { text: 'RF：特征 → log 减速', options: { bold: true, fontSize: 10, color: NAVY, breakLine: true } },
  { text: '公式：−0.0221·ln(1+r/0.2)', options: { bold: true, fontSize: 10, color: NAVY, breakLine: true } },
  { text: '（log 空间平移量）', options: { fontSize: 9, color: MUTED } },
]);
arr(10.15, y2 + h2 / 2, 0.4, 0);
boxR(10.55, y2, 2.4, h2, GREEN_BG, GREEN_LN);
rich(10.55, y2, 2.4, h2, [
  { text: '内插：逐点平均', options: { fontSize: 9.5, breakLine: true } },
  { text: '→ 2.9 / 3.7 pp', options: { bold: true, fontSize: 11, color: DGREEN, breakLine: true } },
  { text: '外推(≥16箱均值差)', options: { fontSize: 9.5, breakLine: true } },
  { text: '→ 3.0 / 2.0 pp', options: { bold: true, fontSize: 11, color: DGREEN } },
]);

// ============ 行3 距离 ============
const y3 = 3.92, h3 = 1.22;
boxR(0.4, y3, 2.35, h3, 'FFFFFF', NAVY);
rich(0.4, y3, 2.35, h3, [
  { text: '距离行程', options: { bold: true, fontSize: 11, color: NAVY, breakLine: true } },
  { text: '每条行程的里程\n（日级聚合, 一天一样本）', options: { fontSize: 9, breakLine: true } },
  { text: '基线 = 晴天 LN 三组分\n+帕累托尾混合模型', options: { fontSize: 9, color: MUTED } },
]);
arr(2.75, y3 + h3 / 2, 0.4, 0);
boxR(3.15, y3, 3.3, h3, 'FFF2CC', 'D6B656');
rich(3.15, y3, 3.3, h3, [
  { text: 'y = 当日 μ1 移位（CDF 匹配）', options: { bold: true, fontSize: 10.5, color: '8C5A00', breakLine: true } },
  { text: '只许基线整体平移, 移到与当日分布\n最贴合 → 平移量即标签', options: { fontSize: 9, breakLine: true } },
  { text: '例：日降水12mm → μ1=−0.031(短3%)', options: { fontSize: 9, color: MUTED } },
]);
arr(6.45, y3 + h3 / 2, 0.4, 0);
boxR(6.85, y3, 3.3, h3, BLUE_BG, NAVY);
rich(6.85, y3, 3.3, h3, [
  { text: 'RF：日降水+日型 → μ1', options: { bold: true, fontSize: 10, color: NAVY, breakLine: true } },
  { text: '公式：三旋钮剂量曲线', options: { bold: true, fontSize: 10, color: PURPLE_TX, breakLine: true } },
  { text: 'μ1: −0.017·ln(1+R/5.4) + w1/机场两条', options: { fontSize: 9, color: MUTED } },
]);
arr(10.15, y3 + h3 / 2, 0.4, 0);
boxR(10.55, y3, 2.4, h3, GREEN_BG, GREEN_LN);
rich(10.55, y3, 2.4, h3, [
  { text: '内插：逐日平均', options: { fontSize: 9.5, breakLine: true } },
  { text: '→ 2.0 / 2.5 pp', options: { bold: true, fontSize: 11, color: DGREEN, breakLine: true } },
  { text: '外推(P25~P75分位差)', options: { fontSize: 9.5, breakLine: true } },
  { text: '→ 2.8 / 1.0 pp', options: { bold: true, fontSize: 11, color: DGREEN } },
]);

// ===== 汇总规则带 =====
const y4 = 5.3;
boxR(0.4, y4, 12.53, 0.56, 'F7F7F5', 'D9D9D9', 1);
s.addText([
  { text: '汇总规则（两选一，取决于域）：', options: { bold: true, fontSize: 10.5, color: NAVY } },
  { text: '内插列 = 域内全部样本 |y−ŷ| 的平均（拟合 MAE）　　　外推列 = 箱内（实测均值 − 预测均值）之差', options: { fontSize: 10.5, color: TEXT } },
], { x: 0.55, y: y4, w: 12.2, h: 0.56, fontFace: F, align: 'left', valign: 'middle', margin: 0 });

// ===== pp 说明 =====
boxR(0.4, 5.98, 12.53, 0.5, YELLOW, 'D6B656', 1);
s.addText([
  { text: 'pp = 百分点（单位，不是误差本身）：', options: { bold: true, fontSize: 10.5, color: '8C5A00' } },
  { text: '表里"偏差/MAE"列的 pp = 误差（预测效应与实际效应差几个百分点）；写 +% 的原值列 = 效应本身。例：预测+10.3% vs 实际+23.8% → 误差 13.5pp', options: { fontSize: 10, color: TEXT } },
], { x: 0.55, y: 5.98, w: 12.2, h: 0.5, fontFace: F, align: 'left', valign: 'middle', margin: 0 });

// ===== 底部 =====
boxR(0.4, 6.6, 12.53, 0.46, GREEN_BG, GREEN_LN, 1.2);
s.addText('读法：任何一格 Excel 数字 = 顺着它所在行从左到右走一遍——数据 ÷ 基线得标签，引擎出预测，按域选汇总规则',
  { x: 0.55, y: 6.6, w: 12.2, h: 0.46, fontSize: 10.5, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/数值生成机器.pptx' }).then(() => console.log('saved'));
