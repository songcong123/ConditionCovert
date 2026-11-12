const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const BLUE_BG = 'D6E4F0', PURPLE_BG = 'E1D5E7', PURPLE_LN = '9673A6',
  ORANGE_BG = 'FCE4D6', ORANGE_LN = 'ED7D31', GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47',
  OUT_BG = 'D5E8D4', OUT_LN = '82B366', GREY = '888888';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('条件转化框架：特征向量 → 可控(RF) / 不可控(function) → 四量系数 → 分布',
  { x: M, y: 0.18, w: W - 2 * M, h: 0.4, fontSize: 17, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });

const CY = 3.3;                       // 流程纵向中心
const c1x = 0.4, c1w = 1.75;          // 特征向量
const diaX = 2.6, diaW = 1.3, diaH = 0.9, diaY = CY - diaH / 2;   // 判断菱形
const c3x = 4.4, c3w = 2.7;           // RF / function
const c4x = 7.6, c4w = 2.5;           // 四量应用
const c5x = 10.6, c5w = 2.33;         // 采样输出
const diaCX = diaX + diaW / 2;

const ln = (x, y, w, h, opt) => s.addShape(p.shapes.LINE, { x, y, w, h, flipV: !!opt.flipV, line: { color: opt.color, width: opt.width || 2, endArrowType: opt.arrow ? 'triangle' : 'none', dashType: opt.dash ? 'dash' : 'solid' } });

// ===== 列1 特征向量 =====
const f1y = CY - 1.02;
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c1x, y: f1y, w: c1w, h: 0.35, fill: { color: NAVY }, rectRadius: 0.04 });
s.addText('特征向量 c', { x: c1x, y: f1y, w: c1w, h: 0.35, fontSize: 11.5, fontFace: F, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
const feats = ['① 雨强 r（如暴雨 30 mm/h）', '② 时段（如早高峰 8 时）', '③ 日型（如工作日周三）', '④ 事件（如无）'];
feats.forEach((t, i) => {
  const y = f1y + 0.39 + i * 0.42;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c1x, y, w: c1w, h: 0.38, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1 }, rectRadius: 0.03 });
  s.addText(t, { x: c1x + 0.06, y, w: c1w - 0.12, h: 0.38, fontSize: 9.5, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0 });
});

// ===== 列2 判断菱形 =====
s.addShape(p.shapes.DIAMOND, { x: diaX, y: diaY, w: diaW, h: diaH, fill: { color: YELLOW }, line: { color: YEL_LN, width: 1.5 } });
s.addText('c 在可控\n范围？', { x: diaX, y: diaY, w: diaW, h: diaH, fontSize: 10, fontFace: F, bold: true, color: TEXT, align: 'center', valign: 'middle', margin: 0 });

// ===== 列3 可控→RF / 不可控→function =====
const rfHy = CY - 2.0;
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c3x, y: rfHy, w: c3w, h: 0.4, fill: { color: BLUE_BG }, line: { color: NAVY, width: 1.5 }, rectRadius: 0.04 });
s.addText('可控 → RF 内插', { x: c3x, y: rfHy, w: c3w, h: 0.4, fontSize: 11.5, fontFace: F, bold: true, color: NAVY, align: 'center', valign: 'middle', margin: 0 });
const rfBy = rfHy + 0.42;
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c3x, y: rfBy, w: c3w, h: 1.28, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1 }, rectRadius: 0.03 });
s.addText([
  { text: '特征：', options: { bold: true, color: NAVY } }, { text: '与 c 对齐（雨强/时段/日型/事件）\n', options: {} },
  { text: '标签：', options: { bold: true, color: NAVY } }, { text: '四类系数（出发时间/空间/距离/速度）\n', options: {} },
  { text: '训练数据：', options: { bold: true, color: NAVY } }, { text: 'NYC 全量（含暴雨 148 mm）\n', options: {} },
  { text: '输出：', options: { bold: true, color: NAVY } }, { text: '四量系数值', options: {} },
], { x: c3x + 0.1, y: rfBy, w: c3w - 0.2, h: 1.28, fontSize: 9.5, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.25 });

const exHy = CY + 0.2;
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c3x, y: exHy, w: c3w, h: 0.4, fill: { color: YELLOW }, line: { color: YEL_LN, width: 1.5 }, rectRadius: 0.04 });
s.addText('不可控 → function 外推', { x: c3x, y: exHy, w: c3w, h: 0.4, fontSize: 11.5, fontFace: F, bold: true, color: YEL_TX, align: 'center', valign: 'middle', margin: 0 });
const exBy = exHy + 0.42;
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c3x, y: exBy, w: c3w, h: 1.13, fill: { color: 'FFFFFF' }, line: { color: YEL_LN, width: 1 }, rectRadius: 0.03 });
s.addText([
  { text: 'function(c)：', options: { bold: true, color: YEL_TX } }, { text: '多维连乘外推\n', options: {} },
  { text: '逻辑：', options: { bold: true, color: YEL_TX } }, { text: '各特征维一个修正因子，连乘得系数\n', options: {} },
  { text: '输出：', options: { bold: true, color: YEL_TX } }, { text: '四量系数值（附置信区间）', options: {} },
], { x: c3x + 0.1, y: exBy, w: c3w - 0.2, h: 1.13, fontSize: 9.5, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.25 });

// ===== 列4 四量应用 =====
const aHy = CY - 1.05;
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c4x, y: aHy, w: c4w, h: 0.4, fill: { color: NAVY }, rectRadius: 0.04 });
s.addText('四量系数 → 应用到分布', { x: c4x, y: aHy, w: c4w, h: 0.4, fontSize: 11.5, fontFace: F, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
const apps = [
  { t: '出发时间：λ(t) × (1+δvol)', fc: BLUE_BG, ec: NAVY },
  { t: '空间分布：λ(s) × γ(s)', fc: PURPLE_BG, ec: PURPLE_LN },
  { t: '距离分布：P(d) × 三旋钮', fc: ORANGE_BG, ec: ORANGE_LN },
  { t: '速度分布：v × exp(-δv)', fc: GREEN_BG, ec: GREEN_LN },
];
apps.forEach((a, i) => {
  const y = aHy + 0.44 + i * 0.42;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c4x, y, w: c4w, h: 0.38, fill: { color: a.fc }, line: { color: a.ec, width: 1 }, rectRadius: 0.03 });
  s.addText(a.t, { x: c4x + 0.07, y, w: c4w - 0.14, h: 0.38, fontSize: 9.5, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0 });
});

// ===== 列5 采样→校验→输出 =====
const sHy = CY - 0.83;
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c5x, y: sHy, w: c5w, h: 0.4, fill: { color: NAVY }, rectRadius: 0.04 });
s.addText('采样 → 校验 → 输出', { x: c5x, y: sHy, w: c5w, h: 0.4, fontSize: 11.5, fontFace: F, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
const sams = [
  { t: '采样：t, s, d, v, 终点', fc: 'FFFFFF', ec: GREY, b: false },
  { t: '校验：三道闸', fc: 'FFFFFF', ec: GREY, b: false },
  { t: '输出：条件轨迹集', fc: OUT_BG, ec: OUT_LN, b: true },
];
sams.forEach((c, i) => {
  const y = sHy + 0.44 + i * 0.42;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c5x, y, w: c5w, h: 0.38, fill: { color: c.fc }, line: { color: c.ec, width: 1 }, rectRadius: 0.03 });
  s.addText(c.t, { x: c5x + 0.07, y, w: c5w - 0.14, h: 0.38, fontSize: 9.5, fontFace: F, bold: c.b, color: TEXT, align: 'left', valign: 'middle', margin: 0 });
});

// ===== 连线 =====
// 特征 → 菱形
ln(c1x + c1w, CY, diaX - c1x - c1w, 0, { color: NAVY, arrow: true });
s.addText('输入 c', { x: c1x + c1w, y: CY - 0.28, w: diaX - c1x - c1w, h: 0.2, fontSize: 8.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });
// 菱形 → RF（上，实线）
ln(diaCX, rfHy + 0.2, 0, diaY - rfHy - 0.2, { color: NAVY });
ln(diaCX, rfHy + 0.2, c3x - diaCX, 0, { color: NAVY, arrow: true });
s.addText('可控', { x: diaCX + 0.06, y: rfHy - 0.06, w: 0.6, h: 0.2, fontSize: 8.5, fontFace: F, bold: true, color: NAVY, margin: 0 });
// 菱形 → function（下，虚线）
ln(diaCX, diaY + diaH, 0, exHy + 0.2 - diaY - diaH, { color: YEL_LN, dash: true });
ln(diaCX, exHy + 0.2, c3x - diaCX, 0, { color: YEL_LN, arrow: true, dash: true });
s.addText('不可控', { x: diaCX + 0.06, y: exHy + 0.26, w: 0.7, h: 0.2, fontSize: 8.5, fontFace: F, bold: true, color: YEL_TX, margin: 0 });
// RF → 四量
ln(c3x + c3w, rfHy + 0.94, c4x - c3x - c3w, 0, { color: NAVY, arrow: true });
// function → 四量（虚线斜上）
ln(c3x + c3w, aHy + 0.44 + 3 * 0.42 + 0.19, c4x - c3x - c3w, exBy + 0.56 - (aHy + 0.44 + 3 * 0.42 + 0.19), { color: YEL_LN, arrow: true, dash: true, flipV: true });
// 四量 → 采样
ln(c4x + c4w, CY, c5x - c4x - c4w, 0, { color: NAVY, arrow: true });

// ===== 底部说明 =====
s.addText('可控：各维度在标定范围内 → RF 内插（精度最优）；不可控：超出范围 → function 多维连乘外推（物理约束保证不爆炸）；殊途同归，输出同一组四量系数',
  { x: M, y: 5.85, w: W - 2 * M, h: 0.3, fontSize: 10.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/条件转化抽象框架.pptx' }).then(() => console.log('saved'));
