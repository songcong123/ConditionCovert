const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const BLUE_BG = 'D6E4F0', PURPLE_BG = 'E1D5E7', PURPLE_TX = '7030A0',
  ORANGE_BG = 'FCE4D6', ORANGE_TX = '843C0C',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', GREY = '999999', GREY_BG = 'F2F2F2', RED = 'C00000';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('训练建模：目标是"条件 → 偏移系数"，生效靠"乘回基线"',
  { x: M, y: 0.16, w: W - 2 * M, h: 0.38, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('标签逐量显性化（黄色列）｜每个量的外推公式单独注明（右列）',
  { x: M, y: 0.56, w: W - 2 * M, h: 0.2, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

// ===== 建模目标横幅（最重要，一眼看到） =====
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 0.88, w: W - 2 * M, h: 0.44, fill: { color: NAVY }, rectRadius: 0.04 });
s.addText([
  { text: '建模目标：', options: { bold: true, fontSize: 12.5, color: 'FFFFFF' } },
  { text: '只学"条件向量 c → 四个偏移系数"的映射 f(c)，不直接学分布——分布永远 = 晴天基线 × 系数，只改参数、不改形状', options: { fontSize: 12.5, color: 'FFFFFF' } },
], { x: M + 0.2, y: 0.88, w: W - 2 * M - 0.4, h: 0.44, fontFace: F, align: 'center', valign: 'middle', margin: 0 });

// ===== 四条泳道 =====
const colX = [0.4, 2.45, 5.7, 8.45], colW = [1.9, 3.1, 2.6, 4.48];
const heads = ['量（晴天基线）', '训练标签 y（显性化）', '怎么拟合', '外推公式（推理时生效）'];
heads.forEach((t, j) => {
  s.addShape(p.shapes.RECTANGLE, { x: colX[j], y: 1.44, w: colW[j], h: 0.26, fill: { color: j === 1 ? YELLOW : GREY_BG }, line: { color: j === 1 ? YEL_LN : GREY, width: 0.75 } });
  s.addText(t, { x: colX[j], y: 1.44, w: colW[j], h: 0.26, fontSize: 9.5, fontFace: F, bold: true, color: j === 1 ? YEL_TX : NAVY, align: 'center', valign: 'middle', margin: 0 });
});

const lanes = [
  {
    name: '① 出发时间', fc: BLUE_BG, ec: NAVY, tc: NAVY,
    base: [['λ(t)：7×24 查表', true], ['（168 值，晴天逐时直读）', false]],
    label: [['y(t) ＝ λ实测(t) ÷ λ基线(t)', true], ['每个雨天×小时的逐时系数', false]],
    fit: [['公式：365 轻中雨样本→2 系数', false], ['RF：内插验证 ｜ γ 表：直读', false]],
    ext: [['✔ 雨天出发变多（乘法放大）', true], ['δvol(r) ＝ 0.052·ln(1+r/0.2)', false], ['例：中雨 10mm/h → 出发 +20%', false], ['超出观测雨强 → 沿曲线延伸', false]],
  },
  {
    name: '② 空间分布', fc: PURPLE_BG, ec: PURPLE_TX, tc: PURPLE_TX,
    base: [['份额(s)：263 区查表', true], ['（晴天直读）', false]],
    label: [['y(s) ＝ 份额实测(s) ÷ 份额基线(s)', true], ['分两个粒度：整天 / 暴雨小时', false]],
    fit: [['日级：直读 ≈1（升消相抵）', false], ['暴雨小时：机场直读 −27% ＋ 梯度 OLS', false], ['事件：锚点查表｜假日：换基线表', false]],
    ext: [['✔ 分层：日级≈1 ｜ 雨时用公式', true], ['γ(s,r) = 1 + Δ(s)·ln(1+r/2)/ln4.55', false], ['机场 −27%、核心 +6%、越远越降', false], ['事件查表；极端 → 崩溃档', false]],
  },
  {
    name: '③ 距离分布', fc: ORANGE_BG, ec: ORANGE_TX, tc: ORANGE_TX,
    base: [['LN 三组分 μ、σ、w', true], ['（晴天拟合，σ 锁死）', false]],
    label: [['y_d ＝ (μ1移位, w1增益, 机场比)', true], ['三旋钮：短途更近、短途更多、机场更少', false]],
    fit: [['分 5 档雨强各直读一组旋钮', false], ['每旋钮拟合饱和曲线（留出验证 0.5%/3.5%）', false]],
    ext: [['✔ 曲线外推', true], ['μ1: −0.017·ln(1+r/5.4)', false], ['w1: +0.022·ln(1+r/38.3)', false], ['w_ap: ×[1−0.073·ln(1+r/5.6)]', false]],
  },
  {
    name: '④ 速度分布', fc: GREEN_BG, ec: GREEN_LN, tc: DGREEN,
    base: [['μ(t)：24 中位数＋σ=0.44', true], ['（σ 跨雨强恒定，锁死）', false]],
    label: [['y_v(t) ＝ μ实测(t) ÷ μ基线(t)', true], ['雨天的逐时减速系数', false]],
    fit: [['公式：同 365 样本→2 系数', false], ['周末反向（雨反而快）：日型表直读', false]],
    ext: [['✔ 雨越大开得越慢（乘法折扣）', true], ['中雨 10mm/h → 慢 8%；30mm/h → 慢 12%', false], ['公式 ×exp(−δv)，δv = 0.025·ln(1+r/0.3)', false], ['周末反转：雨天反而快 3%（日型表）', false]],
  },
];
lanes.forEach((L, i) => {
  const y = 1.76 + i * 0.86, h = 0.78;
  const first = [[L.name, true], ...L.base];
  const cells = [first, L.label, L.fit, L.ext];
  const fills = [L.fc, YELLOW, 'FFFFFF', i === 1 ? GREY_BG : L.fc];
  const ecs = [L.ec, YEL_LN, GREY, L.ec];
  cells.forEach((runs, j) => {
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: colX[j], y, w: colW[j], h, fill: { color: fills[j] }, line: { color: ecs[j], width: 1.1 }, rectRadius: 0.03 });
    const rich = runs.map(([t, b], k) => ({ text: t, options: { fontSize: 8.5, bold: b, color: j === 3 && k === 0 ? (i === 1 ? MUTED : DGREEN) : (j === 0 && k === 0 ? L.tc : TEXT), breakLine: k < runs.length - 1 } }));
    s.addText(rich, { x: colX[j] + 0.09, y, w: colW[j] - 0.18, h, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });
  });
});

// ===== 如何生效 =====
s.addText('如何生效：训练好的系数，推理时"乘回基线"——四步',
  { x: M, y: 5.28, w: 9.5, h: 0.24, fontSize: 12, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
const ln = (x, y, w, h, o = {}) => s.addShape(p.shapes.LINE, { x, y, w, h, line: { color: o.color || NAVY, width: 1.8, endArrowType: 'triangle' } });
const chain = [
  { x: 0.4, w: 2.55, t: '① 输入条件 c', d: '雨强·时段·日型·事件\n（预报＋日历即可得）', fc: 'FFFFFF', ec: NAVY },
  { x: 3.15, w: 3.1, t: '② 查系数（路由）', d: '见过→RF ｜ 没见过→外推公式\n离散维→查表', fc: YELLOW, ec: YEL_LN },
  { x: 6.45, w: 3.6, t: '③ 乘回基线（生效）', d: 'λ(t)×(1+δvol) ｜ 份额×γ(s)\n距离三旋钮 ｜ v×exp(−δv)', fc: BLUE_BG, ec: NAVY },
  { x: 10.25, w: 2.68, t: '④ 输出', d: 'P(c) → 采样 → 三道闸\n→ 条件轨迹集', fc: 'D5E8D4', ec: '82B366' },
];
chain.forEach((b, i) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: b.x, y: 5.58, w: b.w, h: 0.66, fill: { color: b.fc }, line: { color: b.ec, width: 1.2 }, rectRadius: 0.03 });
  s.addText([
    { text: b.t, options: { bold: true, fontSize: 9.5, color: i === 2 ? NAVY : TEXT, breakLine: true } },
    { text: b.d, options: { fontSize: 8, color: MUTED } },
  ], { x: b.x + 0.09, y: 5.58, w: b.w - 0.18, h: 0.66, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });
  if (i < 3) ln(b.x + b.w, 5.91, 0.2, 0);
});

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.5, w: W - 2 * M, h: 0.42, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('目标一句话：把天气/事件的影响压缩成 4 个可乘系数；生效 = 乘回晴天基线——参数变了、形状没变，分布永远合理',
  { x: M, y: 6.5, w: W - 2 * M, h: 0.42, fontSize: 10, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/训练建模目标与生效_v2.pptx' }).then(() => console.log('saved'));
