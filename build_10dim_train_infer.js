const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const BLUE_BG = 'D6E4F0', PURPLE_BG = 'E1D5E7', PURPLE_TX = '7030A0',
  ORANGE_BG = 'FCE4D6', ORANGE_TX = '843C0C',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', OUT_BG = 'D5E8D4', OUT_LN = '82B366', GREY = '999999', RED = 'C00000';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('10 维的训练与推理：分层标定 × 分维路由（不做 10 维联合大模型）',
  { x: M, y: 0.16, w: W - 2 * M, h: 0.38, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('教训先行：10 维联合回归实测失败（MAE 14.9pp，组合样本撑不起）；"边际公式＋直读交互＋RF 内插"的分层结构赢了（MAE 10.7pp）',
  { x: M, y: 0.56, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

const ln = (x, y, w, h, o = {}) => s.addShape(p.shapes.LINE, {
  x, y, w, h, flipV: !!o.flipV,
  line: { color: o.color || NAVY, width: o.width || 1.8, dashType: o.dash ? 'dash' : 'solid', endArrowType: o.end ? 'triangle' : 'none' }
});
const box = (x, y, w, h, runs, fc, ec, dash, align) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: 1.2, dashType: dash ? 'dash' : 'solid' }, rectRadius: 0.03 });
  if (runs) s.addText(runs, { x: x + 0.08, y, w: w - 0.16, h, fontFace: F, color: TEXT, align: align || 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });
};

// ================= 训练（离线）：三层并行 =================
s.addText('训练（离线）：三层结构、逐维标定——不需要指数级联合覆盖',
  { x: M, y: 0.92, w: 9.5, h: 0.26, fontSize: 12, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });

box(M, 1.25, 1.6, 1.35, [
  { text: '训练数据', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: '行程 2.61 亿行', options: { fontSize: 8.5, breakLine: true } },
  { text: '+ 10 维条件记录', options: { fontSize: 8.5, breakLine: true } },
  { text: '（天气/日历/事件…）', options: { fontSize: 8, color: MUTED } },
], 'FFFFFF', NAVY);
const tri = [
  { y: 1.25, fc: PURPLE_BG, ec: PURPLE_TX, runs: [
    { text: '① 逐维边际标定（每维独立）——样本只需各维边际覆盖', options: { bold: true, fontSize: 9.5, color: PURPLE_TX, breakLine: true } },
    { text: '连续维拟合 θ_i, RC_i（每维 2 系数）｜离散维直读查表（基线/γ/事件乘子）', options: { fontSize: 8.5 } } ] },
  { y: 1.74, fc: ORANGE_BG, ec: ORANGE_TX, runs: [
    { text: '② 交互直读（少量、按需）——只直读，不拟合', options: { bold: true, fontSize: 9.5, color: ORANGE_TX, breakLine: true } },
    { text: '强相关维对才建 γ(i,j) ＝ 实测均值 ÷ 全局预测（无拟合误差放大）', options: { fontSize: 8.5 } } ] },
  { y: 2.23, fc: BLUE_BG, ec: NAVY, runs: [
    { text: '③ RF 联合训练（一个模型）——只在已覆盖组合区内使用', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
    { text: 'X = (x1…x10) → y = 四量系数；内插精度最优，不负责外推', options: { fontSize: 8.5 } } ] },
];
tri.forEach((b) => box(2.3, b.y, 6.9, 0.44, b.runs, b.fc, b.ec));
const triFan = [
  ln(2.0, 1.47, 0.3, 0.455, { flipV: true, end: true }),
  ln(2.0, 1.925, 0.3, 0.035, { end: true }),
  ln(2.0, 1.925, 0.3, 0.525, { end: true }),
];
ln(9.2, 1.925, 0.3, 0, { end: true });
box(9.5, 1.25, 3.43, 1.35, [
  { text: '参数资产库', options: { bold: true, fontSize: 10, color: '375623', breakLine: true } },
  { text: '10 行系数表 ｜ γ 交互表 ｜ RF 模型 ｜ 基线 P0', options: { fontSize: 8.5, breakLine: true } },
  { text: '参数量：全局层 ≈ 20 个系数 + 少量 γ 格', options: { fontSize: 8.5, color: MUTED } },
], GREEN_BG, GREEN_LN);
s.addText('对比：10 维联合模型要撑住指数级组合空间（10^8+ 格），样本永远不够——这就是 14.9pp 失败的根源；逐维标定只需各维边际覆盖',
  { x: M, y: 2.74, w: 12.0, h: 0.2, fontSize: 8.5, fontFace: F, color: MUTED, align: 'left', margin: 0 });

// ================= 推理（在线）：逐维查表 → 组合路由 → 分级置信 =================
s.addText('推理（在线）：逐维查表 → 组合路由 → 分级置信',
  { x: M, y: 3.02, w: 9.5, h: 0.26, fontSize: 12, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });

box(M, 3.35, 1.5, 1.3, [
  { text: '新条件 c', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: '(x1 … x10)', options: { fontSize: 9, breakLine: true } },
  { text: '来源：预报＋日历', options: { fontSize: 8, color: MUTED } },
], 'FFFFFF', NAVY);
ln(1.9, 4.0, 0.15, 0, { end: true });
box(2.05, 3.35, 1.95, 1.3, [
  { text: '逐维查登记表', options: { bold: true, fontSize: 9.5, color: YEL_TX, breakLine: true } },
  { text: '每维判断：见过吗？', options: { fontSize: 8.5, breakLine: true } },
  { text: '没见过 → 该维走公式 f_i', options: { fontSize: 8.5 } },
], YELLOW, YEL_LN);
const routes = [
  { y: 3.35, fc: GREEN_BG, ec: GREEN_LN, dash: false, tc: DGREEN, runs: [
    { text: '组合全见过 → RF 直出（精度最优）', options: { bold: true, fontSize: 9.5, color: DGREEN, breakLine: true } },
    { text: '置信等级：高', options: { fontSize: 9, bold: true, color: DGREEN } } ] },
  { y: 3.85, fc: BLUE_BG, ec: NAVY, dash: false, tc: NAVY, runs: [
    { text: '组合没见过、各维都见过 → RF 与连乘互校', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
    { text: '置信等级：中', options: { fontSize: 9, bold: true, color: NAVY } } ] },
  { y: 4.35, fc: YELLOW, ec: YEL_LN, dash: true, tc: YEL_TX, runs: [
    { text: '任一维没见过 → 分维连乘兜底', options: { bold: true, fontSize: 9.5, color: YEL_TX, breakLine: true } },
    { text: '置信等级：低，区间宽', options: { fontSize: 9, bold: true, color: YEL_TX } } ] },
];
routes.forEach((b) => box(4.15, b.y, 4.35, 0.44, b.runs, b.fc, b.ec, b.dash));
const routeFan = [
  ln(4.0, 3.57, 0.15, 0.5, { flipV: true, end: true }),
  ln(4.0, 4.07, 0.15, 0, { end: true }),
  ln(4.0, 4.07, 0.15, 0.5, { end: true, color: YEL_LN, dash: true }),
  ln(8.5, 3.57, 0.2, 0.43, { end: true }),
  ln(8.5, 4.0, 0.2, 0, { end: true }),
  ln(8.5, 4.0, 0.2, 0.57, { flipV: true, end: true, color: YEL_LN, dash: true }),
];
box(8.7, 3.35, 4.23, 1.3, [
  { text: '输出', options: { bold: true, fontSize: 10, color: '375623', breakLine: true } },
  { text: '四量系数 ＋ 置信等级（高/中/低）', options: { fontSize: 8.5, breakLine: true } },
  { text: '区间宽度随"没见过"维数变宽', options: { fontSize: 8.5, breakLine: true } },
  { text: '→ P(c) = P0 × f(c) → 条件轨迹集', options: { fontSize: 8.5, bold: true } },
], OUT_BG, OUT_LN);

// ================= 下：教训 + 代价 =================
box(M, 4.95, 6.2, 1.15, [
  { text: '实测教训：为什么不做 10 维联合大模型', options: { bold: true, fontSize: 10.5, color: YEL_TX, breakLine: true } },
  { text: '· 10 维联合回归：MAE 14.9 pp——组合样本撑不起，外推反噬', options: { fontSize: 9, breakLine: true } },
  { text: '· 分层（边际公式＋直读交互）：MAE 10.7 pp < 自然波动 13.9 pp ✓', options: { fontSize: 9, bold: true } },
], 'FFFFFF', YEL_LN, true);

box(6.73, 4.95, 6.2, 1.15, [
  { text: '10 维时每一维的训练代价', options: { bold: true, fontSize: 10.5, color: NAVY, breakLine: true } },
  { text: '· 连续维：常态数据几百样本拟合 2 系数（θ_i, RC_i）', options: { fontSize: 9, breakLine: true } },
  { text: '· 离散维：每档直读均值即可，不用拟合', options: { fontSize: 9, breakLine: true } },
  { text: '· RF：全量样本训练一次，只在覆盖区内使用', options: { fontSize: 9 } },
], 'FFFFFF', NAVY);

// ================= 底部条 =================
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.3, w: W - 2 * M, h: 0.42, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('一句话：训练 = 每维一行系数 ＋ 少数交互格 ＋ 一个 RF；推理 = 逐维查表 ＋ 组合路由 ＋ 分级置信——维度数只是循环次数',
  { x: M, y: 6.3, w: W - 2 * M, h: 0.42, fontSize: 10, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/10维训练与推理.pptx' }).then(() => console.log('saved'));
