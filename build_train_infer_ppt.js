const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const BLUE_BG = 'D6E4F0', PURPLE_BG = 'E1D5E7', PURPLE_TX = '7030A0',
  ORANGE_BG = 'FCE4D6', ORANGE_TX = '843C0C',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', OUT_BG = 'D5E8D4', OUT_LN = '82B366', GREY = '999999';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('条件转化框架：训练建模 × 推理路由',
  { x: M, y: 0.16, w: W - 2 * M, h: 0.38, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('训练：定义特征维度与标签，同时训练"内插 RF ＋ 外推公式"两个引擎；推理：逐维判断训练是否见过，见过走 RF、没见过走公式',
  { x: M, y: 0.56, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

const ln = (x, y, w, h, o = {}) => s.addShape(p.shapes.LINE, {
  x, y, w, h, flipV: !!o.flipV,
  line: { color: o.color || NAVY, width: o.width || 1.8, dashType: o.dash ? 'dash' : 'solid', endArrowType: o.end ? 'triangle' : 'none', beginArrowType: o.begin ? 'triangle' : 'none' }
});
const tag = (t, x, y, w, c) => s.addText(t, { x, y, w, h: 0.18, fontSize: 8, fontFace: F, bold: true, color: c, align: 'center', valign: 'middle', margin: 0, fill: { color: 'FFFFFF' } });
const box = (x, y, w, h, runs, fc, ec, dash) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: 1.2, dashType: dash ? 'dash' : 'solid' }, rectRadius: 0.03 });
  s.addText(runs, { x: x + 0.1, y, w: w - 0.2, h, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });
};

// ================= 行1：训练（离线） y 0.92..2.16 =================
const R1 = 0.92, H1 = 1.14, R1C = R1 + H1 / 2;
s.addText('训练（离线）', { x: M, y: R1, w: 1.1, h: H1, fontSize: 11, fontFace: F, bold: true, color: NAVY, align: 'left', valign: 'middle', margin: 0 });

box(1.55, R1, 1.6, H1, [
  { text: '训练数据', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: 'NYC TLC 2.61 亿行', options: { fontSize: 8.5, breakLine: true } },
  { text: '＋ ISD 天气 ＋ 日历', options: { fontSize: 8.5 } },
], 'FFFFFF', NAVY);
ln(3.15, R1C, 0.15, 0, { end: true });

box(3.3, R1, 2.45, H1, [
  { text: '特征 X ＝（A, B, C, D）', options: { bold: true, fontSize: 9.5, color: PURPLE_TX, breakLine: true } },
  { text: 'A 雨强 ｜ B 时段', options: { fontSize: 8.5, breakLine: true } },
  { text: 'C 日型 ｜ D 天气/事件', options: { fontSize: 8.5 } },
], PURPLE_BG, PURPLE_TX);
ln(5.75, R1C, 0.15, 0, { end: true });

box(5.9, R1, 2.45, H1, [
  { text: '标签 y ＝ 四量系数', options: { bold: true, fontSize: 9.5, color: ORANGE_TX, breakLine: true } },
  { text: '当天实际 ÷ 同型基线', options: { fontSize: 8.5, breakLine: true } },
  { text: '样本：每个 日期×时段', options: { fontSize: 8.5 } },
], ORANGE_BG, ORANGE_TX);
ln(8.35, R1C, 0.15, 0, { end: true });

box(8.5, R1, 2.9, H1, [
  { text: '双引擎训练', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: 'RF：c → y 回归（内插器）', options: { fontSize: 8.5, breakLine: true } },
  { text: '公式 θ·ln(1+r/RC)（外推器）', options: { fontSize: 8.5 } },
], 'FFFFFF', NAVY);
ln(9.95, 2.06, 0, 0.26, { color: GREEN_LN, end: true });
tag('产出', 10.01, 2.09, 0.5, GREEN_LN);

// ================= 资产库横条 y 2.32..2.62 =================
const BARY = 2.32;
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: BARY, w: W - 2 * M, h: 0.30, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('参数资产库（训练产出 → 推理加载）：RF 模型 ｜ 外推公式 θ, RC ｜ γ 查表 ｜ 基线 P0',
  { x: M, y: BARY, w: W - 2 * M, h: 0.30, fontSize: 10, fontFace: F, bold: true, color: '375623', align: 'center', valign: 'middle', margin: 0 });
ln(4.4, 2.62, 0, 0.24, { color: GREEN_LN, end: true });
tag('加载', 4.46, 2.65, 0.5, GREEN_LN);

// ================= 行2：推理（在线） y 2.72..3.96 =================
const R2 = 2.86, H2 = 1.24, R2C = R2 + H2 / 2;
s.addText('推理（在线）', { x: M, y: R2, w: 1.1, h: H2, fontSize: 11, fontFace: F, bold: true, color: NAVY, align: 'left', valign: 'middle', margin: 0 });

box(1.55, R2, 1.6, H2, [
  { text: '新条件 c', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: '（A, B, C, D）', options: { fontSize: 9, breakLine: true } },
  { text: '如：早高峰 下雨', options: { fontSize: 8, color: MUTED } },
], 'FFFFFF', NAVY);
ln(3.15, R2C, 0.15, 0, { end: true });

box(3.3, R2, 2.2, H2, [
  { text: '路由判断', options: { bold: true, fontSize: 9.5, color: YEL_TX, breakLine: true } },
  { text: '逐维检查每个维度：', options: { fontSize: 8.5, breakLine: true } },
  { text: '该组合训练见过吗？', options: { fontSize: 8.5 } },
], YELLOW, YEL_LN);

// 分支：路由判断 → 上下两框（x 5.65..8.35）
ln(5.5, 3.14, 0.15, 0.34, { flipV: true, end: true });
ln(5.5, 3.48, 0.15, 0.34, { color: YEL_LN, dash: true, end: true });

box(5.65, 2.86, 2.7, 0.56, [
  { text: '见过 → RF 内插', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: '组合在训练范围内，精度最优', options: { fontSize: 8 } },
], BLUE_BG, NAVY);
box(5.65, 3.54, 2.7, 0.56, [
  { text: '没见过 → 公式外推', options: { bold: true, fontSize: 9.5, color: YEL_TX, breakLine: true } },
  { text: '该维超出训练范围，饱和约束', options: { fontSize: 8 } },
], YELLOW, YEL_LN, true);

// 汇合 → 系数合成
ln(8.35, 3.14, 0.15, 0.22, { end: true });
ln(8.35, 3.6, 0.15, 0.22, { color: YEL_LN, dash: true, flipV: true, end: true });

box(8.5, R2, 2.3, H2, [
  { text: '系数合成', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: 'P(c) = P', options: { bold: true, fontSize: 11, color: NAVY } },
  { text: '0', options: { bold: true, fontSize: 11, color: NAVY, subscript: true } },
  { text: ' × f(c)', options: { bold: true, fontSize: 11, color: NAVY, breakLine: true } },
  { text: '只改参数，不改分布形状', options: { fontSize: 8.5, color: MUTED } },
], 'FFFFFF', NAVY);
ln(10.8, R2C, 0.15, 0, { end: true });

box(10.95, R2, 1.98, H2, [
  { text: '输出', options: { bold: true, fontSize: 9.5, color: '375623', breakLine: true } },
  { text: '采样 → 三道闸校验', options: { fontSize: 8.5, breakLine: true } },
  { text: '条件轨迹集', options: { fontSize: 8.5, bold: true } },
], OUT_BG, OUT_LN);

// ================= 下半页：示例 =================
s.addText('示例：同一路由逻辑的两种情况（D 维 = 天气类型 · 假设示例）',
  { x: M, y: 4.18, w: 8.5, h: 0.26, fontSize: 12, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });

box(M, 4.5, 6.0, 1.9, [
  { text: '情形一  中雨 —— 训练见过 → RF 内插', options: { bold: true, fontSize: 10.5, color: NAVY, breakLine: true } },
  { text: 'c =（A 雨强=10 mm/h，B 时段=早高峰，C 日型=工作日，D 天气=中雨）', options: { fontSize: 9, breakLine: true } },
  { text: '逐维判断：A、B、C、D 全部在训练数据范围内', options: { fontSize: 9, breakLine: true } },
  { text: '路由：RF 直接内插这个见过组合 → 四量系数', options: { fontSize: 9, breakLine: true } },
  { text: '输出：如 出发 +12%、速度 −2.5%（中雨见过，精度最优）', options: { fontSize: 9 } },
], 'FFFFFF', NAVY);

box(6.53, 4.5, 6.4, 1.9, [
  { text: '情形二  沙尘暴 —— 训练没见过 → 公式外推', options: { bold: true, fontSize: 10.5, color: YEL_TX, breakLine: true } },
  { text: 'c =（A 雨强=0，B 时段=早高峰，C 日型=工作日，D 天气=沙尘暴）', options: { fontSize: 9, breakLine: true } },
  { text: '逐维判断：A/B/C 都见过，但 D=沙尘暴 训练集从未出现', options: { fontSize: 9, breakLine: true } },
  { text: '路由：D 维走外推公式，其余维用已知规律，连乘合成', options: { fontSize: 9, breakLine: true } },
  { text: '输出：系数＋置信区间（物理约束：单调、饱和，不爆炸）', options: { fontSize: 9 } },
], 'FFFFFF', YEL_LN, true);

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.54, w: W - 2 * M, h: 0.38, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('路由规则一句话：逐维检查——全部见过 → RF（内插精度最优）；任一维没见过 → 该维走外推公式（连乘组合，附置信区间）',
  { x: M, y: 6.54, w: W - 2 * M, h: 0.38, fontSize: 10, fontFace: F, bold: true, color: '375623', align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/条件转化训练与推理_v2.pptx' }).then(() => console.log('saved'));
