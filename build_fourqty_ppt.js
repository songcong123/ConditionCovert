const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const BLUE_BG = 'D6E4F0', PURPLE_BG = 'E1D5E7', PURPLE_TX = '7030A0',
  ORANGE_BG = 'FCE4D6', ORANGE_TX = '843C0C',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', GREY = '999999', GREY_BG = 'F2F2F2';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('模型训练 × 公式生成：出发时间 / 空间 / 距离 / 速度 逐量拆解',
  { x: M, y: 0.16, w: W - 2 * M, h: 0.38, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('每个量 = 晴天基线（直读/拟合）＋ 条件算子（公式或查表）；系数全部实测标定，形状参数锁死、只调位置',
  { x: M, y: 0.56, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

const cols = [
  { x: 0.4, w: 1.15, t: '量' },
  { x: 1.55, w: 2.75, t: '基线 P0（晴天怎么来）' },
  { x: 4.3, w: 3.55, t: '条件算子 f(c)（只调位置不改形状）' },
  { x: 7.85, w: 2.95, t: '系数怎么标定（数据＋方法）' },
  { x: 10.8, w: 2.13, t: '验证成绩' },
];
cols.forEach(c => {
  s.addShape(p.shapes.RECTANGLE, { x: c.x, y: 1.02, w: c.w, h: 0.3, fill: { color: GREY_BG }, line: { color: GREY, width: 0.75 } });
  s.addText(c.t, { x: c.x, y: 1.02, w: c.w, h: 0.3, fontSize: 9.5, fontFace: F, bold: true, color: NAVY, align: 'center', valign: 'middle', margin: 0 });
});

const box = (x, y, w, h, runs, fc, ec) => {
  s.addShape(p.shapes.RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: 1 } });
  if (runs) s.addText(runs, { x: x + 0.07, y, w: w - 0.14, h, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });
};

const rowsQ = [
  {
    name: '① 出发时间', fc: BLUE_BG, ec: NAVY, tc: NAVY,
    base: [['λ(t) 7×24 查表', true], ['168 个值，晴天逐时直读，零参数', false]],
    op: [['×(1+δvol(r))', true], ['δvol = 0.052·ln(1+r/0.2)', false], ['逐时再乘 γ(时段,日型) 直读修正', false]],
    cal: [['365 个轻中雨小时样本', false], ['散点拟合 2 系数（θ 剂量, RC 参考）', false], ['γ 表 10 格：实测均值 ÷ 公式预测', false]],
    val: [['中雨盲测优于基线 76%', false], ['分层外推 MAE 10.7pp', false]],
  },
  {
    name: '② 空间分布', fc: PURPLE_BG, ec: PURPLE_TX, tc: PURPLE_TX,
    base: [['263 区出发份额表', true], ['晴天直读', false]],
    op: [['雨天 γ(s)≈1：JS=0.0002 基本不动', false], ['事件查表：场馆×3.5、马拉松沿线×0.5、跨年封街×0.004', false], ['假日：直接换假日基线表（雨不叠加）', false]],
    cal: [['雨天 JS 散度实测（不动=不调）', false], ['事件日锚点直读——不用拟合', false]],
    val: [['雨不改空间结构（JS≈0）', false], ['假日重排由基线切换吸收', false]],
  },
  {
    name: '③ 距离', fc: ORANGE_BG, ec: ORANGE_TX, tc: ORANGE_TX,
    base: [['LN 三组分混合（短/中/长途）', true], ['晴天拟合 μ、σ、w', false]],
    op: [['三旋钮 → 三条剂量曲线', true], ['μ1: −0.017·ln(1+r/5.4)', false], ['w1: +0.022·ln(1+r/38.3)', false], ['w_ap: ×[1−0.073·ln(1+r/5.6)]', false], ['只动位置与权重，σ 形状锁死', false]],
    cal: [['分 5 档雨强各直读一组旋钮', false], ['每旋钮拟合饱和曲线（五步流水线）', false], ['留出验证：极端档 w1 偏差 0.5%、apr 3.5%', false]],
    val: [['合成距离 KS=0.024', false], ['极端档：不调 0.041 →', false], ['曲线外推 0.015（=直读 0.015）', false], ['外推 R² 0.93~0.98', false]],
  },
  {
    name: '④ 速度', fc: GREEN_BG, ec: GREEN_LN, tc: DGREEN,
    base: [['24 小时中位数 μ(t)＋σ=0.44', true], ['σ 跨雨强恒定 0.434~0.436，锁死', false]],
    op: [['×exp(−δv(r))', true], ['δv = 0.025·ln(1+r/0.3)', false], ['周末雨方向反转：日型参数表修正', false]],
    cal: [['同 365 样本散点拟合 2 系数', false], ['周末反向由日型表直读', false]],
    val: [['速度盲测优于基线 15%', false], ['跨年重复拟合全参数重合', false]],
  },
];
rowsQ.forEach((r, ri) => {
  const y = 1.34 + ri * 1.03, h = 0.98;
  s.addShape(p.shapes.RECTANGLE, { x: cols[0].x, y, w: cols[0].w, h, fill: { color: r.fc }, line: { color: r.ec, width: 1 } });
  s.addText(r.name, { x: cols[0].x, y, w: cols[0].w, h, fontSize: 10.5, fontFace: F, bold: true, color: r.tc, align: 'center', valign: 'middle', margin: 0 });
  [r.base, r.op, r.cal, r.val].forEach((cell, ci) => {
    const runs = cell.map(([t, b], k) => ({ text: t, options: { fontSize: 8.5, bold: b, breakLine: k < cell.length - 1 } }));
    box(cols[ci + 1].x, y, cols[ci + 1].w, h, runs, 'FFFFFF', r.ec);
  });
});

// ================= 公式生成五步 =================
s.addText('公式怎么生成：所有连续维共用同一条五步流水线（δvol、δv 都这么来）',
  { x: M, y: 5.6, w: 10.5, h: 0.24, fontSize: 12, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
const steps = [
  ['① 物理约束选形', '单调＋饱和族：对数 / Hill / 指数饱和'],
  ['② 生成散点', '每档雨强的实测系数点：横轴 r，纵轴增量'],
  ['③ 拟合 2 系数', 'θ 剂量强度、RC 参考浓度（最小二乘）'],
  ['④ 鲁棒性互验', '6 种饱和公式差异 <3pp；多项式反例 785pp 弃用'],
  ['⑤ 盲测', '强雨外推偏差 4%——未见值可信'],
];
steps.forEach(([t, d], i) => {
  const x = 0.4 + i * 2.51;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 5.9, w: 2.31, h: 0.62, fill: { color: i === 4 ? GREEN_BG : 'FFFFFF' }, line: { color: i === 4 ? GREEN_LN : NAVY, width: 1.2 }, rectRadius: 0.03 });
  s.addText([
    { text: t, options: { bold: true, fontSize: 9.5, color: i === 4 ? DGREEN : NAVY, breakLine: true } },
    { text: d, options: { fontSize: 8, color: MUTED } },
  ], { x: x + 0.08, y: 5.9, w: 2.15, h: 0.62, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.1 });
  if (i < 4) s.addShape(p.shapes.LINE, { x: x + 2.31, y: 6.21, w: 0.2, h: 0, line: { color: NAVY, width: 1.8, endArrowType: 'triangle' } });
});

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.72, w: W - 2 * M, h: 0.4, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('四个量的共同范式：基线晴天直读/拟合 ＋ 算子只调位置参数（形状锁死）＋ 系数全部实测标定——没有一个人工拍的数',
  { x: M, y: 6.72, w: W - 2 * M, h: 0.4, fontSize: 10, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/四量训练与公式生成_v2.pptx' }).then(() => console.log('saved'));
