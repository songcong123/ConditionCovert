const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const BLUE_BG = 'D6E4F0', PURPLE_TX = '7030A0',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', GREY = '999999', GREY_BG = 'F2F2F2', RED = 'C00000';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('10 个维度怎么画？——换思路：不画"组合空间"，画"登记表 + 统一路由"',
  { x: M, y: 0.16, w: W - 2 * M, h: 0.38, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('组合空间是指数的（10 维画不完）；处理机制是线性的（画得完）——维度登记表 N 行、路由规则 1 条、公式模板 1 个，全随维数线性增长',
  { x: M, y: 0.56, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

const ln = (x, y, w, h, o = {}) => s.addShape(p.shapes.LINE, {
  x, y, w, h, flipV: !!o.flipV,
  line: { color: o.color || NAVY, width: o.width || 1.8, dashType: o.dash ? 'dash' : 'solid', endArrowType: o.end ? 'triangle' : 'none' }
});
const box = (x, y, w, h, runs, fc, ec, dash, align) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: 1.2, dashType: dash ? 'dash' : 'solid' }, rectRadius: 0.03 });
  if (runs) s.addText(runs, { x: x + 0.08, y, w: w - 0.16, h, fontFace: F, color: TEXT, align: align || 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });
};

// ================= 思路对比：旧 ×  新 √ =================
box(M, 0.95, 6.25, 0.88, [
  { text: '旧思路（× 画不完）：把组合空间画出来', options: { bold: true, fontSize: 10.5, color: RED, breakLine: true } },
  { text: '矩阵 → 切片 → 轴，每维一个图形对象；10 维 × 10 档 = 10^10 个格子，指数爆炸', options: { fontSize: 9, color: MUTED } },
], 'FFFFFF', GREY, true);
box(6.88, 0.95, 6.05, 0.88, null, GREEN_BG, GREEN_LN);
s.addText('新思路（√ 线性可画）：把处理机制画出来', { x: 6.98, y: 1.01, w: 5.9, h: 0.24, fontSize: 10.5, fontFace: F, bold: true, color: DGREEN, align: 'left', margin: 0 });
const chips = [['维度登记表 · N 行', 6.98, 1.62], ['统一路由 · 1 条规则', 8.78, 1.85], ['公式模板 · 1 个形式', 10.81, 1.85]];
chips.forEach(([t, cx, cwid], i) => {
  box(cx, 1.36, cwid, 0.34, [{ text: t, options: { fontSize: 8.5, bold: true, color: NAVY, align: 'center' } }], 'FFFFFF', GREEN_LN, false, 'center');
  if (i < 2) ln(cx + cwid, 1.53, 0.18, 0, { color: GREEN_LN, end: true, width: 1.5 });
});

// ================= 左：维度登记表（10 行） =================
s.addText('维度登记表：第 10 行和第 2 行长得一样（F~J 为示意候选维）',
  { x: M, y: 1.98, w: 6.5, h: 0.24, fontSize: 11, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
const OK = { text: '已标定', options: { color: DGREEN, bold: true } };
const WAIT = { text: '待标定', options: { color: YEL_TX, bold: true } };
const rowsT = [
  ['维度', '类型', '已覆盖范围', '引擎', '状态'],
  ['A 雨强 r', '连续', '0–148 mm/h', 'RF＋公式', OK],
  ['B 时段', '离散', '7×24 全覆盖', '基线查表', OK],
  ['C 日型', '离散', '3 档全覆盖', '基线切换', OK],
  ['D 事件', '离散', '5 类', '事件查表', OK],
  ['E 温度', '连续', '−10~35 °C', 'RF＋公式', { text: 'θ_E 待标定', options: { color: YEL_TX, bold: true } }],
  ['F 风速', '连续', '0~20 m/s', 'RF＋公式', WAIT],
  ['G 湿度', '连续', '30~100 %', 'RF＋公式', WAIT],
  ['H 空气质量', '离散', '4 档', '查表', WAIT],
  ['I 政策管控', '离散', '限行 / 封控', '查表', WAIT],
  ['J 场馆供给', '连续', '动态', '乘子', WAIT],
].map((r, i) => r.map(c => typeof c === 'string' ? { text: c, options: i === 0 ? { bold: true, fill: { color: GREY_BG } } : {} } : [c].map(x => ({ text: x.text, options: { ...x.options, ...(i === 0 ? { fill: { color: GREY_BG } } : {}) } }))[0]));
s.addTable(rowsT, {
  x: M, y: 2.26, w: 6.4, colW: [1.0, 0.7, 1.7, 1.3, 1.7],
  border: { pt: 0.5, color: 'BFBFBF' }, fontFace: F, fontSize: 8, valign: 'middle',
  margin: [0.02, 0.04, 0.02, 0.04], rowH: 0.185, align: 'left', fill: { color: 'FFFFFF' },
});

// ================= 右：统一路由器 =================
s.addText('统一路由器：规则只有一条，维度只决定循环次数',
  { x: 7.0, y: 1.98, w: 5.93, h: 0.24, fontSize: 11, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
box(7.0, 2.26, 5.93, 0.36, [{ text: '条件向量 c = (x1, x2, …, x10)', options: { fontSize: 9.5, bold: true, color: NAVY, align: 'center' } }], 'FFFFFF', NAVY, false, 'center');
ln(9.97, 2.62, 0, 0.12, { end: true });
box(7.0, 2.74, 5.93, 0.9, [
  { text: '统一路由器（同一条规则，循环 N 次）', options: { bold: true, fontSize: 10, color: YEL_TX, breakLine: true } },
  { text: '逐维查登记表：见过 → RF 内插｜没见过 → 公式模板｜离散 → 查表', options: { fontSize: 9 } },
], YELLOW, YEL_LN);
ln(9.97, 3.64, 0, 0.12, { end: true });
box(7.0, 3.76, 5.93, 0.42, [{ text: '统一公式模板：f_i = 1 + θ_i · ln(1 + x_i / RC_i)　——每个连续维一行系数', options: { fontSize: 9.5, color: NAVY, align: 'center' } }], BLUE_BG, NAVY, false, 'center');
ln(9.97, 4.18, 0, 0.12, { end: true });
box(7.0, 4.3, 5.93, 0.42, [{ text: '连乘 f(c) = f_1 × … × f_10 → P(c) = P0 × f(c) → 条件轨迹集', options: { fontSize: 9.5, bold: true, color: TEXT, align: 'center' } }], 'D5E8D4', '82B366', false, 'center');

// ================= 组合空间：数字带 =================
s.addText('组合空间不画——用数字说：量级 10^8~10^10（随档位数指数增长；比例示意）',
  { x: M, y: 4.88, w: 9.5, h: 0.24, fontSize: 11, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
s.addShape(p.shapes.RECTANGLE, { x: M, y: 5.18, w: 9.02, h: 0.42, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1 } });
s.addText('72% 组合见过 → RF（精度最优）', { x: M, y: 5.18, w: 9.02, h: 0.42, fontSize: 9, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });
s.addShape(p.shapes.RECTANGLE, { x: 9.42, y: 5.18, w: 2.76, h: 0.42, fill: { color: YELLOW }, line: { color: YEL_LN, width: 1 } });
s.addText('22% 公式兜底', { x: 9.42, y: 5.18, w: 2.76, h: 0.42, fontSize: 9, fontFace: F, bold: true, color: YEL_TX, align: 'center', valign: 'middle', margin: 0 });
s.addShape(p.shapes.RECTANGLE, { x: 12.18, y: 5.18, w: 0.75, h: 0.42, fill: { color: GREY_BG }, line: { color: GREY, width: 1 } });
s.addText('6% 待标定（暂不可算）', { x: 10.9, y: 5.64, w: 2.03, h: 0.18, fontSize: 8, fontFace: F, color: MUTED, align: 'left', margin: 0 });
s.addText('黄/灰随标定推进逐步转为绿（RF 区）——能力增长 = 表格多行"待标定→已标定"',
  { x: M, y: 5.64, w: 10.4, h: 0.18, fontSize: 8, fontFace: F, color: MUTED, align: 'left', margin: 0 });

// ================= 底部条 =================
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.1, w: W - 2 * M, h: 0.62, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText([
  { text: '为什么扛得住 10 维：', options: { bold: true, breakLine: true } },
  { text: '① 登记表线性增长——10 维 10 行、50 维 50 行，版式不变　② 路由只有一条规则——复杂度 O(n)，不是 O(2^n)　③ 公式模板统一——新维 = 复制一行系数 (θ_i, RC_i)', options: {} },
], { x: M + 0.15, y: 6.1, w: W - 2 * M - 0.3, h: 0.62, fontSize: 9.5, fontFace: F, bold: true, color: DGREEN, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });

p.writeFile({ fileName: 'D:/AI/条件转化/10维怎么画_思路转变.pptx' }).then(() => console.log('saved'));
