const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00',
  BLUE_BG = 'D6E4F0', GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47',
  OR = 'ED7D31', RED = 'C00000', GREY = '999999';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('循环标定：一条流水线逐维建模 → 连乘合成 f(A,B,C) = f(A)·f(B)·f(C)',
  { x: M, y: 0.14, w: W - 2 * M, h: 0.36, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('分别建模 = 队列里每个维各跑一遍同一条标定流水线（循环）；全部跑完 → 各维系数连乘 → 乘回基线生效',
  { x: M, y: 0.52, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

const boxR = (x, y, w, h, fc, ec, lw) => s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: lw || 1.2 }, rectRadius: 0.03 });
const rich = (x, y, w, h, runs, align = 'left', valign = 'middle', spacing = 1.15) => s.addText(runs, { x, y, w, h, fontFace: F, align, valign, margin: 0, lineSpacingMultiple: spacing });
const lbl = (x, y, s2, size, color, bold, w, ha = 'left') => s.addText(s2, { x, y, w: w || 3, h: 0.26, fontFace: F, fontSize: size, color, bold: !!bold, align: ha, valign: 'middle', margin: 0 });
const arrow = (x, y, w, h) => s.addShape(p.shapes.LINE, { x, y, w, h, line: { color: NAVY, width: 2, endArrowType: 'triangle' } });

// ---------- 主行三盒 ----------
const rowY = 1.05, rowH = 1.5;

// B1 维度队列
boxR(M, rowY, 2.35, rowH, 'FFFFFF', NAVY, 1.4);
lbl(M + 0.1, rowY + 0.08, '维度队列', 11, NAVY, true, 2.1);
const queue = [
  ['A 雨强', '✓ 已标定（盲测4%）', GREEN_LN],
  ['B 温度', '✓ 已标定（≈0）', GREEN_LN],
  ['C 时段', '✓ γ 直读', GREEN_LN],
  ['D 事件', '▶ 查表档', OR],
];
queue.forEach((q, k) => {
  lbl(M + 0.12, rowY + 0.42 + k * 0.27, q[0] + '  ', 9.5, TEXT, true, 0.9);
  lbl(M + 0.85, rowY + 0.42 + k * 0.27, q[1], 8.5, q[2], false, 1.5);
});
arrow(2.78, rowY + rowH / 2, 0.25, 0);
lbl(2.55, 0.823, '取第 i 维', 9, MUTED, false, 0.7, 'center');

// B2 流水线
boxR(3.05, rowY, 3.5, rowH, 'FFFFFF', NAVY, 1.4);
lbl(3.17, rowY + 0.08, '统一标定流水线（每维跑一遍）', 11, NAVY, true, 3.3);
rich(3.17, rowY + 0.44, 3.3, 0.95, [
  { text: '① 标签：y(i) = 实测 ÷ 基线（按 x(i) 分箱）', options: { fontSize: 9.5, breakLine: true } },
  { text: '② 拟合：θ(i) · ln(1 + x(i)/RC(i))', options: { fontSize: 9.5, breakLine: true } },
  { text: '③ 系数写入登记表（每维一行）', options: { fontSize: 9.5 } },
], 'left', 'middle');

arrow(6.55, rowY + rowH / 2, 0.28, 0);

// B3 系数登记表
boxR(6.85, rowY, 2.75, rowH, 'FFFFFF', NAVY, 1.4);
lbl(6.97, rowY + 0.08, '系数登记表', 11, NAVY, true, 2.5);
rich(6.97, rowY + 0.44, 2.5, 0.95, [
  { text: '雨: θ=0.052, RC=0.2', options: { fontSize: 9.5, breakLine: true } },
  { text: '温度: 热浪 +12%', options: { fontSize: 9.5, breakLine: true } },
  { text: '时段/事件: 直读查表', options: { fontSize: 9.5 } },
], 'left', 'middle');

// ---------- 循环回环箭头 ----------
const loopY = 0.78;
s.addShape(p.shapes.LINE, { x: 8.22, y: loopY, w: 0, h: rowY - loopY, line: { color: OR, width: 2.2 } });
s.addShape(p.shapes.LINE, { x: 1.55, y: loopY, w: 6.67, h: 0, line: { color: OR, width: 2.2 } });
s.addShape(p.shapes.LINE, { x: 1.55, y: loopY, w: 0, h: rowY - loopY, line: { color: OR, width: 2.2, endArrowType: 'triangle' } });
lbl(3.4, loopY + 0.03, '循环：还有下一维？是 → 取下一维', 10, OR, true, 3.5, 'left');

// ---------- 出口 ----------
s.addShape(p.shapes.LINE, { x: 8.22, y: 2.55, w: 0, h: 0.09, line: { color: NAVY, width: 2 } });
s.addShape(p.shapes.LINE, { x: 7.6, y: 2.64, w: 0.62, h: 0, line: { color: NAVY, width: 2 } });
arrow(7.6, 2.64, 0, 0.21);
lbl(8.32, 2.56, '否（队列空）', 9, NAVY, true, 1.4);

// ---------- 连乘大卡 ----------
boxR(M, 2.85, 8.0, 1.5, BLUE_BG, NAVY, 1.4);
rich(0.55, 2.95, 7.8, 1.3, [
  { text: '循环结束 → 连乘合成', options: { bold: true, fontSize: 11.5, color: NAVY, breakLine: true } },
  { text: 'f(A,B,C) = f(A) · f(B) · f(C)', options: { bold: true, fontSize: 13, color: RED, breakLine: true } },
  { text: '数字例（工作日早峰·暴雨30·热浪35°C）：', options: { fontSize: 9.5, breakLine: true } },
  { text: 'f = [1+0.261×1.52] × [1.12] = 1.40 × 1.12 ≈ 1.57 → 早峰出发 +57%', options: { fontSize: 10, bold: true, color: NAVY } },
], 'left', 'middle');

// 乘回卡
boxR(8.6, 2.85, 4.33, 1.5, GREEN_BG, GREEN_LN, 1.4);
rich(8.75, 2.95, 4.1, 1.3, [
  { text: '乘回基线（生效）', options: { bold: true, fontSize: 11.5, color: DGREEN, breakLine: true } },
  { text: 'λ(工作日,早峰) = 36,000 × 1.57 ≈ 56,500 趟', options: { fontSize: 10, bold: true, breakLine: true } },
  { text: '距离 −3% / 速度 −11% / 空间重分配 → 三道闸 → 轨迹集', options: { fontSize: 9.5, color: TEXT } },
], 'left', 'middle');

// ---------- 真实系数 strip ----------
lbl(M, 4.52, '登记表里的真实三行（系数都有出处）：', 11, NAVY, true, 5);
const coef = [
  ['A 雨维', 'θ=0.052, RC=0.2', '365 轻中雨样本 · 盲测 4%', NAVY],
  ['B 温度', '≈0 平坦（热浪 +12%）', '13 天热浪 · 方式转移', 'ED7D31'],
  ['C 时段', 'γ 直读 1.52/1.34/1.22/0.72', '10 格 · 无拟合误差', '7030A0'],
];
coef.forEach((r, k) => {
  const x = M + k * 4.21;
  boxR(x, 4.85, 4.0, 0.78, 'FFFFFF', r[3], 1.2);
  rich(x + 0.12, 4.85, 3.76, 0.78, [
    { text: r[0] + '：' + r[1], options: { bold: true, fontSize: 10, color: TEXT, breakLine: true } },
    { text: r[2], options: { fontSize: 9, color: MUTED } },
  ], 'left', 'middle');
});

// ---------- 要点 bar ----------
boxR(M, 5.85, W - 2 * M, 0.52, GREEN_BG, GREEN_LN, 1.2);
rich(M + 0.15, 5.85, W - 2 * M - 0.3, 0.52, [
  { text: '循环的意义：', options: { bold: true, fontSize: 10.5, color: DGREEN } },
  { text: '加一维 = 流水线多跑一遍、登记表多一行——线性扩展，联合大模型的指数灾难不存在；', options: { fontSize: 10.5, color: TEXT } },
  { text: '连乘前提：维间近似独立（强相关先合成，如体感温度）', options: { fontSize: 10.5, bold: true, color: DGREEN } },
], 'left', 'middle');

// ---------- 伪代码 ----------
boxR(M, 6.55, W - 2 * M, 0.62, 'F2F2F2', GREY, 1);
s.addText([
  { text: 'for dim in [A, B, C, D]:', options: { breakLine: true } },
  { text: '    y_i = 标签(dim)；(θ_i, RC_i) = fit(y_i)     # 连续维拟合；离散维直读     →  f(c) = ∏ f(i)  →  λ = λ基线 × f(c)', options: {} },
], { x: M + 0.15, y: 6.55, w: W - 2 * M - 0.3, h: 0.62, fontFace: 'Consolas', fontSize: 10, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.2 });

p.writeFile({ fileName: 'D:/AI/条件转化/维度循环标定.pptx' }).then(() => console.log('saved'));
