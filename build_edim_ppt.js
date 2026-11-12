const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const BLUE_BG = 'D6E4F0', PURPLE_BG = 'E1D5E7', PURPLE_TX = '7030A0',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', ORANGE_BG = 'FCE4D6', ORANGE_TX = '843C0C',
  GREY = '999999', GREY_BG = 'F2F2F2', RED = 'C00000';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('框架可扩展：新增 E 维怎么接入与展示（示例：E = 温度）',
  { x: M, y: 0.16, w: W - 2 * M, h: 0.38, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('接入 = 训练特征加一项、路由逐维多查一次、未见值公式兜底、连乘多一个因子；展示 = 清单加一行、矩阵按 E 分片、连续维画覆盖轴',
  { x: M, y: 0.56, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

const ln = (x, y, w, h, o = {}) => s.addShape(p.shapes.LINE, {
  x, y, w, h, flipV: !!o.flipV,
  line: { color: o.color || NAVY, width: o.width || 1.8, dashType: o.dash ? 'dash' : 'solid', endArrowType: o.end ? 'triangle' : 'none' }
});
const box = (x, y, w, h, runs, fc, ec, dash, align) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: 1.2, dashType: dash ? 'dash' : 'solid' }, rectRadius: 0.03 });
  if (runs) s.addText(runs, { x: x + 0.08, y, w: w - 0.16, h, fontFace: F, color: TEXT, align: align || 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });
};

// ================= 第1行：流水线四处变化 =================
const C1Y = 0.98, C1H = 0.62;
box(M, C1Y, 2.7, C1H, [
  { text: '① 训练：特征加一项', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: 'X = (A, B, C, D, ', options: { fontSize: 9 } },
  { text: 'E 温度', options: { fontSize: 9, bold: true, color: ORANGE_TX } },
  { text: ')', options: { fontSize: 9 } },
], 'FFFFFF', NAVY);
ln(3.1, C1Y + C1H / 2, 0.15, 0, { end: true });
box(3.25, C1Y, 2.5, C1H, [
  { text: '② 路由：逐维多查一次', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: '逐维检查 A~E：见过吗？', options: { fontSize: 9 } },
], 'FFFFFF', NAVY);
ln(5.75, C1Y + C1H / 2, 0.15, 0, { end: true });
box(5.9, C1Y, 2.9, C1H, [
  { text: '③ 未见 → 公式兜底', options: { bold: true, fontSize: 9.5, color: YEL_TX, breakLine: true } },
  { text: 'f_E = 1 + θ_E·ln(1 + E/RC_E)', options: { fontSize: 9 } },
], YELLOW, YEL_LN);
ln(8.8, C1Y + C1H / 2, 0.15, 0, { end: true });
box(8.95, C1Y, 3.98, C1H, [
  { text: '④ 连乘合成（多一个因子）', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: 'P(c) = P0 × f_A × f_B × f_C × f_D × f_E', options: { fontSize: 9 } },
], 'FFFFFF', NAVY);

// ================= 第2行：三个展示组件改造 =================
s.addText('展示① 覆盖清单加一行', { x: M, y: 1.86, w: 3.6, h: 0.24, fontSize: 11, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
s.addText('展示② 能力矩阵按 E 分片', { x: 4.2, y: 1.86, w: 4.4, h: 0.24, fontSize: 11, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
s.addText('展示③ 连续维覆盖轴', { x: 8.85, y: 1.86, w: 4.08, h: 0.24, fontSize: 11, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });

// --- 展示①：A~D 灰卡占位 + E 卡 ---
['A', 'B', 'C', 'D'].forEach((t, i) => {
  box(M + i * 0.92, 2.22, 0.84, 0.34, [{ text: t + '（不动）', options: { fontSize: 8, color: MUTED, align: 'center' } }], GREY_BG, GREY, true, 'center');
});
box(M, 2.66, 3.6, 1.02, null, 'FFFFFF', PURPLE_TX);
s.addText([
  { text: '温度（°C）', options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
  { text: '−10~35 实测见过 → RF 内插', options: { fontSize: 8.5, breakLine: true } },
  { text: '>35 热浪没见过 → 公式外推（θ_E 待标定）', options: { fontSize: 8.5, color: YEL_TX } },
], { x: M + 0.58, y: 2.66, w: 3.6 - 0.66, h: 1.02, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.12 });
s.addShape(p.shapes.RECTANGLE, { x: M, y: 2.66, w: 0.5, h: 1.02, fill: { color: PURPLE_BG }, line: { color: PURPLE_TX, width: 1.2 } });
s.addText('E', { x: M, y: 2.66, w: 0.5, h: 1.02, fontSize: 14, fontFace: F, bold: true, color: NAVY, align: 'center', valign: 'middle', margin: 0 });
s.addText('新卡片插在 D 卡后面即可，其余卡片不动', { x: M, y: 3.78, w: 3.6, h: 0.2, fontSize: 8.5, fontFace: F, color: MUTED, align: 'left', margin: 0 });

// --- 展示②：3 张迷你矩阵 ---
const mini = (mx, title, hot) => {
  s.addText(title, { x: mx, y: 2.22, w: 1.4, h: 0.3, fontSize: 8, fontFace: F, bold: true, color: hot ? YEL_TX : DGREEN, align: 'center', margin: 0 });
  const cw = 0.34, rh = 0.26, rx = mx + 0.42, ry = 2.56;
  ['晴', '雨', '极'].forEach((t, j) => {
    s.addShape(p.shapes.RECTANGLE, { x: rx + j * cw, y: ry, w: cw, h: 0.2, fill: { color: GREY_BG }, line: { color: GREY, width: 0.75 } });
    s.addText(t, { x: rx + j * cw, y: ry, w: cw, h: 0.2, fontSize: 7, fontFace: F, color: MUTED, align: 'center', valign: 'middle', margin: 0 });
  });
  ['工', '周', '假'].forEach((t, i) => {
    s.addShape(p.shapes.RECTANGLE, { x: mx, y: ry + 0.2 + i * rh, w: 0.42, h: rh, fill: { color: GREY_BG }, line: { color: GREY, width: 0.75 } });
    s.addText(t, { x: mx, y: ry + 0.2 + i * rh, w: 0.42, h: rh, fontSize: 7, fontFace: F, color: MUTED, align: 'center', valign: 'middle', margin: 0 });
    for (let j = 0; j < 3; j++) {
      const conf = hot ? { fc: YELLOW, ec: YEL_LN, t: '公', tc: YEL_TX, dash: true }
        : (j === 0 ? { fc: BLUE_BG, ec: NAVY, t: '基', tc: NAVY }
          : j === 1 ? { fc: GREEN_BG, ec: GREEN_LN, t: 'RF', tc: DGREEN }
            : { fc: YELLOW, ec: YEL_LN, t: '公', tc: YEL_TX, dash: true });
      s.addShape(p.shapes.RECTANGLE, { x: rx + j * cw, y: ry + 0.2 + i * rh, w: cw, h: rh, fill: { color: conf.fc }, line: { color: conf.ec, width: 0.75, dashType: conf.dash ? 'dash' : 'solid' } });
      s.addText(conf.t, { x: rx + j * cw, y: ry + 0.2 + i * rh, w: cw, h: rh, fontSize: 7.5, fontFace: F, bold: true, color: conf.tc, align: 'center', valign: 'middle', margin: 0 });
    }
  });
  if (hot) {
    s.addShape(p.shapes.RECTANGLE, { x: mx - 0.04, y: 2.52, w: 1.48, h: 1.34, fill: { color: 'FFFFFF', transparency: 100 }, line: { color: YEL_LN, width: 1.5, dashType: 'dash' } });
  }
};
mini(4.2, 'E=低温 −10~5', false);
mini(5.7, 'E=常温 5~25', false);
mini(7.2, 'E=热浪 >35', true);
s.addText('E 落哪档用哪张矩阵；超出最右档 → 整片公式兜底', { x: 4.2, y: 3.88, w: 4.5, h: 0.2, fontSize: 8.5, fontFace: F, color: MUTED, align: 'left', margin: 0 });

// --- 展示③：E 轴覆盖条 ---
s.addText('E 实测区（RF 内插）', { x: 8.85, y: 2.28, w: 3.05, h: 0.2, fontSize: 8.5, fontFace: F, bold: true, color: DGREEN, align: 'center', margin: 0 });
s.addText('公式区', { x: 11.9, y: 2.28, w: 1.03, h: 0.2, fontSize: 8.5, fontFace: F, bold: true, color: YEL_TX, align: 'center', margin: 0 });
const esegs = [
  { x: 8.85, w: 1.0, t: '低温', sub: '−10~5' },
  { x: 9.85, w: 1.15, t: '常温', sub: '5~25' },
  { x: 11.0, w: 0.9, t: '高温', sub: '25~35' },
  { x: 11.9, w: 1.03, t: '热浪', sub: '>35', dash: true },
];
esegs.forEach(g => {
  s.addShape(p.shapes.RECTANGLE, { x: g.x, y: 2.52, w: g.w, h: 0.5, fill: { color: g.dash ? YELLOW : GREEN_BG }, line: { color: g.dash ? YEL_LN : GREEN_LN, width: 1, dashType: g.dash ? 'dash' : 'solid' } });
  s.addText([
    { text: g.t, options: { fontSize: 7.5, bold: true, breakLine: true } },
    { text: g.sub, options: { fontSize: 6.5, color: MUTED } },
  ], { x: g.x, y: 2.52, w: g.w, h: 0.5, fontFace: F, color: g.dash ? YEL_TX : DGREEN, align: 'center', valign: 'middle', margin: 0, lineSpacingMultiple: 1.0 });
});
s.addText('−10°C', { x: 8.8, y: 3.06, w: 0.9, h: 0.16, fontSize: 7.5, fontFace: F, color: MUTED, align: 'left', margin: 0 });
s.addText('35 实测上限', { x: 10.95, y: 3.06, w: 1.2, h: 0.16, fontSize: 7.5, fontFace: F, color: MUTED, align: 'left', margin: 0 });
const emkX = 11.9 + (38 - 35) / 10 * 1.03;
s.addShape(p.shapes.LINE, { x: emkX, y: 2.52, w: 0, h: 0.62, line: { color: RED, width: 1.2, dashType: 'dash' } });
s.addText('示例 38°C 热浪 → f_E = +8%（假设，待标定）', { x: 9.0, y: 3.32, w: 3.9, h: 0.2, fontSize: 8, fontFace: F, bold: true, color: RED, align: 'left', margin: 0 });
s.addText('连续维的展示与雨强轴完全同构', { x: 8.85, y: 3.58, w: 4.08, h: 0.2, fontSize: 8.5, fontFace: F, color: MUTED, align: 'left', margin: 0 });

// ================= 第3行：例子 + 前提 =================
box(M, 4.15, 7.5, 1.72, [
  { text: '例子：热浪日晚峰（D = 无事件）', options: { bold: true, fontSize: 10.5, color: NAVY, breakLine: true } },
  { text: 'c =（A 晴 r=0，B 晚峰，C 工作日，D 无事件，E 温度=38°C）', options: { fontSize: 9, breakLine: true } },
  { text: '逐维路由：A/B/C/D 见过 → RF/基线直读；E=38 > 35 没见过 → 走公式', options: { fontSize: 9, breakLine: true } },
  { text: 'f_E = 1 + θ_E·ln(1+(38−35)/RC_E) ≈ +8%±3%（θ_E 用常态 25~35°C 数据标定，方法与雨强相同）', options: { fontSize: 9, breakLine: true } },
  { text: '连乘：P(c) = P0 × f_A × f_B × f_C × f_D × f_E → 输出条件轨迹集', options: { fontSize: 9, bold: true } },
], 'FFFFFF', NAVY);

box(8.05, 4.15, 4.88, 1.72, [
  { text: '加 E 维的前提（诚实边界）', options: { bold: true, fontSize: 10.5, color: YEL_TX, breakLine: true } },
  { text: '① 训练集要覆盖 E 的常态范围（NYC 温度全年 −10~35°C，天然覆盖）', options: { fontSize: 8.5, breakLine: true } },
  { text: '② θ_E/RC_E 用常态数据即可标定，不用等极端发生', options: { fontSize: 8.5, breakLine: true } },
  { text: '③ 连乘假设各维近似独立——强相关维建议合成一维或查表', options: { fontSize: 8.5 } },
], 'FFFFFF', YEL_LN, true);

// ================= 底部条 =================
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.0, w: W - 2 * M, h: 0.4, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('一句话：E 维不改变框架——清单多一行、矩阵多一片、轴多一条、连乘多一个因子；维度扩到 A~Z 都是同一套展示',
  { x: M, y: 6.0, w: W - 2 * M, h: 0.4, fontSize: 10, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/维度扩展_新增E维.pptx' }).then(() => console.log('saved'));
