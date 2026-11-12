const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47',
  ORANGE_BG = 'FCE4D6', ORANGE_LN = 'ED7D31',
  RED_BG = 'FBE3E3', RED = 'C00000', GREY = '999999', GREY_BG = 'F2F2F2';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('3 维 × 4 取值 = 64 种组合：按"没见过的维数"路由推理步骤',
  { x: M, y: 0.16, w: W - 2 * M, h: 0.38, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('训练集并非每维取值都见过——每个组合查同一条规则（数没见过的维数），落到哪一类就走哪条推理路径；格子颜色 = 路由',
  { x: M, y: 0.56, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

s.addText('训练覆盖声明：A 维见过 a1~a3（a4 未出现）｜B 维见过 b1~b3（b4 未出现）｜C 维见过 c1~c2（c3、c4 未出现）',
  { x: M, y: 0.92, w: 10.2, h: 0.22, fontSize: 10, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
s.addText('共 4×4×4 = 64 格，全部可路由、全部可算', { x: 10.7, y: 0.92, w: 2.23, h: 0.22, fontSize: 10, fontFace: F, bold: true, color: DGREEN, align: 'right', margin: 0 });

// ================= 64 格：4 个 C 切片 × A(列)×B(行) =================
const CS = 0.5, RH = 0.42, LB = 0.55, GY = 1.72;
const sliceX = [0.4, 3.07, 5.74, 8.41];
const sliceTitle = [
  { t: 'C = c1（见过）', seen: true }, { t: 'C = c2（见过）', seen: true },
  { t: 'C = c3（没见过）', seen: false }, { t: 'C = c4（没见过）', seen: false },
];
const styleU = [
  { fc: GREEN_BG, ec: GREEN_LN, dash: false },   // u=0 全见过
  { fc: YELLOW, ec: YEL_LN, dash: false },       // u=1
  { fc: ORANGE_BG, ec: ORANGE_LN, dash: false }, // u=2
  { fc: RED_BG, ec: RED, dash: true },           // u=3
];

sliceX.forEach((sx, sl) => {
  const st = sliceTitle[sl];
  s.addText(st.t, { x: sx, y: 1.42, w: 2.55, h: 0.22, fontSize: 9.5, fontFace: F, bold: true, color: st.seen ? DGREEN : ORANGE_LN, align: 'left', margin: 0 });
  // 列头 a1..a4（a4 未见过 → 红色虚线框）
  ['a1', 'a2', 'a3', 'a4'].forEach((t, j) => {
    const unseen = j === 3;
    s.addShape(p.shapes.RECTANGLE, { x: sx + LB + j * CS, y: GY, w: CS, h: 0.22, fill: { color: GREY_BG }, line: { color: unseen ? RED : GREY, width: 0.75, dashType: unseen ? 'dash' : 'solid' } });
    s.addText(t, { x: sx + LB + j * CS, y: GY, w: CS, h: 0.22, fontSize: 7.5, fontFace: F, bold: unseen, color: unseen ? RED : MUTED, align: 'center', valign: 'middle', margin: 0 });
  });
  // 行 + 格子
  ['b1', 'b2', 'b3', 'b4'].forEach((t, i) => {
    const unseen = i === 3;
    s.addShape(p.shapes.RECTANGLE, { x: sx, y: GY + 0.22 + i * RH, w: LB, h: RH, fill: { color: GREY_BG }, line: { color: unseen ? RED : GREY, width: 0.75, dashType: unseen ? 'dash' : 'solid' } });
    s.addText(t, { x: sx, y: GY + 0.22 + i * RH, w: LB, h: RH, fontSize: 7.5, fontFace: F, bold: unseen, color: unseen ? RED : MUTED, align: 'center', valign: 'middle', margin: 0 });
    for (let j = 0; j < 4; j++) {
      const u = (j === 3 ? 1 : 0) + (i === 3 ? 1 : 0) + (sl >= 2 ? 1 : 0);
      const cf = styleU[u];
      s.addShape(p.shapes.RECTANGLE, { x: sx + LB + j * CS, y: GY + 0.22 + i * RH, w: CS, h: RH, fill: { color: cf.fc }, line: { color: cf.ec, width: 0.75, dashType: cf.dash ? 'dash' : 'solid' } });
    }
  });
});

// ================= 路由表：四条推理步骤 =================
const routes = [
  { fc: GREEN_BG, ec: GREEN_LN, n: '18 格', tc: DGREEN, txt: '触发：三维都见过　→　推理：RF 直接内插，输出四量系数　｜　置信：高' },
  { fc: YELLOW, ec: YEL_LN, n: '30 格', tc: YEL_TX, txt: '触发：恰 1 维没见过（例 a4×b2×c1）　→　推理：见过维走 RF ＋ 该维走公式 f_i，连乘合成　｜　置信：中' },
  { fc: ORANGE_BG, ec: ORANGE_LN, n: '14 格', tc: '843C0C', txt: '触发：恰 2 维没见过　→　推理：两维公式 ＋ 见过维 RF，连乘合成　｜　置信：低（区间宽）' },
  { fc: RED_BG, ec: RED, n: '2 格', tc: RED, txt: '触发：三维都没见过　→　推理：全公式 / 先验兜底，只给量级与方向　｜　仅参考' },
];
routes.forEach((r, i) => {
  const ry = 3.6 + i * 0.5;
  s.addShape(p.shapes.RECTANGLE, { x: M, y: ry, w: 0.3, h: 0.3, fill: { color: r.fc }, line: { color: r.ec, width: 1.2, dashType: (i === 3) ? 'dash' : 'solid' } });
  s.addText([
    { text: r.n + '　', options: { bold: true, fontSize: 10.5, color: r.tc } },
    { text: r.txt, options: { fontSize: 9.5, color: TEXT } },
  ], { x: M + 0.4, y: ry, w: W - 2 * M - 0.4, h: 0.36, fontFace: F, align: 'left', valign: 'middle', margin: 0 });
});
s.addText('格数核对：18 + 30 + 14 + 2 = 64 ✓（覆盖声明决定各类格数；换一组训练覆盖，只需重算格数，规则与图不变）',
  { x: M, y: 5.66, w: 12.0, h: 0.2, fontSize: 8.5, fontFace: F, color: MUTED, align: 'left', margin: 0 });

// ================= 底部条 =================
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.05, w: W - 2 * M, h: 0.42, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('路由引擎只有一条规则：逐维查"是否见过"，按没见过的维数选推理步骤——64 格共用 1 条规则；换成 10 维 × N 取值，只是格子更多，规则与图不变',
  { x: M, y: 6.05, w: W - 2 * M, h: 0.42, fontSize: 10, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/64组合路由图示.pptx' }).then(() => console.log('saved'));
