const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', RED = 'C00000', RED_BG = 'FBE3E3';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('反转一：周末 × 雨——剂量形状的反转（复核版）',
  { x: M, y: 0.14, w: W - 2 * M, h: 0.36, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('稳健结论＝出发量的剂量形状反转（工作日递增，周末台阶）；速度/距离两类日型一致变慢变短；早前"周末更快更远"已撤回',
  { x: M, y: 0.52, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

// 证据总览图（左大）
s.addImage({ path: 'D:/AI/条件转化/反转一_周末方向反转/反转1_证据总览.png', x: M, y: 0.86, w: 9.3, h: 3.1 });

// 撤回说明卡（右上,黄）
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 9.85, y: 0.86, w: 3.08, h: 3.1, fill: { color: YELLOW }, line: { color: YEL_LN, width: 1.2 }, rectRadius: 0.03 });
s.addText([
  { text: '撤回说明', options: { bold: true, fontSize: 10.5, color: YEL_TX, breakLine: true } },
  { text: '早前单季口径记录："周末雨天更快 +3%、更远 +2%"', options: { fontSize: 8.5, breakLine: true } },
  { text: '在两季数据 + 小时匹配口径下未复现（两类日型均变慢变短）→ 撤回汇报引用', options: { fontSize: 8.5, breakLine: true } },
  { text: '保留的稳健反转：出发量的剂量形状（见左图①）', options: { fontSize: 8.5, bold: true, color: DGREEN, breakLine: true } },
  { text: '复核口径：两季数据（419 天）＋ 小时匹配 ÷ 同钟点干日基线（雨时≥0.3mm/h）', options: { fontSize: 8, color: MUTED } },
], { x: 9.97, y: 0.86, w: 2.84, h: 3.1, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });

// 三张数字卡 + 机制卡
const cards = [
  {
    x: 0.4, w: 3.0, tc: NAVY, t: '出发量（剂量形状反转）',
    l1: '工作日：1.096 → 1.183 → 1.305（随雨强递增）', l2: '周末：1.148 → 1.122 → 1.179（台阶，不随强度）', l3: '机制：周末可转移池小，小雨即饱和',
  },
  {
    x: 3.55, w: 3.0, tc: NAVY, t: '中位里程（两类一致）',
    l1: '工作日：−4% → −5.2% → −6.5%', l2: '周末：−5.7% → −5.5% → −6.2%', l3: '方向反转未复现，两类日型相近',
  },
  {
    x: 6.7, w: 3.0, tc: NAVY, t: '中位速度（两类一致）',
    l1: '工作日：−3.1% → −6.3% → −9.8%', l2: '周末：−6.2% → −5.1% → −7.8%', l3: '方向反转未复现，两类日型相近',
  },
  {
    x: 9.85, w: 3.08, tc: DGREEN, t: '建模规则',
    l1: '出发量：按日型两条剂量曲线\n（工作日递增式｜周末台阶式）', l2: '速度/距离：两类日型共用\n同方向曲线（幅度接近）', l3: '禁止用一条公式打所有日型',
  },
];
cards.forEach(c => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c.x, y: 4.14, w: c.w, h: 1.28, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1.1 }, rectRadius: 0.03 });
  const lines = [c.t, c.l1, c.l2, c.l3];
  s.addText(lines.map((t, k) => ({ text: t, options: { fontSize: k === 0 ? 9.5 : 8.5, bold: k === 0, color: k === 0 ? NAVY : TEXT, breakLine: k < lines.length - 1 } })),
    { x: c.x + 0.09, y: 4.14, w: c.w - 0.18, h: 1.28, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });
});

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.62, w: W - 2 * M, h: 0.42, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('样本口径：工作日雨日 148（小70/中52/大26）、周末雨日 57（小25/中21/大11）｜小时匹配 ÷ 同年×日型×同钟点干日基线，雨时≥0.3mm/h｜三量互证 → 中高置信',
  { x: M, y: 5.62, w: W - 2 * M, h: 0.42, fontSize: 9.5, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.2, w: W - 2 * M, h: 0.5, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1.2 }, rectRadius: 0.04 });
s.addText([
  { text: '汇报口径一句话：', options: { bold: true, fontSize: 10, color: NAVY } },
  { text: '"同一个雨，工作日是随雨强越拧越大的递增旋钮，周末是一次拧到位的台阶旋钮——方向相反的不是快慢，而是响应的形状。速度和距离两类日型一致变慢变短，早前的更快更远已复核撤回。"', options: { fontSize: 10, color: TEXT } },
], { x: M + 0.15, y: 6.2, w: W - 2 * M - 0.3, h: 0.5, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });

p.writeFile({ fileName: 'D:/AI/条件转化/反转一_周末方向反转/反转1_周末方向反转.pptx' }).then(() => console.log('saved'));
