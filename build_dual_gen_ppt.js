const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00',
  BLUE_BG = 'D6E4F0', GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', RED = 'C00000';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('泛化维 = 2 实例：雨强 × 温度（双泛化维，均已实测标定）',
  { x: M, y: 0.14, w: W - 2 * M, h: 0.36, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('每个泛化维一条剂量曲线；推理按场景取值连乘（观测域内插值 RF、域外沿曲线外推）',
  { x: M, y: 0.52, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

s.addImage({ path: 'D:/AI/条件转化/温度天气泛化/泛化剂量曲线.png', x: M, y: 0.86, w: 8.6, h: 3.15 });

const cards = [
  {
    x: 9.15, y: 0.86, h: 0.98, ec: NAVY, t: '泛化维① 雨强（强）',
    l1: 'δvol(r) = 0.052·ln(1+r/0.2)',
    l2: '365 轻中雨样本标定 · 强雨盲测偏差 4%',
  },
  {
    x: 9.15, y: 1.94, h: 0.98, ec: 'ED7D31', t: '泛化维② 温度（弱）',
    l1: '观测域 2~40°C 内 ≈0（平坦）',
    l2: '热浪小时 +12%（方式转移）· 闷热 +5pp',
  },
  {
    x: 9.15, y: 3.02, h: 0.99, ec: '375623', t: '连乘示例（闷热小雨晚峰）',
    l1: '×1.09（雨） × 1.05（闷热）≈ ×1.14',
    l2: '两维都温和起效 → L0，RF 直出',
  },
];
cards.forEach(c => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: c.x, y: c.y, w: 3.78, h: c.h, fill: { color: 'FFFFFF' }, line: { color: c.ec, width: 1.2 }, rectRadius: 0.03 });
  s.addText([
    { text: c.t, options: { bold: true, fontSize: 10.5, color: c.ec, breakLine: true } },
    { text: c.l1, options: { fontSize: 9, breakLine: true } },
    { text: c.l2, options: { fontSize: 9, color: '1A1A1A' } },
  ], { x: c.x + 0.12, y: c.y, w: 3.54, h: c.h, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });
});

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 4.2, w: 5.9, h: 1.35, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1.2 }, rectRadius: 0.03 });
s.addText([
  { text: '推理示例：闷热小雨的晚峰（工作日）', options: { bold: true, fontSize: 10.5, color: NAVY, breakLine: true } },
  { text: 'c =（雨 10mm/h，温度 30°C 闷热，工作日，晚峰）', options: { fontSize: 9, breakLine: true } },
  { text: '雨维：10mm/h 在域内 → 1+δvol(10)×γ(晚峰) ≈ ×1.09', options: { fontSize: 9, breakLine: true } },
  { text: '温度维：30°C 闷热 → +5pp（控温对比实测）', options: { fontSize: 9 } },
], { x: 0.52, y: 4.2, w: 5.7, h: 1.35, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.5, y: 4.2, w: 6.43, h: 1.35, fill: { color: 'FFFFFF' }, line: { color: NAVY, width: 1.2 }, rectRadius: 0.03 });
s.addText([
  { text: '结果合成（连乘）', options: { bold: true, fontSize: 10.5, color: NAVY, breakLine: true } },
  { text: '总乘子 = 1.09 × 1.05 ≈ ×1.14 → 晚峰出发量 +14%', options: { fontSize: 9.5, breakLine: true } },
  { text: '置信等级：L0（两维都在观测域内，RF 直出精度最优）', options: { fontSize: 9.5 } },
], { x: 6.65, y: 4.2, w: 6.1, h: 1.35, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.2 });

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.75, w: W - 2 * M, h: 0.5, fill: { color: YELLOW }, line: { color: YEL_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText([
  { text: '诚实边界：', options: { bold: true, fontSize: 10.5, color: YEL_TX } },
  { text: '温度观测域 2~40°C（暖季），冰点以下与 45°C+ 未测 → 曲线延伸挂低置信；热浪样本 13 天（日级）；行人域方向可能反转（闷热压制户外），需自有数据重标——不可继承出租车系数', options: { fontSize: 9.5, color: TEXT } },
], { x: M + 0.15, y: 5.75, w: W - 2 * M - 0.3, h: 0.5, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.45, w: W - 2 * M, h: 0.5, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText([
  { text: '任务计数：', options: { bold: true, fontSize: 10.5, color: DGREEN } },
  { text: '2 个泛化维 × 3 个日型 = 6 条剂量曲线（雨 3 条形状各异、温度 3 条平坦）；极端互斥（暴雨时凉、热浪时晴）→ "双极端格"物理不存在，无需 L3 双超域外推', options: { fontSize: 10, color: TEXT } },
], { x: M + 0.15, y: 6.45, w: W - 2 * M - 0.3, h: 0.5, fontFace: F, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });

p.writeFile({ fileName: 'D:/AI/条件转化/温度天气泛化/泛化维2_雨与温度.pptx' }).then(() => console.log('saved'));
