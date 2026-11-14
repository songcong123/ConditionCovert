const pptxgen = require('pptxgenjs');
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = 'Z.ai';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };
s.addText('手稿流程图 · 原样转绘',
  { x: 0.5, y: 0.2, w: 12.33, h: 0.4, fontSize: 16, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });

const boxR = (x, y, w, h, fc, ec) => s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: 1.6 }, rectRadius: 0.05 });
const line = (x, y, w, h, arrow) => s.addShape(p.shapes.LINE, { x, y, w, h, line: { color: NAVY, width: 2.2, endArrowType: arrow ? 'triangle' : 'none' } });

// ===== 流程元素（横向布局，垂直居中带 =====
// 数据处理（左）
boxR(1.2, 3.1, 2.6, 1.1, 'D6E4F0', NAVY);
s.addText('数据处理', { x: 1.2, y: 3.1, w: 2.6, h: 1.1, fontSize: 16, fontFace: F, bold: true, color: NAVY, align: 'center', valign: 'middle', margin: 0 });

// 箭头 → 菱形
line(3.8, 3.65, 1.5, 0, true);

// 菱形判断框（中）
s.addShape(p.shapes.DIAMOND, { x: 5.35, y: 2.8, w: 2.6, h: 1.7, fill: { color: 'FFF2CC' }, line: { color: 'D6B656', width: 1.6 } });
s.addText('是否存在', { x: 5.35, y: 2.8, w: 2.6, h: 1.7, fontSize: 15, fontFace: F, bold: true, color: TEXT, align: 'center', valign: 'middle', margin: 0 });

// 分支1：否 → 步骤1（右上）：竖线上行 + 横线右行
line(6.65, 2.8, 0, -(2.8 - 2.1), false);            // 菱形上顶点 → 上
line(6.65, 2.1, 9.7 - 6.65, 0, true);                // → 右，箭头入步骤1
s.addText('否', { x: 6.75, y: 2.16, w: 0.5, h: 0.3, fontSize: 13, fontFace: F, bold: true, color: 'C00000', margin: 0 });

// 分支2：是 → 步骤2（右中下）：横线右 + 竖线下 + 横线右（肘形）
line(7.95, 3.65, 0.85, 0, false);                    // 菱形右顶点 → 右
line(8.8, 3.65, 0, 5.05 - 3.65, false);              // → 下
line(8.8, 5.05, 9.7 - 8.8, 0, true);                 // → 右，箭头入步骤2
s.addText('是', { x: 8.0, y: 3.28, w: 0.5, h: 0.3, fontSize: 13, fontFace: F, bold: true, color: '375623', margin: 0 });

// 步骤1（右上）
boxR(9.7, 1.55, 2.6, 1.1, 'FFFFFF', NAVY);
s.addText('步骤1', { x: 9.7, y: 1.55, w: 2.6, h: 1.1, fontSize: 16, fontFace: F, bold: true, color: NAVY, align: 'center', valign: 'middle', margin: 0 });

// 步骤2（右中下）
boxR(9.7, 4.5, 2.6, 1.1, 'FFFFFF', NAVY);
s.addText('步骤2', { x: 9.7, y: 4.5, w: 2.6, h: 1.1, fontSize: 16, fontFace: F, bold: true, color: NAVY, align: 'center', valign: 'middle', margin: 0 });

// 底注
s.addText('按手稿原样转绘（横向：数据处理 → 判断是否存在 → 否/是 二分支）。文字替换成实际业务词，说一声即可。',
  { x: 0.5, y: 6.9, w: 12.33, h: 0.3, fontSize: 10, fontFace: F, color: MUTED, align: 'center', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/手稿_流程图.pptx' }).then(() => console.log('saved'));
