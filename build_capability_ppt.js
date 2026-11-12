const pptxgen = require('pptxgenjs');
const W = 13.33, M = 0.4;
let p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.author = '条件转化';
const F = 'Microsoft YaHei';
const NAVY = '1F4E79', TEXT = '1A1A1A', MUTED = '666666', DGREEN = '375623';
const YELLOW = 'FFF2CC', YEL_LN = 'D6B656', YEL_TX = '8C5A00';
const BLUE_BG = 'D6E4F0', PURPLE_BG = 'E1D5E7', PURPLE_TX = '7030A0',
  GREEN_BG = 'E2F0D9', GREEN_LN = '70AD47', GREY = '999999', GREY_BG = 'F2F2F2';

let s = p.addSlide(); s.background = { color: 'FFFFFF' };

s.addText('推理能力全景：条件组合覆盖图（哪些见过 → RF；哪些没见过 → 公式）',
  { x: M, y: 0.16, w: W - 2 * M, h: 0.38, fontSize: 16.5, fontFace: F, bold: true, color: NAVY, align: 'center', margin: 0 });
s.addText('绿 = 训练见过，RF 内插（精度最优）｜ 黄 = 没见过，公式外推（物理约束兜底）｜ 蓝 = 晴天基线直出｜ 紫 = 事件查表',
  { x: M, y: 0.56, w: W - 2 * M, h: 0.22, fontSize: 9.5, fontFace: F, color: MUTED, align: 'center', margin: 0 });

const box = (x, y, w, h, runs, fc, ec, dash, align) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fc }, line: { color: ec, width: 1.2, dashType: dash ? 'dash' : 'solid' }, rectRadius: 0.03 });
  if (runs) s.addText(runs, { x: x + 0.08, y, w: w - 0.16, h, fontFace: F, color: TEXT, align: align || 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });
};

// ================= 左上：能力矩阵 天气 × 日型 =================
s.addText('能力矩阵：天气强度 × 日型', { x: M, y: 0.9, w: 6.35, h: 0.26, fontSize: 12, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });

const colW = 0.85, gx = 1.5, hy = 1.24, hyH = 0.5;
const cols = ['晴\nr = 0', '小雨\n< 2.5', '中雨\n2.5–8', '大雨\n8–16', '暴雨\n实测 148', '极端\n> 200 假设'];
cols.forEach((t, j) => {
  box(gx + j * colW, hy, colW, hyH, [{ text: t, options: { fontSize: 7.5, align: 'center' } }], GREY_BG, GREY, false, 'center');
});
const rows = ['工作日', '周末', '假日'];
const colStyle = [
  { fc: BLUE_BG, ec: NAVY, main: '基线', sub: null },
  { fc: GREEN_BG, ec: GREEN_LN, main: 'RF', sub: null },
  { fc: GREEN_BG, ec: GREEN_LN, main: 'RF', sub: null },
  { fc: GREEN_BG, ec: GREEN_LN, main: 'RF', sub: null },
  { fc: GREEN_BG, ec: GREEN_LN, main: 'RF', sub: '52 天实测' },
  { fc: YELLOW, ec: YEL_LN, main: '公式外推', sub: '没见过', dash: true },
];
rows.forEach((r, i) => {
  const ry = 1.74 + i * 0.6;
  box(M, ry, 1.1, 0.58, [{ text: r, options: { fontSize: 9, bold: true, align: 'center' } }], GREY_BG, GREY, false, 'center');
  colStyle.forEach((c, j) => {
    const runs = c.sub
      ? [{ text: c.main, options: { fontSize: c.main.length > 3 ? 8 : 10, bold: true, color: c.ec === YEL_LN ? YEL_TX : DGREEN, breakLine: true } },
         { text: c.sub, options: { fontSize: 7, color: MUTED } }]
      : [{ text: c.main, options: { fontSize: 10, bold: true, color: c.ec === NAVY ? NAVY : DGREEN } }];
    box(gx + j * colW, ry, colW, 0.58, runs, c.fc, c.ec, c.dash, 'center');
  });
});

// 图例
const lg = y3 => y3;
s.addShape(p.shapes.RECTANGLE, { x: M, y: 3.66, w: 0.14, h: 0.14, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1 } });
s.addText('见过 → RF 内插', { x: 0.6, y: 3.62, w: 1.5, h: 0.2, fontSize: 8.5, fontFace: F, color: TEXT, margin: 0, valign: 'middle' });
s.addShape(p.shapes.RECTANGLE, { x: 2.15, y: 3.66, w: 0.14, h: 0.14, fill: { color: YELLOW }, line: { color: YEL_LN, width: 1, dashType: 'dash' } });
s.addText('没见过 → 公式外推', { x: 2.35, y: 3.62, w: 1.8, h: 0.2, fontSize: 8.5, fontFace: F, color: TEXT, margin: 0, valign: 'middle' });
s.addShape(p.shapes.RECTANGLE, { x: 4.2, y: 3.66, w: 0.14, h: 0.14, fill: { color: BLUE_BG }, line: { color: NAVY, width: 1 } });
s.addText('晴天基线直出', { x: 4.4, y: 3.62, w: 1.4, h: 0.2, fontSize: 8.5, fontFace: F, color: TEXT, margin: 0, valign: 'middle' });

s.addText([
  { text: '暴雨 52 天实测验证；假日下雨不叠加（差 0.7 pp）→ 直接换假日基线', options: { breakLine: true } },
  { text: '距离分布 KS = 0.024；速度 σ = 0.44 跨雨强恒定（形状参数锁死，只改位置）', options: {} },
], { x: M, y: 3.92, w: 6.35, h: 0.44, fontSize: 8.5, fontFace: F, color: MUTED, align: 'left', margin: 0, lineSpacingMultiple: 1.2 });

// ================= 右上：四个维度覆盖清单 =================
s.addText('四个条件维度的覆盖', { x: 7.0, y: 0.9, w: 5.93, h: 0.26, fontSize: 12, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });
const dims = [
  { t: 'A', n: '雨强 r（mm/h）', d: '0–148 实测见过 → RF 内插；>148 没见过 → 公式外推（饱和，不爆炸）', fc: BLUE_BG, ec: NAVY },
  { t: 'B', n: '时段（24 小时）', d: '7×24 = 168 格基线表，全时段无空档', fc: PURPLE_BG, ec: PURPLE_TX },
  { t: 'C', n: '日型（3 档）', d: '工作日 / 周末 / 假日 三套基线切换；假日 −5.6%，峰位后移 7~10h', fc: GREEN_BG, ec: GREEN_LN },
  { t: 'D', n: '天气类型 / 事件', d: '雨 5 档 ＋ 球赛·演唱会·美网·马拉松·跨年 查表直出', d2: '沙尘暴等未见现象 → 公式框架可挂接（需标定数据）', fc: YELLOW, ec: YEL_LN },
];
dims.forEach((c, i) => {
  const y = 1.24 + i * 0.82, h = i === 3 ? 0.9 : 0.7;
  box(7.0, y, 5.93, h, null, 'FFFFFF', c.ec);
  s.addShape(p.shapes.RECTANGLE, { x: 7.0, y, w: 0.5, h, fill: { color: c.fc }, line: { color: c.ec, width: 1.2 } });
  s.addText(c.t, { x: 7.0, y, w: 0.5, h, fontSize: 14, fontFace: F, bold: true, color: NAVY, align: 'center', valign: 'middle', margin: 0 });
  const runs = [
    { text: c.n, options: { bold: true, fontSize: 9.5, color: NAVY, breakLine: true } },
    { text: c.d, options: { fontSize: 8.5, breakLine: !!c.d2 } },
  ];
  if (c.d2) runs.push({ text: c.d2, options: { fontSize: 8.5, color: YEL_TX } });
  s.addText(runs, { x: 7.62, y, w: 5.2, h, fontFace: F, color: TEXT, align: 'left', valign: 'middle', margin: 0, lineSpacingMultiple: 1.15 });
});

// ================= 下半：雨强连续覆盖轴 =================
s.addText('连续维度怎么延伸：以雨强 r 为例（见过 → 内插；没见过 → 沿公式走）',
  { x: M, y: 4.62, w: 10.5, h: 0.26, fontSize: 12, fontFace: F, bold: true, color: NAVY, align: 'left', margin: 0 });

// 区域标签
s.addText('RF 内插区（训练见过）', { x: 1.38, y: 4.98, w: 7.96, h: 0.2, fontSize: 9, fontFace: F, bold: true, color: DGREEN, align: 'center', margin: 0 });
s.addText('公式外推区（没见过）', { x: 9.34, y: 4.98, w: 3.59, h: 0.2, fontSize: 9, fontFace: F, bold: true, color: YEL_TX, align: 'center', margin: 0 });

// 分段
const segs = [
  { x: 0.4, w: 0.98, fc: BLUE_BG, ec: NAVY, t: '晴', sub: '基线' },
  { x: 1.38, w: 1.42, fc: GREEN_BG, ec: GREEN_LN, t: '小雨', sub: '0–2.5' },
  { x: 2.8, w: 1.42, fc: GREEN_BG, ec: GREEN_LN, t: '中雨', sub: '2.5–8' },
  { x: 4.22, w: 1.42, fc: GREEN_BG, ec: GREEN_LN, t: '大雨', sub: '8–16' },
  { x: 5.64, w: 3.7, fc: GREEN_BG, ec: GREEN_LN, t: '暴雨', sub: '16–148 实测' },
  { x: 9.34, w: 3.59, fc: YELLOW, ec: YEL_LN, t: '极端 200+', sub: '没见过 → 公式', dash: true },
];
segs.forEach(g => {
  s.addShape(p.shapes.RECTANGLE, { x: g.x, y: 5.24, w: g.w, h: 0.52, fill: { color: g.fc }, line: { color: g.ec, width: 1.2, dashType: g.dash ? 'dash' : 'solid' } });
  s.addText([
    { text: g.t, options: { fontSize: 9, bold: true, breakLine: true } },
    { text: g.sub, options: { fontSize: 7.5, color: MUTED } },
  ], { x: g.x, y: 5.24, w: g.w, h: 0.52, fontFace: F, color: g.dash ? YEL_TX : DGREEN, align: 'center', valign: 'middle', margin: 0, lineSpacingMultiple: 1.05 });
});

// 刻度
const ticks = [[0.4, '0', 'left'], [1.38, '2.5', 'left'], [2.8, '8', 'left'], [4.22, '16', 'left'], [8.55, '148 实测上限', 'left'], [11.93, '200+ 假设', 'right']];
ticks.forEach(([tx, tt, al]) => {
  s.addText(tt + ' mm/h', { x: tx - 0.01, y: 5.8, w: 1.0, h: 0.16, fontSize: 8, fontFace: F, color: MUTED, align: al, margin: 0 });
});

// 示例标记：暴雨 30 mm/h
const mkX = 5.64 + (30 - 16) / (148 - 16) * 3.7;
s.addShape(p.shapes.LINE, { x: mkX, y: 5.24, w: 0, h: 0.9, line: { color: 'C00000', width: 1.5, dashType: 'dash' } });
s.addText('示例：暴雨 30 mm/h → 出发 +26%（盲测偏差 4%）',
  { x: mkX + 0.08, y: 6.02, w: 4.6, h: 0.2, fontSize: 8.5, fontFace: F, bold: true, color: 'C00000', align: 'left', margin: 0 });

// ================= 底部证据条 =================
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.5, w: W - 2 * M, h: 0.42, fill: { color: GREEN_BG }, line: { color: GREEN_LN, width: 1.2 }, rectRadius: 0.04 });
s.addText('见过的组合：RF 内插——分层外推 MAE 10.7 pp < 自然波动 13.9 pp（可预测极限）｜没见过的值：公式兜底——强雨盲测偏差 4%，饱和约束不爆炸',
  { x: M, y: 6.5, w: W - 2 * M, h: 0.42, fontSize: 9.5, fontFace: F, bold: true, color: DGREEN, align: 'center', valign: 'middle', margin: 0 });

p.writeFile({ fileName: 'D:/AI/条件转化/条件组合能力全景.pptx' }).then(() => console.log('saved'));
