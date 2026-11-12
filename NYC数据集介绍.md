# NYC TLC 行程数据集介绍

> 整理日期：2026-11-06 ｜ 本项目研究窗口：2024-01 ~ 2025-12
> 官方页面：https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

---

## 1. 数据集概况

纽约市出租车与豪华轿车管理委员会（TLC）官方开放的**逐单行程记录**，按月发布、免费、无需注册，是全球使用最广的城市出行数据集。

| 数据集 | 对应业务 | 起始 | 单量级（近期） | 本项目 |
|---|---|---|---|---|
| **High Volume FHV**（HV FHV）| Uber / Lyft 等高单量网约车 | 2019-02 | **约 2000 万单/月**（450MB parquet） | ✅ 主力，24 个月 |
| Yellow Taxi | 曼哈顿巡游黄出租车 | 2009-01 | 约 300-370 万单/月 | 未下载（可补） |
| Green Taxi | 外区巡游绿出租车 | 2013-08 | 约 4-6 万单/月（已萎缩） | ✅ 24 个月 |
| FHV（非高单量）| 传统网约/租赁车 | 2015 | 约 130 万单/月，**无下车信息** | 未下载 |

- 格式：PARQUET 列式存储；发布滞后约 2 个月
- **下载通道（重要）**：新域名 `https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{yyyy}-{mm}.parquet` 直连可用（约 670KB/s，国内免代理）；老域名 `d37ci6vzury5x3` 已被污染不可用

## 2. 字段说明（HV FHV，实测 25 列，2025 年版）

**时空核心（本研究主要使用）**

| 字段 | 含义 | 备注 |
|---|---|---|
| `pickup_datetime` / `dropoff_datetime` | 上/下车时刻 | **纽约当地时间，精确到秒** |
| `PULocationID` / `DOLocationID` | 上/下车 Taxi Zone（1–263） | ⚠️ 264/265 为 Unknown/Outside，须剔除 |
| `trip_miles` | 行程里程（英里，mph 口径） | 计价里程，含绕行 |
| `trip_time` | 行程时长（秒） | 派生速度 = miles ÷ time |

**供需与费用（可用于条件/分层）**

| 字段 | 含义 |
|---|---|
| `request_datetime` / `on_scene_datetime` | 下单时刻 / 司机到达时刻（等驾时长可分离） |
| `base_passenger_fare` / `tips` / `tolls` / `driver_pay` 等 | 分项车费与司机收入 |
| `congestion_surcharge` | 2019 起曼哈顿 96 街以下附加费（$2.75）——**"是否进入拥堵区"的逐单标记** |
| `cbd_congestion_fee` | **2025-01-05 起拥堵收费**（高峰 $9/非高峰 $2.25）——比 OD 更准的 CBD 进入标记 |
| `airport_fee` | 机场附加费（识别机场单） |
| `shared_*` / `wav_*` / `access_a_ride_flag` | 拼车 / 轮椅无障碍 / 类型标记 |

Green 与 Yellow 为巡游车口径（`trip_distance` 计价器里程、`RatecodeID` 费率类型、`payment_type` 等），结构类似。

## 3. 空间粒度：263 个 Taxi Zone

- 无经纬度坐标，最细粒度为 Zone ID；配套 `taxi_zone_lookup.csv`（Zone↔行政区↔街区名）和 `taxi_zones.zip`（多边形 shapefile，可算质心/面积/邻接）
- Zone 面积差异大（曼哈顿核心区 <1km²，斯塔滕岛大区数十 km²）——做回归需 Zone 固定效应或面积归一化
- 场馆锚点（本项目实测）：时代广场/剧院区=230、MSG=186、巴克莱=189、洋基=69、大都会/花旗球场=253、美网=93（法拉盛）

## 4. 数据量（本机实测 footer 元数据）

| | 行数 | 文件大小 |
|---|---|---|
| HV FHV 2024-06（单月） | 2011 万 | 450MB |
| Yellow 高峰年月（2013-07） | 1382 万 | 165MB |
| 全量四类 2009–2026 累计 | 约 40 亿行 | 约 60–80GB |
| **本项目已下载**（HV FHV+Green 2024-2025 + 分区 + 天气 + 三张外生表） | — | **约 11.3GB** |

## 5. 配套外生变量（本项目已构建）

| 表 | 内容 | 关联键 |
|---|---|---|
| NOAA ISD 逐小时天气 | 中央公园（基准）/拉瓜迪亚/肯尼迪/纽瓦克 4 站 × 2023-2025 | ⚠️ UTC，须转 NY 当地时间（夏令时）后按 (date, hour) join；⚠️ 降水解码**只用 FM-15 报且同小时取 max**（与 FM-16 SPECI 同小时求和会虚高 2~4 倍） |
| `events_calendar.csv` | 592 条：球赛 568 场 + 年度活动 24 天（美网/联大/马拉松/游行/跨年），含 Taxi Zone 映射、到场/散场时间窗、中文列 | (date, taxi_zone) |
| `holidays_us_ny.csv` | 美国联邦假日（NY 口径）2023-2026，60 天 | date |
| `congestion_pricing.csv` | 拥堵收费逐日开关（2025-01-05 生效）+ 峰/谷费率 | date |

## 6. 清洗规则（社区惯例 + 本项目实测验证）

```
0.1 < trip_miles < 100（mi）
60 < trip_time < 10800（秒，1min~3h）
0 < 派生速度 < 100 mph
PU/DO ∈ 1..263（剔 264/265）
时间戳年份月份与文件一致（防跨年污染）
```
清洗后通常保留 85–95% 样本。

## 7. 已实测的分布特征（2024-06 HV FHV，n=2011 万）

| 目标量 | 形态 | 关键参数 |
|---|---|---|
| 行程距离 | **logNormal 主体（峰≈1.2mi）+ 机场分量（14–20mi 鼓包 1.4×）+ Pareto 尾** | 尾 α≈2.76 |
| 平均速度 | **截断 LogNormal** | σ(log v)≈0.46 跨时段稳定；中位 11.5mph（≈18.5km/h）|
| 出发时间 λ(t) | 晚峰最大（18 时 5.7%），午后为速度谷（9.7mph）非早晚高峰 | HV FHV 形态：晚峰>早峰 |
| ⚠️ 域提示 | 以上为**网约车域**实测；个体移动域距离应采用截断幂律（文献一致），参数需在个体数据上重测 | |

## 8. 外生冲击窗口（天然实验）

| 冲击 | 时间 | 设计 |
|---|---|---|
| **CBD 拥堵收费** | **2025-01-05 起** | 2024 vs 2025 同月 DiD；处理组用 `cbd_congestion_fee>0` 逐单识别 |
| 极端天气 | 剂量-响应实测锚点（2026-11-06 FM-15 修复后重算）：雨日(≥5mm) 总量 +7.7%、距离中位 −0.17mi、短途 +2.3pp；**强雨时段(≥4mm/h) λ +22.8%、速度 −8.2%**；暖季最大日降水 147.6mm（2024-08-18） | 分层对比 |
| 事件六型 | 点状场馆/路线/区域封锁/全城/多日持续/夜间封锁 | 详见《研究难点.md》《极端案例实测.png》 |

## 9. 引用与来源

- 数据: NYC TLC Trip Record Data（上文官方页）
- 分区: `misc/taxi_zone_lookup.csv`、`misc/taxi_zones.zip`（同 CDN 域名 `/misc/` 路径）
- 天气: NOAA ISD（AWS 镜像 `noaa-global-hourly-pds.s3.amazonaws.com` 直连可用）
- 赛程: MLB statsapi、ESPN 公开接口；假日: Python `holidays` 库
- 本项目下载/处理脚本: `data/download_robust.sh`、`data/exogenous/*.py`
