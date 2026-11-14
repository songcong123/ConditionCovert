# -*- coding: utf-8 -*-
"""特征构造模块: 三组正交特征体系
=============================
时段组(索引维, 离散, 值域封闭):
  hour               小时 0-23(同时给 sin/cos 圆编码)
  segment            时段 5 档: 凌晨/早峰/白天/晚峰/夜间
  is_weekday         周中(周一~五=1)
  is_weekend         周末(周六日=1)

天气组(泛化维, 连续, 值域开放 → RF 外推风险组):
  rain_now           当前小时雨强 mm/h (log1p 编码)
  rain_lag1/lag2     前 1/2 小时雨强(滞后)
  temp               气温 °C
  humidity           相对湿度 %
  wind               风速 m/s

节假日组(混合):
  is_holiday         假日旗标
  hol_offset         假日内第几天(1,2,...; 非假日=0)
  hol_len            该假日总长(天; 调休连休型>1)
  hol_dist           距最近假日的天数(前负后正, 当天=0)
  weekend_share      假日窗口内周末占比(区分"插空型"vs"连休型")
=============================
用法: build_features(dates, hour, weather_lookup, holiday_lookup) -> dict[str, array]
验证: validate_41 Feature NYC 数据, 标签=出发量效应, 打印相关性
"""
import numpy as np

SEG_DEF = [(0, 6, '凌晨'), (7, 9, '早峰'), (10, 16, '白天'), (17, 20, '晚峰'), (21, 23, '夜间')]

def seg_of(h):
    for lo, hi, nm in SEG_DEF:
        if lo <= h <= hi: return nm
    return '夜间'

def build_features(dates, hours, wx, hol):
    hours = np.asarray(hours, dtype=float)
    """dates: list[str YYYY-MM-DD]; hours: array[int]
    wx: {(date, hour): dict(rain=, temp=, humidity=, wind=)}
    hol: {date: dict(name=, offset=, length=, dist=, weekend_share=)}  缺省非假日
    返回: dict[特征名 -> array]"""
    n = len(dates)
    F = {}
    dow = np.array([(np.datetime64(d, 'D').view('int64') + 3) % 7 for d in dates])  # 0=周一
    # ---- 时段组 ----
    F['hour'] = np.array(hours, float)
    F['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    F['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    seg_map = {nm: i for i, (_, _, nm) in enumerate(SEG_DEF)}
    F['segment'] = np.array([seg_map[seg_of(h)] for h in hours], float)
    F['is_weekday'] = (dow < 5).astype(float)
    F['is_weekend'] = (dow >= 5).astype(float)
    # ---- 天气组 ----
    r = np.array([wx.get((d, h), {}).get('rain', 0.0) for d, h in zip(dates, hours)])
    rl1 = np.array([wx.get((d, h - 1), {}).get('rain', 0.0) if h >= 1 else 0.0
                    for d, h in zip(dates, hours)])
    rl2 = np.array([wx.get((d, h - 2), {}).get('rain', 0.0) if h >= 2 else 0.0
                    for d, h in zip(dates, hours)])
    F['rain_now'] = np.log1p(r)
    F['rain_lag1'] = np.log1p(rl1)
    F['rain_lag2'] = np.log1p(rl2)
    F['temp'] = np.array([wx.get((d, h), {}).get('temp', np.nan) for d, h in zip(dates, hours)])
    F['humidity'] = np.array([wx.get((d, h), {}).get('humidity', np.nan) for d, h in zip(dates, hours)])
    F['wind'] = np.array([wx.get((d, h), {}).get('wind', np.nan) for d, h in zip(dates, hours)])
    # ---- 节假日组 ----
    F['is_holiday'] = np.array([1.0 if d in hol else 0.0 for d in dates])
    F['hol_offset'] = np.array([hol[d]['offset'] if d in hol else 0.0 for d in dates])
    F['hol_len'] = np.array([hol[d]['length'] if d in hol else 0.0 for d in dates])
    F['hol_dist'] = np.array([hol[d]['dist'] if d in hol else 0.0 for d in dates])
    F['weekend_share'] = np.array([hol[d]['weekend_share'] if d in hol else 0.0 for d in dates])
    return F


def build_holiday_table(holiday_dates, all_dates):
    """从假日日期列表构建节假日特征表
    holiday_dates: list[(date_str, name, length)]  length=假日总长(连休型>1)
    返回 {date: {name, offset, length, dist, weekend_share}}"""
    import datetime as dt
    hol = {}
    hset = set(d for d, _, _ in holiday_dates)
    hinfo = {d: (nm, ln) for d, nm, ln in holiday_dates}
    # 假日内 offset 与窗口 weekend_share
    for d, nm, ln in holiday_dates:
        do = dt.date.fromisoformat(d)
        # 找连块起点(同一段假期的第一天): 简化=把同 length 的连续日期分组
        off = 1
        prev = (do - dt.timedelta(days=1)).isoformat()
        while prev in hset:
            off += 1
            prev = (dt.date.fromisoformat(prev) - dt.timedelta(days=1)).isoformat()
        # 窗口 = 该假日及其相邻周末
        nd = 0; nw = 0
        for k in range(ln):
            dd = do + dt.timedelta(days=k)
            if dd.isoformat() in hset:
                nd += 1
                if dd.weekday() >= 5: nw += 1
        hol[d] = dict(name=nm, offset=off, length=ln, dist=0,
                      weekend_share=nw / max(nd, 1))
    # 非假日的 dist: 距最近假日(前负后正)
    hlist = sorted(hset)
    for d in all_dates:
        if d in hset: continue
        do = dt.date.fromisoformat(d)
        best = None
        for hd in hlist:
            delta = (do - dt.date.fromisoformat(hd)).days
            if best is None or abs(delta) < abs(best): best = delta
        hol[d] = dict(name='', offset=0, length=0, dist=best if best is not None else 99,
                      weekend_share=0.0)
    return hol
