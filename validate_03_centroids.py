# -*- coding: utf-8 -*-
"""流程验证: taxi_zones shapefile → 264 zone 质心(经纬度) → haversine 距离矩阵(264x264 km)
输出 .flow_zones.npz"""
import numpy as np, geopandas as gpd, zipfile, os

tmp = 'D:/AI/条件转化/data/tlc/zones/_shp/taxi_zones'
shp = tmp + '/taxi_zones.shp'
if not os.path.exists(shp):
    with zipfile.ZipFile('D:/AI/条件转化/data/tlc/zones/taxi_zones.zip') as z:
        z.extractall('D:/AI/条件转化/data/tlc/zones/_shp')
g = gpd.read_file(shp)
g = g.to_crs(4326)
cen = g.geometry.representative_point()
lat = cen.y.values; lon = cen.x.values; lid = g['LocationID'].values.astype(int)

# 填 1..263 (263 zones, id 连续)
lat263 = np.full(264, np.nan); lon263 = np.full(264, np.nan)
for i, l in enumerate(lid):
    if 1 <= l <= 263:
        lat263[l] = lat[i]; lon263[l] = lon[i]
assert not np.isnan(lat263[1:264]).any(), '有 zone 缺质心'

# haversine 距离矩阵 (km)
R = 6371.0
la = np.radians(lat263); lo = np.radians(lon263)
dla = la[:, None] - la[None, :]
dlo = lo[:, None] - lo[None, :]
a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
dist_km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
np.fill_diagonal(dist_km, 0.0)

np.savez_compressed('D:/AI/条件转化/.flow_zones.npz',
                    lat=lat263, lon=lon263, dist_km=dist_km.astype(np.float32))
print('zones:', len(g), '| dist matrix', dist_km.shape,
      '| 对角线外最小/中位/最大 km:', dist_km[dist_km > 0].min().round(2),
      np.median(dist_km[dist_km > 0]).round(2), dist_km.max().round(2))
