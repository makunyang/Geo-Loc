#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scenes.py - 在Karlsruhe知识图谱区域内随机选取100个点位，
为每个点位搜索周边建筑和POI，生成英文场景描述。

输出: scenes_100.json
"""

import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "bbox": {
        "north": 49.05,
        "south": 48.98,
        "east":  8.45,
        "west":  8.34,
    },
    "num_points": 100,
    "search_radius_m": 100,
    "seed": 42,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILDINGS_CSV = os.path.join(SCRIPT_DIR, "buildings_with_colors.csv")
POIS_CSV     = os.path.join(SCRIPT_DIR, "pois_raw.csv")
OUTPUT_JSON  = os.path.join(SCRIPT_DIR, "scenes_100.json")

# ============================================================
# 颜色翻译表（中文 -> 英文）
# ============================================================
CN_COLOR_MAP = {
    "白色": "white",
    "灰色": "gray",
    "浅灰色": "light gray",
    "深灰色": "dark gray",
    "银灰色": "silver gray",
    "黑色": "black",
    "红色": "red",
    "深红色": "dark red",
    "蓝色": "blue",
    "浅蓝色": "light blue",
    "深蓝色": "dark blue",
    "深蓝": "dark blue",
    "绿色": "green",
    "浅绿色": "light green",
    "深绿色": "dark green",
    "黄色": "yellow",
    "棕色": "brown",
    "橙色": "orange",
    "紫色": "purple",
    "紫灰": "purple-gray",
    "紫灰色": "purple-gray",
    "粉紫": "pink-purple",
    "粉色": "pink",
    "米色": "beige",
    "米白色": "off-white",
    "米白": "off-white",
    "米黄": "cream",
    "米黄色": "cream",
    "青色": "cyan",
    "金色": "gold",
    "银色": "silver",
    "铜色": "copper",
    "红棕色": "red-brown",
    "灰白色": "gray-white",
    "灰白": "gray-white",
    "黄褐色": "tan",
    "砖红色": "brick red",
    "砖色": "brick",
    "浅棕色": "light brown",
    "深棕色": "dark brown",
    "浅黄色": "light yellow",
    "土黄色": "khaki",
    "暗红色": "dark red",
    "暗灰色": "dark gray",
    "淡灰色": "pale gray",
    "淡蓝色": "pale blue",
    "淡绿色": "pale green",
    "蓝灰": "blue-gray",
    "蓝灰色": "blue-gray",
    "灰蓝": "gray-blue",
    "灰蓝色": "gray-blue",
    "灰紫": "gray-purple",
    "棕灰": "brown-gray",
    "棕灰色": "brown-gray",
    "绿灰": "green-gray",
    "绿灰色": "green-gray",
    "灰绿": "gray-green",
    "灰绿色": "gray-green",
    "红灰": "red-gray",
    "红灰色": "red-gray",
    "黄灰": "yellow-gray",
    "黄灰色": "yellow-gray",
    "深灰": "dark gray",
    "浅灰": "light gray",
    "深紫": "dark purple",
    "深紫色": "dark purple",
    "浅紫": "light purple",
    "浅蓝": "light blue",
    "浅灰蓝": "light gray-blue",
    "紫红": "purple-red",
    "紫红色": "purple-red",
    "粉红": "pink",
    "粉红色": "pink",
    "蓝紫": "blue-purple",
    "蓝紫色": "blue-purple",
    "蓝白相间": "blue and white",
    "红白相间": "red and white",
    "红棕": "red-brown",
    "红褐色": "red-brown",
    "灰褐色": "gray-brown",
    "深红": "dark red",
    "深绿": "dark green",
    "深蓝灰": "dark blue-gray",
    "深蓝灰色": "dark blue-gray",
    "深色": "dark",
    "灰": "gray",
    "灰紫": "gray-purple",
    "灰紫色": "gray-purple",
    "蓝绿": "blue-green",
    "蓝绿色": "blue-green",
    "黄绿": "yellow-green",
    "黄绿色": "yellow-green",
    "浅粉": "light pink",
    "浅粉色": "light pink",
    "橙红": "orange-red",
    "橙红色": "orange-red",
    "未知": "unknown",
    "无色": "colorless",
}


def translate_color(cn_color):
    """将中文颜色翻译为英文，未知颜色保留原文"""
    if not cn_color:
        return "unknown"
    cn_color = cn_color.strip()
    return CN_COLOR_MAP.get(cn_color, cn_color)


# ============================================================
# Haversine 距离计算 (返回米)
# ============================================================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000  # 地球半径（米）
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# 方位计算
# ============================================================
def bearing_deg(lat1, lon1, lat2, lon2):
    """计算从点1到点2的方位角（0-360度，北为0，顺时针）"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    deg = math.degrees(math.atan2(x, y))
    return (deg + 360) % 360


def angle_to_direction(angle_deg):
    """将角度转换为8方位字符串"""
    dirs = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    idx = int(((angle_deg + 22.5) % 360) / 45)
    return dirs[idx]


# ============================================================
# WKT 解析
# ============================================================
def parse_wkt_point(wkt_str):
    """解析 WKT POINT (x y) -> (lon, lat)"""
    m = re.search(r'POINT\s*\(\s*([\d.]+)\s+([\d.]+)\s*\)', str(wkt_str), re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_wkt_polygon(wkt_str):
    """
    解析 WKT POLYGON，返回外环坐标列表 [(lon, lat), ...]
    支持带孔的多边形（只取外环）
    """
    wkt = str(wkt_str).strip()
    # 提取 POLYGON (( ... ))
    m = re.search(r'POLYGON\s*\(\((.*?)\)\)', wkt, re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    # 外环以第一个 ) 结束（忽略内环）
    outer_text = m.group(1)
    # 如果有多重括号，取第一个完整的环
    # 找到第一个 ) 的位置来截取外环
    depth = 0
    end_idx = 0
    started = False
    for i, ch in enumerate(outer_text):
        if ch == '(':
            depth += 1
            started = True
        elif ch == ')':
            depth -= 1
            if started and depth == 0:
                end_idx = i
                break

    if end_idx > 0:
        ring_text = outer_text[:end_idx]
    else:
        ring_text = outer_text

    # 清理并解析坐标
    ring_text = ring_text.strip().strip('(').strip(')')
    coords = []
    for pair in ring_text.split(','):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return coords if len(coords) >= 3 else None


def point_in_polygon(px, py, polygon):
    """
    射线法判断点是否在多边形内
    polygon: [(x, y), ...]
    返回 True/False
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ============================================================
# 数据加载
# ============================================================
def load_buildings(csv_path):
    """加载建筑数据"""
    buildings = []
    print(f"Loading buildings from {csv_path} ...")
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                b = {
                    "id": row.get("id", "").strip(),
                    "wkt": row.get("WKT坐标", "").strip(),
                    "lon": float(row.get("lon", 0)),
                    "lat": float(row.get("lat", 0)),
                    "color_side": translate_color(row.get("color_side", "")),
                    "color_top": translate_color(row.get("color_top", "")),
                    # 预解析多边形坐标（用于point-in-polygon）
                    "_polygon": parse_wkt_polygon(row.get("WKT坐标", "")),
                }
                buildings.append(b)
            except (ValueError, KeyError) as e:
                continue
    print(f"  Loaded {len(buildings)} buildings.")
    return buildings


def load_pois(csv_path):
    """加载POI数据"""
    pois = []
    print(f"Loading POIs from {csv_path} ...")
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p = {
                    "code": row.get("code", "").strip(),
                    "fclass": row.get("fclass", "").strip(),
                    "name": row.get("name", "").strip(),
                    "lon": float(row.get("lon", 0)),
                    "lat": float(row.get("lat", 0)),
                }
                pois.append(p)
            except (ValueError, KeyError):
                continue
    print(f"  Loaded {len(pois)} POIs.")
    return pois


# ============================================================
# 构建空间索引（简化的网格索引）
# ============================================================
def build_grid_index(items, lon_key, lat_key, cell_size_deg=0.005):
    """
    构建网格索引以加速空间查询
    cell_size_deg: 约500m（在49度纬度处）
    """
    grid = defaultdict(list)
    for item in items:
        lon = item[lon_key]
        lat = item[lat_key]
        cell_x = int(lon / cell_size_deg)
        cell_y = int(lat / cell_size_deg)
        grid[(cell_x, cell_y)].append(item)
    return grid, cell_size_deg


def query_grid_index(grid, cell_size_deg, center_lon, center_lat, radius_m):
    """从网格索引中查询候选对象（粗略过滤）"""
    # 估算半径对应的度数
    deg_margin = (radius_m * 2) / 111000  # 留2倍余量
    min_lon = center_lon - deg_margin
    max_lon = center_lon + deg_margin
    min_lat = center_lat - deg_margin
    max_lat = center_lat + deg_margin

    min_cx = int(min_lon / cell_size_deg) - 1
    max_cx = int(max_lon / cell_size_deg) + 1
    min_cy = int(min_lat / cell_size_deg) - 1
    max_cy = int(max_lat / cell_size_deg) + 1

    candidates = []
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            candidates.extend(grid.get((cx, cy), []))
    return candidates


# ============================================================
# 按方向排序 (N, NE, E, SE, S, SW, W, NW)
# ============================================================
DIR_ORDER = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}

DIR_FULL_NAME = {
    'N': 'north', 'NE': 'northeast', 'E': 'east', 'SE': 'southeast',
    'S': 'south', 'SW': 'southwest', 'W': 'west', 'NW': 'northwest',
}


def sort_by_direction(items):
    """按8方位顺序排序"""
    return sorted(items, key=lambda x: DIR_ORDER.get(x[0], 99))


# ============================================================
# 生成英文描述
# ============================================================
def generate_description(point_lat, point_lon, grouped_buildings):
    """
    按方向分组生成英文描述
    grouped_buildings: {direction: [(building_info, distance_m), ...]}
    """
    parts = []
    for direction in sort_by_direction(grouped_buildings.items()):
        dir_name = direction[0]
        for bldg_info, dist_m in direction[1]:
            color_side = bldg_info.get("color_side", "unknown")
            color_top = bldg_info.get("color_top", "unknown")
            pois = bldg_info.get("_pois", [])
            dist_int = int(round(dist_m))

            if color_side == color_top:
                bldg_desc = f"a fully {color_side} building"
            else:
                bldg_desc = f"a building with {color_side} sides and {color_top} roof"

            poi_str = ""
            if pois:
                poi_types = [p["fclass"].replace("_", " ") for p in pois]
                seen = set()
                unique_poi_types = []
                for pt in poi_types:
                    if pt not in seen:
                        seen.add(pt)
                        unique_poi_types.append(pt)
                if len(unique_poi_types) == 1:
                    poi_str = f", containing a {unique_poi_types[0]}"
                elif len(unique_poi_types) == 2:
                    poi_str = f", containing a {unique_poi_types[0]} and a {unique_poi_types[1]}"
                else:
                    poi_str = f", containing a {', a '.join(unique_poi_types[:-1])}, and a {unique_poi_types[-1]}"

            dir_full = DIR_FULL_NAME.get(dir_name, dir_name.lower())
            parts.append(
                f"To the {dir_full}, there is {bldg_desc}{poi_str}, "
                f"about {dist_int} meters away."
            )

    # 如果附近没有建筑
    if not parts:
        return "I am on an east-west road. There are no buildings within 100 meters."

    # 拼接
    desc = "I am on an east-west road. " + " ".join(parts)
    return desc


# ============================================================
# 主流程
# ============================================================
def main():
    random.seed(CONFIG["seed"])

    # Step 0: 加载数据
    buildings = load_buildings(BUILDINGS_CSV)
    pois = load_pois(POIS_CSV)

    # 构建空间索引
    print("Building spatial index for buildings ...")
    bldg_grid, bldg_cell = build_grid_index(buildings, "lon", "lat")
    print("Building spatial index for POIs ...")
    poi_grid, poi_cell = build_grid_index(pois, "lon", "lat")

    # Step 1: 生成100个随机点位
    print(f"\nGenerating {CONFIG['num_points']} random points ...")
    points = []
    for i in range(CONFIG["num_points"]):
        lat = random.uniform(CONFIG["bbox"]["south"], CONFIG["bbox"]["north"])
        lon = random.uniform(CONFIG["bbox"]["west"], CONFIG["bbox"]["east"])
        points.append({"id": i, "lon": lon, "lat": lat})

    # Step 2-4: 处理每个点位
    print("Processing points ...")
    results = []
    radius = CONFIG["search_radius_m"]

    for idx, pt in enumerate(points):
        if (idx + 1) % 10 == 0:
            print(f"  Processing point {idx + 1}/{CONFIG['num_points']} ...")

        plat, plon = pt["lat"], pt["lon"]

        # 2a: 搜索周边建筑
        nearby_buildings = []
        candidates_b = query_grid_index(bldg_grid, bldg_cell, plon, plat, radius)
        for b in candidates_b:
            dist = haversine_m(plat, plon, b["lat"], b["lon"])
            if dist < radius:
                nearby_buildings.append((b, dist))

        # 2b: 搜索周边POI
        nearby_pois = []
        candidates_p = query_grid_index(poi_grid, poi_cell, plon, plat, radius)
        for p in candidates_p:
            dist = haversine_m(plat, plon, p["lat"], p["lon"])
            if dist < radius:
                nearby_pois.append((p, dist))

        # 2c: 将POI匹配到建筑（点是否在建筑多边形内）
        # 为每个建筑初始化 _pois 列表
        for b, _ in nearby_buildings:
            if "_pois" not in b:
                b["_pois"] = []

        for p, pdist in nearby_pois:
            matched = False
            for b, _ in nearby_buildings:
                poly = b.get("_polygon")
                if poly and point_in_polygon(p["lon"], p["lat"], poly):
                    b.setdefault("_pois", []).append(p)
                    matched = True
                    break
            # 如果没有匹配到任何建筑，POI仍保留但关联到最近建筑
            if not matched and nearby_buildings:
                # 找到最近的建筑
                nearest_b = min(nearby_buildings, key=lambda x: x[1])
                nearest_b[0].setdefault("_pois", []).append(p)

        # Step 3: 计算方位并分组
        direction_groups = defaultdict(list)
        nearby_bldg_info = []

        for b, dist in nearby_buildings:
            bearing = bearing_deg(plat, plon, b["lat"], b["lon"])
            direction = angle_to_direction(bearing)
            direction_groups[direction].append((b, dist))

            # 构建 nearby_buildings 输出
            poi_list = b.get("_pois", [])
            nearby_bldg_info.append({
                "id": b["id"],
                "distance_m": round(dist, 1),
                "direction": direction,
                "color_side": b.get("color_side", ""),
                "color_top": b.get("color_top", ""),
                "pois": [
                    {"fclass": p["fclass"], "name": p["name"]}
                    for p in poi_list
                ],
            })

        # Step 4: 生成英文描述
        description = generate_description(plat, plon, direction_groups)

        # 组装结果
        results.append({
            "id": pt["id"],
            "lon": round(pt["lon"], 6),
            "lat": round(pt["lat"], 6),
            "ground_truth_lon": round(pt["lon"], 6),
            "ground_truth_lat": round(pt["lat"], 6),
            "description": description,
            "nearby_buildings": nearby_bldg_info,
        })

    # Step 5: 输出JSON
    output = {
        "config": {
            "bbox": CONFIG["bbox"],
            "num_points": CONFIG["num_points"],
            "search_radius_m": CONFIG["search_radius_m"],
            "seed": CONFIG["seed"],
        },
        "points": results,
    }

    print(f"\nWriting output to {OUTPUT_JSON} ...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 统计信息
    total_nearby = sum(len(p["nearby_buildings"]) for p in results)
    pts_with_bldg = sum(1 for p in results if len(p["nearby_buildings"]) > 0)
    print(f"\nDone!")
    print(f"  Total points: {len(results)}")
    print(f"  Points with at least one nearby building: {pts_with_bldg}/{len(results)}")
    print(f"  Average buildings per point: {total_nearby / len(results):.1f}")
    print(f"  Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
