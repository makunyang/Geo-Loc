#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scenes_llm.py - Neo4j 驱动：100个随机点位场景描述生成。

流程：
1. 随机生成点位 → 检查是否在建筑内部（用 buildings_with_colors.csv 的 WKT 多边形）
2. Neo4j 查询点位周边 30m 的建筑 + 内部 POI
3. 按方向分组，POI 多的优先保留（最多 8 栋）
4. LLM 生成 / 模板回退描述

输出: scenes_100_llm.json
"""

import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

from openai import OpenAI
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon


# ============================================================
# Neo4j
# ============================================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j"

try:
    from neo4j import GraphDatabase
    _neo4j_available = True
except ImportError:
    _neo4j_available = False
    print("WARNING: neo4j 未安装，将以 CSV 模式运行")


class Neo4jClient:
    """Neo4j 查询客户端（轻量版）"""
    def __init__(self):
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD))
        return self._driver

    def query(self, cql, params=None):
        with self.driver.session() as session:
            result = session.run(cql, params or {})
            return [record.data() for record in result]

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None


# ============================================================
# 配置
# ============================================================
CONFIG = {
    "bbox": {
        "north": 49.05,
        "south": 48.98,
        "east": 8.45,
        "west": 8.34,
    },
    "num_points": 100,
    "search_radius_m": 30,
    "max_buildings": 8,
    "seed": None,  # None = 随机种子
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILDINGS_CSV = os.path.join(SCRIPT_DIR, "buildings_with_colors.csv")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "scenes_100_llm.json")

# LLM API
OPENAI_API_KEY = "8O9vGvq0gQ93aaSS2f2WvzkPuP8qNrBxdRKsUKJXCeXa4toN"
OPENAI_BASE_URL = "https://www.autodl.art/api/v1"
GPT_MODEL = "DeepSeek-V4-Flash"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ============================================================
# 颜色翻译
# ============================================================
CN_COLOR_MAP = {
    "白色": "white", "灰色": "gray", "浅灰色": "light gray", "深灰色": "dark gray",
    "银灰色": "silver gray", "黑色": "black", "红色": "red", "深红色": "dark red",
    "蓝色": "blue", "浅蓝色": "light blue", "深蓝色": "dark blue", "深蓝": "dark blue",
    "绿色": "green", "浅绿色": "light green", "深绿色": "dark green",
    "黄色": "yellow", "棕色": "brown", "橙色": "orange",
    "紫色": "purple", "紫灰": "purple-gray", "紫灰色": "purple-gray",
    "粉紫": "pink-purple", "粉色": "pink", "米色": "beige",
    "米白色": "off-white", "米白": "off-white", "米黄": "cream", "米黄色": "cream",
    "青色": "cyan", "金色": "gold", "银色": "silver", "铜色": "copper",
    "红棕色": "red-brown", "灰白色": "gray-white", "灰白": "gray-white",
    "黄褐色": "tan", "砖红色": "brick red", "砖色": "brick",
    "浅棕色": "light brown", "深棕色": "dark brown",
    "浅黄色": "light yellow", "土黄色": "khaki",
    "暗红色": "dark red", "暗灰色": "dark gray",
    "淡灰色": "pale gray", "淡蓝色": "pale blue", "淡绿色": "pale green",
    "蓝灰": "blue-gray", "蓝灰色": "blue-gray", "灰蓝": "gray-blue", "灰蓝色": "gray-blue",
    "灰紫": "gray-purple", "棕灰": "brown-gray", "棕灰色": "brown-gray",
    "绿灰": "green-gray", "绿灰色": "green-gray", "灰绿": "gray-green", "灰绿色": "gray-green",
    "红灰": "red-gray", "红灰色": "red-gray", "黄灰": "yellow-gray", "黄灰色": "yellow-gray",
    "深灰": "dark gray", "浅灰": "light gray",
    "深紫": "dark purple", "深紫色": "dark purple",
    "浅紫": "light purple", "浅蓝": "light blue", "浅灰蓝": "light gray-blue",
    "紫红": "purple-red", "紫红色": "purple-red",
    "粉红": "pink", "粉红色": "pink",
    "蓝紫": "blue-purple", "蓝紫色": "blue-purple",
    "蓝白相间": "blue and white", "红白相间": "red and white",
    "红棕": "red-brown", "红褐色": "red-brown",
    "灰褐色": "gray-brown", "深红": "dark red", "深绿": "dark green",
    "深蓝灰": "dark blue-gray", "深蓝灰色": "dark blue-gray",
    "深色": "dark", "灰": "gray",
    "灰紫": "gray-purple", "灰紫色": "gray-purple",
    "蓝绿": "blue-green", "蓝绿色": "blue-green",
    "黄绿": "yellow-green", "黄绿色": "yellow-green",
    "浅粉": "light pink", "浅粉色": "light pink",
    "橙红": "orange-red", "橙红色": "orange-red",
    "未知": "unknown", "无色": "colorless",
}

DIR_FULL_NAME = {
    "N": "north", "NE": "northeast", "E": "east", "SE": "southeast",
    "S": "south", "SW": "southwest", "W": "west", "NW": "northwest",
}


# ============================================================
# 工具函数
# ============================================================
def translate_color(cn_color):
    if not cn_color:
        return "unknown"
    return CN_COLOR_MAP.get(cn_color.strip(), cn_color.strip())


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_to_direction(angle_deg):
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((angle_deg + 22.5) % 360) / 45)
    return dirs[idx]


def parse_wkt_polygon(wkt_str):
    """解析 WKT Polygon → shapely Polygon 对象"""
    if not wkt_str or not isinstance(wkt_str, str):
        return None
    try:
        geom = wkt_loads(wkt_str)
        if geom.geom_type == "Polygon":
            return geom
        elif geom.geom_type == "MultiPolygon":
            return geom.geoms[0]
        return None
    except Exception:
        return None


# ============================================================
# 建筑描述（严格区分 side/top，POI 用原始 fclass）
# ============================================================
def describe_building_colors(cs_cn, ct_cn):
    """
    - 不同色: "with a dark blue side and light gray roof"
    - 同色: "with both sides and roof in gray"
    """
    cs = translate_color(cs_cn)
    ct = translate_color(ct_cn)
    if cs.lower() == ct.lower():
        return f"with both sides and roof in {cs}"
    else:
        return f"with a {cs} side and {ct} roof"


def describe_building_pois_text(poi_fclasses):
    """
    POI 用原始 fclass（保留下划线）。
    ["fast_food", "doctors"] → "a fast_food and a doctors"
    """
    if not poi_fclasses:
        return ""
    uniq = list(dict.fromkeys(poi_fclasses))  # 去重保序
    if len(uniq) == 1:
        return f", inside which there is a {uniq[0]}"
    return (", inside which there are " +
            ", a ".join(uniq[:-1]) + f", and a {uniq[-1]}")


# ============================================================
# Neo4j 查询：点位周边 30m 建筑 + 内部 POI
# ============================================================
def query_nearby_buildings_neo4j(neo, plat, plon, radius_m):
    """
    查询 (plat, plon) 周边 radius_m 内的所有建筑，
    以及每个建筑内部的 POI（通过 INSIDE 关系）。
    返回: [(building_dict, distance_m), ...]
    """
    deg = radius_m / 111320.0  # 纬度每度约 111.32 km
    lat_min, lat_max = plat - deg * 1.5, plat + deg * 1.5
    lon_min, lon_max = plon - deg * 1.5, plon + deg * 1.5

    cql = """
    MATCH (b:Building)
    WHERE b.lat >= $lat_min AND b.lat <= $lat_max
      AND b.lon >= $lon_min AND b.lon <= $lon_max
    OPTIONAL MATCH (b)-[:INSIDE]-(p:POI)
    RETURN b.id AS id, b.lon AS lon, b.lat AS lat,
           b.color_side AS color_side, b.color_top AS color_top,
           collect(DISTINCT {fclass: p.fclass, name: p.name}) AS pois
    """
    rows = neo.query(cql, {
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max})

    results = []
    for r in rows:
        dist = haversine_m(plat, plon, r["lat"], r["lon"])
        if dist <= radius_m:
            # 过滤掉 None POI
            pois = [p for p in (r.get("pois") or []) if p.get("fclass")]
            results.append(({
                "id": r["id"],
                "lon": r["lon"],
                "lat": r["lat"],
                "color_side": r.get("color_side", ""),
                "color_top": r.get("color_top", ""),
                "_pois": pois,
            }, dist))
    return results


# ============================================================
# 点位验证：是否在建筑内部
# ============================================================
def load_building_polygons(csv_path):
    """加载 all 建筑的 WKT 多边形（用于点位验证）"""
    polys = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            wkt = row.get("WKT坐标", row.get("WKT", ""))
            poly = parse_wkt_polygon(wkt)
            if poly:
                polys.append(poly)
    print(f"Loaded {len(polys)} building polygons for point validation")
    return polys


def is_point_inside_any_building(lat, lon, polygons):
    """检查点 (lat, lon) 是否在任何建筑多边形内"""
    pt = ShapelyPoint(lon, lat)
    for poly in polygons:
        if poly.contains(pt):
            return True
    return False


# ============================================================
# 结构化上下文（供 LLM）
# ============================================================
def build_structured_context(direction_groups):
    lines = []
    lines.append("Buildings around the user (direction/distance from user):")
    lines.append("")
    all_buildings = []
    for dir_name, items in direction_groups.items():
        for b, dist in items:
            all_buildings.append((dir_name, b, dist))
    all_buildings.sort(key=lambda x: x[2])

    for dir_name, b, dist in all_buildings:
        dist_int = int(round(dist))
        dir_full = DIR_FULL_NAME.get(dir_name, dir_name.lower())
        color_desc = describe_building_colors(
            b.get("color_side", ""), b.get("color_top", ""))
        poi_fcs = [p["fclass"] for p in b.get("_pois", [])]
        poi_text = describe_building_pois_text(poi_fcs)
        lines.append(
            f"  {b['id']}: {dir_full} {dist_int}m, {color_desc}{poi_text}")
    return "\n".join(lines)


# ============================================================
# LLM Prompt
# ============================================================
SYSTEM_PROMPT = (
    "You generate natural, first-person urban scene descriptions for geo-localization.\n\n"
    "CRITICAL RULES:\n\n"
    "1. Describe buildings relative to YOU. Use phrasing like:\n"
    "   'I am approximately 13 meters north of a building...'\n"
    "   'Twenty meters to my north is another building...'\n"
    "   'Thirty-five meters northeast of me stands a building...'\n\n"
    "2. Colors: COPY EXACTLY from the input:\n"
    "   - 'with a {color} side and {color} roof' (different colors)\n"
    "   - 'with both sides and roof in {color}' (same colors)\n"
    "   NEVER rephrase or merge colors.\n\n"
    "3. POI types: COPY EXACTLY with underscores. 'fast_food' NOT 'fast food'.\n"
    "   Use varied phrasing for POIs: 'inside which there is a...',\n"
    "   'containing a...', 'housing a...', 'which has a...'.\n\n"
    "4. Output ONLY the description — no markdown, no JSON, no labels.\n\n"
    "5. Describe EVERY building in its own clause or sentence.\n"
    "6. Vary sentence openers naturally.\n"
    "7. Describe buildings in the order they appear — do not reorder them."
)


def generate_description_llm(structured_context):
    user_prompt = (
        "Here are buildings around a person:\n\n"
        f"{structured_context}\n\n"
        "Generate a natural first-person scene description following ALL rules."
    )
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    LLM API error: {e}")
        return None


# ============================================================
# 模板回退（精确格式）
# ============================================================
def generate_description_template(direction_groups):
    """模板生成描述（严格 follow requested format）"""
    all_buildings = []
    for dir_name, items in direction_groups.items():
        for b, dist in items:
            all_buildings.append((dir_name, b, dist))
    all_buildings.sort(key=lambda x: x[2])  # 按距离

    parts = []
    for i, (dir_name, b, dist) in enumerate(all_buildings):
        dist_int = int(round(dist))
        dir_full = DIR_FULL_NAME.get(dir_name, dir_name.lower())
        color_desc = describe_building_colors(
            b.get("color_side", ""), b.get("color_top", ""))
        poi_fcs = [p["fclass"] for p in b.get("_pois", [])]
        poi_text = describe_building_pois_text(poi_fcs)

        if i == 0:
            # first building: "I am approximately X meters N of a building..."
            part = (f"I am approximately {dist_int} meters {dir_full} of "
                    f"a building {color_desc}{poi_text}.")
        else:
            # subsequent buildings: "Twenty meters to my N is another building..."
            # Distance in words
            dist_words = str(dist_int)
            part = (f"{dist_words.capitalize()} meters to my {dir_full} "
                    f"{'stands' if i % 2 == 0 else 'is'} a building "
                    f"{color_desc}{poi_text}.")
        parts.append(part)

    return " ".join(parts)


# ============================================================
# 主流程
# ============================================================
def main():
    seed = CONFIG["seed"]
    if seed is not None:
        random.seed(seed)
    else:
        seed = random.randint(1, 999999)
        random.seed(seed)
        CONFIG["seed"] = seed

    print(f"LLM model: {GPT_MODEL}")
    print(f"Seed: {seed}, Points: {CONFIG['num_points']}, Radius: {CONFIG['search_radius_m']}m")

    # ── 加载建筑多边形（用于点位验证） ──
    building_polys = load_building_polygons(BUILDINGS_CSV)

    # ── Neo4j ──
    if not _neo4j_available:
        print("FATAL: neo4j is required. Install with: pip install neo4j")
        sys.exit(1)
    neo = Neo4jClient()
    print("Neo4j connected")

    # ── 生成 100 个有效点位 ──
    radius = CONFIG["search_radius_m"]
    results = []
    errors = 0
    attempts = 0
    max_attempts = CONFIG["num_points"] * 20  # 防止死循环

    print(f"\nGenerating {CONFIG['num_points']} valid points...")

    pt_idx = 0
    while pt_idx < CONFIG["num_points"] and attempts < max_attempts:
        attempts += 1
        lat = random.uniform(CONFIG["bbox"]["south"], CONFIG["bbox"]["north"])
        lon = random.uniform(CONFIG["bbox"]["west"], CONFIG["bbox"]["east"])

        # 检查是否在建筑内部
        if is_point_inside_any_building(lat, lon, building_polys):
            continue  # 作废，重试

        # ── Neo4j 查询周边建筑 ──
        nearby = query_nearby_buildings_neo4j(neo, lat, lon, radius)
        if not nearby:
            continue  # 周边无建筑，跳过

        # ── 方位分组 ──
        dir_groups_raw = defaultdict(list)
        for b, dist in nearby:
            bearing = bearing_deg(lat, lon, b["lat"], b["lon"])
            d = angle_to_direction(bearing)
            dir_groups_raw[d].append((b, dist))

        # 每个方向只取最近的
        dir_groups = {}
        for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            if d in dir_groups_raw:
                nearest = min(dir_groups_raw[d], key=lambda x: x[1])
                dir_groups[d] = [nearest]

        # ── POI 优先保留（最多 8 栋） ──
        all_flat = []
        for d, items in dir_groups.items():
            for b, dist in items:
                all_flat.append((d, b, dist))
        all_flat.sort(key=lambda x: x[2])

        max_b = CONFIG["max_buildings"]
        if len(all_flat) > max_b:
            with_poi = [(d, b, dist) for d, b, dist in all_flat if b.get("_pois")]
            without_poi = [(d, b, dist) for d, b, dist in all_flat if not b.get("_pois")]
            selected = with_poi[:max_b]
            if len(selected) < max_b:
                selected += without_poi[:(max_b - len(selected))]
            keep_ids = {x[1]["id"] for x in selected}
            dir_groups = {}
            for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                filtered = [(b, dist) for b, dist in dir_groups_raw[d]
                            if b["id"] in keep_ids]
                if filtered:
                    filtered.sort(key=lambda x: x[1])
                    dir_groups[d] = [filtered[0]]

        num_selected = sum(len(v) for v in dir_groups.values())
        if num_selected == 0:
            continue

        total_pois = sum(len(b.get("_pois", []))
                         for items in dir_groups.values() for b, _ in items)

        # ── 构建 nearby_buildings 输出 ──
        nearby_info = []
        for d in dir_groups:
            for b, dist in dir_groups[d]:
                nearby_info.append({
                    "id": b["id"],
                    "distance_m": round(dist, 1),
                    "direction": d,
                    "color_side": b.get("color_side", ""),
                    "color_top": b.get("color_top", ""),
                    "pois": [{"fclass": p["fclass"], "name": p.get("name","")}
                             for p in b.get("_pois", [])],
                })

        # ── 生成描述 ──
        structured_ctx = build_structured_context(dir_groups)

        print(f"  [{pt_idx + 1}/{CONFIG['num_points']}] "
              f"lon={lon:.4f} lat={lat:.4f}, "
              f"bld={num_selected}/{len(nearby)}, poi={total_pois}", end="")

        description = generate_description_llm(structured_ctx)
        if not description:
            description = generate_description_template(dir_groups)
            errors += 1
            print("  -> LLM FAILED, using template fallback")
        else:
            print("  -> OK")

        results.append({
            "id": pt_idx,
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "ground_truth_lon": round(lon, 6),
            "ground_truth_lat": round(lat, 6),
            "description": description,
            "nearby_buildings": nearby_info,
        })
        pt_idx += 1

        if pt_idx % 10 == 0:
            time.sleep(0.5)

    neo.close()

    # ── 输出 ──
    output = {
        "config": {
            "bbox": CONFIG["bbox"],
            "num_points": CONFIG["num_points"],
            "search_radius_m": CONFIG["search_radius_m"],
            "max_buildings": CONFIG["max_buildings"],
            "seed": CONFIG["seed"],
            "generation_method": "LLM+Neo4j",
            "model": GPT_MODEL,
        },
        "points": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(results)} points → {OUTPUT_JSON}")
    print(f"LLM errors: {errors}, Total attempts: {attempts}")


if __name__ == "__main__":
    main()
