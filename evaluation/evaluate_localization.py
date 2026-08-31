"""
============================================================
自然语言地理定位批量评估脚本
============================================================

功能：读取场景描述JSON文件，对N个点位逐一执行自然语言地理定位，
      汇总评估指标并输出JSON结果。

知识图谱适配说明：
--------------------
本脚本支持两种知识图谱类型，通过 KG_TYPE 配置切换：

1. KG_TYPE = "delaunay" (Delaunay三角网图谱)
   - Building-Building 关系: DELAUNAY (属性: direction, distance_m, angle_deg)
   - Building-POI 关系: INSIDE / NEAR

2. KG_TYPE = "buffer" (缓冲区图谱)
   - Building-Building 关系: BUFFER_NEAR (属性: direction, distance_m)
   - Building-POI 关系: INSIDE / NEAR
   - 切换时需修改 KGMatcher 中 Neo4j 查询的关系类型: DELAUNAY -> BUFFER_NEAR

关联文件：
- 场景描述: d:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\scenes_100.json
- 定位逻辑源自: d:\TTT\GeoKG\Helsinki data\test1\geolocalization_delaunay2.py
============================================================
"""

import os
import json
import math
import time
import warnings
KG_SHOW_RESULTS = True  # 是否打印 Neo4j 查询结果（匹配到的实体ID）

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

from neo4j import GraphDatabase
from openai import OpenAI, APIConnectionError, APITimeoutError

warnings.filterwarnings("ignore")

# ===================== 1. 配置参数（用户可修改） =====================

SCENES_JSON = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\scenes_100_llm.json"
OUTPUT_JSON = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe_test\evaluation_results.json"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j"

OPENAI_API_KEY = "8O9vGvq0gQ93aaSS2f2WvzkPuP8qNrBxdRKsUKJXCeXa4toN"
OPENAI_BASE_URL = "https://www.autodl.art/api/v1"
GPT_MODEL = "DeepSeek-V4-Flash"

KG_TYPE = "buffer"  # "delaunay" 或 "buffer"

# 根据 KG_TYPE 自动确定 Neo4j 关系类型
if KG_TYPE == "buffer":
    BLD_BLD_REL = "CONTACT"   # 缓冲区KG的建筑-建筑关系类型
else:
    BLD_BLD_REL = "DELAUNAY"  # Delaunay KG的建筑-建筑关系类型

EVAL_DISTANCE_THRESHOLDS = [5, 10, 15, 20, 25]

# ===================== 2. 距离约束配置 =====================

MAX_NEIGHBOR_DISTANCE = 200
MAX_2HOP_DISTANCE = 300
DISTANCE_TOLERANCE_RATIO = 0.5
KNN_FALLBACK_COUNT = 20
KNN_FALLBACK_MAX_DISTANCE = 500

# ===================== 3. 方位容差配置 =====================

DIRECTION_TOLERANCE_ANGLES = {
    "E": [0, 45], "NE": [22.5, 67.5], "N": [45, 135], "NW": [112.5, 157.5],
    "W": [135, 225], "SW": [202.5, 247.5], "S": [225, 315], "SE": [292.5, 337.5],
}

DIRECTION_MULTI_MAP = {
    "E": ["E", "NE", "SE"], "NE": ["NE", "E", "N"], "N": ["N", "NE", "NW"],
    "NW": ["NW", "N", "W"], "W": ["W", "NW", "SW"], "SW": ["SW", "W", "S"],
    "S": ["S", "SW", "SE"], "SE": ["SE", "S", "E"],
}

# ===================== 4. POI类型映射（中文 -> fclass） =====================

POI_TYPE_MAP = {
    # 餐饮
    "快餐店": "fast_food", "快餐": "fast_food",
    "餐厅": "restaurant", "饭馆": "restaurant",
    "咖啡厅": "cafe", "咖啡店": "cafe",
    "酒吧": "pub", "酒馆": "bar", "面包店": "bakery",
    # 零售
    "超市": "supermarket", "便利店": "convenience",
    "服装店": "clothes", "书店": "books",
    "洗衣店": "laundry", "干洗店": "laundry",
    "当铺": "pawnbroker", "典当行": "pawnbroker",
    # 服务
    "银行": "bank", "自动取款机": "atm", "取款机": "atm",
    "药店": "pharmacy", "理发店": "hairdresser", "美容院": "beauty",
    "酒店": "hotel", "医院": "hospital",
    "牙医诊所": "dentist", "牙医": "dentist",
    "诊所": "doctors", "医生": "doctors",
    # 公共设施
    "学校": "school", "幼儿园": "kindergarten",
    "电影院": "cinema", "剧院": "theatre",
    "邮箱": "post_box", "邮局": "post_office",
    "图书馆": "library",
    # 运动/休闲
    "健身房": "gym", "运动中心": "sports_centre",
    "游泳池": "swimming_pool",
    "俱乐部": "club", "俱乐部会所": "club",
     # 数据库中有 nightclub/clubhouse，CONTAINS fallback 会匹配
    # 交通
    "加油站": "fuel", "充电站": "charging_station",
    "停车场": "parking",
    "共享汽车": "car_sharing",
    # 其他
    "大使馆": "embassy", "教堂": "place_of_worship",
    "社区中心": "community_centre",
    "监控": "camera_surveillance", "摄像头": "camera_surveillance",
    "自动售货机": "vending_machine", "vending_machine": "vending_machine",
    "vending_parking": "vending_parking",
    "户外用品店": "outdoor_shop", "运动用品店": "sports_shop",
    "厕所": "toilets", "卫生间": "toilets",
    "药店": "pharmacy",
    "公共设备": "social_facility",
    "数据中心": "data_centre",
    "轮胎店": "tyres",
    "物资仓库": "post_depot",
    "出版社": "bookmaker"
}

# ===================== 5. 基础常量和工具函数 =====================

EARTH_RADIUS = 6371000
DIRECTIONS_8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
DIRECTION_ANGLE = {
    "E": 0, "NE": 45, "N": 90, "NW": 135,
    "W": 180, "SW": 225, "S": 270, "SE": 315
}
OPPOSITE_DIRECTION = {
    "N": "S", "S": "N", "E": "W", "W": "E",
    "NE": "SW", "SW": "NE", "NW": "SE", "SE": "NW"
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的球面距离（米）"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS * c


def direction_to_offset(direction, distance_meters, lat=60.0):
    """将方位和距离转换为经纬度偏移"""
    angle = DIRECTION_ANGLE.get(direction, 0)
    angle_rad = math.radians(angle)
    meters_per_deg_lon = 111000 * math.cos(math.radians(lat))
    meters_per_deg_lat = 111000
    dlon = (distance_meters * math.cos(angle_rad)) / meters_per_deg_lon
    dlat = (distance_meters * math.sin(angle_rad)) / meters_per_deg_lat
    return dlon, dlat


def calculate_relative_direction(dir1, dir2):
    """计算两个实体相对于彼此的方位"""
    if not dir1 or not dir2:
        return ""
    angle1 = DIRECTION_ANGLE.get(dir1, 0)
    angle2 = DIRECTION_ANGLE.get(dir2, 0)
    relative_angle = (angle2 - angle1 + 360) % 360
    idx = int(((relative_angle + 22.5) % 360) / 45)
    return DIRECTIONS_8[idx]


def compute_angle_deg(lon1, lat1, lon2, lat2):
    """计算从点1到点2的角度（度）"""
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    angle = math.degrees(math.atan2(dlat, dlon))
    return (angle + 360) % 360


def compute_direction_8(angle_deg):
    """将角度转换为8方位"""
    idx = int(((angle_deg + 22.5) % 360) / 45)
    return DIRECTIONS_8[idx]


def calculate_possible_directions(dir1, dist1, dir2, dist2):
    """根据两个实体相对于用户的方位和距离，计算它们之间可能的相对方位"""
    if not dir1 or not dir2 or dist1 <= 0 or dist2 <= 0:
        return [calculate_relative_direction(dir1, dir2)]
    angle1 = DIRECTION_ANGLE.get(dir1, 0)
    angle2 = DIRECTION_ANGLE.get(dir2, 0)
    x1 = dist1 * math.cos(math.radians(angle1))
    y1 = dist1 * math.sin(math.radians(angle1))
    x2 = dist2 * math.cos(math.radians(angle2))
    y2 = dist2 * math.sin(math.radians(angle2))
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    angle = (angle + 360) % 360
    main_idx = int(((angle + 22.5) % 360) / 45)
    main_dir = DIRECTIONS_8[main_idx]
    dist_ratio = max(dist1, dist2) / min(dist1, dist2) if min(dist1, dist2) > 0 else 1
    if dist_ratio < 1.5:
        tolerance = 1
    elif dist_ratio < 3:
        tolerance = 2
    else:
        tolerance = 3
    possible_dirs = []
    for i in range(-tolerance, tolerance + 1):
        idx = (main_idx + i) % 8
        possible_dirs.append(DIRECTIONS_8[idx])
    possible_dirs.remove(main_dir)
    possible_dirs.insert(0, main_dir)
    return possible_dirs


def resolve_chain_entities(entities):
    """
    递推解析链式描述的实体，将 relative_to 转换为 direction_from_user
    返回: True 表示全部解析成功
    """
    positions = {}
    for e in entities:
        if e.direction_from_user and e.estimated_distance and e.estimated_distance > 0:
            angle = math.radians(DIRECTION_ANGLE.get(e.direction_from_user, 0))
            x = e.estimated_distance * math.cos(angle)
            y = e.estimated_distance * math.sin(angle)
            positions[e.entity_id] = (x, y)
            e.resolved = True

    max_iterations = len(entities) * 2
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        any_resolved = False
        for e in entities:
            if e.resolved:
                continue
            if not e.relative_to or not e.direction_from_ref:
                continue
            ref_id = e.relative_to
            if ref_id not in positions:
                continue
            ref_x, ref_y = positions[ref_id]
            chain_angle = math.radians(DIRECTION_ANGLE.get(e.direction_from_ref, 0))
            chain_dx = e.distance_from_ref * math.cos(chain_angle)
            chain_dy = e.distance_from_ref * math.sin(chain_angle)
            new_x = ref_x + chain_dx
            new_y = ref_y + chain_dy
            new_dist = math.sqrt(new_x ** 2 + new_y ** 2)
            new_angle = math.degrees(math.atan2(new_y, new_x))
            new_angle = (new_angle + 360) % 360
            idx = int(((new_angle + 22.5) % 360) / 45)
            e.direction_from_user = DIRECTIONS_8[idx]
            e.estimated_distance = round(new_dist, 1)
            positions[e.entity_id] = (new_x, new_y)
            e.resolved = True
            any_resolved = True
        if not any_resolved:
            break

    unresolved = [e.entity_id for e in entities if not e.resolved and not e.direction_from_user]
    if unresolved:
        for e in entities:
            if not e.resolved and not e.direction_from_user:
                e.direction_from_user = ""
                e.estimated_distance = 0.0
    return len(unresolved) == 0


def normalize_poi_type(poi_type):
    """将POI类型转换为fclass。支持中文映射表 + 英文fclass直接透传。"""
    if isinstance(poi_type, list):
        poi_type = poi_type[0] if len(poi_type) > 0 else ""
        return ""
    poi_type = poi_type.strip()
    # 1. 查中文映射表
    if poi_type in POI_TYPE_MAP:
        return POI_TYPE_MAP[poi_type]
    # 2. 纯ASCII → 直接作为fclass透传（如LLM输出的"restaurant"/"laundry"等）
    if poi_type.isascii():
        return poi_type
    # 3. 中文但不在映射表中 → 尝试fuzzy match（下划线→空格）
    # 这种情况极少，返回原值让CONTAINS查
    return poi_type


# ===================== 6. 数据类 =====================

@dataclass
class SpatialEntity:
    """空间实体"""
    entity_id: str = ""
    entity_type: str = "Building"
    lon: float = 0.0
    lat: float = 0.0
    color_side: str = ""
    color_top: str = ""
    fclass: str = ""
    poi_name: str = ""
    road_type: str = ""
    road_orientation: str = ""
    direction_from_user: str = ""
    estimated_distance: float = 0.0
    relative_to: str = ""
    direction_from_ref: str = ""
    distance_from_ref: float = 0.0
    resolved: bool = False
    associated_poi: Any = field(default_factory=dict)
    possible_relative_directions: Dict[str, List[str]] = field(default_factory=dict)

    def _get_poi_dict(self):
        if isinstance(self.associated_poi, dict):
            return self.associated_poi
        elif isinstance(self.associated_poi, list) and len(self.associated_poi) > 0:
            return self.associated_poi[0]
        return None

    def _get_all_poi_dicts(self):
        if isinstance(self.associated_poi, dict):
            return [self.associated_poi]
        elif isinstance(self.associated_poi, list):
            return self.associated_poi
        return []

    def has_poi_constraint(self):
        poi_dict = self._get_poi_dict()
        return bool(poi_dict and poi_dict.get("poi_type"))

    def get_poi_type(self):
        poi_dict = self._get_poi_dict()
        if poi_dict and poi_dict.get("poi_type"):
            return normalize_poi_type(poi_dict["poi_type"])
        return ""

    def get_all_poi_types(self):
        return [normalize_poi_type(p.get("poi_type", ""))
                for p in self._get_all_poi_dicts() if p.get("poi_type")]

    def get_first_poi_type(self):
        all_types = self.get_all_poi_types()
        return all_types[0] if all_types else ""


@dataclass
class MatchedEntity:
    """匹配到的实体"""
    query_id: str = ""
    entity_id: str = ""
    entity_type: str = ""
    lon: float = 0.0
    lat: float = 0.0
    color_side: str = ""
    color_top: str = ""
    fclass: str = ""
    poi_name: str = ""
    poi_validation: bool = False
    required_poi_type: str = ""
    required_poi_types: List[str] = field(default_factory=list)
    matched_poi_names: List[str] = field(default_factory=list)


@dataclass
class CandidateCombination:
    """候选组合"""
    entities: List[MatchedEntity] = field(default_factory=list)
    used_ids: Set[str] = field(default_factory=set)
    total_score: float = 0.0
    confidence: float = 0.0
    poi_constraint_satisfied: bool = False
    poi_satisfaction_count: int = 0
    poi_total_count: int = 0
    satisfied_poi_types: List[str] = field(default_factory=list)

    def add_entity(self, entity):
        if entity.entity_id in self.used_ids:
            return False
        self.entities.append(entity)
        self.used_ids.add(entity.entity_id)
        return True


@dataclass
class SubgraphTemplate:
    """用户描述的子图模板"""
    entities: List[SpatialEntity] = field(default_factory=list)
    entity_relations: Dict[Tuple[str, str], Dict] = field(default_factory=dict)

    def add_relation(self, e1_id, e2_id, possible_directions, distance_range):
        key = (min(e1_id, e2_id), max(e1_id, e2_id))
        self.entity_relations[key] = {
            "possible_directions": possible_directions,
            "distance_range": distance_range,
        }

    def get_relation(self, e1_id, e2_id):
        key = (min(e1_id, e2_id), max(e1_id, e2_id))
        return self.entity_relations.get(key)

    def has_relation(self, e1_id, e2_id):
        key = (min(e1_id, e2_id), max(e1_id, e2_id))
        return key in self.entity_relations


# ===================== 7. 子图变体生成 =====================

def generate_subgraph_variants(base_template):
    """根据方位多解生成多个子图变体（限制最大数量防爆炸）"""
    MAX_VARIANTS = 20
    variants = []
    edges_with_directions = []
    for (e1_id, e2_id), relation in base_template.entity_relations.items():
        if relation.get("possible_directions"):
            edges_with_directions.append(((e1_id, e2_id), relation["possible_directions"]))
    if not edges_with_directions:
        return [base_template]
    from itertools import product
    direction_lists = [dirs for (_, dirs) in edges_with_directions]
    # 计算总变体数，超过阈值则随机抽样
    total_combos = 1
    for dl in direction_lists:
        total_combos *= len(dl)
    if total_combos <= MAX_VARIANTS:
        combos_iter = product(*direction_lists)
    else:
        # 随机抽样
        import random as _random
        idx_lists = [_random.sample(range(len(dl)), min(len(dl), 2)) for dl in direction_lists]
        combos_iter = product(*[
            [direction_lists[i][j] for j in idx_lists[i]]
            for i in range(len(direction_lists))
        ])
    for direction_combo in combos_iter:
        variant = SubgraphTemplate(entities=base_template.entities[:])
        for i, ((e1_id, e2_id), _) in enumerate(edges_with_directions):
            fixed_dir = direction_combo[i]
            orig = base_template.get_relation(e1_id, e2_id)
            d_range = orig.get("distance_range", (0, 1000)) if orig else (0, 1000)
            variant.add_relation(e1_id, e2_id, [fixed_dir], d_range)
        for (e1_id, e2_id), rel in base_template.entity_relations.items():
            if not rel.get("possible_directions"):
                variant.add_relation(e1_id, e2_id, [], rel.get("distance_range", (0, 1000)))
        variants.append(variant)
        if len(variants) >= MAX_VARIANTS:
            break
    return variants


# ===================== 8. Neo4j连接 =====================

class Neo4jConnector:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cql, parameters=None, quiet=False):
        if not quiet and KG_SHOW_RESULTS:
            with self.driver.session() as session:
                result = session.run(cql, parameters or {})
                rows = [record.data() for record in result]
                self._print_result_summary(rows)
                return rows
        with self.driver.session() as session:
            result = session.run(cql, parameters or {})
            return [record.data() for record in result]

    def _print_result_summary(self, rows):
        """打印查询结果摘要（建筑ID、POI名称等）"""
        if not rows:
            print(f"  [KG] 0 rows")
            return
        ids = []
        for r in rows[:30]:
            if "id" in r and isinstance(r["id"], str):
                ids.append(r["id"])
            elif "poi_id" in r:
                name = r.get("poi_name", "")
                ids.append(f"POI({r['poi_id']}{'='+str(name) if name else ''})")
        suffix = f"...(+{len(rows)-30})" if len(rows) > 30 else ""
        print(f"  [KG] {len(rows)} rows {ids}{suffix}")

    def test_connection(self):
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                return result.single()["num"] == 1
        except Exception as e:
            print(f"   [Neo4j] 连接失败: {e}")
            return False


# ===================== 9. LLM客户端 =====================

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def call(self, prompt, temperature=0.1, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, timeout=120
                )
                content = response.choices[0].message.content.strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    raise ValueError(f"无法解析JSON: {content}")
            except (APIConnectionError, APITimeoutError) as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM调用失败: {e}")
                print(f"   [LLM] 重试第{attempt+1}次...")
                time.sleep(2)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM调用失败: {e}")
                print(f"   [LLM] 重试第{attempt+1}次: {e}")
                time.sleep(1)


# ===================== 10. 用户输入解析器 =====================

class UserInputParser:
    def __init__(self, llm_client):
        self.llm = llm_client

    def parse(self, user_input):
        prompt = f"""你是专业的地理空间描述解析专家。请严格按以下规则解析用户输入。

【核心原则：每个建筑是一个参照物，POI是建筑的属性】
- 每个建筑（Building）是一个独立的参照物，按出现顺序赋予ID（ref_0, ref_1, ref_2...）
- 建筑内提到的所有店铺/设施都是该建筑的POI属性，不是独立参照物

【POI类型必须转换为英文fclass格式】
快餐店→fast_food | 餐厅/饭馆→restaurant | 咖啡店/咖啡厅→cafe
酒吧→pub | 酒馆→bar | 面包店→bakery
超市→supermarket | 便利店→convenience | 服装店→clothes | 书店→books
银行→bank | ATM/取款机→atm | 药店→pharmacy | 理发店→hairdresser
美容院→beauty | 酒店→hotel | 医院→hospital
牙医/牙医诊所→dentist | 诊所/医生→doctors
学校→school | 幼儿园→kindergarten | 电影院→cinema | 剧院→theatre
邮箱→post_box | 邮局→post_office
健身房→gym | 运动中心→sports_centre | 游泳池→swimming_pool
加油站→fuel | 充电站→charging_station | 停车场→parking
共享汽车→car_sharing
大使馆→embassy | 教堂→place_of_worship | 社区中心→community_centre
监控/摄像头→camera_surveillance | 自动售货机→vending_machine
户外用品店→outdoor_shop | 运动用品店→sports_shop
厕所/卫生间→toilets
洗衣店/干洗店→laundry | 当铺/典当行→pawnbroker
俱乐部/俱乐部会所→club | 图书馆→library
公共设备→social_facility | 轮胎店→tyres
数据中心→data_centre
物资仓库→post_depot
出版社→bookmaker


【解析规则】
1. 道路信息：提取道路走向（东西向/南北向等）
2. 建筑信息（两种方式）：
   方式A（相对用户）：direction_from_user(N/S/E/W/NE/NW/SE/SW), estimated_distance(米), relative_to=null
   方式B（链式描述）：relative_to(参照物ID), direction_from_ref(方位), estimated_distance_from_ref(米), direction_from_user=null
   公共字段：color_side, color_top, associated_poi
3. POI：单POI用{{"poi_type":"fast_food"}}，多POI用[{{}},...]

【输出格式】仅返回JSON：
{{
    "user_context": {{"road": {{"road_orientation": "东西向"}}}},
    "reference_objects": [
        {{"entity_type":"Building","direction_from_user":"NE","estimated_distance":20,
          "relative_to":null,"direction_from_ref":null,"estimated_distance_from_ref":null,
          "color_side":"深蓝色","color_top":"灰色","associated_poi":{{"poi_type":"restaurant"}}}}
    ]
}}

用户输入：{user_input}
"""
        result = self.llm.call(prompt)
        result = self._merge_poi_entities(result)
        return result

    def _merge_poi_entities(self, parsed):
        ref_objects = parsed.get("reference_objects", [])
        if not ref_objects or len(ref_objects) <= 1:
            return parsed
        merged, used = [], set()
        for i in range(len(ref_objects)):
            if i in used:
                continue
            cur = ref_objects[i]
            cur_pois = self._extract_pois(cur)
            for j in range(i + 1, len(ref_objects)):
                if j in used:
                    continue
                oth = ref_objects[j]
                if (cur.get("direction_from_user") == oth.get("direction_from_user") and
                        cur.get("estimated_distance") == oth.get("estimated_distance")):
                    cur_pois.extend(self._extract_pois(oth))
                    used.add(j)
                    if not cur.get("color_side") and oth.get("color_side"):
                        cur["color_side"] = oth["color_side"]
                    if not cur.get("color_top") and oth.get("color_top"):
                        cur["color_top"] = oth["color_top"]
            if cur_pois:
                cur["associated_poi"] = cur_pois[0] if len(cur_pois) == 1 else cur_pois
            merged.append(cur)
            used.add(i)
        parsed["reference_objects"] = merged
        return parsed

    def _extract_pois(self, ref_obj):
        poi = ref_obj.get("associated_poi")
        if not poi:
            return []
        if isinstance(poi, dict):
            return [poi] if poi.get("poi_type") else []
        elif isinstance(poi, list):
            return [p for p in poi if isinstance(p, dict) and p.get("poi_type")]
        return []


# ===================== 11. 知识图谱匹配器 =====================

class KGMatcher:
    """
    知识图谱匹配器（子图同构匹配）

    KG类型适配：
    - Delaunay KG: Building-Building 关系为 DELAUNAY
    - Buffer KG: 需将 Neo4j 查询中 'DELAUNAY' 全局替换为 'BUFFER_NEAR'
    """

    # 匹配性能限制
    MAX_BACKTRACKS_PER_EXPANSION = 200  # 单次展开的最大回溯次数（全局回退需要更多）
    MAX_MATCH_TIME_SECONDS = 120        # match阶段总超时（秒）
    _backtrack_count = 0
    _match_start_time = 0

    # 中文颜色 → 英文（统一映射，与 KGdata 翻译脚本一致）
    COLOR_CN_TO_EN = {
        "深蓝色": "dark blue", "蓝色": "blue", "浅蓝色": "light blue",
        "深蓝": "dark blue", "蓝": "blue", "浅蓝": "light blue",
        "蓝灰": "blue-gray", "蓝灰色": "blue-gray",
        "深蓝灰": "dark blue-gray", "浅蓝灰": "light blue-gray",
        "蓝紫": "blue-purple", "蓝紫色": "blue-purple",
        "蓝绿": "blue-green", "蓝绿色": "blue-green",
        "灰色": "gray", "灰": "gray",
        "浅灰": "light gray", "浅灰色": "light gray",
        "深灰": "dark gray", "深灰色": "dark gray",
        "银灰": "silver gray", "银灰色": "silver gray",
        "灰白": "gray-white", "灰白色": "gray-white",
        "灰蓝": "gray-blue", "灰蓝色": "gray-blue",
        "灰紫": "gray-purple", "灰紫色": "gray-purple",
        "灰绿": "gray-green", "灰绿色": "gray-green",
        "灰黑": "dark gray",
        "白色": "white", "白": "white",
        "黑色": "black", "黑": "black",
        "深黑": "black", "米白": "off-white", "米白色": "off-white",
        "米色": "beige", "米黄": "cream", "米黄色": "cream",
        "红色": "red", "深红": "dark red", "深红色": "dark red",
        "暗红": "dark red", "暗红色": "dark red", "浅红": "light red",
        "粉色": "pink", "粉红": "pink", "粉红色": "pink",
        "粉紫": "pink-purple", "粉紫色": "pink-purple",
        "紫色": "purple", "深紫": "dark purple", "深紫色": "dark purple",
        "浅紫": "light purple", "浅紫色": "light purple",
        "紫红": "purple-red", "紫红色": "purple-red",
        "紫灰": "purple-gray", "紫灰色": "purple-gray",
        "绿色": "green", "深绿": "dark green", "深绿色": "dark green",
        "浅绿": "light green", "浅绿色": "light green",
        "黄绿": "yellow-green", "黄绿色": "yellow-green",
        "黄色": "yellow", "浅黄": "light yellow", "浅黄色": "light yellow",
        "橙色": "orange", "橙红": "orange-red", "橙红色": "orange-red",
        "棕色": "brown", "浅棕": "light brown", "浅棕色": "light brown",
        "深棕": "dark brown", "深棕色": "dark brown",
        "棕红": "brown-red", "棕黄": "brown-yellow", "棕褐": "brown",
        "红棕": "red-brown", "红棕色": "red-brown",
        "红褐": "red-brown", "红褐色": "red-brown",
        "黄褐": "tan", "黄褐色": "tan",
        "青色": "cyan", "金色": "gold", "银色": "silver",
        "未知": "unknown", "无色": "colorless",
    }

    @staticmethod
    def _to_en_color(cn_color):
        """将中文颜色翻译为英文（数据库查询用）"""
        if not cn_color:
            return "unknown"
        cn_color = cn_color.strip()
        # 已经是英文 → 直接返回
        if cn_color.isascii():
            return cn_color
        # 查表
        if cn_color in KGMatcher.COLOR_CN_TO_EN:
            return KGMatcher.COLOR_CN_TO_EN[cn_color]
        # 去渐变/条纹后缀再查
        for suffix in ["渐变", "条纹"]:
            if cn_color.endswith(suffix):
                base = cn_color[:-len(suffix)]
                if base in KGMatcher.COLOR_CN_TO_EN:
                    return KGMatcher.COLOR_CN_TO_EN[base]
        return cn_color  # fallback

    def __init__(self, neo4j_conn):
        self.neo4j = neo4j_conn
        self._shortest_path_cache = {}

    @staticmethod
    def _color_word_boundary_condition(field, param_name):
        """生成词边界颜色条件，防止 'blue' 匹配 'dark blue'"""
        return (
            f"({field} = $param OR "
            f"{field} STARTS WITH ($param + ' ') OR "
            f"{field} ENDS WITH (' ' + $param) OR "
            f"{field} CONTAINS (' ' + $param + ' '))"
        ).replace("$param", f"${param_name}")

    def _get_color_condition(self, entity):
        """生成严格词边界颜色 Cypher 条件"""
        cs_param, ct_param = "color_side", "color_top"
        if entity.color_side and entity.color_top:
            return (f"AND ({KGMatcher._color_word_boundary_condition('n.color_side', cs_param)}"
                    f" AND {KGMatcher._color_word_boundary_condition('n.color_top', ct_param)})")
        elif entity.color_side:
            return f"AND {KGMatcher._color_word_boundary_condition('n.color_side', cs_param)}"
        elif entity.color_top:
            return f"AND {KGMatcher._color_word_boundary_condition('n.color_top', ct_param)}"
        return ""

    def _get_adjacent_directions(self, direction):
        adjacent_map = {
            "E": ["NE", "SE"], "NE": ["E", "N"], "N": ["NE", "NW"],
            "NW": ["N", "W"], "W": ["NW", "SW"], "SW": ["W", "S"],
            "S": ["SW", "SE"], "SE": ["S", "E"],
        }
        return adjacent_map.get(direction, [])

    def _get_shortest_path(self, start_id, end_id):
        cache_key = (min(start_id, end_id), max(start_id, end_id))
        if cache_key in self._shortest_path_cache:
            return self._shortest_path_cache[cache_key]
        query = f"""
        MATCH (start:Building {{id: $start_id}}), (end:Building {{id: $end_id}})
        MATCH p = shortestPath((start)-[:{BLD_BLD_REL}*1..4]-(end))
        RETURN [node in nodes(p) | node.id] as path,
               [rel in relationships(p) | {{direction: rel.direction, distance_m: rel.distance_m}}] as edges,
               reduce(dist = 0, rel in relationships(p) | dist + rel.distance_m) as total_distance
        LIMIT 1
        """
        try:
            results = self.neo4j.query(query, {"start_id": start_id, "end_id": end_id})
            if results:
                r = results[0]
                path_info = {"path": r["path"], "edges": r["edges"], "distance": r["total_distance"]}
                self._shortest_path_cache[cache_key] = path_info
                return path_info
        except Exception:
            pass
        return None

    def _infer_direction_from_path(self, path_info):
        edges = path_info.get("edges", [])
        if not edges:
            return ""
        dx_total, dy_total = 0.0, 0.0
        for edge in edges:
            direction = edge.get("direction", "")
            distance = edge.get("distance_m", 0)
            angle = DIRECTION_ANGLE.get(direction, 0)
            dx = distance * math.cos(math.radians(angle))
            dy = distance * math.sin(math.radians(angle))
            dx_total += dx
            dy_total += dy
        if dx_total == 0 and dy_total == 0:
            return ""
        angle = math.degrees(math.atan2(dy_total, dx_total))
        angle = (angle + 360) % 360
        idx = int(((angle + 22.5) % 360) / 45)
        return DIRECTIONS_8[idx]

    def _select_key_edges(self, subgraph):
        entities = subgraph.entities
        if len(entities) <= 1:
            return set()
        all_edges = []
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            dist_range = relation.get("distance_range", (0, 1000))
            avg_dist = (dist_range[0] + dist_range[1]) / 2
            all_edges.append((avg_dist, (e1_id, e2_id)))
        all_edges.sort(key=lambda x: x[0])
        parent = {e.entity_id: e.entity_id for e in entities}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        key_edges = set()
        for _, (e1_id, e2_id) in all_edges:
            if find(e1_id) != find(e2_id):
                union(e1_id, e2_id)
                key_edges.add((min(e1_id, e2_id), max(e1_id, e2_id)))
        return key_edges

    def _check_relation_via_path(self, c1_id, c2_id, expected_relation):
        direct_query = f"""
        MATCH (a:Building {{id: $id1}})-[r:{BLD_BLD_REL}]-(b:Building {{id: $id2}})
        RETURN r.direction as direction, r.distance_m as distance
        """
        direct_results = self.neo4j.query(direct_query, {"id1": c1_id, "id2": c2_id})
        if direct_results:
            r = direct_results[0]
            dirs = expected_relation.get("possible_directions", [])
            min_dist, max_dist = expected_relation.get("distance_range", (0, 1000))
            dir_match = r["direction"] in dirs if dirs else True
            dist_match = min_dist <= r["distance"] <= max_dist
            score = 0.0
            if dir_match:
                score += 1.0
            elif r["direction"] in self._get_adjacent_directions(dirs[0] if dirs else ""):
                score += 0.5
            if dist_match:
                score += 1.0
            elif min_dist * 0.5 <= r["distance"] <= max_dist * 2:
                score += 0.5
            return (dir_match or dist_match), score
        path_info = self._get_shortest_path(c1_id, c2_id)
        if path_info and len(path_info["path"]) <= 4:
            inferred_dir = self._infer_direction_from_path(path_info)
            actual_dist = path_info["distance"]
            dirs = expected_relation.get("possible_directions", [])
            min_dist, max_dist = expected_relation.get("distance_range", (0, 1000))
            dir_match = inferred_dir in dirs if dirs else True
            dir_score = 1.0 if dir_match else (0.5 if inferred_dir in self._get_adjacent_directions(dirs[0] if dirs else "") else 0)
            path_len = len(path_info["path"]) - 1
            tolerance = 1 + 0.3 * path_len
            dist_match = min_dist <= actual_dist <= max_dist * tolerance
            dist_score = 1.0 if dist_match else (0.5 if min_dist * 0.5 <= actual_dist <= max_dist * tolerance * 2 else 0)
            confidence = 1.0 / path_len
            total_score = (dir_score + dist_score) * confidence
            return (dir_match or dist_match), total_score
        
        # === KG 图中无路径 → 几何验证回退（全局匹配场景） ===
        c1 = self._get_building_geodata(c1_id)
        c2 = self._get_building_geodata(c2_id)
        if not c1 or not c2:
            return True, 0.3  # 无法验证 → 宽松通过
        
        actual_dist = haversine_distance(c1["lat"], c1["lon"], c2["lat"], c2["lon"])
        actual_angle = compute_angle_deg(c1["lon"], c1["lat"], c2["lon"], c2["lat"])
        actual_dir = compute_direction_8(actual_angle)
        
        dirs = expected_relation.get("possible_directions", [])
        min_dist, max_dist = expected_relation.get("distance_range", (0, 1000))
        
        dir_score = 1.0 if (not dirs or actual_dir in dirs) else 0.0
        dist_score = 1.0 if min_dist <= actual_dist <= max_dist * 3 else (0.5 if min_dist * 0.2 <= actual_dist <= max_dist * 5 else 0.0)
        total_score = (dir_score + dist_score) / 2.0
        return (total_score >= 0.3), total_score

    def _get_building_geodata(self, building_id):
        """获取建筑的坐标"""
        cached = getattr(self, '_geo_cache', None)
        if cached is None:
            self._geo_cache = {}
        if building_id in self._geo_cache:
            return self._geo_cache[building_id]
        try:
            results = self.neo4j.query(
                "MATCH (b:Building {id: $id}) RETURN b.lon AS lon, b.lat AS lat",
                {"id": building_id})
            if results:
                self._geo_cache[building_id] = results[0]
                return results[0]
        except:
            pass
        self._geo_cache[building_id] = None
        return None

    def normalize_color(self, color):
        color = color.strip()
        color_normalize_map = {
            "蓝": "蓝色", "灰": "灰色", "白": "白色", "黑": "黑色",
            "红": "红色", "绿": "绿色", "黄": "黄色", "棕": "棕色",
        }
        if color.endswith("色"):
            return color
        for short, standard in color_normalize_map.items():
            if short in color:
                if color.startswith("深"):
                    return f"深{standard}"
                elif color.startswith("浅"):
                    return f"浅{standard}"
                else:
                    return standard
        return color

    def check_poi_in_building(self, building_id, poi_type):
        fclass = normalize_poi_type(poi_type)
        # 缓存键
        cache_key = (str(building_id), fclass)
        if not hasattr(self, '_poi_cache'):
            self._poi_cache = {}
        if cache_key in self._poi_cache:
            cached = self._poi_cache[cache_key]
            # DEBUG: 检查关键 POI 是否被缓存为 False
            if not cached[0]:
                print(f"\n  [DBG] POI_CACHE_HIT: {cache_key} -> MISS (cached)", end="", flush=True)
            return cached
        
        ids_to_try = [building_id]
        try:
            ids_to_try.append(int(building_id))
        except:
            pass
        for bid in ids_to_try:
            for rel_type in ["INSIDE", "NEAR"]:
                try:
                    result = self.neo4j.query(
                        f"MATCH (b:Building {{id: $bid}})-[:{rel_type}]-(p:POI) "
                        f"WHERE p.fclass = $fclass "
                        f"RETURN p.id AS poi_id, p.name AS poi_name LIMIT 1",
                        {"bid": bid, "fclass": fclass}, quiet=True)
                    if result and result[0].get("poi_id"):
                        val = (True, result[0].get("poi_name", result[0].get("poi_id")))
                        self._poi_cache[cache_key] = val
                        return val
                except Exception:
                    pass
            try:
                result = self.neo4j.query(
                    "MATCH (b:Building {id: $bid})-[r:INSIDE]-(p:POI) "
                    "WHERE p.fclass CONTAINS $fclass "
                    "RETURN p.id AS poi_id, p.name AS poi_name LIMIT 1",
                    {"bid": bid, "fclass": fclass}, quiet=True)
                if result and result[0].get("poi_id"):
                    val = (True, result[0].get("poi_name", result[0].get("poi_id")))
                    self._poi_cache[cache_key] = val
                    return val
            except Exception:
                pass
        self._poi_cache[cache_key] = (False, None)
        return False, None

    def check_all_pois_in_building(self, building_id, poi_types):
        if not poi_types:
            return True, []
        satisfied_pois = []
        for poi_type in poi_types:
            has_poi, poi_name = self.check_poi_in_building(building_id, poi_type)
            if has_poi and poi_name:
                satisfied_pois.append(poi_name)
            else:
                return False, []
        return True, satisfied_pois

    def _build_subgraph_template(self, entities):
        template = SubgraphTemplate(entities=entities)
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue
                possible_dirs = self._calculate_possible_directions_for_pair(e1, e2)
                dist_range = self._calculate_distance_range_for_pair(e1, e2)
                template.add_relation(e1.entity_id, e2.entity_id, possible_dirs, dist_range)
        return template

    def _calculate_possible_directions_for_pair(self, e1, e2):
        if e1.direction_from_user and e2.direction_from_user:
            dir1 = e1.direction_from_user
            dir2 = e2.direction_from_user
            possible_dirs1 = DIRECTION_MULTI_MAP.get(dir1, [dir1])
            possible_dirs2 = DIRECTION_MULTI_MAP.get(dir2, [dir2])
            possible_relative_dirs = []
            for d1 in possible_dirs1:
                for d2 in possible_dirs2:
                    angle1 = DIRECTION_ANGLE.get(d1, 0)
                    angle2 = DIRECTION_ANGLE.get(d2, 0)
                    relative_angle = (angle2 - angle1) % 360
                    idx = int(((relative_angle + 22.5) % 360) / 45)
                    relative_dir = DIRECTIONS_8[idx]
                    if relative_dir not in possible_relative_dirs:
                        possible_relative_dirs.append(relative_dir)
            return possible_relative_dirs
        return DIRECTIONS_8[:]

    def _calculate_distance_range_for_pair(self, e1, e2):
        if e1.estimated_distance and e2.estimated_distance:
            dist1 = e1.estimated_distance
            dist2 = e2.estimated_distance
            min_dist = abs(dist1 - dist2) * (1 - DISTANCE_TOLERANCE_RATIO)
            max_dist = (dist1 + dist2) * (1 + DISTANCE_TOLERANCE_RATIO)
            return (min_dist, max_dist)
        return (0, 1000)

    def _select_anchor_entity(self, entities):
        # 优先选择有 POI 约束的实体
        poi_entities = [e for e in entities if e.has_poi_constraint()]
        # 再选有颜色的
        color_entities = [e for e in entities if e.color_side or e.color_top]
        # 返回优先级列表：POI实体 > 颜色实体 > 任意实体
        return poi_entities + color_entities + list(entities)

    def _search_anchor_candidates(self, anchor):
        """锚点候选搜索：颜色+POI 一次性在 Cypher 中过滤"""
        color_condition = self._get_color_condition(anchor)
        
        if anchor.has_poi_constraint():
            poi_types = anchor.get_all_poi_types()
            poi_list = [t for t in poi_types if t]  # 有效 fclass
            if not poi_list:
                # 无有效 POI → 纯颜色查询
                query = f"""
                MATCH (n:Building) WHERE 1=1 {color_condition}
                RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                       n.lon AS lon, n.lat AS lat LIMIT 50
                """
                params = {"color_side": self._to_en_color(anchor.color_side or ""),
                          "color_top": self._to_en_color(anchor.color_top or "")}
                try:
                    return self.neo4j.query(query, params)[:30]
                except:
                    return []

            # === POI 推入 Cypher：颜色+POI 一次性查询 ===
            placeholders = ", ".join([f"$poi_{i}" for i in range(len(poi_list))])
            poi_params = {f"poi_{i}": poi_list[i] for i in range(len(poi_list))}
            
            query = f"""
            MATCH (n:Building) WHERE 1=1 {color_condition}
            MATCH (n)-[:INSIDE|NEAR]-(p:POI)
            WHERE p.fclass IN [{placeholders}]
            WITH n, collect(DISTINCT p.fclass) AS matched_pois
            // 至少匹配到 1 个 POI（不要求全部）
            RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                   n.lon AS lon, n.lat AS lat, matched_pois
            ORDER BY n.id
            LIMIT 200
            """
            params = {"color_side": self._to_en_color(anchor.color_side or ""),
                      "color_top": self._to_en_color(anchor.color_top or "")}
            params.update(poi_params)
            try:
                results = self.neo4j.query(query, params)
                validated = []
                for r in results:
                    r["poi_validated"] = True
                    r["matched_poi_names"] = r.get("matched_pois", [])
                    validated.append(r)
                return validated[:50]
            except:
                return []
        else:
            # 无 POI → 纯颜色
            query = f"""
            MATCH (n:Building) WHERE 1=1 {color_condition}
            RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                   n.lon AS lon, n.lat AS lat LIMIT 50
            """
            try:
                return self.neo4j.query(query, {
                    "color_side": self._to_en_color(anchor.color_side or ""),
                    "color_top": self._to_en_color(anchor.color_top or "")})[:30]
            except:
                return []

    def _search_entity_candidates(self, entity, limit=20):
        """实体候选搜索：颜色+POI 一次性在 Cypher 中过滤"""
        color_condition = self._get_color_condition(entity)
        
        if entity.has_poi_constraint():
            poi_types = entity.get_all_poi_types()
            poi_list = [t for t in poi_types if t]
            if not poi_list:
                query = f"""
                MATCH (n:Building) WHERE 1=1 {color_condition}
                RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                       n.lon AS lon, n.lat AS lat LIMIT $limit
                """
                try:
                    return self.neo4j.query(query, {
                        "color_side": self._to_en_color(entity.color_side or ""),
                        "color_top": self._to_en_color(entity.color_top or ""),
                        "limit": limit})
                except:
                    return []

            placeholders = ", ".join([f"$poi_{i}" for i in range(len(poi_list))])
            poi_params = {f"poi_{i}": poi_list[i] for i in range(len(poi_list))}
            query = f"""
            MATCH (n:Building) WHERE 1=1 {color_condition}
            MATCH (n)-[:INSIDE|NEAR]-(p:POI)
            WHERE p.fclass IN [{placeholders}]
            WITH n, collect(DISTINCT p.fclass) AS matched_pois
            RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                   n.lon AS lon, n.lat AS lat, matched_pois
            LIMIT $limit
            """
            params = {"color_side": self._to_en_color(entity.color_side or ""),
                      "color_top": self._to_en_color(entity.color_top or ""),
                      "limit": limit}
            params.update(poi_params)
            try:
                results = self.neo4j.query(query, params)
                for r in results:
                    r["poi_validated"] = True
                    r["matched_poi_names"] = r.get("matched_pois", [])
                return results
            except:
                return []
        else:
            query = f"""
            MATCH (n:Building) WHERE 1=1 {color_condition}
            RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                   n.lon AS lon, n.lat AS lat LIMIT $limit
            """
            try:
                return self.neo4j.query(query, {
                    "color_side": self._to_en_color(entity.color_side or ""),
                    "color_top": self._to_en_color(entity.color_top or ""),
                    "limit": limit})
            except:
                return []

    def _search_entity_candidates_nearby(self, entity, anchor_lon, anchor_lat, max_dist=300, limit=50):
        """在锚点附近搜索实体候选：空间距离 + 颜色 + POI（Cypher层面过滤，避免LIMIT截断）"""
        color_condition = self._get_color_condition(entity)

        # Neo4j 5.x 空间距离过滤（米）
        # toFloat 兼容数据库中字符串/数值两种存储格式
        spatial_condition = """
        AND point.distance(
          point({latitude: toFloat($anchor_lat), longitude: toFloat($anchor_lon)}),
          point({latitude: toFloat(n.lat), longitude: toFloat(n.lon)})
        ) <= $max_dist
        """

        if entity.has_poi_constraint():
            poi_types = entity.get_all_poi_types()
            poi_list = [t for t in poi_types if t]
            if not poi_list:
                query = f"""
                MATCH (n:Building)
                WHERE 1=1 {spatial_condition} {color_condition}
                RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                       n.lon AS lon, n.lat AS lat
                LIMIT $limit
                """
                try:
                    return self.neo4j.query(query, {
                        "anchor_lat": anchor_lat, "anchor_lon": anchor_lon,
                        "max_dist": max_dist,
                        "color_side": self._to_en_color(entity.color_side or ""),
                        "color_top": self._to_en_color(entity.color_top or ""),
                        "limit": limit})
                except Exception as e:
                    print(f"\n  [WARN] nearby query failed: {e}")
                    return []

            placeholders = ", ".join([f"$poi_{i}" for i in range(len(poi_list))])
            poi_params = {f"poi_{i}": poi_list[i] for i in range(len(poi_list))}
            query = f"""
            MATCH (n:Building)
            WHERE 1=1 {spatial_condition} {color_condition}
            MATCH (n)-[:INSIDE|NEAR]-(p:POI)
            WHERE p.fclass IN [{placeholders}]
            WITH n, collect(DISTINCT p.fclass) AS matched_pois
            RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                   n.lon AS lon, n.lat AS lat, matched_pois
            LIMIT $limit
            """
            params = {"anchor_lat": anchor_lat, "anchor_lon": anchor_lon,
                      "max_dist": max_dist,
                      "color_side": self._to_en_color(entity.color_side or ""),
                      "color_top": self._to_en_color(entity.color_top or ""),
                      "limit": limit}
            params.update(poi_params)
            try:
                results = self.neo4j.query(query, params)
                for r in results:
                    r["poi_validated"] = True
                    r["matched_poi_names"] = r.get("matched_pois", [])
                return results
            except Exception as e:
                print(f"\n  [WARN] nearby query failed: {e}")
                return []
        else:
            query = f"""
            MATCH (n:Building)
            WHERE 1=1 {spatial_condition} {color_condition}
            RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                   n.lon AS lon, n.lat AS lat
            LIMIT $limit
            """
            try:
                return self.neo4j.query(query, {
                    "anchor_lat": anchor_lat, "anchor_lon": anchor_lon,
                    "max_dist": max_dist,
                    "color_side": self._to_en_color(entity.color_side or ""),
                    "color_top": self._to_en_color(entity.color_top or ""),
                    "limit": limit})
            except Exception as e:
                print(f"\n  [WARN] nearby query failed: {e}")
                return []

    def _search_constrained_neighbors(self, source_id, relation, target_entity_id, entities):
        """邻居候选搜索：颜色+POI 一次性在 Cypher 中过滤"""
        target_entity = next((e for e in entities if e.entity_id == target_entity_id), None)
        if not target_entity:
            return []
        dirs = relation.get("possible_directions", [])
        dir_condition = ""
        if dirs:
            dir_list = "'" + "','".join(dirs) + "'"
            dir_condition = f"AND r.direction IN [{dir_list}]"
        dist_range = relation.get("distance_range", (0, 1000))
        min_dist, max_dist = dist_range
        color_condition = self._get_color_condition(target_entity)

        if target_entity.has_poi_constraint():
            poi_types = target_entity.get_all_poi_types()
            poi_list = [t for t in poi_types if t]
            if not poi_list:
                query = f"""
                MATCH (s:Building {{id: $source_id}})-[r:{BLD_BLD_REL}]-(n:Building)
                WHERE r.distance_m >= $min_dist AND r.distance_m <= $max_dist
                {dir_condition} {color_condition}
                RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                       n.lon AS lon, n.lat AS lat, r.direction AS direction, r.distance_m AS distance
                LIMIT 10
                """
                try:
                    return self.neo4j.query(query, {
                        "source_id": source_id, "min_dist": min_dist * 0.5, "max_dist": max_dist * 2,
                        "color_side": self._to_en_color(target_entity.color_side or ""),
                        "color_top": self._to_en_color(target_entity.color_top or "")})
                except:
                    return []

            placeholders = ", ".join([f"$poi_{i}" for i in range(len(poi_list))])
            poi_params = {f"poi_{i}": poi_list[i] for i in range(len(poi_list))}
            query = f"""
            MATCH (s:Building {{id: $source_id}})-[r:{BLD_BLD_REL}]-(n:Building)
            WHERE r.distance_m >= $min_dist AND r.distance_m <= $max_dist
            {dir_condition} {color_condition}
            MATCH (n)-[:INSIDE|NEAR]-(p:POI)
            WHERE p.fclass IN [{placeholders}]
            WITH n, r, collect(DISTINCT p.fclass) AS matched_pois
            RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                   n.lon AS lon, n.lat AS lat, r.direction AS direction, 
                   r.distance_m AS distance, matched_pois
            LIMIT 10
            """
            params = {"source_id": source_id, "min_dist": min_dist * 0.5, "max_dist": max_dist * 2,
                      "color_side": self._to_en_color(target_entity.color_side or ""),
                      "color_top": self._to_en_color(target_entity.color_top or "")}
            params.update(poi_params)
            try:
                results = self.neo4j.query(query, params)
                for r in results:
                    r["poi_validated"] = True
                    r["matched_poi_names"] = r.get("matched_pois", [])
                return results
            except:
                return []
        else:
            query = f"""
            MATCH (s:Building {{id: $source_id}})-[r:{BLD_BLD_REL}]-(n:Building)
            WHERE r.distance_m >= $min_dist AND r.distance_m <= $max_dist
            {dir_condition} {color_condition}
            RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
                   n.lon AS lon, n.lat AS lat, r.direction AS direction, r.distance_m AS distance
            LIMIT 10
            """
            try:
                return self.neo4j.query(query, {
                    "source_id": source_id, "min_dist": min_dist * 0.5, "max_dist": max_dist * 2,
                    "color_side": self._to_en_color(target_entity.color_side or ""),
                    "color_top": self._to_en_color(target_entity.color_top or "")})
            except:
                return []
            return []

    def _expand_match_iteratively_relaxed(self, partial_match, subgraph, entities, key_edges):
        matched_ids = set(partial_match.keys())
        
        # 超时保护
        if time.time() - self._match_start_time > self.MAX_MATCH_TIME_SECONDS:
            return False
        
        # 回溯计数保护
        self._backtrack_count += 1
        if self._backtrack_count > self.MAX_BACKTRACKS_PER_EXPANSION:
            return False
        frontier = []
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            if e1_id in matched_ids and e2_id not in matched_ids:
                frontier.append((e2_id, e1_id, relation))
            elif e2_id in matched_ids and e1_id not in matched_ids:
                frontier.append((e1_id, e2_id, relation))
        if not frontier:
            return self._validate_non_key_edges(partial_match, subgraph, key_edges)
        
        for target_id, source_id, relation in frontier:
            if target_id in partial_match:
                continue
            source_cand = partial_match[source_id]
            
            # === 候选搜索：CONTACT 邻居优先 + 全局回退（参考 geo_localization_fixed2.py） ===
            candidates = self._search_constrained_neighbors(
                source_cand["id"], relation, target_id, entities)
            
            if not candidates:
                # CONTACT 图中找不到 → 回退到全局独立颜色+POI搜索
                target_entity = next((e for e in entities if e.entity_id == target_id), None)
                if target_entity:
                    candidates = self._search_entity_candidates(target_entity, limit=50)
                    if candidates:
                        print(f"!", end="", flush=True)  # debug: 触发 fallback
            
            if not candidates:
                continue
            
            for neighbor in candidates[:20]:  # 限制每个 frontier 最多尝试 20 个候选
                if neighbor["id"] in [c["id"] for c in partial_match.values()]:
                    continue
                partial_match[target_id] = neighbor
                if self._expand_match_iteratively_relaxed(partial_match, subgraph, entities, key_edges):
                    return True
                del partial_match[target_id]
        return len(partial_match) == len(entities)

    def _validate_non_key_edges(self, partial_match, subgraph, key_edges):
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            edge_key = (min(e1_id, e2_id), max(e1_id, e2_id))
            if edge_key in key_edges:
                continue
            if e1_id not in partial_match or e2_id not in partial_match:
                return False
            c1 = partial_match[e1_id]
            c2 = partial_match[e2_id]
            valid, score = self._check_relation_via_path(c1["id"], c2["id"], relation)
            if not valid or score < 0.3:
                return False
        return True

    def _subgraph_isomorphism_match(self, subgraph, entities, max_matches, entity_candidates_cache=None):
        """
        匹配核心：独立实体匹配 + 锚点枢纽 + 拓扑搜索（参考 geo_localization_fixed2.py）
        1. 每个实体独立全局匹配（颜色+POI 一次性 Cypher）
        2. 候选最少者作为锚点
        3. 在锚点 CONTACT 邻居中搜索其他实体（拓扑约束）
        4. 邻居中找不到则全局回退
        5. 笛卡尔积生成组合（空间关系在 _calculate_match_score 中评估）
        """
        # Step 1: 独立查找所有实体的全局候选（优先使用缓存）
        if entity_candidates_cache is not None:
            entity_candidates = entity_candidates_cache
        else:
            entity_candidates = {}
            for entity in entities:
                candidates = self._search_entity_candidates(entity, limit=200)
                entity_candidates[entity.entity_id] = candidates
        
        entities_with_cands = [
            (e, len(entity_candidates.get(e.entity_id, [])))
            for e in entities
            if len(entity_candidates.get(e.entity_id, [])) > 0
        ]
        if not entities_with_cands:
            print(f"(anchor:0)", end="", flush=True)
            return []
        
        # Step 2: 候选最少者作为锚点
        entities_with_cands.sort(key=lambda x: x[1])
        min_entity = entities_with_cands[0][0]
        min_candidates = entity_candidates[min_entity.entity_id]
        other_entities = [e for e, _ in entities_with_cands[1:]]
        
        # 限制锚点数量，按评分降序排列（精确颜色+POI优先）
        MAX_ANCHORS = 10
        min_candidates_sorted = sorted(
            min_candidates,
            key=lambda c: self._score_candidate(c, min_entity),
            reverse=True)
        min_candidates_limited = min_candidates_sorted[:MAX_ANCHORS]
        
        print(f"(ac={len(min_candidates_limited)}", end="", flush=True)
        
        # Step 3: 逐个锚点候选，在 CONTACT 邻居中搜索其他实体
        self._match_start_time = time.time()
        import itertools
        matches = []
        matched_count = 0
        
        for i, anchor_cand in enumerate(min_candidates_limited):
            if time.time() - self._match_start_time > self.MAX_MATCH_TIME_SECONDS:
                print(f" T/O", end="", flush=True)
                break
            if len(matches) >= max_matches:
                break
            
            # 拓扑搜索：在锚点的 CONTACT 邻居中查找其他实体
            neighbors = self._get_neighbors_dict(anchor_cand["id"])
            all_matches_lists = []
            topo_hit_count = 0  # 通过拓扑（邻居）找到的实体数
            
            for other_entity in other_entities:
                neigh_matches = []
                for nid, neighbor in neighbors.items():
                    if self._building_matches_criteria(neighbor, other_entity):
                        # 记录通过验证的 POI 名称（用于输出显示）
                        if other_entity.has_poi_constraint():
                            poi_types = other_entity.get_all_poi_types()
                            _, poi_names = self.check_all_pois_in_building(neighbor["id"], poi_types)
                            neighbor = dict(neighbor)  # shallow copy
                            neighbor["poi_validated"] = True
                            neighbor["matched_poi_names"] = poi_names
                        neigh_matches.append(neighbor)
                
                # DEBUG
                neigh_ids = [n["id"] for n in neigh_matches]
                print(f"\n  [DBG] anchor={anchor_cand['id']} entity={other_entity.entity_id}: neigh_matches={neigh_ids}", end="", flush=True)

                if neigh_matches:
                    all_matches_lists.append(neigh_matches)
                    topo_hit_count += 1
                else:
                    # 邻居中找不到 → Cypher层面空间查询（避免LIMIT截断）
                    anchor_lat = float(anchor_cand.get("lat", 0))
                    anchor_lon = float(anchor_cand.get("lon", 0))
                    max_fallback_dist = 300
                    glob_filtered = self._search_entity_candidates_nearby(
                        other_entity, anchor_lon, anchor_lat,
                        max_dist=max_fallback_dist, limit=50)
                    print(f"\n  [DBG] global_fallback: anchor={anchor_cand['id']} entity={other_entity.entity_id}: nearby={len(glob_filtered)} (<=300m)", end="", flush=True)
                    if glob_filtered:
                        all_matches_lists.append(glob_filtered[:5])
                    else:
                        all_matches_lists = None
                        break
            
            if all_matches_lists is None:
                continue
            
            combo_limit = max(3, max_matches // max(len(min_candidates_limited), 1) + 1)
            for combo_tuple in itertools.product(*all_matches_lists):
                if len(matches) >= max_matches:
                    break
                combo_ids = [c["id"] for c in combo_tuple]
                combo_ids.append(anchor_cand["id"])  # 锚点也参与去重
                if len(set(combo_ids)) != len(combo_ids):
                    continue
                
                matched_entities = [
                    MatchedEntity(
                        entity_id=anchor_cand["id"], entity_type="Building",
                        lon=float(anchor_cand.get("lon", 0)), lat=float(anchor_cand.get("lat", 0)),
                        color_side=anchor_cand.get("color_side", ""), color_top=anchor_cand.get("color_top", ""),
                        query_id=min_entity.entity_id,
                        poi_validation=anchor_cand.get("poi_validated", False),
                        required_poi_type=min_entity.get_first_poi_type() if min_entity.has_poi_constraint() else "",
                        matched_poi_names=anchor_cand.get("matched_poi_names", []))
                ]
                for j, cand in enumerate(combo_tuple):
                    matched_entities.append(MatchedEntity(
                        entity_id=cand["id"], entity_type="Building",
                        lon=float(cand.get("lon", 0)), lat=float(cand.get("lat", 0)),
                        color_side=cand.get("color_side", ""), color_top=cand.get("color_top", ""),
                        query_id=other_entities[j].entity_id,
                        poi_validation=cand.get("poi_validated", False),
                        required_poi_type=other_entities[j].get_first_poi_type() if other_entities[j].has_poi_constraint() else "",
                        matched_poi_names=cand.get("matched_poi_names", [])))
                
                # 拓扑约束已由 CONTACT 邻居搜索隐式保证（与 geo_localization_fixed2.py 一致）
                # 实体间方向+距离验证在评分阶段 _calculate_match_score 中做软评估
                matches.append(CandidateCombination(entities=matched_entities))
                matched_count += 1
        
        # 拓扑命中率标记（t=N 表示 N 个实体通过 CONTACT 邻居找到）
        topo_tag = f"t{topo_hit_count}" if topo_hit_count > 0 else ""
        print(f" mc={matched_count}{topo_tag})", end="", flush=True)
        return matches

    def _validate_topology(self, matched_entities, subgraph):
        """
        拓扑验证：检查匹配组合中实体对间的实际空间关系
        是否满足描述中推导出的方向+距离约束。
        类似 geo_localization_fixed2.py 的 CONTACT 邻居匹配，
        但这里用 haversine 几何计算而非图遍历。
        """
        entity_map = {m.query_id: m for m in matched_entities}
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            m1 = entity_map.get(e1_id)
            m2 = entity_map.get(e2_id)
            if not m1 or not m2:
                continue  # 该对未全部匹配，跳过
            
            actual_dist = haversine_distance(m1.lat, m1.lon, m2.lat, m2.lon)
            actual_angle = compute_angle_deg(m1.lon, m1.lat, m2.lon, m2.lat)
            actual_dir = compute_direction_8(actual_angle)
            
            dirs = relation.get("possible_directions", [])
            min_d, max_d = relation.get("distance_range", (0, 1000))
            
            # 方向验证：允许相邻方向作为容差
            dir_ok = False
            if not dirs:
                dir_ok = True
            elif actual_dir in dirs:
                dir_ok = True
            elif dirs and actual_dir in self._get_adjacent_directions(dirs[0]):
                dir_ok = True
            
            # 距离验证：宽松容差（全城范围内全局回退的场景）
            dist_ok = (min_d * 0.3 <= actual_dist <= max_d * 2.5)
            
            if not (dir_ok and dist_ok):
                return False
        return True

    def _get_neighbors_dict(self, building_id):
        """获取某个建筑的 CONTACT 邻居（{neighbor_id: data}），带缓存"""
        if not hasattr(self, '_neighbor_cache'):
            self._neighbor_cache = {}
        if building_id in self._neighbor_cache:
            return self._neighbor_cache[building_id]
        query = f"""
        MATCH (s:Building {{id: $id}})-[r:{BLD_BLD_REL}]-(n:Building)
        RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
               n.lon AS lon, n.lat AS lat
        """
        try:
            result = {r["id"]: r for r in self.neo4j.query(query, {"id": building_id})}
            self._neighbor_cache[building_id] = result
            return result
        except:
            self._neighbor_cache[building_id] = {}
            return {}

    def _building_color_matches(self, building, entity):
        """严格颜色匹配：建筑每个面必须分别包含实体描述的对应面"""
        ent_cs = (self._to_en_color(entity.color_side or "")).lower()
        ent_ct = (self._to_en_color(entity.color_top or "")).lower()
        bld_cs = (building.get("color_side", "") or "").lower()
        bld_ct = (building.get("color_top", "") or "").lower()
        if not ent_cs and not ent_ct:
            return True
        def _contains(description_color, actual_face_color):
            if not description_color or not actual_face_color:
                return False
            return description_color == actual_face_color or description_color in actual_face_color
        cs_ok = (not ent_cs) or _contains(ent_cs, bld_cs)
        ct_ok = (not ent_ct) or _contains(ent_ct, bld_ct)
        return cs_ok and ct_ok

    def _score_candidate(self, building, entity):
        """对候选建筑评分，用于锚点排序：精确匹配优先"""
        score = 0
        ent_cs = (self._to_en_color(entity.color_side or "")).lower()
        ent_ct = (self._to_en_color(entity.color_top or "")).lower()
        bld_cs = (building.get("color_side", "") or "").lower()
        bld_ct = (building.get("color_top", "") or "").lower()

        for ent_c, bld_c in [(ent_cs, bld_cs), (ent_ct, bld_ct)]:
            if not ent_c:
                continue
            if not bld_c:
                continue
            if ent_c == bld_c:
                score += 2  # 精确相等
            elif ent_c in bld_c or bld_c in ent_c:
                score += 1  # 子字符串包含

        # POI 匹配加分
        if entity.has_poi_constraint():
            poi_types = entity.get_all_poi_types()
            poi_valid, _ = self.check_all_pois_in_building(building["id"], poi_types)
            if poi_valid:
                score += 3  # POI 全部满足
            elif building.get("poi_validated"):
                score += 1  # POI 部分满足（Cypher 预过滤通过）

        return score

    def _validate_topology(self, matched_entities, subgraph):
        """
        拓扑验证：检查匹配组合中实体对间的实际空间关系
        是否满足 subgraph 中推导出的方向+距离约束。
        用 haversine 几何计算，和评分阶段的 _calculate_match_score 中拓扑评分一致。
        """
        entity_map = {m.query_id: m for m in matched_entities}
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            m1 = entity_map.get(e1_id)
            m2 = entity_map.get(e2_id)
            if not m1 or not m2:
                continue
            actual_dist = haversine_distance(m1.lat, m1.lon, m2.lat, m2.lon)
            actual_angle = compute_angle_deg(m1.lon, m1.lat, m2.lon, m2.lat)
            actual_dir = compute_direction_8(actual_angle)
            dirs = relation.get("possible_directions", [])
            min_d, max_d = relation.get("distance_range", (0, 1000))
            # 方向验证：允许相邻方向容差
            dir_ok = True
            if dirs:
                dir_ok = (actual_dir in dirs or
                          actual_dir in self._get_adjacent_directions(dirs[0]))
            # 距离验证：宽松容差（全城范围可能跨度较大）
            dist_ok = (min_d * 0.3 <= actual_dist <= max_d * 3.0)
            if not (dir_ok and dist_ok):
                return False
        return True

    def _building_matches_criteria(self, building, entity):
        """
        完整属性匹配：颜色 + POI（与 geo_localization_fixed2.py 的 _building_matches_criteria 一致）
        """
        # 颜色匹配
        if not self._building_color_matches(building, entity):
            return False
        # POI 验证
        if entity.has_poi_constraint():
            poi_types = entity.get_all_poi_types()
            poi_valid, _ = self.check_all_pois_in_building(building["id"], poi_types)
            if not poi_valid:
                return False
        return True

    def _deduplicate_matches(self, matches):
        seen = set()
        unique = []
        for combo in matches:
            key = tuple(sorted([e.entity_id for e in combo.entities]))
            if key not in seen:
                seen.add(key)
                unique.append(combo)
        return unique

    def _count_satisfied_poi(self, combo, entities):
        entity_map = {e.query_id: e for e in combo.entities}
        total_satisfied = 0
        for entity in entities:
            if entity.has_poi_constraint():
                eid = entity.entity_id
                if eid in entity_map:
                    matched = entity_map[eid]
                    if matched.poi_validation:
                        total_satisfied += len(entity.get_all_poi_types())
        return total_satisfied

    def _get_satisfied_poi_types(self, combo, entities):
        entity_map = {e.query_id: e for e in combo.entities}
        satisfied_types = []
        for entity in entities:
            if entity.has_poi_constraint():
                eid = entity.entity_id
                if eid in entity_map:
                    matched = entity_map[eid]
                    if matched.poi_validation:
                        satisfied_types.extend(entity.get_all_poi_types())
        return satisfied_types

    def _get_total_poi_count(self, entities):
        total = 0
        for entity in entities:
            if entity.has_poi_constraint():
                total += len(entity.get_all_poi_types())
        return total

    def _calculate_match_score(self, combo, subgraph, entities):
        score = 0.0
        for matched in combo.entities:
            entity = next((e for e in entities if e.entity_id == matched.query_id), None)
            if entity:
                if entity.color_side and matched.color_side:
                    if entity.color_side.lower() in matched.color_side.lower():
                        score += 1.0
                if entity.color_top and matched.color_top:
                    if entity.color_top.lower() in matched.color_top.lower():
                        score += 1.0
        
        # === POI 加权：核心改进 ===
        poi_total = self._get_total_poi_count(entities)
        poi_satisfied = combo.poi_satisfaction_count
        if poi_total > 0:
            # 每个满足的POI +15分
            score += poi_satisfied * 15.0
            # 全部POI满足额外 +30分大奖
            if poi_satisfied == poi_total:
                score += 30.0
            # 未满足的POI 扣分
            score -= (poi_total - poi_satisfied) * 5.0
        
        matched_map = {m.query_id: m for m in combo.entities}
        topology_score = 0.0
        topology_count = 0
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            if e1_id not in matched_map or e2_id not in matched_map:
                continue
            m1 = matched_map[e1_id]
            m2 = matched_map[e2_id]
            actual_dist = haversine_distance(m1.lat, m1.lon, m2.lat, m2.lon)
            actual_angle = compute_angle_deg(m1.lon, m1.lat, m2.lon, m2.lat)
            actual_dir = compute_direction_8(actual_angle)
            dirs = relation.get("possible_directions", [])
            if dirs and actual_dir in dirs:
                topology_score += 1.0
            elif dirs and actual_dir in self._get_adjacent_directions(dirs[0]):
                topology_score += 0.5
            min_dist, max_dist = relation.get("distance_range", (0, 1000))
            if min_dist <= actual_dist <= max_dist:
                topology_score += 1.0
            elif min_dist * 0.5 <= actual_dist <= max_dist * 2:
                topology_score += 0.5
            topology_count += 1
        if topology_count > 0:
            score += (topology_score / topology_count) * 2.0
        return score

    def find_candidate_combinations(self, entities, max_total_combinations=50):
        """多子图同构匹配定位（公开入口）"""
        if not entities:
            return []
        print(f"(e={len(entities)}", end="", flush=True)
        base_template = self._build_subgraph_template(entities)
        subgraph_variants = generate_subgraph_variants(base_template)
        print(f" v={len(subgraph_variants)}", end="", flush=True)
        
        # 实体候选缓存：所有子图变体共享，避免重复 Cypher 查询
        entity_candidates_cache = {}
        for entity in entities:
            entity_candidates_cache[entity.entity_id] = self._search_entity_candidates(entity, limit=200)
        
        all_matches = []
        for subgraph in subgraph_variants:
            matches = self._subgraph_isomorphism_match(
                subgraph, entities, max_total_combinations // len(subgraph_variants) + 5,
                entity_candidates_cache)
            all_matches.extend(matches)
        unique_matches = self._deduplicate_matches(all_matches)
        # 计算POI满足情况
        poi_total = self._get_total_poi_count(entities)
        for combo in unique_matches:
            combo.poi_satisfaction_count = self._count_satisfied_poi(combo, entities)
            combo.poi_total_count = poi_total
            combo.poi_constraint_satisfied = (
                combo.poi_satisfaction_count == poi_total and poi_total > 0)
            combo.satisfied_poi_types = self._get_satisfied_poi_types(combo, entities)
            combo.total_score = self._calculate_match_score(combo, base_template, entities)
        
        # === POI排序（已通过Cypher预过滤，无需严格全部满足） ===
        # POI满足数已由Cypher中 IN 子句保证 ≥1
        # 只做降序排列，不丢弃部分匹配
        unique_matches.sort(key=lambda c: (-c.poi_satisfaction_count, -c.total_score))
        return unique_matches[:max_total_combinations]


# ===================== 12. 坐标推算器 =====================

class CoordinateEstimator:
    """根据候选组合推算用户坐标"""

    def estimate_user_position(self, combo, entities):
        positions = []
        weights = []
        entity_map = {e.query_id: e for e in combo.entities}
        for query_entity in entities:
            eid = query_entity.entity_id
            if eid not in entity_map:
                continue
            matched = entity_map[eid]
            b_lon = matched.lon
            b_lat = matched.lat
            user_dir = query_entity.direction_from_user
            if not user_dir:
                continue
            opposite_dir = OPPOSITE_DIRECTION.get(user_dir, user_dir)
            est_distance = query_entity.estimated_distance
            if est_distance <= 0:
                est_distance = 20
            dlon, dlat = direction_to_offset(opposite_dir, est_distance, b_lat)
            user_lon = b_lon + dlon
            user_lat = b_lat + dlat
            positions.append((user_lon, user_lat))
            weights.append(1.0 / max(est_distance, 5))
        if not positions:
            return 0.0, 0.0, 0.0
        total_weight = sum(weights)
        avg_lon = sum(p[0] * w for p, w in zip(positions, weights)) / total_weight
        avg_lat = sum(p[1] * w for p, w in zip(positions, weights)) / total_weight
        if len(positions) > 1:
            distances = []
            for lon, lat in positions:
                dist = haversine_distance(lat, lon, avg_lat, avg_lon)
                distances.append(dist)
            avg_dist = sum(distances) / len(distances)
            confidence = max(0, 1 - avg_dist / 300)
        else:
            confidence = 0.5
        return avg_lon, avg_lat, confidence


# ===================== 13. 批量定位器 =====================

class BatchLocalizer:
    """批量定位器：初始化组件并逐一执行定位"""

    def __init__(self):
        self.neo4j = Neo4jConnector()
        self.llm = LLMClient()
        self.parser = UserInputParser(self.llm)
        self.matcher = KGMatcher(self.neo4j)
        self.estimator = CoordinateEstimator()

    def localize_single(self, description):
        """执行单个点位的定位，返回统一格式的结果字典"""
        try:
            t0 = time.time()
            print("parse", end="", flush=True)
            parsed = self.parser.parse(description)
            print(f"({time.time()-t0:.1f}s)", end="", flush=True)
            
            ref_objects = parsed.get('reference_objects', [])
            if not ref_objects:
                print("->no_ref", end="", flush=True)
                return {"status": "no_reference", "message": "未解析到参照物"}

            entities = []
            for i, ref in enumerate(ref_objects):
                entity_id = ref.get("entity_id", f"ref_{i}")
                entity = SpatialEntity(
                    entity_id=entity_id,
                    entity_type=ref.get("entity_type", "Building"),
                    color_side=ref.get("color_side", ""),
                    color_top=ref.get("color_top", ""),
                    fclass=ref.get("poi_type", ""),
                    road_type=ref.get("road_type", ""),
                    road_orientation=ref.get("road_orientation"))
                entity.direction_from_user = ref.get("direction_from_user") or ""
                est_dist = ref.get("estimated_distance")
                entity.estimated_distance = float(est_dist) if est_dist is not None else 0.0
                entity.relative_to = ref.get("relative_to") or ""
                entity.direction_from_ref = ref.get("direction_from_ref") or ""
                ref_dist = ref.get("estimated_distance_from_ref")
                entity.distance_from_ref = float(ref_dist) if ref_dist is not None else 0.0
                entity.associated_poi = ref.get("associated_poi") or {}
                entities.append(entity)

            has_chain = any(e.relative_to for e in entities)
            if has_chain:
                resolve_chain_entities(entities)

            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if i == j:
                        continue
                    dir1 = e1.direction_from_user
                    dir2 = e2.direction_from_user
                    dist1 = e1.estimated_distance
                    dist2 = e2.estimated_distance
                    if dir1 and dir2 and dist1 and dist2 and dist1 > 0 and dist2 > 0:
                        possible_dirs = calculate_possible_directions(dir1, dist1, dir2, dist2)
                        e1.possible_relative_directions[e2.entity_id] = possible_dirs

            t1 = time.time()
            print(f" match", end="", flush=True)
            combinations = self.matcher.find_candidate_combinations(entities)
            match_elapsed = time.time() - t1
            print(f"({match_elapsed:.1f}s)", end="", flush=True)
            
            if not combinations:
                print("->no_match", end="", flush=True)
                return {"status": "no_match", "message": "未找到匹配组合"}

            best_combo = combinations[0]
            lon, lat, conf = self.estimator.estimate_user_position(best_combo, entities)

            # 输出最终匹配的组合详情
            matched_info = []
            entity_map = {m.query_id: m for m in best_combo.entities}
            for entity in entities:
                if entity.entity_id in entity_map:
                    m = entity_map[entity.entity_id]
                    colors = f"{m.color_side}/{m.color_top}"
                    poi_str = ""
                    if entity.has_poi_constraint():
                        poi_types = entity.get_all_poi_types()
                        satisfied = m.matched_poi_names if hasattr(m, 'matched_poi_names') and m.matched_poi_names else []
                        poi_str = f" [{', '.join(poi_types)}]"
                        if satisfied:
                            poi_str += f" = {'✓' * len(satisfied)}"
                    matched_info.append({
                        "desc_id": entity.entity_id,
                        "building_id": m.entity_id,
                        "colors": colors,
                        "poi": poi_str.strip()
                    })

            if lon is None or lat is None:
                print("->no_pos", end="", flush=True)
                return {"status": "no_position", "message": "无法推算用户坐标"}

            print(f"->OK({len(combinations)}combos)", end="", flush=True)
            return {
                "status": "success", "lon": lon, "lat": lat,
                "confidence": conf, "num_references": len(ref_objects),
                "num_matches": len(combinations),
                "matched_combo": matched_info}
        except Exception as e:
            print(f"->ERR", end="", flush=True)
            return {"status": "error", "message": str(e)}

    def close(self):
        self.neo4j.close()


# ===================== 14. 评估指标 =====================

def evaluate_results(results, ground_truths):
    """
    计算评估指标（对标MambaPlace和Where am I?）

    参数:
        results: 定位结果列表 [{"status":"success","lon":...,"lat":...}, ...]
        ground_truths: 真值列表 [{"lon":...,"lat":...}, ...]
    """
    k_values = [1, 5]
    thresholds = EVAL_DISTANCE_THRESHOLDS
    num_total = len(ground_truths)

    distances = []
    for r, gt in zip(results, ground_truths):
        pred = r.get('predicted', {})
        if r.get('status') == 'success' and pred.get('lon') is not None and pred.get('lat') is not None:
            dist = haversine_distance(gt['lat'], gt['lon'], pred['lat'], pred['lon'])
            distances.append(dist)

    recall = {}
    for k in k_values:
        for t in thresholds:
            correct = sum(1 for d in distances if d <= t)
            recall[f"Recall@{k}({t}m)"] = correct / num_total if num_total > 0 else 0.0

    if distances:
        mean_error = float(np.mean(distances))
        median_error = float(np.median(distances))
        p25 = float(np.percentile(distances, 25))
        p50 = float(np.percentile(distances, 50))
        p75 = float(np.percentile(distances, 75))
        p90 = float(np.percentile(distances, 90))
    else:
        mean_error = median_error = p25 = p50 = p75 = p90 = None

    success_rate = len(distances) / num_total if num_total > 0 else 0.0

    return {
        "recall": recall,
        "mean_distance_error_m": round(mean_error, 2) if mean_error is not None else None,
        "median_distance_error_m": round(median_error, 2) if median_error is not None else None,
        "success_rate": round(success_rate, 4),
        "num_success": len(distances),
        "num_total": num_total,
        "distance_percentiles": {
            "p25": round(p25, 2) if p25 is not None else None,
            "p50": round(p50, 2) if p50 is not None else None,
            "p75": round(p75, 2) if p75 is not None else None,
            "p90": round(p90, 2) if p90 is not None else None
        }
    }


# ===================== 15. 主程序 =====================

def main():
    """批量定位评估主程序"""
    print("=" * 70)
    print("  自然语言地理定位 - 批量评估系统")
    print("=" * 70)

    print(f"\n[Config] 场景文件: {SCENES_JSON}")
    print(f"[Config] 输出路径: {OUTPUT_JSON}")
    print(f"[Config] KG类型: {KG_TYPE}")
    print(f"[Config] Neo4j: {NEO4J_URI}")
    print(f"[Config] LLM模型: {GPT_MODEL}")

    if not os.path.exists(SCENES_JSON):
        print(f"\n[ERROR] 场景描述文件不存在: {SCENES_JSON}")
        print("请确保已生成场景描述JSON文件后再运行本脚本。")
        return

    print(f"\n[Step 2] 读取场景描述...")
    with open(SCENES_JSON, 'r', encoding='utf-8') as f:
        scenes_data = json.load(f)

    if isinstance(scenes_data, dict) and 'points' in scenes_data:
        scenes = scenes_data['points']
    elif isinstance(scenes_data, dict) and 'scenes' in scenes_data:
        scenes = scenes_data['scenes']
    elif isinstance(scenes_data, list):
        scenes = scenes_data
    else:
        print(f"[ERROR] 无法识别的JSON格式, keys={list(scenes_data.keys()) if isinstance(scenes_data, dict) else type(scenes_data)}")
        return

    num_points = len(scenes)
    print(f"   加载 {num_points} 个点位")

    print(f"\n[Step 3] 初始化定位器...")
    localizer = None
    try:
        localizer = BatchLocalizer()
        if not localizer.neo4j.test_connection():
            print("[ERROR] 无法连接到Neo4j数据库")
            return
        print("   Neo4j连接成功，定位器初始化完成")
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        if localizer:
            localizer.close()
        return

    print(f"\n[Step 4] 开始批量定位 ({num_points} 个点位)...")
    ground_truths = []
    localization_results = []

    for idx, scene in enumerate(scenes):
        point_id = scene.get('point_id', scene.get('id', idx))
        description = scene.get('description', scene.get('text', ''))
        gt_lon = scene.get('ground_truth_lon', scene.get('lon', scene.get('longitude', None)))
        gt_lat = scene.get('ground_truth_lat', scene.get('lat', scene.get('latitude', None)))

        if not description:
            print(f"   [{idx+1}/{num_points}] Point {point_id}: SKIP (无描述)")
            continue

        ground_truths.append({"lon": gt_lon, "lat": gt_lat})

        print(f"   [{idx+1}/{num_points}] Point {point_id}: ", end="", flush=True)
        start_time = time.time()
        result = localizer.localize_single(description)
        elapsed = time.time() - start_time

        detail = {
            "point_id": point_id,
            "ground_truth": {"lon": gt_lon, "lat": gt_lat},
            "predicted": {"lon": result.get("lon"), "lat": result.get("lat")},
            "status": result["status"],
        }

        if result["status"] == "success":
            dist = haversine_distance(gt_lat, gt_lon, result["lat"], result["lon"])
            detail["distance_error_m"] = round(dist, 2)
            detail["confidence"] = round(result.get("confidence", 0), 4)
            detail["num_references"] = result.get("num_references", 0)
            detail["num_matches"] = result.get("num_matches", 0)
            print(f"SUCCESS (距离:{dist:.1f}m, 置信度:{result.get('confidence',0):.2%}, 耗时:{elapsed:.1f}s)")
        else:
            detail["distance_error_m"] = None
            detail["message"] = result.get("message", "")
            print(f"{result['status'].upper()} ({result.get('message','')}, 耗时:{elapsed:.1f}s)")

        localization_results.append(detail)

    print(f"\n[Step 5] 计算评估指标...")
    metrics = evaluate_results(localization_results, ground_truths)

    print(f"\n[Step 6] 输出评估结果...")
    output = {
        "config": {
            "scenes_file": SCENES_JSON, "kg_type": KG_TYPE,
            "num_points": num_points, "neo4j_uri": NEO4J_URI,
            "llm_model": GPT_MODEL, "eval_thresholds_m": EVAL_DISTANCE_THRESHOLDS
        },
        "evaluation": metrics,
        "details": localization_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("  评估结果汇总")
    print(f"{'='*70}")
    print(f"  总样本数:     {metrics['num_total']}")
    print(f"  成功定位:     {metrics['num_success']}")
    print(f"  成功率:       {metrics['success_rate']:.2%}")
    if metrics['mean_distance_error_m']:
        print(f"  平均距离误差: {metrics['mean_distance_error_m']} m")
        print(f"  中位距离误差: {metrics['median_distance_error_m']} m")
    else:
        print("  平均距离误差: N/A")
        print("  中位距离误差: N/A")
    print(f"\n  Recall@K:")
    for key, value in metrics['recall'].items():
        print(f"    {key}: {value:.4f}")
    print(f"\n  距离分位数:")
    p = metrics.get('distance_percentiles', {})
    print(f"    P25: {p.get('p25')} m, P50: {p.get('p50')} m, P75: {p.get('p75')} m, P90: {p.get('p90')} m")
    print(f"\n  结果已保存至: {OUTPUT_JSON}")
    print(f"{'='*70}")

    if localizer:
        localizer.close()


if __name__ == "__main__":
    main()
