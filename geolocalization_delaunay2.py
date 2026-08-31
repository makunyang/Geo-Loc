"""
基于Delaunay三角网知识图谱的自然语言地理定位系统 (V3 - 子图匹配版)
适配Delaunay三角网关系结构：
- Building-Building关系类型: DELAUNAY (属性: distance_m, direction, angle_deg)
- Building-POI关系类型: INSIDE / NEAR (属性: relation, distance_m, poi_fclass, poi_name)

V3改进内容（子图匹配定位逻辑）：
1. 构建用户描述的子图模板：包含实体属性（颜色、POI）和实体间拓扑关系（方位、距离）
2. 全局候选生成：为每个实体独立搜索候选，不依赖邻居关系
3. 组合拓扑验证：验证候选组合是否满足子图的拓扑约束
4. 方位容差处理：考虑方位的多解性（如"东边"可能是E或NE）
5. 距离模糊匹配：使用容差范围而非精确约束
6. 拓扑一致性评分：基于整体拓扑结构匹配度评分
"""

import os
import json
import math
import time
import warnings
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from neo4j import GraphDatabase, exceptions
from openai import OpenAI, APIConnectionError, APITimeoutError

warnings.filterwarnings("ignore")

# ===================== 1. 全局配置 =====================

OPENAI_API_KEY = "8O9vGvq0gQ93aaSS2f2WvzkPuP8qNrBxdRKsUKJXCeXa4toN"
OPENAI_BASE_URL = "https://www.autodl.art/api/v1"
GPT_MODEL = "DeepSeek-V3.2"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "neo4j"

# ===================== 距离约束配置 =====================
MAX_NEIGHBOR_DISTANCE = 200     # Delaunay邻居最大距离阈值（米），超过此距离的边不视为有效邻居
MAX_2HOP_DISTANCE = 300         # 两跳邻居最大距离阈值（米）
DISTANCE_TOLERANCE_RATIO = 0.5  # 距离容差比例，用户描述距离与实际距离的允许偏差比例
KNN_FALLBACK_COUNT = 20         # KNN兜底搜索时返回的最近建筑数量
KNN_FALLBACK_MAX_DISTANCE = 500 # KNN兜底搜索的最大距离（米）

# ===================== 方位容差配置 =====================
DIRECTION_TOLERANCE_ANGLES = {
    # 每个方位允许的偏差角度范围（度）
    "E": [0, 45],      # 东：0-45度（含NE）
    "NE": [22.5, 67.5],
    "N": [45, 135],    # 北：45-135度（含NW）
    "NW": [112.5, 157.5],
    "W": [135, 225],   # 西：135-225度（含SW）
    "SW": [202.5, 247.5],
    "S": [225, 315],   # 南：225-315度（含SE）
    "SE": [292.5, 337.5],
}

# 方位多解映射：用户描述的方位可能对应多个实际方位
DIRECTION_MULTI_MAP = {
    "E": ["E", "NE", "SE"],      # 东边可能是E、NE或SE
    "NE": ["NE", "E", "N"],
    "N": ["N", "NE", "NW"],      # 北边可能是N、NE或NW
    "NW": ["NW", "N", "W"],
    "W": ["W", "NW", "SW"],      # 西边可能是W、NW或SW
    "SW": ["SW", "W", "S"],
    "S": ["S", "SW", "SE"],      # 南边可能是S、SW或SE
    "SE": ["SE", "S", "E"],
}

# POI类型映射（中文 -> fclass）
POI_TYPE_MAP = {
    "快餐店": "fast_food",
    "餐厅": "restaurant",
    "咖啡厅": "cafe",
    "咖啡店": "cafe",
    "银行": "bank",
    "超市": "supermarket",
    "药店": "pharmacy",
    "便利店": "convenience",
    "加油站": "fuel",
    "酒店": "hotel",
    "快餐": "fast_food",
    "饭馆": "restaurant",
    "理发店": "hairdresser",
    "美容院": "beauty",
    "服装店": "clothes",
    "书店": "books",
    "药店": "pharmacy",
    "医院": "hospital",
    "学校": "school",
    "电影院": "cinema",
    "健身房": "gym",
    "邮箱": "post_box",
    "监控": "camera_surveillance",
    "自动售货机": "vending_parking",
    "户外用品店": "outdoor_shop",
    "运动用品店": "sports_shop",
    "大使馆": "embassy",
    "牙医诊所": "dentist",
    "剧院": "theatre"
}

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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的球面距离（米）"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS * c


def direction_to_offset(direction: str, distance_meters: float, lat: float = 60.0) -> Tuple[float, float]:
    """将方位和距离转换为经纬度偏移"""
    angle = DIRECTION_ANGLE.get(direction, 0)
    angle_rad = math.radians(angle)
    meters_per_deg_lon = 111000 * math.cos(math.radians(lat))
    meters_per_deg_lat = 111000
    dlon = (distance_meters * math.cos(angle_rad)) / meters_per_deg_lon
    dlat = (distance_meters * math.sin(angle_rad)) / meters_per_deg_lat
    return dlon, dlat


def calculate_relative_direction(dir1: str, dir2: str) -> str:
    """计算两个实体相对于彼此的方位（简化版）"""
    if not dir1 or not dir2:
        return ""

    angle1 = DIRECTION_ANGLE.get(dir1, 0)
    angle2 = DIRECTION_ANGLE.get(dir2, 0)

    relative_angle = (angle2 - angle1 + 360) % 360
    idx = int(((relative_angle + 22.5) % 360) / 45)
    return DIRECTIONS_8[idx]


def compute_angle_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """计算从点1到点2的角度（度）"""
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    angle = math.degrees(math.atan2(dlat, dlon))
    return (angle + 360) % 360


def compute_direction_8(angle_deg: float) -> str:
    """将角度转换为8方位"""
    idx = int(((angle_deg + 22.5) % 360) / 45)
    return DIRECTIONS_8[idx]


def calculate_possible_directions(dir1: str, dist1: float, dir2: str, dist2: float) -> List[str]:
    """
    根据两个实体相对于用户的方位和距离，计算它们之间可能的相对方位范围
    """
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


def resolve_chain_entities(entities: List['SpatialEntity']) -> bool:
    """
    递推解析链式描述的实体，将 relative_to 关系转换为 direction_from_user
    
    算法：
    1. 对每个有 direction_from_user 的实体标记为已解析，计算其相对于用户的笛卡尔坐标
    2. 对每个 relative_to 指向已解析实体的未解析实体，计算其 direction_from_user
    3. 重复直到全部解析或无法继续
    
    返回: True 表示全部解析成功
    """
    import math
    
    # 计算每个已解析实体相对于用户的笛卡尔坐标（单位：米，以北为y轴正方向）
    # 坐标系: x = 东, y = 北  (与 matlab 的 atan2(y, x) 对应)
    positions = {}  # entity_id -> (x, y)
    
    for e in entities:
        if e.direction_from_user and e.estimated_distance and e.estimated_distance > 0:
            angle = math.radians(DIRECTION_ANGLE.get(e.direction_from_user, 0))
            x = e.estimated_distance * math.cos(angle)
            y = e.estimated_distance * math.sin(angle)
            positions[e.entity_id] = (x, y)
            e.resolved = True
    
    # 迭代解析链式实体
    max_iterations = len(entities) * 2  # 防止死循环
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
                continue  # 参照实体还未解析
            
            # 参照实体的坐标
            ref_x, ref_y = positions[ref_id]
            
            # 链式偏移量
            chain_angle = math.radians(DIRECTION_ANGLE.get(e.direction_from_ref, 0))
            chain_dx = e.distance_from_ref * math.cos(chain_angle)
            chain_dy = e.distance_from_ref * math.sin(chain_angle)
            
            # 当前实体的绝对坐标
            new_x = ref_x + chain_dx
            new_y = ref_y + chain_dy
            
            # 计算相对于用户的方位和距离
            new_dist = math.sqrt(new_x**2 + new_y**2)
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
    
    # 检查是否全部解析
    unresolved = [e.entity_id for e in entities if not e.resolved and not e.direction_from_user]
    if unresolved:
        print(f"   ⚠️ 无法解析的实体: {unresolved}")
        # 对于无法解析的实体，尝试给默认值
        for e in entities:
            if not e.resolved and not e.direction_from_user:
                e.direction_from_user = ""
                e.estimated_distance = 0.0
    
    return len(unresolved) == 0


def normalize_poi_type(poi_type: str) -> str:
    """将中文POI类型转换为fclass"""
    # 处理poi_type可能是列表的情况
    if isinstance(poi_type, list):
        # 如果是列表，取第一个元素并标准化
        if len(poi_type) > 0:
            poi_type = poi_type[0]
        else:
            return ""
    return POI_TYPE_MAP.get(poi_type, poi_type)


# ===================== 2. 数据类 =====================

@dataclass
class SpatialEntity:
    """空间实体"""
    entity_id: str
    entity_type: str  # Building, POI, Road
    lon: float = 0.0
    lat: float = 0.0
    # Building属性
    color_side: str = ""
    color_top: str = ""
    # POI属性
    fclass: str = ""
    poi_name: str = ""
    # Road属性
    road_type: str = ""
    road_orientation: str = ""
    # 用户参照系
    direction_from_user: str = ""
    estimated_distance: float = 0.0
    # 链式参照系（相对于其他实体）
    relative_to: str = ""           # 被参照的实体ID（如"ref_0"）
    direction_from_ref: str = ""    # 相对于被参照实体的方位
    distance_from_ref: float = 0.0  # 相对于被参照实体的距离（米）
    resolved: bool = False          # 是否已递推解析为用户参照系
    associated_poi: Any = field(default_factory=dict)  # 可能是dict或list
    # 实体间参照系
    possible_relative_directions: Dict[str, List[str]] = field(default_factory=dict)

    def _get_poi_dict(self) -> Optional[Dict]:
        """
        获取POI字典（兼容列表和字典两种格式）
        - 如果是字典，直接返回
        - 如果是列表，返回第一个元素
        - 否则返回None
        """
        if isinstance(self.associated_poi, dict):
            return self.associated_poi
        elif isinstance(self.associated_poi, list) and len(self.associated_poi) > 0:
            return self.associated_poi[0]
        return None

    def _get_all_poi_dicts(self) -> List[Dict]:
        """
        获取所有POI字典（支持多个POI）
        - 如果是字典，返回包含该字典的列表
        - 如果是列表，返回所有元素
        - 否则返回空列表
        """
        if isinstance(self.associated_poi, dict):
            return [self.associated_poi]
        elif isinstance(self.associated_poi, list):
            return self.associated_poi
        return []

    def has_poi_constraint(self) -> bool:
        """检查该实体是否有POI约束"""
        poi_dict = self._get_poi_dict()
        return bool(poi_dict and poi_dict.get("poi_type"))

    def get_poi_type(self) -> str:
        """获取该实体的第一个POI类型约束"""
        poi_dict = self._get_poi_dict()
        if poi_dict and poi_dict.get("poi_type"):
            return normalize_poi_type(poi_dict["poi_type"])
        return ""

    def get_all_poi_types(self) -> List[str]:
        """获取该实体的所有POI类型约束"""
        return [normalize_poi_type(p.get("poi_type", "")) for p in self._get_all_poi_dicts() if p.get("poi_type")]

    def get_first_poi_type(self) -> str:
        """获取该实体的第一个POI类型约束（用于MatchedEntity的required_poi_type字段）"""
        all_types = self.get_all_poi_types()
        return all_types[0] if all_types else ""

    def to_dict(self) -> Dict:
        return {
            "id": self.entity_id,
            "type": self.entity_type,
            "lon": self.lon,
            "lat": self.lat,
            "color_side": self.color_side,
            "color_top": self.color_top,
            "fclass": self.fclass,
            "direction_from_user": self.direction_from_user,
            "estimated_distance": self.estimated_distance,
            "has_poi_constraint": self.has_poi_constraint(),
            "poi_type": self.get_poi_type()
        }


@dataclass
class MatchedEntity:
    """匹配到的实体"""
    query_id: str
    entity_id: str
    entity_type: str
    lon: float
    lat: float
    color_side: str = ""
    color_top: str = ""
    fclass: str = ""
    poi_name: str = ""
    poi_validation: bool = False  # 是否通过了POI验证
    required_poi_type: str = ""   # 该实体要求的第一个POI类型
    required_poi_types: List[str] = field(default_factory=list)  # 所有要求的POI类型
    matched_poi_names: List[str] = field(default_factory=list)   # 满足的POI名称列表

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "lon": self.lon,
            "lat": self.lat,
            "color_side": self.color_side,
            "color_top": self.color_top,
            "fclass": self.fclass,
            "poi_validation": self.poi_validation,
            "required_poi_type": self.required_poi_type,
            "required_poi_types": self.required_poi_types,
            "matched_poi_names": self.matched_poi_names
        }


@dataclass
class CandidateCombination:
    """候选组合"""
    entities: List[MatchedEntity] = field(default_factory=list)
    used_ids: Set[str] = field(default_factory=set)
    total_score: float = 0.0
    confidence: float = 0.0
    poi_constraint_satisfied: bool = False
    poi_satisfaction_count: int = 0  # 新增：满足的POI约束数量
    poi_total_count: int = 0         # 新增：总共需要的POI约束数量
    satisfied_poi_types: List[str] = field(default_factory=list)  # 新增：满足的POI类型列表

    def add_entity(self, entity: MatchedEntity) -> bool:
        if entity.entity_id in self.used_ids:
            return False
        self.entities.append(entity)
        self.used_ids.add(entity.entity_id)
        return True

    def to_dict(self) -> Dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "total_score": self.total_score,
            "confidence": self.confidence,
            "poi_constraint_satisfied": self.poi_constraint_satisfied,
            "poi_satisfaction_count": self.poi_satisfaction_count,
            "poi_total_count": self.poi_total_count,
            "satisfied_poi_types": self.satisfied_poi_types
        }


@dataclass
class SubgraphTemplate:
    """用户描述的子图模板"""
    entities: List[SpatialEntity]  # 实体列表
    entity_relations: Dict[Tuple[str, str], Dict] = field(default_factory=dict)  # 实体间关系
    
    def add_relation(self, entity1_id: str, entity2_id: str, 
                     possible_directions: List[str], estimated_distance_range: Tuple[float, float]):
        """添加实体间关系"""
        key = (min(entity1_id, entity2_id), max(entity1_id, entity2_id))
        self.entity_relations[key] = {
            "possible_directions": possible_directions,  # 可能的方位列表（考虑多解）
            "distance_range": estimated_distance_range,  # 距离范围（min, max）
        }
    
    def get_relation(self, entity1_id: str, entity2_id: str) -> Optional[Dict]:
        """获取两个实体间的关系约束"""
        key = (min(entity1_id, entity2_id), max(entity1_id, entity2_id))
        return self.entity_relations.get(key)
    
    def has_relation(self, entity1_id: str, entity2_id: str) -> bool:
        """判断两个实体间是否有关系约束"""
        key = (min(entity1_id, entity2_id), max(entity1_id, entity2_id))
        return key in self.entity_relations


def generate_subgraph_variants(base_template: SubgraphTemplate) -> List[SubgraphTemplate]:
    """
    根据方位多解生成多个子图变体
    每个变体对应一种具体的方位关系组合
    """
    variants = []
    
    # 获取所有有方位关系约束的边
    edges_with_directions = []
    for (e1_id, e2_id), relation in base_template.entity_relations.items():
        if relation.get("possible_directions"):
            edges_with_directions.append(((e1_id, e2_id), relation["possible_directions"]))
    
    if not edges_with_directions:
        # 没有方位约束，返回原始模板
        return [base_template]
    
    # 生成所有方位组合（笛卡尔积）
    from itertools import product
    direction_lists = [dirs for (_, dirs) in edges_with_directions]
    
    for direction_combo in product(*direction_lists):
        # 创建新变体
        variant = SubgraphTemplate(entities=base_template.entities[:])
        
        # 复制所有关系，但将方位固定为具体值
        for i, ((e1_id, e2_id), _) in enumerate(edges_with_directions):
            fixed_dir = direction_combo[i]
            # 获取原始距离范围
            orig_relation = base_template.get_relation(e1_id, e2_id)
            dist_range = orig_relation.get("distance_range", (0, 1000)) if orig_relation else (0, 1000)
            
            # 添加固定方位的关系
            variant.add_relation(e1_id, e2_id, [fixed_dir], dist_range)
        
        # 复制其他没有方位约束的关系
        for (e1_id, e2_id), relation in base_template.entity_relations.items():
            if not relation.get("possible_directions"):
                variant.add_relation(e1_id, e2_id, [], relation.get("distance_range", (0, 1000)))
        
        variants.append(variant)
    
    return variants


# ===================== 3. Neo4j连接 =====================

class Neo4jConnector:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PWD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cql: str, parameters: Dict = None) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(cql, parameters or {})
            return [record.data() for record in result]

    def test_connection(self) -> bool:
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                return result.single()["num"] == 1
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False


# ===================== 4. LLM客户端 =====================

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

    def call(self, prompt: str, temperature: float = 0.1, max_retries: int = 3) -> Dict:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=120
                )
                content = response.choices[0].message.content.strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    raise ValueError(f"无法解析JSON响应: {content}")
            except (APIConnectionError, APITimeoutError) as e:
                if attempt == max_retries - 1:
                    raise Exception(f"大模型调用失败（重试{max_retries}次）: {e}")
                print(f"⚠️ 大模型调用超时，重试第{attempt + 1}次...")
                time.sleep(2)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"大模型调用失败: {e}")
                print(f"⚠️ 大模型调用错误，重试第{attempt + 1}次: {e}")
                time.sleep(1)


# ===================== 5. 用户输入解析器 =====================

class UserInputParser:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def parse(self, user_input: str) -> Dict:
        prompt = f"""你是专业的地理空间描述解析专家。请严格按以下规则解析用户输入。

【核心原则：每个建筑是一个参照物，POI是建筑的属性】
- 每个建筑（Building）是一个独立的参照物，按出现顺序赋予ID（ref_0, ref_1, ref_2...）
- 建筑内提到的所有店铺/设施都是该建筑的POI属性，不是独立参照物
- 例如："一栋建筑里有理发店和自动售货机" → 这是一个参照物，包含两个POI
- 绝对不要把同一建筑内的多个POI拆分成多个参照物

【重要：POI类型必须转换为英文fclass格式】
- 快餐店 → fast_food    | 餐厅/饭馆 → restaurant    | 咖啡店/咖啡厅 → cafe
- 银行 → bank           | 超市 → supermarket       | 药店 → pharmacy
- 便利店 → convenience  | 理发店 → hairdresser     | 美容院 → beauty
- 服装店 → clothes      | 书店 → books            | 医院 → hospital
- 学校 → school         | 电影院 → cinema         | 健身房 → gym
- 自动售货机 → vending_parking | 户外用品店 → outdoor_shop | 运动用品店 → sports_shop
- 大使馆 → embassy      | 牙医诊所 → dentist       | 剧院 → theatre
- 社区中心 → community_centre

【解析规则】
1. **道路信息**：提取用户当前所在的道路
   - road_orientation: 道路走向（东西向/南北向等）

2. **建筑信息**（Building）—— 支持两种描述方式：

   **方式A：相对于用户描述（默认）**
   例如"北边有一栋建筑"、"我左边有一栋建筑"
   - direction_from_user: 相对于用户的方位（N/S/E/W/NE/NW/SE/SW）
   - estimated_distance: 到用户的估计距离（米）
   - relative_to: 留空 null

   **方式B：相对于其他建筑描述（链式描述）**
   例如"A的东边有B"、"B的南边有C"、"东边那栋建筑的旁边还有一栋"
   - relative_to: 被参照的建筑ID（如"ref_0"）
   - direction_from_ref: 相对于被参照建筑的方位（N/S/E/W/NE/NW/SE/SW）
   - estimated_distance_from_ref: 两建筑间的估计距离（米）
   - direction_from_user: 留空 null
   - estimated_distance: 留空 null

   **公共字段（两种方式都需要）**：
   - color_side: 侧面颜色（标准化：灰色、蓝色等）
   - color_top: 顶面颜色
   - associated_poi: 建筑内包含的POI列表

3. **POI规则（极其重要）**：
   - 同一句话中描述的建筑，其内部的所有店铺/设施都属于同一个参照物
   - 单个POI用字典格式：associated_poi: {{"poi_type": "fast_food"}}
   - 多个POI用列表格式：associated_poi: [{{"poi_type": "hairdresser"}}, {{"poi_type": "vending_parking"}}]
   - 即使POI类型不在上述列表中，也要生成associated_poi（保留原始中文类型）
   - 任何提到的店铺、设施都必须生成associated_poi，不可遗漏

【输出格式】
仅返回JSON，不要其他文字：
{{
    "user_context": {{
        "road": {{
            "road_orientation": "东西向"
        }}
    }},
    "reference_objects": [
        {{
            "entity_type": "Building",
            "direction_from_user": "NE",
            "estimated_distance": 20,
            "relative_to": null,
            "direction_from_ref": null,
            "estimated_distance_from_ref": null,
            "color_side": "深蓝色",
            "color_top": "灰色",
            "associated_poi": {{"poi_type": "restaurant"}}
        }},
        {{
            "entity_type": "Building",
            "direction_from_user": "SE",
            "estimated_distance": 25,
            "relative_to": null,
            "direction_from_ref": null,
            "estimated_distance_from_ref": null,
            "color_side": "蓝色",
            "color_top": "灰色"
        }},
        {{
            "entity_type": "Building",
            "direction_from_user": null,
            "estimated_distance": null,
            "relative_to": "ref_1",
            "direction_from_ref": "SW",
            "estimated_distance_from_ref": 30,
            "color_side": "灰色",
            "color_top": "灰色",
            "associated_poi": [
                {{"poi_type": "hairdresser"}},
                {{"poi_type": "vending_parking"}}
            ]
        }}
    ]
}}

【示例】对于"我在A北边10米，A的东边20米有B，B的南边15米有C"：
- ref_0(A): direction_from_user="S", estimated_distance=10, relative_to=null
- ref_1(B): relative_to="ref_0", direction_from_ref="E", estimated_distance_from_ref=20
- ref_2(C): relative_to="ref_1", direction_from_ref="S", estimated_distance_from_ref=15

注意：direction_from_user 和 relative_to 互斥，每个建筑只用一种方式定位。

用户输入：{user_input}
"""
        result = self.llm.call(prompt)

        # 后处理：合并同一建筑的多个POI参照物
        result = self._merge_poi_entities(result)

        return result

    def _merge_poi_entities(self, parsed: Dict) -> Dict:
        """
        后处理：检测并合并被LLM错误拆分的同一建筑的多个POI参照物
        判断依据：如果两个参照物方位相同且距离相同，则认为是同一建筑
        """
        ref_objects = parsed.get("reference_objects", [])
        if not ref_objects or len(ref_objects) <= 1:
            return parsed

        merged = []
        used_indices = set()

        for i in range(len(ref_objects)):
            if i in used_indices:
                continue

            current = ref_objects[i]
            # 收集当前实体的所有POI
            current_pois = self._extract_pois(current)

            # 查找方位和距离相同的其他参照物
            for j in range(i + 1, len(ref_objects)):
                if j in used_indices:
                    continue

                other = ref_objects[j]

                # 判断是否为同一建筑：方位相同且距离相同
                same_direction = (current.get("direction_from_user") == other.get("direction_from_user"))
                same_distance = (current.get("estimated_distance") == other.get("estimated_distance"))

                if same_direction and same_distance:
                    # 合并POI
                    other_pois = self._extract_pois(other)
                    current_pois.extend(other_pois)
                    used_indices.add(j)

                    # 合并颜色信息（取更详细的）
                    if not current.get("color_side") and other.get("color_side"):
                        current["color_side"] = other["color_side"]
                    if not current.get("color_top") and other.get("color_top"):
                        current["color_top"] = other["color_top"]

                    print(f"     🔗 合并参照物: ref_{i} + ref_{j} (方位={current.get('direction_from_user')}, 距离={current.get('estimated_distance')}m)")

            # 更新合并后的POI
            if current_pois:
                if len(current_pois) == 1:
                    current["associated_poi"] = current_pois[0]
                else:
                    current["associated_poi"] = current_pois

            merged.append(current)
            used_indices.add(i)

        if len(merged) != len(ref_objects):
            print(f"     ✅ 参照物合并: {len(ref_objects)} → {len(merged)} 个")

        parsed["reference_objects"] = merged
        return parsed

    def _extract_pois(self, ref_obj: Dict) -> List[Dict]:
        """从参照物中提取POI列表"""
        poi = ref_obj.get("associated_poi")
        if not poi:
            return []

        if isinstance(poi, dict):
            if poi.get("poi_type"):
                return [poi]
            return []
        elif isinstance(poi, list):
            return [p for p in poi if isinstance(p, dict) and p.get("poi_type")]
        return []


# ===================== 6. 知识图谱匹配器 =====================

class KGMatcher:
    def __init__(self, neo4j_conn: Neo4jConnector):
        self.neo4j = neo4j_conn
        # 预计算的最短路径缓存 { (node1_id, node2_id): {path: [...], distance: float} }
        self._shortest_path_cache = {}
    
    def _get_shortest_path(self, start_id: str, end_id: str) -> Optional[Dict]:
        """
        获取两个节点之间的最短路径（使用BFS）
        返回: {path: [node_id, ...], distance: float, edges: [{direction, distance_m}, ...]}
        """
        cache_key = (min(start_id, end_id), max(start_id, end_id))
        if cache_key in self._shortest_path_cache:
            return self._shortest_path_cache[cache_key]
        
        # 使用Neo4j查询最短路径
        query = """
        MATCH (start:Building {id: $start_id}), (end:Building {id: $end_id})
        MATCH p = shortestPath((start)-[:DELAUNAY*1..4]-(end))
        RETURN [node in nodes(p) | node.id] as path,
               [rel in relationships(p) | {direction: rel.direction, distance_m: rel.distance_m}] as edges,
               reduce(dist = 0, rel in relationships(p) | dist + rel.distance_m) as total_distance
        LIMIT 1
        """
        
        try:
            results = self.neo4j.query(query, {"start_id": start_id, "end_id": end_id})
            if results:
                result = results[0]
                path_info = {
                    "path": result["path"],
                    "edges": result["edges"],
                    "distance": result["total_distance"]
                }
                self._shortest_path_cache[cache_key] = path_info
                return path_info
        except Exception as e:
            # 最短路径查询失败，可能是节点间无连接
            pass
        
        return None
    
    def _infer_direction_from_path(self, path_info: Dict) -> str:
        """
        从路径推断整体方位
        基于路径中各边的方向向量合成
        """
        edges = path_info.get("edges", [])
        if not edges:
            return ""
        
        # 将各边的方向转换为向量并求和
        dx_total, dy_total = 0.0, 0.0
        for edge in edges:
            direction = edge.get("direction", "")
            distance = edge.get("distance_m", 0)
            angle = DIRECTION_ANGLE.get(direction, 0)
            dx = distance * math.cos(math.radians(angle))
            dy = distance * math.sin(math.radians(angle))
            dx_total += dx
            dy_total += dy
        
        # 计算合成方向
        if dx_total == 0 and dy_total == 0:
            return ""
        
        angle = math.degrees(math.atan2(dy_total, dx_total))
        angle = (angle + 360) % 360
        idx = int(((angle + 22.5) % 360) / 45)
        return DIRECTIONS_8[idx]
    
    def _select_key_edges(self, subgraph: SubgraphTemplate) -> Set[Tuple[str, str]]:
        """
        选择关键边（用于松弛约束匹配）
        策略：选择形成最小生成树的边，优先选择距离较近的边
        关键边必须在KG中存在直接DELAUNAY关系
        """
        entities = subgraph.entities
        if len(entities) <= 1:
            return set()
        
        # 获取所有边并按距离排序
        all_edges = []
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            dist_range = relation.get("distance_range", (0, 1000))
            avg_dist = (dist_range[0] + dist_range[1]) / 2
            all_edges.append((avg_dist, (e1_id, e2_id)))
        
        # 按距离升序排序
        all_edges.sort(key=lambda x: x[0])
        
        # 使用并查集选择最小生成树的边
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
    
    def _check_relation_via_path(self, c1_id: str, c2_id: str, 
                                  expected_relation: Dict) -> Tuple[bool, float]:
        """
        通过路径检查两个候选节点是否满足预期的关系约束
        返回: (是否满足, 匹配得分)
        """
        # 首先检查是否有直接边
        direct_query = """
        MATCH (a:Building {id: $id1})-[r:DELAUNAY]-(b:Building {id: $id2})
        RETURN r.direction as direction, r.distance_m as distance
        """
        direct_results = self.neo4j.query(direct_query, {"id1": c1_id, "id2": c2_id})
        
        if direct_results:
            # 有直接边，直接验证
            result = direct_results[0]
            actual_dir = result["direction"]
            actual_dist = result["distance"]
            
            dirs = expected_relation.get("possible_directions", [])
            min_dist, max_dist = expected_relation.get("distance_range", (0, 1000))
            
            dir_match = actual_dir in dirs if dirs else True
            dist_match = min_dist <= actual_dist <= max_dist
            
            score = 0.0
            if dir_match:
                score += 1.0
            elif actual_dir in self._get_adjacent_directions(dirs[0] if dirs else ""):
                score += 0.5
            
            if dist_match:
                score += 1.0
            elif min_dist * 0.5 <= actual_dist <= max_dist * 2:
                score += 0.5
            
            return (dir_match or dist_match), score
        
        # 没有直接边，通过路径推断
        path_info = self._get_shortest_path(c1_id, c2_id)
        if not path_info:
            return False, 0.0
        
        # 路径长度限制（最多3跳）
        if len(path_info["path"]) > 4:  # 节点数 > 4 表示 > 3跳
            return False, 0.0
        
        # 推断方位
        inferred_dir = self._infer_direction_from_path(path_info)
        actual_dist = path_info["distance"]
        
        dirs = expected_relation.get("possible_directions", [])
        min_dist, max_dist = expected_relation.get("distance_range", (0, 1000))
        
        # 验证方位（宽松约束）
        dir_match = inferred_dir in dirs if dirs else True
        dir_score = 1.0 if dir_match else (0.5 if inferred_dir in self._get_adjacent_directions(dirs[0] if dirs else "") else 0)
        
        # 验证距离（考虑路径累加误差）
        path_len = len(path_info["path"]) - 1
        tolerance = 1 + 0.3 * path_len  # 每多一跳增加30%容差
        dist_match = min_dist <= actual_dist <= max_dist * tolerance
        dist_score = 1.0 if dist_match else (0.5 if min_dist * 0.5 <= actual_dist <= max_dist * tolerance * 2 else 0)
        
        # 路径越长，置信度越低
        confidence = 1.0 / path_len
        
        total_score = (dir_score + dist_score) * confidence
        return (dir_match or dist_match), total_score

    def expand_color(self, color: str, strict: bool = True) -> List[str]:
        """扩展颜色匹配"""
        color = color.strip()

        if strict:
            return [color]

        color_map = {
            "蓝": ["蓝色", "深蓝", "浅蓝", "天蓝", "深蓝色", "浅蓝色"],
            "灰": ["灰色", "深灰", "浅灰", "银灰", "深灰色", "浅灰色"],
            "白": ["白色", "米白", "乳白"],
            "黑": ["黑色", "深黑", "墨黑"],
            "红": ["红色", "深红", "浅红", "砖红"],
            "绿": ["绿色", "深绿", "浅绿", "草绿"],
            "黄": ["黄色", "深黄", "浅黄", "米黄"],
            "棕": ["棕色", "深棕", "浅棕", "褐色"],
        }

        for base_color, variants in color_map.items():
            if color == base_color or color in variants:
                return variants

        return [color]

    def normalize_color(self, color: str) -> str:
        """将颜色标准化为数据库中的格式"""
        color = color.strip()

        color_normalize_map = {
            "蓝": "蓝色",
            "灰": "灰色",
            "白": "白色",
            "黑": "黑色",
            "红": "红色",
            "绿": "绿色",
            "黄": "黄色",
            "棕": "棕色",
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

    def match_buildings_strict(self, color_side: str = None, color_top: str = None,
                                limit: int = 10000) -> List[Dict]:
        """严格匹配候选建筑"""
        query_parts = ["MATCH (b:Building)"]
        where_conditions = []
        params = {}

        if color_side and color_side.strip():
            normalized_side = self.normalize_color(color_side)
            where_conditions.append("b.color_side = $color_side")
            params["color_side"] = normalized_side

        if color_top and color_top.strip():
            normalized_top = self.normalize_color(color_top)
            where_conditions.append("b.color_top = $color_top")
            params["color_top"] = normalized_top

        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))

        query_parts.append(f"""
        RETURN b.id AS id, b.color_side AS color_side, b.color_top AS color_top,
               b.lon AS lon, b.lat AS lat
        LIMIT {limit}
        """)

        query = "\n".join(query_parts)
        results = self.neo4j.query(query, params)

        if not results:
            print(f"      ⚠️ 严格匹配无结果，尝试宽松匹配...")
            return self.match_buildings(color_side, color_top, limit)

        return results

    def match_buildings(self, color_side: str = None, color_top: str = None,
                       limit: int = 10000) -> List[Dict]:
        """匹配候选建筑"""
        query_parts = ["MATCH (b:Building)"]
        where_conditions = []
        params = {}

        if color_side and color_side.strip():
            colors = self.expand_color(color_side)
            if len(colors) > 1:
                cond = " OR ".join([f"b.color_side CONTAINS $cs_{i}" for i in range(len(colors))])
                where_conditions.append(f"({cond})")
                for i, c in enumerate(colors):
                    params[f"cs_{i}"] = c
            else:
                where_conditions.append("b.color_side CONTAINS $color_side")
                params["color_side"] = color_side

        if color_top and color_top.strip():
            colors = self.expand_color(color_top)
            if len(colors) > 1:
                cond = " OR ".join([f"b.color_top CONTAINS $ct_{i}" for i in range(len(colors))])
                where_conditions.append(f"({cond})")
                for i, c in enumerate(colors):
                    params[f"ct_{i}"] = c
            else:
                where_conditions.append("b.color_top CONTAINS $color_top")
                params["color_top"] = color_top

        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))

        query_parts.append(f"""
        RETURN b.id AS id, b.color_side AS color_side, b.color_top AS color_top,
               b.lon AS lon, b.lat AS lat
        LIMIT {limit}
        """)

        query = "\n".join(query_parts)
        return self.neo4j.query(query, params)

    def check_poi_in_building(self, building_id: str, poi_type: str) -> Tuple[bool, Optional[str]]:
        """
        检查建筑内是否有指定类型的POI
        返回: (是否找到POI, POI的id或名称)
        """
        fclass = normalize_poi_type(poi_type)

        ids_to_try = [building_id]
        try:
            ids_to_try.append(int(building_id))
        except:
            pass

        for bid in ids_to_try:
            # Delaunay KG: INSIDE是关系类型，不是属性
            query1 = """
            MATCH (b:Building {id: $building_id})-[r:INSIDE]-(p:POI)
            WHERE p.fclass = $fclass
            RETURN p.id AS poi_id, p.name AS poi_name
            LIMIT 1
            """
            try:
                result = self.neo4j.query(query1, {"building_id": bid, "fclass": fclass})
                if result and result[0].get("poi_id"):
                    return True, result[0].get("poi_name", result[0].get("poi_id"))
            except Exception as e:
                pass

            # 尝试使用NEAR关系类型作为备选查询
            query2 = """
            MATCH (b:Building {id: $building_id})-[r:NEAR]-(p:POI)
            WHERE p.fclass = $fclass
            RETURN p.id AS poi_id, p.name AS poi_name
            LIMIT 1
            """
            try:
                result = self.neo4j.query(query2, {"building_id": bid, "fclass": fclass})
                if result and result[0].get("poi_id"):
                    return True, result[0].get("poi_name", result[0].get("poi_id"))
            except Exception as e:
                pass

            # 尝试模糊匹配fclass
            query3 = """
            MATCH (b:Building {id: $building_id})-[r:INSIDE]-(p:POI)
            WHERE p.fclass CONTAINS $fclass
            RETURN p.id AS poi_id, p.name AS poi_name
            LIMIT 1
            """
            try:
                result = self.neo4j.query(query3, {"building_id": bid, "fclass": fclass})
                if result and result[0].get("poi_id"):
                    return True, result[0].get("poi_name", result[0].get("poi_id"))
            except Exception as e:
                pass

        return False, None

    def check_all_pois_in_building(self, building_id: str, poi_types: List[str]) -> Tuple[bool, List[str]]:
        """
        检查建筑内是否包含所有指定类型的POI（支持多个POI）
        返回: (是否包含所有POI, 满足的POI名称列表)
        """
        if not poi_types:
            return True, []

        satisfied_pois = []
        for poi_type in poi_types:
            has_poi, poi_name = self.check_poi_in_building(building_id, poi_type)
            if has_poi and poi_name:
                satisfied_pois.append(poi_name)
            else:
                # 如果任何一个POI不满足，返回False
                return False, []

        return True, satisfied_pois

    def get_building_pois(self, building_id: str) -> List[Dict]:
        """
        获取建筑内所有POI的详细信息
        返回: POI列表
        """
        ids_to_try = [building_id]
        try:
            ids_to_try.append(int(building_id))
        except:
            pass

        for bid in ids_to_try:
            # Delaunay KG: INSIDE是关系类型
            query = """
            MATCH (b:Building {id: $building_id})-[r:INSIDE]-(p:POI)
            RETURN p.id AS poi_id, p.name AS poi_name, p.fclass AS fclass
            """
            try:
                results = self.neo4j.query(query, {"building_id": bid})
                if results:
                    return results
            except Exception as e:
                pass

            # 尝试NEAR关系作为备选
            query2 = """
            MATCH (b:Building {id: $building_id})-[r:NEAR]-(p:POI)
            RETURN p.id AS poi_id, p.name AS poi_name, p.fclass AS fclass
            """
            try:
                results = self.neo4j.query(query2, {"building_id": bid})
                if results:
                    return results
            except Exception as e:
                pass

        return []

    def _build_subgraph_template(self, entities: List[SpatialEntity]) -> SubgraphTemplate:
        """根据用户描述构建子图模板"""
        template = SubgraphTemplate(entities=entities)
        
        # 为每对实体计算可能的方位关系和距离范围
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue
                
                # 计算可能的相对方位
                possible_dirs = self._calculate_possible_directions_for_pair(e1, e2)
                
                # 计算距离范围（基于用户估计距离）
                dist_range = self._calculate_distance_range_for_pair(e1, e2)
                
                template.add_relation(e1.entity_id, e2.entity_id, possible_dirs, dist_range)
        
        return template
    
    def _calculate_possible_directions_for_pair(self, e1: SpatialEntity, e2: SpatialEntity) -> List[str]:
        """计算两个实体间可能的相对方位（考虑方位多解）"""
        # 如果两个实体都有相对于用户的方位信息
        if e1.direction_from_user and e2.direction_from_user:
            dir1 = e1.direction_from_user
            dir2 = e2.direction_from_user
            
            # 获取方位多解
            possible_dirs1 = DIRECTION_MULTI_MAP.get(dir1, [dir1])
            possible_dirs2 = DIRECTION_MULTI_MAP.get(dir2, [dir2])
            
            # 计算所有可能的相对方位
            possible_relative_dirs = []
            for d1 in possible_dirs1:
                for d2 in possible_dirs2:
                    angle1 = DIRECTION_ANGLE.get(d1, 0)
                    angle2 = DIRECTION_ANGLE.get(d2, 0)
                    relative_angle = (angle2 - angle1) % 360
                    
                    # 将角度转换为方位
                    dirs = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
                    idx = int(((relative_angle + 22.5) % 360) / 45)
                    relative_dir = dirs[idx]
                    if relative_dir not in possible_relative_dirs:
                        possible_relative_dirs.append(relative_dir)
            
            return possible_relative_dirs
        
        # 如果没有方位信息，返回所有方位
        return DIRECTIONS_8
    
    def _calculate_distance_range_for_pair(self, e1: SpatialEntity, e2: SpatialEntity) -> Tuple[float, float]:
        """计算两个实体间的距离范围（基于用户估计距离）"""
        if e1.estimated_distance and e2.estimated_distance:
            dist1 = e1.estimated_distance
            dist2 = e2.estimated_distance
            
            # 根据三角不等式计算距离范围
            min_dist = abs(dist1 - dist2) * (1 - DISTANCE_TOLERANCE_RATIO)
            max_dist = (dist1 + dist2) * (1 + DISTANCE_TOLERANCE_RATIO)
            
            return (min_dist, max_dist)
        
        # 如果没有距离信息，使用宽松范围
        return (0, 1000)  # 默认1000米范围内

    def _subgraph_isomorphism_match(self, subgraph: SubgraphTemplate, 
                                     entities: List[SpatialEntity],
                                     max_matches: int) -> List[CandidateCombination]:
        """
        松弛约束的子图同构匹配（支持路径推断）
        
        算法：
        1. 选择关键边（最小生成树）- 这些边必须有直接DELAUNAY关系
        2. 选择锚点（约束最强的实体）
        3. 在Neo4j中搜索锚点候选
        4. 对每个锚点候选，使用松弛约束迭代扩展
           - 关键边：必须有直接DELAUNAY关系
           - 非关键边：可以通过路径推断验证
        5. 验证完整匹配
        """
        matches = []
        
        # Step 1: 选择关键边（最小生成树）
        key_edges = self._select_key_edges(subgraph)
        print(f"       关键边: {len(key_edges)} 条 (总边数: {len(subgraph.entity_relations)})")
        
        # Step 2: 选择锚点（有POI约束或颜色最具体的实体）
        anchor_entity = self._select_anchor_entity(entities)
        if not anchor_entity:
            return []
        
        print(f"       锚点实体: {anchor_entity.entity_id}")
        
        # Step 3: 搜索锚点候选
        anchor_candidates = self._search_anchor_candidates(anchor_entity)
        print(f"       锚点候选: {len(anchor_candidates)} 个")
        
        if not anchor_candidates:
            return []
        
        # Step 4: 对每个锚点候选进行松弛约束迭代扩展
        for anchor_cand in anchor_candidates[:20]:  # 限制锚点候选数
            partial_match = {anchor_entity.entity_id: anchor_cand}
            
            # 使用松弛约束迭代扩展
            if self._expand_match_iteratively_relaxed(partial_match, subgraph, entities, key_edges):
                # 构建完整匹配
                matched_entities = []
                for entity in entities:
                    if entity.entity_id in partial_match:
                        cand = partial_match[entity.entity_id]
                        matched = MatchedEntity(
                            entity_id=cand["id"],
                            entity_type="Building",
                            lon=float(cand.get("lon", 0)),
                            lat=float(cand.get("lat", 0)),
                            color_side=cand.get("color_side", ""),
                            color_top=cand.get("color_top", ""),
                            query_id=entity.entity_id,
                            poi_validation=cand.get("poi_validated", False),
                            required_poi_type=entity.get_first_poi_type() if entity.has_poi_constraint() else "",
                            matched_poi_names=cand.get("matched_poi_names", [])
                        )
                        matched_entities.append(matched)
                
                if len(matched_entities) == len(entities):
                    combo = CandidateCombination(entities=matched_entities)
                    matches.append(combo)
                    
                    if len(matches) >= max_matches:
                        break
        
        return matches
    
    def _select_anchor_entity(self, entities: List[SpatialEntity]) -> Optional[SpatialEntity]:
        """选择锚点实体（约束最强的）"""
        # 优先选择有POI约束的
        for e in entities:
            if e.has_poi_constraint():
                return e
        # 其次选择有颜色的
        for e in entities:
            if e.color_side or e.color_top:
                return e
        # 最后返回第一个
        return entities[0] if entities else None
    
    def _search_anchor_candidates(self, anchor: SpatialEntity) -> List[Dict]:
        """搜索锚点候选"""
        # 构建查询条件
        color_condition = ""
        if anchor.color_side and anchor.color_top:
            color_condition = """
            AND (n.color_side CONTAINS $color_side OR n.color_top CONTAINS $color_top
                 OR n.color_side CONTAINS $color_top OR n.color_top CONTAINS $color_side)
            """
        elif anchor.color_side:
            color_condition = "AND (n.color_side CONTAINS $color_side OR n.color_top CONTAINS $color_side)"
        elif anchor.color_top:
            color_condition = "AND (n.color_side CONTAINS $color_top OR n.color_top CONTAINS $color_top)"
        
        query = f"""
        MATCH (n:Building)
        WHERE 1=1 {color_condition}
        RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
               n.lon AS lon, n.lat AS lat
        LIMIT 50
        """
        
        try:
            results = self.neo4j.query(query, {
                "color_side": anchor.color_side or "",
                "color_top": anchor.color_top or ""
            })
            
            # 如果有POI约束，验证POI
            if anchor.has_poi_constraint():
                poi_types = anchor.get_all_poi_types()
                validated = []
                for r in results:
                    poi_valid, matched_names = self._check_multiple_poi_types(r["id"], poi_types)
                    if poi_valid:
                        r["poi_validated"] = True
                        r["matched_poi_names"] = matched_names
                        validated.append(r)
                return validated
            
            return results
        except:
            return []
    
    def _expand_match_iteratively_relaxed(self, partial_match: Dict, subgraph: SubgraphTemplate,
                                           entities: List[SpatialEntity],
                                           key_edges: Set[Tuple[str, str]]) -> bool:
        """
        松弛约束的迭代扩展匹配
        
        改进点：
        1. 关键边（最小生成树边）必须在KG中存在直接DELAUNAY关系
        2. 非关键边可以通过路径推断验证
        3. 支持多跳路径推断
        """
        # 获取已匹配的实体ID
        matched_ids = set(partial_match.keys())
        
        # 找到查询子图中与已匹配节点相邻的未匹配节点
        frontier = []
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            edge_key = (min(e1_id, e2_id), max(e1_id, e2_id))
            is_key_edge = edge_key in key_edges
            
            if e1_id in matched_ids and e2_id not in matched_ids:
                frontier.append((e2_id, e1_id, relation, is_key_edge))
            elif e2_id in matched_ids and e1_id not in matched_ids:
                frontier.append((e1_id, e2_id, relation, is_key_edge))
        
        if not frontier:
            # 所有节点都已匹配，验证所有非关键边
            return self._validate_non_key_edges(partial_match, subgraph, key_edges)
        
        # 优先处理关键边（必须有直接DELAUNAY关系）
        frontier.sort(key=lambda x: not x[3])  # 关键边排在前面
        
        # 对前沿中的每个节点进行匹配
        for target_id, source_id, relation, is_key_edge in frontier:
            if target_id in partial_match:
                continue
            
            source_cand = partial_match[source_id]
            
            if is_key_edge:
                # 关键边：必须在KG中存在直接DELAUNAY关系
                neighbors = self._search_constrained_neighbors(
                    source_cand["id"], relation, target_id, entities
                )
                
                if not neighbors:
                    return False
                
                # 尝试每个邻居
                for neighbor in neighbors:
                    if neighbor["id"] in [c["id"] for c in partial_match.values()]:
                        continue
                    
                    partial_match[target_id] = neighbor
                    
                    if self._expand_match_iteratively_relaxed(partial_match, subgraph, entities, key_edges):
                        return True
                    
                    del partial_match[target_id]
                
                return False
            
            else:
                # 非关键边：可以通过路径推断，先跳过，最后统一验证
                # 尝试从已匹配的候选中找一个合适的
                target_entity = next((e for e in entities if e.entity_id == target_id), None)
                if not target_entity:
                    return False
                
                # 搜索满足颜色和POI约束的候选（不考虑邻接关系）
                candidates = self._search_entity_candidates(target_entity)
                
                for cand in candidates:
                    if cand["id"] in [c["id"] for c in partial_match.values()]:
                        continue
                    
                    # 验证该候选与source_cand的关系（通过路径）
                    valid, score = self._check_relation_via_path(
                        source_cand["id"], cand["id"], relation
                    )
                    
                    if valid and score >= 0.5:  # 最低得分阈值
                        partial_match[target_id] = cand
                        
                        if self._expand_match_iteratively_relaxed(partial_match, subgraph, entities, key_edges):
                            return True
                        
                        del partial_match[target_id]
                
                return False
        
        return len(partial_match) == len(entities)
    
    def _validate_non_key_edges(self, partial_match: Dict, subgraph: SubgraphTemplate,
                                 key_edges: Set[Tuple[str, str]]) -> bool:
        """
        验证所有非关键边是否满足约束（通过路径推断）
        """
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            edge_key = (min(e1_id, e2_id), max(e1_id, e2_id))
            if edge_key in key_edges:
                continue  # 关键边已经在扩展过程中验证
            
            if e1_id not in partial_match or e2_id not in partial_match:
                return False
            
            c1 = partial_match[e1_id]
            c2 = partial_match[e2_id]
            
            valid, score = self._check_relation_via_path(c1["id"], c2["id"], relation)
            if not valid or score < 0.3:  # 非关键边允许更低的阈值
                return False
        
        return True
    
    def _search_entity_candidates(self, entity: SpatialEntity, limit: int = 50) -> List[Dict]:
        """
        搜索满足实体属性约束的候选（不考虑邻接关系）
        """
        # 构建颜色约束
        color_condition = ""
        if entity.color_side and entity.color_top:
            color_condition = """
            AND (n.color_side CONTAINS $color_side OR n.color_top CONTAINS $color_top
                 OR n.color_side CONTAINS $color_top OR n.color_top CONTAINS $color_side)
            """
        elif entity.color_side:
            color_condition = "AND (n.color_side CONTAINS $color_side OR n.color_top CONTAINS $color_side)"
        elif entity.color_top:
            color_condition = "AND (n.color_side CONTAINS $color_top OR n.color_top CONTAINS $color_top)"
        
        query = f"""
        MATCH (n:Building)
        WHERE 1=1 {color_condition}
        RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
               n.lon AS lon, n.lat AS lat
        LIMIT $limit
        """
        
        try:
            results = self.neo4j.query(query, {
                "color_side": entity.color_side or "",
                "color_top": entity.color_top or "",
                "limit": limit
            })
            
            # POI验证
            if entity.has_poi_constraint():
                poi_types = entity.get_all_poi_types()
                validated = []
                for r in results:
                    poi_valid, matched_names = self._check_multiple_poi_types(r["id"], poi_types)
                    if poi_valid:
                        r["poi_validated"] = True
                        r["matched_poi_names"] = matched_names
                        validated.append(r)
                return validated
            
            return results
        except:
            return []
    
    def _get_adjacent_directions(self, direction: str) -> List[str]:
        """获取相邻方位"""
        adjacent_map = {
            "E": ["NE", "SE"],
            "NE": ["E", "N"],
            "N": ["NE", "NW"],
            "NW": ["N", "W"],
            "W": ["NW", "SW"],
            "SW": ["W", "S"],
            "S": ["SW", "SE"],
            "SE": ["S", "E"],
        }
        return adjacent_map.get(direction, [])
    
    def _search_constrained_neighbors(self, source_id: str, relation: Dict,
                                       target_entity_id: str,
                                       entities: List[SpatialEntity]) -> List[Dict]:
        """搜索满足约束的邻居节点"""
        # 获取目标实体的属性约束
        target_entity = next((e for e in entities if e.entity_id == target_entity_id), None)
        if not target_entity:
            return []
        
        # 构建方位约束
        dirs = relation.get("possible_directions", [])
        dir_condition = ""
        if dirs:
            dir_list = "'" + "','".join(dirs) + "'"
            dir_condition = f"AND r.direction IN [{dir_list}]"
        
        # 构建距离约束
        dist_range = relation.get("distance_range", (0, 1000))
        min_dist, max_dist = dist_range
        
        # 构建颜色约束
        color_condition = ""
        if target_entity.color_side and target_entity.color_top:
            color_condition = """
            AND (n.color_side CONTAINS $color_side OR n.color_top CONTAINS $color_top
                 OR n.color_side CONTAINS $color_top OR n.color_top CONTAINS $color_side)
            """
        elif target_entity.color_side:
            color_condition = "AND (n.color_side CONTAINS $color_side OR n.color_top CONTAINS $color_side)"
        elif target_entity.color_top:
            color_condition = "AND (n.color_side CONTAINS $color_top OR n.color_top CONTAINS $color_top)"
        
        query = f"""
        MATCH (s:Building {{id: $source_id}})-[r:DELAUNAY]-(n:Building)
        WHERE r.distance_m >= $min_dist AND r.distance_m <= $max_dist
        {dir_condition}
        {color_condition}
        RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
               n.lon AS lon, n.lat AS lat, r.direction AS direction, r.distance_m AS distance
        LIMIT 20
        """
        
        try:
            results = self.neo4j.query(query, {
                "source_id": source_id,
                "min_dist": min_dist * 0.5,  # 宽松约束
                "max_dist": max_dist * 2,
                "color_side": target_entity.color_side or "",
                "color_top": target_entity.color_top or ""
            })
            
            # POI验证
            if target_entity.has_poi_constraint():
                poi_types = target_entity.get_all_poi_types()
                validated = []
                for r in results:
                    poi_valid, matched_names = self._check_multiple_poi_types(r["id"], poi_types)
                    if poi_valid:
                        r["poi_validated"] = True
                        r["matched_poi_names"] = matched_names
                        validated.append(r)
                return validated
            
            return results
        except:
            return []

    def _deduplicate_matches(self, matches: List[CandidateCombination]) -> List[CandidateCombination]:
        """去重匹配结果（基于实体ID组合）"""
        seen = set()
        unique = []
        
        for combo in matches:
            # 生成唯一键（排序后的实体ID元组）
            key = tuple(sorted([e.entity_id for e in combo.entities]))
            if key not in seen:
                seen.add(key)
                unique.append(combo)
        
        return unique

    def _calculate_match_score(self, combo: CandidateCombination,
                                subgraph: SubgraphTemplate,
                                entities: List[SpatialEntity]) -> float:
        """计算匹配评分"""
        score = 0.0
        
        # 1. 实体属性匹配（权重2.0）
        for matched in combo.entities:
            entity = next((e for e in entities if e.entity_id == matched.query_id), None)
            if entity:
                if entity.color_side and matched.color_side:
                    if entity.color_side.lower() in matched.color_side.lower():
                        score += 1.0
                if entity.color_top and matched.color_top:
                    if entity.color_top.lower() in matched.color_top.lower():
                        score += 1.0
        
        # 2. POI满足（权重3.0）
        score += combo.poi_satisfaction_count * 3.0
        
        # 3. 拓扑一致性（权重2.0）
        matched_map = {m.query_id: m for m in combo.entities}
        topology_score = 0.0
        topology_count = 0
        
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            if e1_id not in matched_map or e2_id not in matched_map:
                continue
            
            m1 = matched_map[e1_id]
            m2 = matched_map[e2_id]
            
            # 实际距离和方位
            actual_dist = haversine_distance(m1.lat, m1.lon, m2.lat, m2.lon)
            actual_angle = compute_angle_deg(m1.lon, m1.lat, m2.lon, m2.lat)
            actual_dir = compute_direction_8(actual_angle)
            
            # 方位匹配
            dirs = relation.get("possible_directions", [])
            if dirs and actual_dir in dirs:
                topology_score += 1.0
            elif dirs and actual_dir in self._get_adjacent_directions(dirs[0]):
                topology_score += 0.5
            
            # 距离匹配
            min_dist, max_dist = relation.get("distance_range", (0, 1000))
            if min_dist <= actual_dist <= max_dist:
                topology_score += 1.0
            elif min_dist * 0.5 <= actual_dist <= max_dist * 2:
                topology_score += 0.5
            
            topology_count += 1
        
        if topology_count > 0:
            score += (topology_score / topology_count) * 2.0
        
        return score

    def find_candidate_combinations(self, entities: List[SpatialEntity],
                                     max_total_combinations: int = 50) -> List[CandidateCombination]:
        """
        多子图同构匹配定位
        
        核心逻辑：
        1. 构建基础子图模板
        2. 根据方位多解生成多个子图变体
        3. 对每个子图变体执行同构匹配
        4. 合并所有匹配结果并去重评分
        """
        print(f"\n{'='*60}")
        print(f"🔎 多子图同构匹配定位开始")
        print(f"{'='*60}")
        
        if not entities:
            return []
        
        # Step 1: 构建基础子图模板
        print(f"\n📋 Step 1: 构建基础子图模板...")
        base_template = self._build_subgraph_template(entities)
        
        print(f"   实体数量: {len(entities)}")
        print(f"   关系约束数量: {len(base_template.entity_relations)}")
        
        # Step 2: 生成子图变体（方位多解）
        print(f"\n🔄 Step 2: 根据方位多解生成子图变体...")
        subgraph_variants = generate_subgraph_variants(base_template)
        print(f"   生成 {len(subgraph_variants)} 个子图变体")
        
        # Step 3: 对每个子图变体执行同构匹配
        print(f"\n🔍 Step 3: 执行子图同构匹配...")
        all_matches = []
        
        for idx, subgraph in enumerate(subgraph_variants):
            print(f"\n   子图变体 {idx+1}/{len(subgraph_variants)}:")
            # 打印当前变体的方位关系
            for (e1, e2), rel in subgraph.entity_relations.items():
                dirs = rel.get("possible_directions", [])
                if dirs:
                    print(f"     {e1} -{dirs[0]}-> {e2}")
            
            matches = self._subgraph_isomorphism_match(subgraph, entities, max_total_combinations // len(subgraph_variants) + 5)
            print(f"     找到 {len(matches)} 个匹配")
            all_matches.extend(matches)
        
        # Step 4: 合并结果并去重
        print(f"\n📊 Step 4: 合并匹配结果...")
        unique_matches = self._deduplicate_matches(all_matches)
        print(f"   去重后: {len(unique_matches)} 个唯一匹配")
        
        # Step 5: 计算完整评分
        print(f"\n⭐ Step 5: 计算评分...")
        for combo in unique_matches:
            combo.poi_satisfaction_count = self._count_satisfied_poi(combo, entities)
            combo.poi_total_count = self._get_total_poi_count(entities)
            combo.poi_constraint_satisfied = (combo.poi_satisfaction_count == combo.poi_total_count and combo.poi_total_count > 0)
            combo.satisfied_poi_types = self._get_satisfied_poi_types(combo, entities)
            combo.total_score = self._calculate_match_score(combo, base_template, entities)
        
        # Step 6: 按评分排序
        unique_matches.sort(key=lambda c: (
            -c.poi_satisfaction_count,
            -c.total_score,
            -c.poi_constraint_satisfied
        ))
        
        print(f"\n✅ 最终返回 {min(len(unique_matches), max_total_combinations)} 个有效匹配")
        
        return unique_matches[:max_total_combinations]

    def _find_buildings_by_color(self, color_side: str = None, color_top: str = None,
                                  strict: bool = False) -> List[Dict]:
        """根据颜色搜索候选建筑"""
        if strict:
            return self.match_buildings_strict(color_side, color_top)
        else:
            return self.match_buildings(color_side, color_top)
    
    def _check_multiple_poi_types(self, building_id: str, poi_types: List[str]) -> Tuple[bool, List[str]]:
        """检查建筑是否包含所有指定的POI类型"""
        return self.check_all_pois_in_building(str(building_id), poi_types)

    def _count_satisfied_poi(self, combo: CandidateCombination, entities: List[SpatialEntity]) -> int:
        """
        统计组合中满足的POI约束总数（支持多个POI）
        例如：如果一个建筑需要 restaurant 和 cafe 两个POI，都满足则计数为2
        """
        entity_map = {e.query_id: e for e in combo.entities}
        total_satisfied = 0

        for entity in entities:
            if entity.has_poi_constraint():
                eid = entity.entity_id
                if eid in entity_map:
                    matched = entity_map[eid]
                    if matched.poi_validation:
                        # 如果POI验证通过，统计该实体要求的所有POI类型数量
                        required_types = entity.get_all_poi_types()
                        total_satisfied += len(required_types)

        return total_satisfied

    def _get_satisfied_poi_types(self, combo: CandidateCombination, entities: List[SpatialEntity]) -> List[str]:
        """
        获取组合中满足的所有POI类型列表（支持多个POI）
        """
        entity_map = {e.query_id: e for e in combo.entities}
        satisfied_types = []

        for entity in entities:
            if entity.has_poi_constraint():
                eid = entity.entity_id
                if eid in entity_map:
                    matched = entity_map[eid]
                    if matched.poi_validation:
                        # 添加该实体所有满足的POI类型
                        satisfied_types.extend(entity.get_all_poi_types())

        return satisfied_types

    def _get_total_poi_count(self, entities: List[SpatialEntity]) -> int:
        """
        获取所有实体要求的POI总数（支持多个POI）
        """
        total = 0
        for entity in entities:
            if entity.has_poi_constraint():
                total += len(entity.get_all_poi_types())
        return total

    # ===================== 已废弃方法（V4多子图同构匹配替代） =====================
    # 以下方法已被V4多子图同构匹配逻辑替代，保留注释作为参考
    # 
    # 被替代的方法包括：
    # - _get_neighbors_as_dict: 获取一跳邻居
    # - _get_knn_neighbors: KNN兜底搜索
    # - _get_2hop_neighbors_as_dict: 获取两跳邻居
    # - _get_neighbors_with_poi: 获取带POI信息的邻居
    # - _find_matches_in_neighbors: 在邻居中查找匹配
    #
    # 新方法使用基于锚点的迭代扩展子图同构匹配，更高效准确

    def _building_matches_criteria(self, building: Dict, entity: SpatialEntity, strict: bool = True) -> bool:
        """检查建筑是否符合实体特征要求"""
        color_side = building.get("color_side", "")
        color_top = building.get("color_top", "")

        side_match = False
        top_match = False

        if entity.color_side:
            if strict:
                normalized_expected = self.normalize_color(entity.color_side)
                side_match = (color_side == normalized_expected)
                if not side_match:
                    side_match = normalized_expected in color_side or color_side in normalized_expected
            else:
                expanded = self.expand_color(entity.color_side)
                side_match = any(c in color_side for c in expanded)
        else:
            side_match = True

        if entity.color_top:
            if strict:
                normalized_expected = self.normalize_color(entity.color_top)
                top_match = (color_top == normalized_expected)
                if not top_match:
                    top_match = normalized_expected in color_top or color_top in normalized_expected
            else:
                expanded = self.expand_color(entity.color_top)
                top_match = any(c in color_top for c in expanded)
        else:
            top_match = True

        return side_match and top_match

    def _validate_direction_constraints(self, combo: CandidateCombination,
                                        entities: List[SpatialEntity]) -> bool:
        """验证实体间的方位约束"""
        if len(combo.entities) < 2:
            return True

        entity_map = {e.query_id: e for e in combo.entities}

        for e1 in combo.entities:
            query_entity1 = next((en for en in entities if en.entity_id == e1.query_id), None)
            if not query_entity1:
                continue

            for other_id, possible_dirs in query_entity1.possible_relative_directions.items():
                e2 = entity_map.get(other_id)
                if not e2:
                    continue

                actual_dir = self._calculate_actual_direction(e1.lon, e1.lat, e2.lon, e2.lat)
                is_match = actual_dir in possible_dirs

                if not is_match:
                    return False

        return True

    def _validate_distance_constraints(self, combo: CandidateCombination,
                                       entities: List[SpatialEntity]) -> bool:
        """验证候选组合中实体间的距离与用户描述的距离是否一致"""
        entity_map = {e.query_id: e for e in combo.entities}

        for e1 in combo.entities:
            query_entity1 = next((en for en in entities if en.entity_id == e1.query_id), None)
            if not query_entity1 or not query_entity1.estimated_distance:
                continue

            for e2 in combo.entities:
                if e1.entity_id == e2.entity_id:
                    continue
                query_entity2 = next((en for en in entities if en.entity_id == e2.query_id), None)
                if not query_entity2 or not query_entity2.estimated_distance:
                    continue

                # 计算两实体间的实际距离
                actual_dist = haversine_distance(e1.lat, e1.lon, e2.lat, e2.lon)

                # 获取用户描述的两实体各自到用户的距离
                dist1 = query_entity1.estimated_distance
                dist2 = query_entity2.estimated_distance

                # 根据三角不等式估算两实体间的预期距离范围
                min_expected = abs(dist1 - dist2) * (1 - DISTANCE_TOLERANCE_RATIO)
                max_expected = (dist1 + dist2) * (1 + DISTANCE_TOLERANCE_RATIO)

                if actual_dist > max_expected or actual_dist < min_expected:
                    return False

        return True

    def _validate_subgraph_topology(self, combo: CandidateCombination,
                                     subgraph: SubgraphTemplate,
                                     entities: List[SpatialEntity]) -> bool:
        """验证候选组合是否满足子图的拓扑约束"""
        
        # 构建实体ID到匹配实体的映射
        matched_map = {}
        entity_map = {}
        for matched in combo.entities:
            matched_map[matched.query_id] = matched
        for entity in entities:
            entity_map[entity.entity_id] = entity
        
        # 验证每对实体的拓扑约束
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            if e1_id not in matched_map or e2_id not in matched_map:
                continue
            
            m1 = matched_map[e1_id]
            m2 = matched_map[e2_id]
            
            # 计算实际距离
            actual_dist = haversine_distance(m1.lat, m1.lon, m2.lat, m2.lon)
            
            # 验证距离范围（宽松约束）
            min_dist, max_dist = relation["distance_range"]
            # 距离容差扩大到2倍
            if actual_dist > max_dist * 2 or actual_dist < min_dist * 0.5:
                # 距离超出范围，但不是硬性拒绝，只是降低评分
                pass  # 不直接返回False
            
            # 验证方位（宽松约束）
            possible_dirs = relation["possible_directions"]
            actual_angle = self._calculate_actual_direction(m1.lon, m1.lat, m2.lon, m2.lat)
            # 将角度转换为方位（使用DIRECTIONS_8）
            actual_dir = actual_angle
            
            # 方位不在可能列表中，但考虑相邻方位
            adjacent_dirs = self._get_adjacent_directions(actual_dir)
            all_possible = set(possible_dirs + adjacent_dirs)
            
            # 如果实际方位及其相邻方位都不在可能列表中，返回False
            if actual_dir not in all_possible:
                return False
        
        return True
    
    def _get_adjacent_directions(self, direction: str) -> List[str]:
        """获取相邻方位"""
        adjacent_map = {
            "E": ["NE", "SE"],
            "NE": ["E", "N"],
            "N": ["NE", "NW"],
            "NW": ["N", "W"],
            "W": ["NW", "SW"],
            "SW": ["W", "S"],
            "S": ["SW", "SE"],
            "SE": ["S", "E"],
        }
        return adjacent_map.get(direction, [])

    def _calculate_actual_direction(self, lon1: float, lat1: float,
                                    lon2: float, lat2: float) -> str:
        """计算从点1到点2的实际方位"""
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        angle = math.degrees(math.atan2(dlat, dlon))
        angle = (angle + 360) % 360

        idx = int(((angle + 22.5) % 360) / 45)
        return DIRECTIONS_8[idx]

    def _check_poi_constraints(self, combo: CandidateCombination,
                               entities: List[SpatialEntity]) -> bool:
        """检查POI约束是否满足"""
        entity_map = {e.query_id: e for e in combo.entities}

        for entity in entities:
            if entity.has_poi_constraint():
                eid = entity.entity_id
                if eid in entity_map:
                    matched = entity_map[eid]
                    if not matched.poi_validation:
                        return False
        return True

    def _calculate_score(self, combo: CandidateCombination,
                        entities: List[SpatialEntity]) -> float:
        """计算组合得分"""
        if not combo.entities:
            return 0.0

        score = len(combo.entities) * 0.2

        # POI约束满足度 - 更重要的权重
        poi_satisfied = self._count_satisfied_poi(combo, entities)
        poi_total = len([e for e in entities if e.has_poi_constraint()])
        if poi_total > 0:
            score += (poi_satisfied / poi_total) * 1.5  # 提高权重

        # 空间紧凑性
        building_coords = [(e.lon, e.lat) for e in combo.entities]
        if len(building_coords) >= 2:
            total_dist = 0
            count = 0
            for i in range(len(building_coords)):
                for j in range(i + 1, len(building_coords)):
                    dist = haversine_distance(
                        building_coords[i][1], building_coords[i][0],
                        building_coords[j][1], building_coords[j][0]
                    )
                    total_dist += dist
                    count += 1

            if count > 0:
                avg_dist = total_dist / count
                if avg_dist < 50:
                    score += 0.5
                elif avg_dist < 100:
                    score += 0.3
                elif avg_dist < 200:
                    score += 0.1

        return score

    def _calculate_detailed_score(self, combo: CandidateCombination,
                                  entities: List[SpatialEntity]) -> float:
        """计算组合的详细得分"""
        if not combo.entities:
            return 0.0

        score = 0.0
        entity_map = {e.query_id: e for e in combo.entities}

        # 1. 实体匹配完整度
        matched_count = len(combo.entities)
        expected_count = len(entities)
        completeness = matched_count / expected_count if expected_count > 0 else 0
        score += completeness * 2.0

        # 2. POI约束满足度 - 重要权重
        poi_satisfied = self._count_satisfied_poi(combo, entities)
        poi_total = len([e for e in entities if e.has_poi_constraint()])
        if poi_total > 0:
            poi_score = (poi_satisfied / poi_total) * 2.0  # 提高权重
            score += poi_score

        # 3. 方位关系准确度
        direction_score = 0.0
        direction_count = 0
        for e1 in combo.entities:
            query_entity1 = next((en for en in entities if en.entity_id == e1.query_id), None)
            if not query_entity1:
                continue

            for other_id, possible_dirs in query_entity1.possible_relative_directions.items():
                e2 = entity_map.get(other_id)
                if not e2:
                    continue

                actual_dir = self._calculate_actual_direction(e1.lon, e1.lat, e2.lon, e2.lat)

                if actual_dir in possible_dirs:
                    match_idx = possible_dirs.index(actual_dir)
                    direction_score += 1.0 / (match_idx + 1)

                direction_count += 1

        if direction_count > 0:
            score += (direction_score / direction_count) * 2.0

        # 3.5 距离一致性评分
        distance_consistency_score = 0.0
        distance_check_count = 0
        for e1 in combo.entities:
            query_entity1 = next((en for en in entities if en.entity_id == e1.query_id), None)
            if not query_entity1 or not query_entity1.estimated_distance:
                continue
            for e2 in combo.entities:
                if e1.entity_id == e2.entity_id:
                    continue
                query_entity2 = next((en for en in entities if en.entity_id == e2.query_id), None)
                if not query_entity2 or not query_entity2.estimated_distance:
                    continue
                actual_dist = haversine_distance(e1.lat, e1.lon, e2.lat, e2.lon)
                dist1 = query_entity1.estimated_distance
                dist2 = query_entity2.estimated_distance
                min_expected = abs(dist1 - dist2) * (1 - DISTANCE_TOLERANCE_RATIO)
                max_expected = (dist1 + dist2) * (1 + DISTANCE_TOLERANCE_RATIO)
                if min_expected <= actual_dist <= max_expected:
                    distance_consistency_score += 1.0
                distance_check_count += 1
        if distance_check_count > 0:
            score += (distance_consistency_score / distance_check_count) * 1.5

        # 4. 空间紧凑性
        building_coords = [(e.lon, e.lat) for e in combo.entities]
        if len(building_coords) >= 2:
            distances = []
            for i in range(len(building_coords)):
                for j in range(i + 1, len(building_coords)):
                    dist = haversine_distance(
                        building_coords[i][1], building_coords[i][0],
                        building_coords[j][1], building_coords[j][0]
                    )
                    distances.append(dist)

            if distances:
                avg_dist = sum(distances) / len(distances)
                if avg_dist < 30:
                    score += 1.0
                elif avg_dist < 60:
                    score += 0.8
                elif avg_dist < 100:
                    score += 0.6
                elif avg_dist < 150:
                    score += 0.4
                elif avg_dist < 200:
                    score += 0.2

        return score

    def _calculate_subgraph_score(self, combo: CandidateCombination,
                                   subgraph: SubgraphTemplate,
                                   entities: List[SpatialEntity]) -> float:
        """基于子图匹配的评分"""
        score = 0.0
        
        # 1. 实体属性匹配评分（权重2.0）
        for matched in combo.entities:
            entity = next((e for e in entities if e.entity_id == matched.query_id), None)
            if entity:
                # 颜色匹配
                if entity.color_side and matched.color_side:
                    if entity.color_side.lower() in matched.color_side.lower():
                        score += 1.0
                if entity.color_top and matched.color_top:
                    if entity.color_top.lower() in matched.color_top.lower():
                        score += 1.0
        
        # 2. POI满足评分（权重3.0）
        poi_score = combo.poi_satisfaction_count * 3.0
        score += poi_score
        
        # 3. 拓扑一致性评分（权重2.0）
        matched_map = {m.query_id: m for m in combo.entities}
        topology_score = 0.0
        topology_count = 0
        
        for (e1_id, e2_id), relation in subgraph.entity_relations.items():
            if e1_id not in matched_map or e2_id not in matched_map:
                continue
            
            m1 = matched_map[e1_id]
            m2 = matched_map[e2_id]
            
            # 计算实际距离和方位
            actual_dist = haversine_distance(m1.lat, m1.lon, m2.lat, m2.lon)
            actual_dir = self._calculate_actual_direction(m1.lon, m1.lat, m2.lon, m2.lat)
            
            # 方位匹配评分
            possible_dirs = relation["possible_directions"]
            if actual_dir in possible_dirs:
                topology_score += 1.0
            elif actual_dir in self._get_adjacent_directions(possible_dirs[0]) if possible_dirs else []:
                topology_score += 0.5
            
            # 距离匹配评分（宽松）
            min_dist, max_dist = relation["distance_range"]
            if min_dist <= actual_dist <= max_dist:
                topology_score += 1.0
            elif min_dist * 0.5 <= actual_dist <= max_dist * 2:
                topology_score += 0.5
            
            topology_count += 1
        
        if topology_count > 0:
            score += (topology_score / topology_count) * 2.0
        
        return score

    def _calculate_confidence(self, combo: CandidateCombination,
                             entities: List[SpatialEntity]) -> float:
        """计算组合的置信度"""
        if not combo.entities:
            return 0.0

        confidence = 0.0

        # 基础置信度：实体匹配比例
        matched_ratio = len(combo.entities) / len(entities) if entities else 0
        confidence += matched_ratio * 0.2

        # POI约束满足度 - 重要权重
        poi_satisfied = self._count_satisfied_poi(combo, entities)
        poi_total = len([e for e in entities if e.has_poi_constraint()])
        if poi_total > 0:
            poi_conf = (poi_satisfied / poi_total) * 0.3
            confidence += poi_conf

        # 方位匹配准确度
        entity_map = {e.query_id: e for e in combo.entities}
        direction_accuracy = 0.0
        direction_count = 0

        for e1 in combo.entities:
            query_entity1 = next((en for en in entities if en.entity_id == e1.query_id), None)
            if not query_entity1:
                continue

            for other_id, possible_dirs in query_entity1.possible_relative_directions.items():
                e2 = entity_map.get(other_id)
                if not e2:
                    continue

                actual_dir = self._calculate_actual_direction(e1.lon, e1.lat, e2.lon, e2.lat)

                if actual_dir in possible_dirs:
                    match_idx = possible_dirs.index(actual_dir)
                    direction_accuracy += max(0, 1.0 - match_idx * 0.2)

                direction_count += 1

        if direction_count > 0:
            confidence += (direction_accuracy / direction_count) * 0.3

        # 距离一致性置信度
        distance_consistency = 0.0
        distance_check_count = 0
        for e1 in combo.entities:
            query_entity1 = next((en for en in entities if en.entity_id == e1.query_id), None)
            if not query_entity1 or not query_entity1.estimated_distance:
                continue
            for e2 in combo.entities:
                if e1.entity_id == e2.entity_id:
                    continue
                query_entity2 = next((en for en in entities if en.entity_id == e2.query_id), None)
                if not query_entity2 or not query_entity2.estimated_distance:
                    continue
                actual_dist = haversine_distance(e1.lat, e1.lon, e2.lat, e2.lon)
                dist1 = query_entity1.estimated_distance
                dist2 = query_entity2.estimated_distance
                min_expected = abs(dist1 - dist2) * (1 - DISTANCE_TOLERANCE_RATIO)
                max_expected = (dist1 + dist2) * (1 + DISTANCE_TOLERANCE_RATIO)
                if min_expected <= actual_dist <= max_expected:
                    distance_consistency += 1.0
                distance_check_count += 1
        if distance_check_count > 0:
            confidence += (distance_consistency / distance_check_count) * 0.15

        # 空间紧凑性
        building_coords = [(e.lon, e.lat) for e in combo.entities]
        if len(building_coords) >= 2:
            distances = []
            for i in range(len(building_coords)):
                for j in range(i + 1, len(building_coords)):
                    dist = haversine_distance(
                        building_coords[i][1], building_coords[i][0],
                        building_coords[j][1], building_coords[j][0]
                    )
                    distances.append(dist)

            if distances:
                avg_dist = sum(distances) / len(distances)
                if avg_dist < 50:
                    confidence += 0.2
                elif avg_dist < 100:
                    confidence += 0.15
                elif avg_dist < 150:
                    confidence += 0.1

        return min(confidence, 1.0)


# ===================== 7. 坐标推算器 =====================

class CoordinateEstimator:
    def estimate_user_position(self, combo: CandidateCombination,
                             entities: List[SpatialEntity]) -> Tuple[float, float, float]:
        """根据候选组合推算用户位置"""
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
            return None, None, 0.0

        total_weight = sum(weights)
        avg_lon = sum(p[0] * w for p, w in zip(positions, weights)) / total_weight
        avg_lat = sum(p[1] * w for p, w in zip(positions, weights)) / total_weight

        if len(positions) > 1:
            distances = []
            for lon, lat in positions:
                dist = haversine_distance(lat, lon, avg_lat, avg_lon)
                distances.append(dist)
            avg_dist = sum(distances) / len(distances)
            confidence = max(0, 1 - avg_dist / 100)
        else:
            confidence = 0.5

        return avg_lon, avg_lat, confidence


# ===================== 8. 主控制器 =====================

class GeoLocalizationSystem:
    def __init__(self):
        print("🚀 初始化地理定位系统...")
        self.llm = LLMClient()
        self.neo4j = Neo4jConnector()
        self.parser = UserInputParser(self.llm)
        self.matcher = KGMatcher(self.neo4j)
        self.estimator = CoordinateEstimator()

        if not self.neo4j.test_connection():
            raise Exception("无法连接到Neo4j数据库")
        print("✅ 数据库连接成功")
        print("✅ 系统初始化完成\n")

    def localize(self, user_input: str) -> Dict:
        """主定位函数"""
        result = {
            "status": "processing",
            "user_input": user_input,
            "parsed_input": None,
            "candidate_combinations": [],
            "selected_combination": None,
            "estimated_position": None,
            "confidence": 0.0,
            "message": ""
        }

        try:
            # Step 1: 解析用户输入
            print("📝 Step 1: 解析用户输入...")
            parsed = self.parser.parse(user_input)
            result["parsed_input"] = parsed

            ref_objects = parsed.get('reference_objects', [])
            print(f"✅ 解析完成，发现 {len(ref_objects)} 个参照物")

            if not ref_objects:
                result["status"] = "error"
                result["message"] = "未从输入中解析到任何参照物"
                return result

            # 创建实体列表
            entities = []
            chain_count = 0
            poi_count = 0
            for i, ref in enumerate(ref_objects):
                entity_id = ref.get("entity_id", f"ref_{i}")
                entity = SpatialEntity(
                    entity_id=entity_id,
                    entity_type=ref.get("entity_type", "Building"),
                    color_side=ref.get("color_side", ""),
                    color_top=ref.get("color_top", ""),
                    fclass=ref.get("poi_type", ""),
                    road_type=ref.get("road_type", ""),
                    road_orientation=ref.get("road_orientation")
                )
                # 方式A：相对于用户描述
                entity.direction_from_user = ref.get("direction_from_user") or ""
                est_dist = ref.get("estimated_distance")
                entity.estimated_distance = float(est_dist) if est_dist is not None else 0.0
                
                # 方式B：链式描述（相对于其他实体）
                entity.relative_to = ref.get("relative_to") or ""
                entity.direction_from_ref = ref.get("direction_from_ref") or ""
                ref_dist = ref.get("estimated_distance_from_ref")
                entity.distance_from_ref = float(ref_dist) if ref_dist is not None else 0.0
                
                if entity.relative_to:
                    chain_count += 1
                
                entity.associated_poi = ref.get("associated_poi") or {}
                if entity.has_poi_constraint():
                    poi_count += 1
                entities.append(entity)

            print(f"📊 实体统计: {len(entities)} 个, 其中链式描述 {chain_count} 个, {poi_count} 个有POI约束")
            
            # Step 1.5: 递推解析链式描述（将 relative_to 转换为 direction_from_user）
            if chain_count > 0:
                print("\n🔄 Step 1.5: 递推解析链式描述...")
                all_resolved = resolve_chain_entities(entities)
                
                # 打印解析结果
                for e in entities:
                    if e.relative_to:
                        print(f"   {e.entity_id}: relative_to={e.relative_to}({e.direction_from_ref}, {e.distance_from_ref}m)"
                              f" → direction_from_user={e.direction_from_user}, estimated_distance={e.estimated_distance}m")
                
                if all_resolved:
                    print(f"   ✅ 全部链式实体解析完成")
                else:
                    print(f"   ⚠️ 部分实体未能解析，可能影响后续匹配")

            # Step 2: 将用户参照系转换为实体间参照系
            print("\n🔄 Step 2: 转换参照系（用户→实体间）...")

            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if i == j:
                        continue
                    dir1 = e1.direction_from_user
                    dir2 = e2.direction_from_user
                    dist1 = e1.estimated_distance
                    dist2 = e2.estimated_distance

                    # 添加空值检查，防止 None 比较错误
                    if dir1 and dir2 and dist1 is not None and dist2 is not None and dist1 > 0 and dist2 > 0:
                        possible_dirs = calculate_possible_directions(dir1, dist1, dir2, dist2)
                        e1.possible_relative_directions[e2.entity_id] = possible_dirs

            # 打印方位关系
            print(f"\n🧭 实体间方位关系:")
            for e in entities:
                if e.possible_relative_directions:
                    user_dir = e.direction_from_user
                    user_dist = e.estimated_distance
                    poi_info = f" [含{e.get_poi_type()}]" if e.has_poi_constraint() else ""
                    chain_info = f" (链式: {e.relative_to}的{e.direction_from_ref}边)" if e.relative_to else ""
                    src_info = f" (递推解析: {e.direction_from_user}, {e.estimated_distance}m)" if e.relative_to else ""
                    print(f"   {e.entity_id} (用户{user_dir}边, {user_dist}m){chain_info}{src_info}{poi_info}:")
                    for other_id, possible_dirs in e.possible_relative_directions.items():
                        other_entity = next((en for en in entities if en.entity_id == other_id), None)
                        if other_entity:
                            main_dir = possible_dirs[0]
                            if len(possible_dirs) > 1:
                                print(f"      → 相对于 {other_id}: {main_dir} [{', '.join(possible_dirs)}]")
                            else:
                                print(f"      → 相对于 {other_id}: {main_dir}")

            # Step 3: 在知识图谱中匹配
            print("\n🔍 Step 3: 在知识图谱中匹配候选参照物...")
            combinations = self.matcher.find_candidate_combinations(entities)
            result["candidate_combinations"] = [c.to_dict() for c in combinations]

            if not combinations:
                result["status"] = "error"
                result["message"] = "未找到匹配的参照物组合"
                return result

            # Step 4: 推算坐标
            print("\n📍 Step 4: 推算用户坐标...")
            best_combo = combinations[0]
            lon, lat, conf = self.estimator.estimate_user_position(best_combo, entities)

            result["selected_combination"] = best_combo.to_dict()
            result["estimated_position"] = {
                "longitude": lon,
                "latitude": lat
            }
            result["confidence"] = conf
            result["status"] = "success"
            result["message"] = "定位成功"

            print(f"✅ 定位完成！")

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"定位失败: {str(e)}"
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()

        return result

    def print_result(self, result: Dict):
        """格式化输出"""
        print("\n" + "=" * 70)
        print("📊 地理定位结果 (V3 - 子图匹配版)")
        print("=" * 70)

        if result["status"] != "success":
            print(f"❌ 定位失败: {result['message']}")
            return

        print(f"\n📝 用户输入:")
        print(f"   {result['user_input']}")

        print(f"\n🎯 解析到的参照物:")
        for i, ref in enumerate(result['parsed_input'].get('reference_objects', []), 1):
            entity_type = ref.get('entity_type', 'N/A')
            direction = ref.get('direction_from_user', 'N/A')
            distance = ref.get('estimated_distance', 0)

            if entity_type == "Building":
                color = f"{ref.get('color_side', '')}/{ref.get('color_top', '')}"
                poi_info = ""
                associated_poi = ref.get('associated_poi')
                if associated_poi:
                    # 兼容列表和字典两种格式
                    if isinstance(associated_poi, dict):
                        poi_type = associated_poi.get('poi_type', 'POI')
                        poi_info = f" (含{poi_type})"
                    elif isinstance(associated_poi, list) and len(associated_poi) > 0:
                        poi_types = [p.get('poi_type', 'POI') for p in associated_poi if isinstance(p, dict)]
                        if poi_types:
                            poi_info = f" (含{', '.join(poi_types)})"
                print(f"   {i}. [Building] 方向:{direction}, 颜色:{color}, 距离:{distance}米{poi_info}")

        # 显示所有有效组合
        all_combos = result.get('candidate_combinations', [])
        if all_combos:
            print(f"\n🏆 找到 {len(all_combos)} 个有效参照物组合:")
            print(f"   排序规则：POI满足数优先，然后是置信度")

            for rank, combo in enumerate(all_combos[:10], 1):
                print(f"\n   【第 {rank} 名】")
                print(f"   POI满足: {combo.get('poi_satisfaction_count', 0)}/{combo.get('poi_total_count', 0)}")
                if combo.get('satisfied_poi_types'):
                    print(f"   满足的POI类型: {', '.join(combo['satisfied_poi_types'])}")
                print(f"   置信度: {combo.get('confidence', 0):.2%}")
                print(f"   得分: {combo.get('total_score', 0):.2f}")

                for entity in combo['entities']:
                    entity_type = entity.get('entity_type', 'N/A')
                    entity_id = entity.get('entity_id', 'N/A')
                    lon = entity.get('lon', 0)
                    lat = entity.get('lat', 0)

                    if entity_type == "Building":
                        color = f"{entity.get('color_side', '')}/{entity.get('color_top', '')}"
                        poi_info = ""
                        # 支持多个POI类型显示
                        required_types = entity.get('required_poi_types', [])
                        if required_types:
                            poi_info = f" [需POI:{', '.join(required_types)}]"
                        poi_valid = " ✓" if entity.get('poi_validation') else ""
                        # 显示满足的POI名称
                        matched_names = entity.get('matched_poi_names', [])
                        if matched_names:
                            poi_info += f" ({', '.join(matched_names)})"
                        print(f"      - ID={entity_id}, 颜色={color}{poi_info}{poi_valid}, 坐标=({lon:.6f}, {lat:.6f})")

            if len(all_combos) > 10:
                print(f"\n   ... 还有 {len(all_combos) - 10} 个组合未显示")

        # 显示最佳组合
        print(f"\n🥇 最佳组合:")
        best_combo = result.get('selected_combination')
        if best_combo:
            print(f"   POI满足: {best_combo.get('poi_satisfaction_count', 0)}/{best_combo.get('poi_total_count', 0)}")
            print(f"   置信度: {best_combo.get('confidence', 0):.2%}")

            for entity in best_combo['entities']:
                if entity.get('entity_type') == "Building":
                    color = f"{entity.get('color_side', '')}/{entity.get('color_top', '')}"
                    print(f"   - ID={entity['entity_id']}, 颜色={color}, 坐标=({entity.get('lon', 0):.6f}, {entity.get('lat', 0):.6f})")

        print(f"\n📍 推算的用户坐标:")
        pos = result['estimated_position']
        if pos:
            print(f"   经度: {pos['longitude']:.6f}")
            print(f"   纬度: {pos['latitude']:.6f}")

        print(f"\n✅ 整体置信度: {result['confidence']:.2%}")
        print("=" * 70)

    def close(self):
        self.neo4j.close()


# ===================== 9. 主程序 =====================

def main():
    example_input = """在我的北边是一栋侧面深蓝色顶面浅灰色的建筑，距离我大概10米，他的东边紧邻这一栋灰色的建筑，我的南边10是一栋灰色的建筑，里面有一个服装店有一个便利店，它的东边是一栋灰色的建筑。"""

    system = None
    try:
        system = GeoLocalizationSystem()

        import sys
        if len(sys.argv) > 1:
            user_input = " ".join(sys.argv[1:])
        else:
            user_input = example_input

        print(f"\n{'='*70}")
        print("🌍 自然语言地理定位系统 (V3 - 子图匹配版)")
        print(f"{'='*70}\n")

        result = system.localize(user_input)
        system.print_result(result)

        return result

    except Exception as e:
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if system:
            system.close()


if __name__ == "__main__":
    main()
