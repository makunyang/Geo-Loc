"""
基于知识图谱的自然语言地理定位系统 - POI约束增强版 + 链式描述V3
V3新增：
1. 链式参照系支持（方式B：相对于其他建筑描述）
2. 递推解析链式描述（resolve_chain_entities）
3. 支持 relative_to / direction_from_ref / estimated_distance_from_ref 字段
4. 修复内容：
   - 增强POI约束处理：多个POI的优先级管理
   - POI约束优先搜索策略
   - 统计并优先展示满足最多POI约束的组合
   - 实体优先排序：有POI约束的实体优先匹配
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
    positions = {}  # entity_id -> (x, y)
    
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
    
    unresolved = [e.entity_id for e in entities if not e.resolved and not e.direction_from_user]
    if unresolved:
        print(f"   ⚠️ 无法解析的实体: {unresolved}")
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
- 酒吧 → bar            | 社区中心 → community_centre

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
            # 使用通用关系类型，通过relation属性过滤
            query1 = """
            MATCH (b:Building {id: $building_id})-[r]-(p:POI)
            WHERE r.relation = 'INSIDE' AND p.fclass = $fclass
            RETURN p.id AS poi_id, p.name AS poi_name
            LIMIT 1
            """
            try:
                result = self.neo4j.query(query1, {"building_id": bid, "fclass": fclass})
                if result and result[0].get("poi_id"):
                    return True, result[0].get("poi_name", result[0].get("poi_id"))
            except Exception as e:
                pass
            
            # 尝试使用BUILDING_POI_REL关系类型
            query2 = """
            MATCH (b:Building {id: $building_id})-[r:BUILDING_POI_REL]-(p:POI)
            WHERE r.relation = 'INSIDE' AND p.fclass = $fclass
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
            MATCH (b:Building {id: $building_id})-[r]-(p:POI)
            WHERE r.relation = 'INSIDE' AND p.fclass CONTAINS $fclass
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
            query = """
            MATCH (b:Building {id: $building_id})-[r]-(p:POI)
            WHERE r.relation = 'INSIDE'
            RETURN p.id AS poi_id, p.name AS poi_name, p.fclass AS fclass
            """
            try:
                results = self.neo4j.query(query, {"building_id": bid})
                if results:
                    return results
            except Exception as e:
                pass
            
            # 尝试BUILDING_POI_REL关系
            query2 = """
            MATCH (b:Building {id: $building_id})-[r:BUILDING_POI_REL]-(p:POI)
            WHERE r.relation = 'INSIDE'
            RETURN p.id AS poi_id, p.name AS poi_name, p.fclass AS fclass
            """
            try:
                results = self.neo4j.query(query2, {"building_id": bid})
                if results:
                    return results
            except Exception as e:
                pass
        
        return []

    def find_candidate_combinations(self, entities: List[SpatialEntity]) -> List[CandidateCombination]:
        """
        查找候选参照物组合（POI约束优先策略）
        
        策略：
        1. 识别所有有POI约束的实体
        2. 有POI约束的实体优先匹配（因为它们更具体）
        3. 按候选数量排序，从最少的组开始
        4. 对POI约束满足情况进行统计和排名
        """
        print("\n🔍 开始匹配候选参照物（POI约束优先策略）...")
        
        # 统计POI约束
        entities_with_poi = [e for e in entities if e.has_poi_constraint()]
        entities_without_poi = [e for e in entities if not e.has_poi_constraint()]
        
        print(f"📊 POI约束统计:")
        print(f"   - 有POI约束的实体: {len(entities_with_poi)} 个")
        for e in entities_with_poi:
            all_types = e.get_all_poi_types()
            poi_types_str = ', '.join(all_types) if len(all_types) > 1 else all_types[0] if all_types else "N/A"
            print(f"     • {e.entity_id}: 需要 POI类型={poi_types_str}")
        print(f"   - 无POI约束的实体: {len(entities_without_poi)} 个")
        
        # Step 1: 为每个实体查找所有候选
        entity_candidates = {}
        for entity in entities:
            candidates = []
            
            if entity.entity_type == "Building":
                print(f"\n  🔍 实体 {entity.entity_id}:")
                
                # 检查是否有POI约束
                if entity.has_poi_constraint():
                    all_poi_types = entity.get_all_poi_types()
                    poi_types_str = ', '.join(all_poi_types) if len(all_poi_types) > 1 else all_poi_types[0]
                    print(f"     [POI约束] 需要包含: {poi_types_str}")
                    
                    # 首先查找所有颜色匹配的建筑
                    print(f"     步骤1: 颜色匹配...")
                    color_matches = self.match_buildings_strict(
                        color_side=entity.color_side,
                        color_top=entity.color_top,
                        limit=100000
                    )
                    
                    # 验证所有POI约束
                    print(f"     步骤2: 验证POI约束（{len(color_matches)} 个候选中）...")
                    validated = []
                    for b in color_matches:
                        # 使用新方法检查所有POI
                        all_satisfied, satisfied_names = self.check_all_pois_in_building(
                            str(b["id"]), all_poi_types
                        )
                        if all_satisfied:
                            b["poi_validated"] = True
                            b["matched_poi_names"] = satisfied_names
                            b["matched_poi_types"] = all_poi_types
                            validated.append(b)
                            poi_str = ', '.join(satisfied_names)
                            print(f"       ✓ {b['id']}: 含{poi_str}")
                    
                    if validated:
                        candidates = validated
                        print(f"     ✅ POI约束满足: {len(candidates)} 个有效候选")
                    else:
                        print(f"     ⚠️ POI约束无满足！使用全部颜色匹配候选: {len(color_matches)} 个")
                        candidates = color_matches  # 如果没有POI满足，使用全部颜色匹配
                else:
                    print(f"     无POI约束，直接颜色匹配...")
                    candidates = self.match_buildings_strict(
                        color_side=entity.color_side,
                        color_top=entity.color_top,
                        limit=100000
                    )
                    print(f"     找到 {len(candidates)} 个候选")
            
            entity_candidates[entity.entity_id] = candidates
        
        # Step 2: POI优先排序策略
        # 排序规则：
        # 1. 有POI约束的实体优先（poi_penalty=0）
        # 2. 无POI约束的实体靠后（poi_penalty=1000）
        # 3. 同类中按候选数量升序
        def sort_key(e):
            poi_penalty = 0 if e.has_poi_constraint() else 1000
            candidate_count = len(entity_candidates.get(e.entity_id, []))
            return (poi_penalty, candidate_count)
        
        sorted_entities = sorted(entities, key=sort_key)
        
        print(f"\n🎯 POI优先排序后的匹配顺序:")
        for i, e in enumerate(sorted_entities, 1):
            poi_info = ""
            if e.has_poi_constraint():
                all_types = e.get_all_poi_types()
                poi_info = f"[含POI: {', '.join(all_types)}]"
            candidates = entity_candidates.get(e.entity_id, [])
            print(f"   {i}. {e.entity_id} {poi_info}: {len(candidates)} 个候选")
        
        # 确定最少候选的实体（跳过候选数量为0的实体）
        # 首先过滤出候选数量>0的实体
        entities_with_candidates = [
            (e, len(entity_candidates.get(e.entity_id, [])))
            for e in sorted_entities
            if len(entity_candidates.get(e.entity_id, [])) > 0
        ]
        
        if not entities_with_candidates:
            print(f"\n⚠️ 所有实体的候选数量均为0，尝试全局搜索...")
            # 如果所有实体都没有候选，返回空结果
            return []
        
        # 选择候选数量最少且>0的实体
        min_entity, min_count = entities_with_candidates[0]
        
        # 确定other_entities（排除min_entity）
        min_entity_id = min_entity.entity_id
        other_entities = [e for e in sorted_entities if e.entity_id != min_entity_id]
        
        min_candidates = entity_candidates.get(min_entity.entity_id, [])
        
        print(f"\n🎯 从 {min_entity.entity_id} 开始搜索...")
        print(f"   最少候选组: {len(min_candidates)} 个")
        
        # 输出Cypher语句示例
        print(f"\n📜 匹配Cypher语句示例:")
        for i, entity in enumerate(other_entities[:2]):
            print(f"   -- 匹配 {entity.entity_id}:")
            print(f"   MATCH (b:Building)-[r]-(n:Building)")
            poi_clause = f" WHERE n:Building" if not entity.has_poi_constraint() else f" WHERE n:Building"
            print(f"   {poi_clause}...")
        
        # Step 3: 从最少候选组开始搜索
        valid_combinations = []
        max_combinations_per_candidate = 50
        max_total_combinations = 500

        print(f"\n🔎 开始搜索有效组合...")
        
        for idx, min_candidate in enumerate(min_candidates):
            if len(valid_combinations) >= max_total_combinations:
                break
            
            neighbors = self._get_neighbors_as_dict(min_candidate["id"])
            
            matched = self._find_matches_in_neighbors(
                min_entity, min_candidate,
                other_entities, entity_candidates,
                neighbors, entities,
                max_combinations_per_candidate
            )
            
            if matched:
                valid_combinations.extend(matched)
                print(f"    ✓ {min_candidate['id']} ({idx+1}/{len(min_candidates)}): 找到 {len(matched)} 个，累计 {len(valid_combinations)} 个")
        
        print(f"\n📊 总共找到 {len(valid_combinations)} 个有效组合")
        
        # Step 4: 统计POI满足情况（使用POI总数而不是实体数）
        poi_total_count = self._get_total_poi_count(entities)
        for combo in valid_combinations:
            combo.poi_total_count = poi_total_count
            combo.poi_satisfaction_count = self._count_satisfied_poi(combo, entities)
            combo.poi_constraint_satisfied = (combo.poi_satisfaction_count == poi_total_count and poi_total_count > 0)
            combo.satisfied_poi_types = self._get_satisfied_poi_types(combo, entities)
        
        # Step 5: 按POI满足数和置信度排序
        if valid_combinations:
            print(f"\n📈 置信度评估和排序...")
            for combo in valid_combinations:
                combo.total_score = self._calculate_detailed_score(combo, entities)
                combo.confidence = self._calculate_confidence(combo, entities)
            
            # 排序：POI满足数优先，然后是置信度
            valid_combinations.sort(key=lambda x: (x.poi_satisfaction_count, x.confidence), reverse=True)
            
            # 输出排名
            print(f"\n🏆 组合排名（POI满足数优先）:")
            for i, combo in enumerate(valid_combinations[:10], 1):
                poi_status = f"✓POI({combo.poi_satisfaction_count}/{combo.poi_total_count})"
                if combo.satisfied_poi_types:
                    poi_status += f": {', '.join(combo.satisfied_poi_types)}"
                entity_ids = [e.entity_id for e in combo.entities]
                print(f"   第{i}名: POI满足={combo.poi_satisfaction_count}/{combo.poi_total_count}, 置信度={combo.confidence:.2%}")
                print(f"          满足的POI: {poi_status}")
                print(f"          建筑: {', '.join(entity_ids[:5])}{'...' if len(entity_ids) > 5 else ''}")
        
        return valid_combinations
    
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
    
    def _get_neighbors_as_dict(self, building_id: str) -> Dict[str, Dict]:
        """获取建筑的一跳邻居"""
        query = """
        MATCH (b:Building {id: $building_id})-[r]-(n)
        WHERE n:Building
        RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
               n.lon AS lon, n.lat AS lat, r.direction AS direction, r.distance AS distance
        LIMIT 1000
        """
        try:
            results = self.neo4j.query(query, {"building_id": building_id})
            return {str(r["id"]): r for r in results}
        except:
            return {}
    
    def _get_neighbors_with_poi(self, building_id: str) -> Dict[str, Dict]:
        """获取建筑的一跳邻居，包含POI信息"""
        query = """
        MATCH (b:Building {id: $building_id})-[r]-(n:Building)
        OPTIONAL MATCH (n)-[r2]-(p:POI)
        WHERE r.relation = 'INSIDE'
        RETURN n.id AS id, n.color_side AS color_side, n.color_top AS color_top,
               n.lon AS lon, n.lat AS lat, r.direction AS direction, r.distance AS distance,
               COLLECT(CASE WHEN p.fclass IS NOT NULL THEN p.fclass END) AS poi_fclasses
        LIMIT 1000
        """
        try:
            results = self.neo4j.query(query, {"building_id": building_id})
            return {str(r["id"]): r for r in results}
        except:
            return {}
    
    def _find_matches_in_neighbors(self, min_entity: SpatialEntity, min_candidate: Dict,
                                    other_entities: List[SpatialEntity],
                                    entity_candidates: Dict[str, List[Dict]],
                                    neighbors: Dict[str, Dict],
                                    all_entities: List[SpatialEntity],
                                    max_combinations: int = 5) -> List[CandidateCombination]:
        """在邻居中查找满足所有实体特征的匹配"""
        combinations = []
        entity_matches = {}
        
        print(f"    🔎 检查候选 {min_candidate['id']} 的一跳邻居 ({len(neighbors)}个):")
        
        for entity in other_entities:
            matches = []
            
            for nid, neighbor in neighbors.items():
                if self._building_matches_criteria(neighbor, entity):
                    matches.append(neighbor)
            
            match_ids = [str(m["id"]) for m in matches[:10]]
            poi_info = ""
            if entity.has_poi_constraint():
                all_types = entity.get_all_poi_types()
                poi_info = f" [需POI:{', '.join(all_types)}]"
            print(f"      - {entity.entity_id}{poi_info}: 邻居匹配 {len(matches)} 个")
            if matches:
                print(f"        ID: {', '.join(match_ids)}{'...' if len(matches) > 10 else ''}")
            
            if not matches:
                global_candidates = entity_candidates.get(entity.entity_id, [])
                matches = global_candidates[:50]
                if matches:
                    print(f"        ⚠️ 邻居中未找到，从全局候选取 {len(matches)} 个")
            
            if not matches:
                print(f"        ❌ 无匹配，放弃该候选")
                return []
            
            entity_matches[entity.entity_id] = matches
        
        # 生成组合
        import itertools
        entity_ids = list(entity_matches.keys())
        match_lists = [entity_matches[eid] for eid in entity_ids]
        
        for combo_tuple in itertools.product(*match_lists):
            combo_ids = [c["id"] for c in combo_tuple]
            if len(set(combo_ids)) != len(combo_ids):
                continue
            
            combo_entities = [
                MatchedEntity(
                    query_id=min_entity.entity_id,
                    entity_id=str(min_candidate["id"]),
                    entity_type=min_entity.entity_type,
                    lon=min_candidate["lon"],
                    lat=min_candidate["lat"],
                    color_side=min_candidate.get("color_side", ""),
                    color_top=min_candidate.get("color_top", ""),
                    poi_validation=min_candidate.get("poi_validated", False),
                    required_poi_type=min_entity.get_poi_type() if min_entity.has_poi_constraint() else "",
                    required_poi_types=min_entity.get_all_poi_types() if min_entity.has_poi_constraint() else [],
                    matched_poi_names=min_candidate.get("matched_poi_names", [])
                )
            ]
            
            for i, entity_id in enumerate(entity_ids):
                entity = next(e for e in other_entities if e.entity_id == entity_id)
                candidate = combo_tuple[i]
                combo_entities.append(
                    MatchedEntity(
                        query_id=entity_id,
                        entity_id=str(candidate["id"]),
                        entity_type=entity.entity_type,
                        lon=candidate["lon"],
                        lat=candidate["lat"],
                        color_side=candidate.get("color_side", ""),
                        color_top=candidate.get("color_top", ""),
                        poi_validation=candidate.get("poi_validated", False),
                        required_poi_type=entity.get_poi_type() if entity.has_poi_constraint() else "",
                        required_poi_types=entity.get_all_poi_types() if entity.has_poi_constraint() else [],
                        matched_poi_names=candidate.get("matched_poi_names", [])
                    )
                )
            
            combo = CandidateCombination(entities=combo_entities)
            
            if self._validate_direction_constraints(combo, all_entities):
                combo.poi_satisfaction_count = self._count_satisfied_poi(combo, all_entities)
                combo.poi_constraint_satisfied = (combo.poi_satisfaction_count > 0)
                combo.total_score = self._calculate_score(combo, all_entities)
                combinations.append(combo)
                
                if len(combinations) >= max_combinations:
                    break
        
        return combinations
    
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
                    road_orientation=ref.get("road_orientation", "")
                )
                entity.direction_from_user = ref.get("direction_from_user") or ""
                est_dist = ref.get("estimated_distance")
                entity.estimated_distance = float(est_dist) if est_dist is not None else 0.0
                
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
            
            # Step 1.5: 递推解析链式描述
            if chain_count > 0:
                print("\n🔄 Step 1.5: 递推解析链式描述...")
                all_resolved = resolve_chain_entities(entities)
                
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
        print("📊 地理定位结果 (POI约束增强版)")
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
    example_input = """我在一条南北向的道路上，东边是一栋侧面浅蓝顶面浅绿色的建筑，这个建筑里有两个理发店，距离我大概10米；西边是一栋侧面灰色顶面深灰色的建筑，距离我大概15米；
    东北是一栋顶面侧面深灰色的建筑，里面有一个饭店，距离我大概35米；我的西南是一栋侧面深灰顶面浅灰色的建筑，距离我大概30米；东南是一栋侧面灰色顶面深灰色的建筑，距离我大概35米。"""

    system = None
    try:
        system = GeoLocalizationSystem()

        import sys
        if len(sys.argv) > 1:
            user_input = " ".join(sys.argv[1:])
        else:
            user_input = example_input

        print(f"\n{'='*70}")
        print("🌍 自然语言地理定位系统 (POI约束增强版)")
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
