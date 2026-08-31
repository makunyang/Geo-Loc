"""
基于Delaunay三角网的知识图谱构建
构建原则：
1. Building-Building: Delaunay三角网
2. Building-POI: 包含关系(INSIDE) + 最近邻关系(NEAR)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt, geometry
from scipy.spatial import Delaunay
import math
import os
import warnings

warnings.filterwarnings('ignore')

# ================== 【1】配置参数 ==================
BUILDING_CSV = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\buildings_with_colors.csv"
POI_CSV = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\pois_raw.csv"
OUTPUT_DIR = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\KGdata\delaunay"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 阈值参数（单位：米，EPSG:3857投影后）
THRESHOLD_BUILDING_POI = 50  # 建筑-POI最近邻搜索半径（用于NEAR关系）

EARTH_RADIUS = 6371000  # 地球半径（米）


# ================== 【2】工具函数 ==================
def safe_wkt_loads(s):
    """安全解析WKT"""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return wkt.loads(s)
    except:
        return None


def read_csv_fixed(path):
    """修复CSV编码问题"""
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            df = pd.read_csv(path, encoding=encoding)
            # 验证WKT列是否可解析
            if 'WKT' in df.columns:
                from shapely import wkt
                test = df['WKT'].dropna().iloc[0] if len(df['WKT'].dropna()) > 0 else None
                if test and wkt.loads(str(test)):
                    return df
            return df
        except:
            continue
    raise ValueError(f"无法读取文件: {path}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的球面距离（米）"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS * c


def compute_angle_deg(lon1, lat1, lon2, lat2):
    """计算方位角（0-360度）"""
    dx = lon2 - lon1
    dy = lat2 - lat1
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return round(angle, 6)


def compute_direction_8(angle_deg):
    """将角度转换为8方位"""
    dirs = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    idx = int(((angle_deg + 22.5) % 360) / 45)
    return dirs[idx]


# ================== 【3】加载数据 ==================
def load_data():
    """加载所有实体数据"""
    print("===== 加载实体数据 =====")
    
    # 加载建筑
    df_bld = read_csv_fixed(BUILDING_CSV)
    df_bld = df_bld.rename(columns={"WKT坐标": "WKT"})
    df_bld['geometry'] = df_bld['WKT'].apply(safe_wkt_loads)
    df_bld = df_bld[df_bld['geometry'].notna()].copy()
    gdf_bld = gpd.GeoDataFrame(df_bld, geometry='geometry', crs="EPSG:4326")
    gdf_bld['centroid'] = gdf_bld.geometry.centroid
    gdf_bld['lon'] = gdf_bld.centroid.x
    gdf_bld['lat'] = gdf_bld.centroid.y
    gdf_bld['id'] = gdf_bld['id'].astype(str)
    print(f"  建筑: {len(gdf_bld)} 个")
    
    # 加载POI
    df_poi = read_csv_fixed(POI_CSV)
    df_poi = df_poi.rename(columns={"code": "id"})
    df_poi['geometry'] = df_poi['WKT'].apply(safe_wkt_loads)
    df_poi = df_poi[df_poi['geometry'].notna()].copy()
    gdf_poi = gpd.GeoDataFrame(df_poi, geometry='geometry', crs="EPSG:4326")
    gdf_poi['lon'] = gdf_poi.geometry.x
    gdf_poi['lat'] = gdf_poi.geometry.y
    gdf_poi['id'] = gdf_poi['id'].astype(str)
    print(f"  POI: {len(gdf_poi)} 个")
    
    return gdf_bld, gdf_poi


# ================== 【4】构建空间索引 ==================
def build_spatial_indexes(gdf_bld, gdf_poi):
    """构建空间索引（R-tree）"""
    print("\n===== 构建空间索引 =====")
    
    # 转换为投影坐标系（米为单位）
    gdf_bld_3857 = gdf_bld.to_crs(epsg=3857)
    gdf_poi_3857 = gdf_poi.to_crs(epsg=3857)
    
    print("  空间索引构建完成")
    return gdf_bld_3857, gdf_poi_3857


# ================== 【5】Delaunay三角网构建 ==================
def build_delaunay_triangulation(gdf_bld):
    """
    使用Delaunay三角网构建建筑-建筑关系
    返回：边列表 [(building_id1, building_id2, distance_m, angle_deg), ...]
    """
    print("\n===== 构建Building-Building Delaunay三角网 =====")
    
    # 获取建筑centroid坐标和ID映射
    ids = gdf_bld['id'].values
    points = np.array([[b.lon, b.lat] for _, b in gdf_bld.iterrows()])
    
    # 构建Delaunay三角网
    tri = Delaunay(points)
    
    # 提取边（去重）
    edges_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = simplex[i], simplex[j]
                # 使用索引对作为边的唯一标识
                edge_key = (min(p1, p2), max(p1, p2))
                edges_set.add(edge_key)
    
    # 构建边列表
    edges = []
    for p1, p2 in edges_set:
        id1 = ids[p1]  # 直接使用points数组的索引对应的ID
        id2 = ids[p2]
        lon1, lat1 = points[p1]
        lon2, lat2 = points[p2]
        
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        angle = compute_angle_deg(lon1, lat1, lon2, lat2)
        direction = compute_direction_8(angle)
        
        edges.append({
            'start_id': id1,
            'end_id': id2,
            'distance_m': dist,
            'angle_deg': angle,
            'direction': direction
        })
    
    print(f"  Delaunay边数: {len(edges)}")
    return edges


# ================== 【6】Building-POI关系构建 ==================
def build_building_poi_relations(gdf_bld, gdf_poi):
    """
    构建Building-POI关系
    1. INSIDE: POI在建筑内部
    2. NEAR: POI不在任何建筑内，找最近的建筑
    """
    print("\n===== 构建Building-POI关系 =====")
    
    inside_edges = []  # INSIDE关系
    near_edges = []    # NEAR关系
    poi_ids = set()
    
    # 方法1: 使用sjoin检测POI是否在建筑内部（更可靠）
    print("  步骤1: 检测POI是否在建筑内部 (INSIDE)...")
    
    # 转换为投影坐标系以提高准确性
    gdf_bld_proj = gdf_bld.to_crs(epsg=3857)
    gdf_poi_proj = gdf_poi.to_crs(epsg=3857)
    
    # 使用sjoin进行空间连接
    try:
        joined = gpd.sjoin(gdf_poi_proj, gdf_bld_proj, how='left', predicate='within')
        # 找出匹配到建筑物的POI
        inside_pois = joined[joined['id_right'].notna()]
        
        for _, row in inside_pois.iterrows():
            inside_edges.append({
                'start_id': str(row['id_right']),
                'end_id': str(row['id_left']),
                'relation': 'INSIDE',
                'distance_m': 0.0,
                'poi_fclass': row.get('fclass', ''),
                'poi_name': row.get('name', '')
            })
            poi_ids.add(str(row['id_left']))
        
        print(f"    INSIDE关系数: {len(inside_edges)}")
    except Exception as e:
        print(f"    sjoin失败，使用备选方法: {e}")
        # 备选方法：逐个检测
        for _, poi in gdf_poi.iterrows():
            poi_geom = poi.geometry
            for _, bld in gdf_bld.iterrows():
                if bld.geometry.contains(poi_geom):
                    inside_edges.append({
                        'start_id': str(bld['id']),
                        'end_id': str(poi['id']),
                        'relation': 'INSIDE',
                        'distance_m': 0.0,
                        'poi_fclass': poi.get('fclass', ''),
                        'poi_name': poi.get('name', '')
                    })
                    poi_ids.add(str(poi['id']))
                    break
        print(f"    INSIDE关系数(备选): {len(inside_edges)}")
    
    # 方法2: 对于不在任何建筑内的POI，找最近的建筑
    print("  步骤2: 检测POI最近建筑 (NEAR)...")
    
    # 找出不在任何建筑内的POI
    outside_pois = gdf_poi[~gdf_poi['id'].isin(poi_ids)]
    
    # 为每个POI找最近的建筑
    bld_ids = gdf_bld['id'].values
    bld_points = np.array([[b.lon, b.lat] for _, b in gdf_bld.iterrows()])
    
    for _, poi in outside_pois.iterrows():
        poi_lon, poi_lat = poi.lon, poi.lat
        
        # 计算到所有建筑的距离
        distances = [
            haversine_distance(poi_lat, poi_lon, lat, lon)
            for lon, lat in bld_points
        ]
        
        # 找到最近的建筑
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        if min_dist < THRESHOLD_BUILDING_POI:
            near_edges.append({
                'start_id': bld_ids[min_idx],
                'end_id': poi['id'],
                'relation': 'NEAR',
                'distance_m': round(min_dist, 6),
                'poi_fclass': poi.get('fclass', ''),
                'poi_name': poi.get('name', '')
            })
    
    print(f"    NEAR关系数: {len(near_edges)}")
    
    return inside_edges, near_edges


# ================== 【7】导出节点CSV ==================
def export_nodes(gdf_bld, gdf_poi):
    """导出节点数据为Neo4j格式"""
    print("\n===== 导出节点数据 =====")
    
    # 建筑节点
    bld_nodes = gdf_bld[['id', 'color_side', 'color_top', 'geometry', 'lon', 'lat']].copy()
    bld_nodes.columns = ['id:ID', 'color_side', 'color_top', 'geometry', 'lon', 'lat']
    bld_nodes[':LABEL'] = 'Building'
    bld_nodes.to_csv(f"{OUTPUT_DIR}/buildings.csv", index=False, encoding='utf-8-sig')
    print(f"  建筑节点: {len(bld_nodes)}")
    
    # POI节点
    poi_nodes = gdf_poi[['id', 'name', 'fclass', 'geometry', 'lon', 'lat']].copy()
    poi_nodes.columns = ['id:ID', 'name', 'fclass', 'geometry', 'lon', 'lat']
    poi_nodes[':LABEL'] = 'POI'
    poi_nodes.to_csv(f"{OUTPUT_DIR}/pois.csv", index=False, encoding='utf-8-sig')
    print(f"  POI节点: {len(poi_nodes)}")


# ================== 【8】导出关系CSV ==================
def export_relations(delaunay_edges, inside_edges, near_edges):
    """导出关系数据为Neo4j格式"""
    print("\n===== 导出关系数据 =====")
    
    # 1. Delaunay边 (Building-Building)
    if delaunay_edges:
        df_delaunay = pd.DataFrame(delaunay_edges)
        df_delaunay.columns = [':START_ID', ':END_ID', 'distance_m', 'angle_deg', 'direction']
        df_delaunay[':TYPE'] = 'DELAUNAY'
        df_delaunay = df_delaunay[[':START_ID', ':END_ID', 'distance_m', 'angle_deg', 'direction', ':TYPE']]
        df_delaunay.to_csv(f"{OUTPUT_DIR}/building_building_delaunay.csv", index=False, encoding='utf-8-sig')
        print(f"  Building-DELAUNAY-Building: {len(df_delaunay)}")
    
    # 2. INSIDE边 (Building-POI)
    if inside_edges:
        df_inside = pd.DataFrame(inside_edges)
        df_inside.columns = [':START_ID', ':END_ID', 'relation', 'distance_m', 'poi_fclass', 'poi_name']
        df_inside = df_inside[[':START_ID', ':END_ID', 'relation', 'distance_m', 'poi_fclass', 'poi_name']]
        df_inside.to_csv(f"{OUTPUT_DIR}/building_poi_inside.csv", index=False, encoding='utf-8-sig')
        print(f"  Building-INSIDE-POI: {len(df_inside)}")
    
    # 3. NEAR边 (Building-POI)
    if near_edges:
        df_near = pd.DataFrame(near_edges)
        df_near.columns = [':START_ID', ':END_ID', 'relation', 'distance_m', 'poi_fclass', 'poi_name']
        df_near = df_near[[':START_ID', ':END_ID', 'relation', 'distance_m', 'poi_fclass', 'poi_name']]
        df_near.to_csv(f"{OUTPUT_DIR}/building_poi_near.csv", index=False, encoding='utf-8-sig')
        print(f"  Building-NEAR-POI: {len(df_near)}")
    
    # 4. 导出统一格式的所有关系
    all_edges = []
    for e in delaunay_edges:
        all_edges.append({':START_ID': e['start_id'], ':END_ID': e['end_id'], 
                         'relation': 'DELAUNAY', 'distance_m': e['distance_m'],
                         'angle_deg': e['angle_deg'], 'direction': e['direction']})
    for e in inside_edges:
        all_edges.append({':START_ID': e['start_id'], ':END_ID': e['end_id'],
                         'relation': e['relation'], 'distance_m': e['distance_m']})
    for e in near_edges:
        all_edges.append({':START_ID': e['start_id'], ':END_ID': e['end_id'],
                         'relation': e['relation'], 'distance_m': e['distance_m']})
    
    if all_edges:
        df_all = pd.DataFrame(all_edges)
        df_all.to_csv(f"{OUTPUT_DIR}/all_relations.csv", index=False, encoding='utf-8-sig')
        print(f"  总关系数: {len(all_edges)}")


# ================== 【9】主流程 ==================
def main():
    print("=" * 60)
    print("基于Delaunay三角网的知识图谱构建")
    print("=" * 60)
    
    # 1. 加载数据
    gdf_bld, gdf_poi = load_data()
    
    # 2. 构建空间索引
    gdf_bld_3857, gdf_poi_3857 = build_spatial_indexes(gdf_bld, gdf_poi)
    
    # 3. 构建Delaunay三角网 (Building-Building)
    delaunay_edges = build_delaunay_triangulation(gdf_bld)
    
    # 4. 构建Building-POI关系
    inside_edges, near_edges = build_building_poi_relations(gdf_bld, gdf_poi)
    
    # 5. 导出节点
    export_nodes(gdf_bld, gdf_poi)
    
    # 6. 导出关系
    export_relations(delaunay_edges, inside_edges, near_edges)
    
    print("\n" + "=" * 60)
    print(f"KG construction completed!")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
