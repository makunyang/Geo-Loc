
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.strtree import STRtree
import os
import warnings
warnings.filterwarnings('ignore')

# ================== 【1】配置参数 ==================
# 四个建筑-建筑缓冲半径（米）
BUILDING_RADII = [50]
THRESHOLD_NEAR = 50   # 建筑-POI NEAR距离

BUILDING_CSV = r"buildings_with_colors.csv"
POI_CSV = r"pois_raw.csv"
BASE_OUTPUT = r"./KGdata"

# ================== 【2】工具函数 ==================
def safe_wkt_loads(s):
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return wkt.loads(s)
    except:
        return None

def read_csv_fixed(path):
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except:
        return pd.read_csv(path, encoding='gbk')

def compute_angle(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = np.arctan2(dy, dx) * 180 / np.pi
    dirs = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    idx = int(((angle + 22.5) % 360) / 45)
    return dirs[idx]

# ================== 【3】加载数据（一次加载，四套共用） ==================
def load_buildings():
    df = read_csv_fixed(BUILDING_CSV)
    if 'WKT坐标' in df.columns:
        df = df.rename(columns={"WKT坐标": "WKT"})
    if 'id' not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'id'})
    df['geometry'] = df['WKT'].apply(safe_wkt_loads)
    df = df[df['geometry'].notna()].copy()
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf_m = gdf.to_crs(epsg=3857)
    gdf['centroid'] = gdf.geometry.centroid
    gdf['lon'] = gdf.centroid.x
    gdf['lat'] = gdf.centroid.y
    return gdf, gdf_m

def load_pois():
    df = read_csv_fixed(POI_CSV)
    if 'id' not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'id'})
    df['geometry'] = df['WKT'].apply(safe_wkt_loads)
    df = df[df['geometry'].notna()].copy()
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf_m = gdf.to_crs(epsg=3857)
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    return gdf, gdf_m

def export_nodes(buildings, pois, out_dir):
    """导出建筑和POI节点（所有半径共用）"""
    os.makedirs(out_dir, exist_ok=True)
    
    bld_nodes = buildings[['id', 'color_side', 'color_top', 'WKT', 'lon', 'lat']].copy()
    bld_nodes.columns = ['id:ID', 'color_side', 'color_top', 'geometry', 'lon', 'lat']
    bld_nodes[':LABEL'] = 'Building'
    bld_nodes.to_csv(f"{out_dir}/buildings.csv", index=False, encoding='utf-8-sig')
    
    poi_nodes = pois[['id', 'name', 'fclass', 'WKT', 'lon', 'lat']].copy()
    poi_nodes.columns = ['id:ID', 'name', 'fclass', 'geometry', 'lon', 'lat']
    poi_nodes[':LABEL'] = 'POI'
    poi_nodes.to_csv(f"{out_dir}/pois.csv", index=False, encoding='utf-8-sig')
    
    return len(bld_nodes), len(poi_nodes)

def compute_building_relations(buildings, buildings_m, radius_m):
    """计算建筑-建筑关系（指定半径内）"""
    edges = []
    # STRtree空间索引（EPSG:4326做粗筛，度数近似转换） 
    bld_tree = STRtree(buildings.geometry.values)
    
    for idx, row in buildings.iterrows():
        geom = row.geometry
        geom_m = buildings_m.iloc[idx].geometry
        candidates = bld_tree.query(geom.buffer(radius_m * 0.00001))
        for cand_idx in candidates:
            if idx >= cand_idx:
                continue
            other = buildings.iloc[cand_idx]
            other_m = buildings_m.iloc[cand_idx].geometry
            dist_m = geom_m.distance(other_m)
            if dist_m >= radius_m:
                continue
            dir_angle = compute_angle(row.centroid, other.centroid)
            edges.append((row['id'], other['id'], round(dist_m, 2), dir_angle))
    return edges

def compute_poi_relations(buildings, buildings_m, pois, pois_m):
    """计算建筑-POI关系（INSIDE/NEAR）"""
    bld_tree = STRtree(buildings.geometry.values)
    edges = []
    
    for idx, poi in pois.iterrows():
        pt = poi.geometry
        pt_m = pois_m.iloc[idx].geometry
        candidates = bld_tree.query(pt.buffer(THRESHOLD_NEAR * 0.00001))
        inside = None
        min_dist_m = float('inf')
        best_bld = None
        for cand_idx in candidates:
            bld = buildings.iloc[cand_idx]
            if bld.geometry.contains(pt):
                inside = bld['id']
                break
            bld_m = buildings_m.iloc[cand_idx].geometry
            dist_m = pt_m.distance(bld_m)
            if dist_m < min_dist_m:
                min_dist_m = dist_m
                best_bld = bld['id']
        if inside:
            edges.append((inside, poi['id'], 'INSIDE', 0.0))
        elif best_bld and min_dist_m < THRESHOLD_NEAR:
            edges.append((best_bld, poi['id'], 'NEAR', round(min_dist_m, 2)))
    return edges

# ================== 【4】主流程 ==================
if __name__ == "__main__":
    print("=" * 60)
    print("多半径知识图谱构建")
    print("=" * 60)
    
    print("\n===== 加载实体数据 =====")
    buildings, buildings_m = load_buildings()
    pois, pois_m = load_pois()
    print(f"建筑总数: {len(buildings)}")
    print(f"POI总数: {len(pois)}")
    
    # 计算POI关系（一次性，所有半径共用）
    print("\n===== 计算建筑-POI关系（共用） =====")
    bld_poi_edges = compute_poi_relations(buildings, buildings_m, pois, pois_m)
    print(f"  建筑-POI关系数: {len(bld_poi_edges)}")
    
    # 汇总
    summary = []
    prev_edges = set()  # 上一个半径的边集，用于增量统计
    
    for radius_m in BUILDING_RADII:
        out_dir = os.path.join(BASE_OUTPUT, f"r{radius_m}")
        print(f"\n{'=' * 60}")
        print(f"构建 r={radius_m}m 知识图谱 → {out_dir}")
        print(f"{'=' * 60}")
        
        print(f"\n  计算建筑-建筑关系（r={radius_m}m）...")
        bld_edges = compute_building_relations(buildings, buildings_m, radius_m)
        n_new = len(set((a, b) for a, b, _, _ in bld_edges) - prev_edges)
        prev_edges = set((a, b) for a, b, _, _ in bld_edges)
        print(f"  建筑-建筑关系数: {len(bld_edges)} (新增 {n_new} 条)")
        
        # 导出节点
        n_bld, n_poi = export_nodes(buildings, pois, out_dir)
        
        # 导出国关系
        pd.DataFrame(bld_edges, 
                     columns=[':START_ID', ':END_ID', 'distance_m', 'direction']
                    ).to_csv(f"{out_dir}/building_building.csv", index=False, encoding='utf-8-sig')
        
        pd.DataFrame(bld_poi_edges,
                     columns=[':START_ID', ':END_ID', 'relation', 'distance_m']
                    ).to_csv(f"{out_dir}/building_poi.csv", index=False, encoding='utf-8-sig')
        
        summary.append((radius_m, len(bld_edges), n_new))
    
    print(f"\n{'=' * 60}")
    print("构建完成！汇总：")
    print(f"{'=' * 60}")
    print(f"  建筑节点: {n_bld} 栋 (所有半径共用)")
    print(f"  POI节点: {n_poi} 个 (所有半径共用)")
    print(f"  建筑-POI: {len(bld_poi_edges)} 条 (所有半径共用)")
    print()
    for r, n, n_new in summary:
        print(f"  r={r:4d}m → {n:6d} 条建筑-建筑关系 (净增 {n_new:5d})")
