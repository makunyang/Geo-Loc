import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely import wkt
from shapely.ops import linemerge
from shapely.errors import GEOSException
from shapely.geometry import LineString
import warnings
warnings.filterwarnings('ignore')

# ===================== 【配置项】按需修改 =====================
INPUT_CSV = r"D:\TTT\GeoKG\Helsinki data\test1\roads_cleaned.csv"
OUTPUT_CSV = r"D:\TTT\GeoKG\Helsinki data\roads_cleaned_final.csv"
MIN_ROAD_LENGTH = 5  # 小于5米的道路删除
HELSINKI_CRS = "EPSG:32635"
WGS84_CRS = "EPSG:4326"

# ===================== 工具函数 =====================
def safe_load_wkt(wkt_str):
    """安全解析WKT，自动过滤无效/空值"""
    try:
        if pd.isna(wkt_str) or str(wkt_str).strip() == "":
            return None
        return wkt.loads(str(wkt_str).strip())
    except:
        return None

def get_all_endpoints(geom):
    """【核心修复】兼容 LineString + MultiLineString，提取所有端点"""
    endpoints = []
    if geom.type == "LineString":
        coords = list(geom.coords)
        if len(coords) >= 2:
            endpoints.append(coords[0])
            endpoints.append(coords[-1])
    elif geom.type == "MultiLineString":
        for part in geom.geoms:
            coords = list(part.coords)
            if len(coords) >= 2:
                endpoints.append(coords[0])
                endpoints.append(coords[-1])
    return endpoints

def get_connected_groups(gdf):
    """按空间端点连通性分组（修复多段线兼容）"""
    G = nx.Graph()
    # 添加所有节点
    for idx in range(len(gdf)):
        G.addNode(idx)

    # 提取所有道路的端点
    geom_list = list(gdf.geometry)
    all_endpoints = [get_all_endpoints(geom) for geom in geom_list]

    # 两两判断是否连通
    for i in range(len(geom_list)):
        for j in range(i + 1, len(geom_list)):
            # 有共同端点 = 连通
            if set(all_endpoints[i]) & set(all_endpoints[j]):
                G.addEdge(i, j)

    # 返回连通分组
    return list(nx.connectedComponents(G))

# ===================== 主清理流程 =====================
def clean_roads_fully():
    print("===== 道路全自动清理（已修复MultiLineString兼容） =====")

    # 1. 读取CSV
    df = pd.read_csv(INPUT_CSV)
    required_cols = ["osm_id", "code", "fclass", "name", "WKT"]
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：缺少字段 {col}")
            return

    # 2. 解析WKT + 过滤无效数据
    print("解析WKT几何数据...")
    df["geometry"] = df["WKT"].apply(safe_load_wkt)
    df = df[df["geometry"].notna()].copy()
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84_CRS)
    print(f"有效道路数量：{len(gdf)}")

    # 3. 投影转换 → 计算长度 → 删除短道路
    gdf_proj = gdf.to_crs(HELSINKI_CRS)
    gdf_proj["length"] = gdf_proj.length
    gdf_filtered = gdf_proj[gdf_proj["length"] >= MIN_ROAD_LENGTH].copy()
    print(f"过滤短道路后：{len(gdf_filtered)}")

    # 4. 合并连通道路（有名按名称，无名按空间）
    print("合并空间连通道路...")
    merged = []

    # --- 有名称道路：按名称分组 + 连通合并 ---
    named_roads = gdf_filtered[gdf_filtered["name"].notna() & (gdf_filtered["name"] != "")]
    for name, group in named_roads.groupby("name"):
        group = group.reset_index(drop=True)
        try:
            conn_groups = get_connected_groups(group)
            for cg in conn_groups:
                sub = group.iloc[cg]
                line = linemerge(sub.geometry.unary_union)
                row = sub.iloc[0].copy()
                row.geometry = line
                merged.append(row)
        except:
            merged.extend([row for _, row in group.iterrows()])

    # --- 无名称道路：纯空间连通合并（核心需求） ---
    unnamed_roads = gdf_filtered[gdf_filtered["name"].isna() | (gdf_filtered["name"] == "")]
    if len(unnamed_roads) > 0:
        unnamed_roads = unnamed_roads.reset_index(drop=True)
        try:
            conn_groups = get_connected_groups(unnamed_roads)
            for cg in conn_groups:
                sub = unnamed_roads.iloc[cg]
                line = linemerge(sub.geometry.unary_union)
                row = sub.iloc[0].copy()
                row.geometry = line
                merged.append(row)
        except:
            merged.extend([row for _, row in unnamed_roads.iterrows()])

    # 5. 生成最终数据
    final_gdf = gpd.GeoDataFrame(merged, crs=HELSINKI_CRS)
    final_gdf_wgs = final_gdf.to_crs(WGS84_CRS)
    final_gdf_wgs["WKT"] = final_gdf_wgs.geometry.to_wkt()

    # 6. 导出CSV（保持原始格式）
    output_df = final_gdf_wgs[required_cols].copy()
    output_df.reset_index(drop=True, inplace=True)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 7. 输出结果
    print(f"\n===== ✅ 清理完成 =====")
    print(f"原始有效道路：{len(gdf)}")
    print(f"最终清理后：{len(output_df)}")
    print(f"所有连通道路已合并！无名断路已修复！")
    print(f"文件保存至：{OUTPUT_CSV}")

if __name__ == "__main__":
    clean_roads_fully()