import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.ops import linemerge
from shapely.errors import GEOSException
import warnings

warnings.filterwarnings('ignore')

# ===================== 【用户配置】自行修改路径 =====================
INPUT_CSV = r"D:\TTT\GeoKG\Helsinki data\test1\test-road.csv"  # 你的道路CSV
OUTPUT_CSV = r"D:\TTT\GeoKG\Helsinki data\test1\roads_cleaned.csv"  # 输出路径
MIN_ROAD_LENGTH = 5  # 小于5米的道路删除（可修改）
HELSINKI_CRS = "EPSG:32635"  # 赫尔辛基投影（算长度用）
WGS84_CRS = "EPSG:4326"  # OSM默认坐标系


# ===================== 安全解析WKT（核心修复） =====================
def safe_load_wkt(wkt_str):
    """安全解析WKT，跳过空值、无效格式"""
    try:
        # 过滤空字符串、纯空格
        if pd.isna(wkt_str) or str(wkt_str).strip() == "":
            return None
        # 解析WKT
        return wkt.loads(str(wkt_str).strip())
    except GEOSException:
        # 解析失败返回None，后续自动过滤
        return None
    except Exception:
        return None


# ===================== 核心处理流程 =====================
def clean_road_data():
    print("===== 开始道路数据清理（已修复WKT解析报错） =====")

    # 1. 读取CSV
    df = pd.read_csv(INPUT_CSV)
    required_cols = ["osm_id", "code", "fclass", "name", "WKT"]

    # 校验字段
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：缺少字段 {col}")
            return

    # 2. 【修复】安全解析WKT，自动跳过无效数据
    print("正在校验并解析WKT几何数据...")
    df["geometry"] = df["WKT"].apply(safe_load_wkt)

    # 过滤掉WKT解析失败的行（核心！）
    original_count = len(df)
    df = df[df["geometry"].notna()].copy()
    invalid_count = original_count - len(df)
    print(f"过滤无效WKT数据：{invalid_count} 条")

    # 3. 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84_CRS)

    # 4. 转换投影，计算长度（米）
    gdf_proj = gdf.to_crs(HELSINKI_CRS)
    gdf_proj["length_m"] = gdf_proj.length

    # 5. 删除过短道路
    print(f"过滤长度小于 {MIN_ROAD_LENGTH} 米的道路...")
    gdf_filtered = gdf_proj[gdf_proj["length_m"] >= MIN_ROAD_LENGTH].copy()

    # 6. 合并同名道路（有名字才合并，无名不合并）
    print("合并同名连通道路...")
    merged_roads = []

    # 分组：有名称 / 无名称
    named_roads = gdf_filtered[gdf_filtered["name"].notna() & (gdf_filtered["name"] != "")]
    unnamed_roads = gdf_filtered[gdf_filtered["name"].isna() | (gdf_filtered["name"] == "")]

    # 合并有名称的道路
    for name, group in named_roads.groupby("name"):
        try:
            merged_geom = linemerge(group.geometry.unary_union)
            row = group.iloc[0].copy()
            row.geometry = merged_geom
            merged_roads.append(row)
        except:
            # 合并失败则保留原线段
            merged_roads.extend([r for _, r in group.iterrows()])

    # 无名道路直接保留
    merged_roads.extend([r for _, r in unnamed_roads.iterrows()])

    # 7. 生成最终数据
    final_gdf = gpd.GeoDataFrame(merged_roads, crs=HELSINKI_CRS)
    final_gdf_wgs = final_gdf.to_crs(WGS84_CRS)
    final_gdf_wgs["WKT"] = final_gdf_wgs.geometry.to_wkt()

    # 8. 导出（保留原始字段）
    output_df = final_gdf_wgs[required_cols].copy()
    output_df.reset_index(drop=True, inplace=True)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 统计结果
    print(f"\n===== 清理完成 =====")
    print(f"原始道路总数：{original_count}")
    print(f"过滤无效WKT：{invalid_count}")
    print(f"过滤短道路后：{len(gdf_filtered)}")
    print(f"合并后最终：{len(output_df)}")
    print(f"清理完成！文件保存至：{OUTPUT_CSV}")


if __name__ == "__main__":
    clean_road_data()