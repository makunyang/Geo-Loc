import pandas as pd
import rasterio
import numpy as np
from shapely.wkt import loads
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform
from rasterio.mask import mask
from rasterio.crs import CRS
import base64
from PIL import Image
import csv
import os
import json
from openai import OpenAI
from pyproj import Transformer

# ===================== 【固定】你的文件路径 =====================
EXCEL_PATH = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\buildings_raw.csv"
RASTER_PATH = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\yingxiang\1.tif"
OUTPUT_CSV = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\建筑颜色分析结果.csv"
TEMP_IMG_DIR = r"D:\TTT\GeoKG\Helsinki data\test1\Karlsruhe-data-prepare\temp_crop"
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

# ===================== 坐标系转换器（全局，启动时初始化一次） =====================
_RASTER_CRS = None
_GEO_TO_RASTER = None  # EPSG:4326 → 影像坐标系的转换器

def _init_transform():
    """初始化坐标系转换器"""
    global _RASTER_CRS, _GEO_TO_RASTER
    with rasterio.open(RASTER_PATH) as src:
        _RASTER_CRS = src.crs
    _GEO_TO_RASTER = Transformer.from_crs("EPSG:4326", _RASTER_CRS, always_xy=True)
    print(f"影像CRS: {_RASTER_CRS}")

# ===================== 【你的AutoDL配置（不变）】 =====================
OPENAI_API_KEY = "8O9vGvq0gQ93aaSS2f2WvzkPuP8qNrBxdRKsUKJXCeXa4toN"
OPENAI_BASE_URL = "https://www.autodl.art/api/v1"
GPT_MODEL = "qwen3-vl-plus"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

PROMPT = """分析这张建筑图片，识别建筑的侧面主体颜色和顶面主体颜色。
只输出JSON格式，不要任何其他文字，格式如下：
{"side_color":"颜色","top_color":"颜色"}"""


# ===================== 核心函数 =====================
def crop_building_from_raster(raster, geom, save_path):
    """裁切影像 + 返回图片尺寸（自动转换坐标系）"""
    try:
        # 将几何从 EPSG:4326 转换到影像坐标系
        if _GEO_TO_RASTER and _RASTER_CRS and _RASTER_CRS != CRS.from_epsg(4326):
            geom_proj = shapely_transform(_GEO_TO_RASTER.transform, geom)
        else:
            geom_proj = geom

        out_image, out_transform = mask(
            raster, [mapping(geom_proj)], crop=True, nodata=0, filled=True
        )
        out_image = np.moveaxis(out_image, 0, -1)
        if out_image.max() == 0:
            return None, (0, 0)

        img = Image.fromarray(out_image.astype(np.uint8))
        img.save(save_path)
        return save_path, img.size
    except Exception as e:
        print(f"裁切失败: {str(e)}")
        return None, (0, 0)


def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_autodl_vision_api(base64_img):
    """调用AutoDL API"""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]
            }
        ]

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()
        color_json = json.loads(content)
        return color_json.get("side_color", "未知"), color_json.get("top_color", "未知")

    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return "识别失败", "识别失败"


# ===================== 主流程 =====================
if __name__ == "__main__":
    print("===== 开始处理建筑数据 =====")

    # 初始化坐标系转换器
    _init_transform()

    df = pd.read_csv(EXCEL_PATH)
    if "id" not in df.columns or "WKT坐标" not in df.columns:
        print("错误：CSV必须包含 id 和 WKT坐标 字段")
        exit()

    with rasterio.open(RASTER_PATH) as src:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["id", "WKT坐标", "侧面主体颜色", "顶面主体颜色", "处理状态"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, row in df.iterrows():
                building_id = row["id"]
                wkt = row["WKT坐标"]
                print(f"\n处理建筑 ID: {building_id}")

                # 1. 解析WKT
                try:
                    geom = loads(wkt)
                except:
                    writer.writerow({"id": building_id, "WKT坐标": wkt, "侧面主体颜色": "", "顶面主体颜色": "",
                                     "处理状态": "WKT解析失败"})
                    continue

                # 2. 裁切影像 + 获取尺寸
                crop_path, (img_width, img_height) = crop_building_from_raster(src, geom, os.path.join(TEMP_IMG_DIR,
                                                                                                       f"building_{building_id}.png"))
                if not crop_path:
                    writer.writerow({"id": building_id, "WKT坐标": wkt, "侧面主体颜色": "", "顶面主体颜色": "",
                                     "处理状态": "裁切失败"})
                    continue

                # ===================== 【核心修复】过滤尺寸过小的图片（宽/高 < 10 直接跳过） =====================
                if img_width < 10 or img_height < 10:
                    print(f"⚠️ 图片尺寸过小({img_width}x{img_height})，跳过API调用")
                    writer.writerow({
                        "id": building_id, "WKT坐标": wkt,
                        "侧面主体颜色": "尺寸过小", "顶面主体颜色": "尺寸过小",
                        "处理状态": "图片尺寸过小"
                    })
                    continue

                # 3. 调用API
                base64_img = image_to_base64(crop_path)
                side_color, top_color = call_autodl_vision_api(base64_img)

                # 4. 写入结果
                writer.writerow({
                    "id": building_id,
                    "WKT坐标": wkt,
                    "侧面主体颜色": side_color,
                    "顶面主体颜色": top_color,
                    "处理状态": "处理成功"
                })
                print(f"✅ 完成：侧面={side_color} | 顶面={top_color}")

    print(f"\n===== 全部处理完成！结果保存至：{OUTPUT_CSV} =====")
