# -*- coding: utf-8 -*-
"""单条描述测试 - 调用 evaluate_localization.py 的定位组件"""
import sys, os, time

# 确保能导入 evaluate_localization
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_localization import BatchLocalizer

# ========== 在这里粘贴你的描述 ==========
USER_DESCRIPTION = """
Seven meters to my north is a building with blue sides and a gray roof, adjacent to which on the west side stands another building with blue sides and a gray roof, containing a police. Slightly southwest of me is another building with blue sides and a gray roof, housing a restaurant, approximately five meters away.
"""
# =======================================

print("=" * 70)
print("单条自然语言地理定位测试")
print("=" * 70)
print(f"\n输入描述:\n{USER_DESCRIPTION.strip()}\n")

print("初始化定位器...", end="", flush=True)
localizer = BatchLocalizer()
print(" 完成\n")

t0 = time.time()
result = localizer.localize_single(USER_DESCRIPTION.strip())
elapsed = time.time() - t0

print(f"\n{'='*70}")
print("定位结果")
print(f"{'='*70}")
print(f"状态:       {result['status']}")
if result['status'] == 'success':
    print(f"预测坐标:   lon={result['lon']:.6f}, lat={result['lat']:.6f}")
    print(f"置信度:     {result.get('confidence', 0):.2%}")
    print(f"参照物数:   {result.get('num_references', 0)}")
    print(f"匹配组合数: {result.get('num_matches', 0)}")
    combo = result.get('matched_combo', [])
    if combo:
        print(f"\n{'─'*70}")
        print("最终匹配的实体组合 (排名第1):")
        print(f"{'─'*70}")
        for m in combo:
            print(f"  {m['desc_id']:8s} → {m['building_id']:8s}  colors={m['colors']:25s} {m['poi']}")
        print(f"\n  > 坐标推算: 从 {len(combo)} 个实体的「用户→建筑」反方向联合计算")
else:
    print(f"错误信息:   {result.get('message', '')}")
print(f"耗时:       {elapsed:.1f}s")
print(f"{'='*70}")

localizer.close()
