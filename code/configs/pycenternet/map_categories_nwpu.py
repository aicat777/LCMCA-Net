import json
import copy

# 输入原始 NWPU 测试集 annotation
with open('/mnt/cs_dsns_public/share/group_lvxiaodan/NWPU_VHR10/instances_val2017.json', 'r') as f:
    nwpu_data = json.load(f)

# NWPU 类别 ID -> DIOR 类别 ID （严格对齐）
# 以下映射根据两个数据集的类别名称手动对齐，务必确认正确性
mapping = {
    0: 0,    # airplane -> airplane
    1: 13,   # ship -> ship
    2: 15,   # storage tank -> storage tank
    3: 2,    # baseball diamond -> baseball field (DIOR 中是 baseball field)
    4: 16,   # tennis court -> tennis court
    5: 3,    # basketball court -> basketball court
    6: 10,   # ground track field -> ground track field
    7: 11,   # harbor -> harbor
    8: 4,    # bridge -> bridge
    9: 18    # vehicle -> vehicle
}
# 共有类别列表
common_categories = list(mapping.keys())

# 1. 过滤 annotations，只保留共有类别，并映射其 category_id
new_annotations = []
for ann in nwpu_data['annotations']:
    if ann['category_id'] in mapping:
        new_ann = copy.deepcopy(ann)
        new_ann['category_id'] = mapping[ann['category_id']]
        new_annotations.append(new_ann)

# 2. 过滤 categories，只保留映射后的 DIOR 类别
dior_categories_to_keep = set(mapping.values())
new_categories = []
for cat in nwpu_data['categories']:
    if cat['id'] in mapping:
        new_cat = copy.deepcopy(cat)
        new_cat['id'] = mapping[cat['id']]
        # 可选：将名称也改为 DIOR 中的名称，以便查看
        # 但评估时主要靠 ID
        new_categories.append(new_cat)

# 3. 更新 JSON
new_data = copy.deepcopy(nwpu_data)
new_data['annotations'] = new_annotations
new_data['categories'] = new_categories

# 保存为新文件
with open('data/nwpu/annotations/instances_test_dior_mapped.json', 'w') as f:
    json.dump(new_data, f)

print(f'原始标注数: {len(nwpu_data["annotations"])}, 映射后标注数: {len(new_annotations)}')