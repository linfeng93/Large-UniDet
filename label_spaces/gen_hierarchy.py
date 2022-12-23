import json
import codecs
import csv
import numpy as np
import pdb

oid_hierarchy_file = "challenge-2019-label500-hierarchy.json"
unified_space_file = "class_names.txt"
label_mapping_file = "obj_det_mapping_540.csv"

hirarchy_oid_file = "hierarchy_oid.json"
hirarchy_rvc_file = "hierarchy_rvc.json"

C = 540
is_parents = np.zeros((C + 1, C + 1), dtype='float')  # the last row / column is the background
is_childs = np.zeros((C + 1, C + 1), dtype='float')

# merge category 
# egg. {"a" : "A_super"}
# TODO:
# 1. for "a", treat "A_super" and all parents of "A_super" as parents, treat all childs of "A_super" as childs
# 2. for "A_super", treat "a" as a parent
# 3. for parents of "A_super", treat "a" as a child
# 4. for childs of "A_super", treat "a" as a parent
merged_categories = {
    'ball': 'ball_super', 
    'bear': 'bear_super',
    'bed': 'bed_super',
    'bird': 'bird_super',
    'boat': 'boat_super',
    'car': 'car_super',
    'clock': 'clock_super',
    'person': 'person_super',
}

# insert category 
# egg. {"a" : "A_super"}
# TODO:
# 1. for "a", treat "A_super" and all parents of "A_super" as parents
# 2. for "A_super", treat "a" as a child
# 3. for parents of "A_super", treat "a" as a child
inserted_categories = {
    'caravan': 'land_vehicle_super',
    'cow': 'animal_super',
    'ground_animal': 'animal_super',
    'handcart': 'land_vehicle_super',
    'land_vehicle': 'land_vehicle_super',
    'traffic_sign_backside': 'traffic_sign_super',
    'traffic_sign_frame': 'traffic_sign_super',
    'traffic_sign_front': 'traffic_sign_super',
    'trailer': 'land_vehicle_super',
}


def find_all_parents(hierarchy, label, parents):
    if hierarchy['LabelName'] == label:
        return True
    else:
        if hierarchy.get('Subcategory', None) is None:
            return False
        hit = False
        for sub_hierarchy in hierarchy['Subcategory']:
            if find_all_parents(sub_hierarchy, label, parents):
                parents.append(hierarchy['LabelName'])
                hit = True
        return hit


def add_all_child(hierarchy, childs):
    if hierarchy.get('Subcategory', None) is not None:
        for sub_hierarchy in hierarchy['Subcategory']:
            childs.append(sub_hierarchy['LabelName'])
            add_all_child(sub_hierarchy, childs)
    return


def find_all_childs(hierarchy, label, childs):
    if hierarchy['LabelName'] == label:
        add_all_child(hierarchy, childs)
    else:
        if hierarchy.get('Subcategory', None) is not None:
            for sub_hierarchy in hierarchy['Subcategory']:
                find_all_childs(sub_hierarchy, label, childs)
    return True


oid_hierarchy = json.load(open(oid_hierarchy_file, 'r'))
with open(unified_space_file, 'r') as f:
    unified_space_names = f.readlines()[0].split(', ')
with codecs.open(label_mapping_file, encoding='utf-8-sig') as f:
    mapping_labels_csv = []
    for row in csv.DictReader(f, skipinitialspace=True):
        mapping_labels_csv.append(row)
mapping_labels = {
    _label['key']:{
        'oid_boxable_leaf': _label['oid_boxable_leaf'],
        'coco_boxable_name': _label['coco_boxable_name'],
        'mvd_boxable_name': _label['mvd_boxable_name']
    } for _label in mapping_labels_csv}
oid_cat2labels = {
    _label['oid_boxable_leaf']: _label['key']
    for _label in mapping_labels_csv if len(_label['oid_boxable_leaf']) > 0}

# oid hierarchy 
for idx, name in enumerate(unified_space_names):
    
    oid_label = mapping_labels[name]['oid_boxable_leaf']
    coco_label = mapping_labels[name]['coco_boxable_name']
    mvd_label = mapping_labels[name]['mvd_boxable_name']

    parents, childs = [], []
    if len(oid_label) > 0:
        # parents
        assert find_all_parents(oid_hierarchy, oid_label, parents), "something wrong on {} when searching parents".format(oid_label)
        while '/m/0bl9f' in parents:
            parents.remove('/m/0bl9f')
        idx_parents = []
        for parent in parents:
            idx_parents.append(unified_space_names.index(oid_cat2labels[parent]))
        if len(idx_parents) > 0:
            is_parents[idx][idx_parents] = 1

        # childs
        assert find_all_childs(oid_hierarchy, oid_label, childs), "something wrong on {} when searching childs".format(oid_label)
        idx_childs = []
        for child in childs:
            idx_childs.append(unified_space_names.index(oid_cat2labels[child]))
        if len(idx_childs) > 0:
            is_childs[idx][idx_childs] = 1
    # else:
        # super_categories = [_name for _name in unified_space_names if _name[-6:] == '_super']
        # print("category: {}, \n coco label: {}, \n mvd label: {}".format(name, coco_label, mvd_label))
        # print(super_categories)
        # pdb.set_trace()

hirarchy_oid = {'is_parents': is_parents.tolist(), 'is_childs': is_childs.tolist()}
with open(hirarchy_oid_file, 'w') as f:
    json.dump(hirarchy_oid, f)

# merge category 
for label_src, label_target in merged_categories.items():

    # step 1
    parents = []
    oid_label = mapping_labels[label_target]['oid_boxable_leaf']
    assert find_all_parents(oid_hierarchy, oid_label, parents), "something wrong on {} when searching parents".format(oid_label)
    while '/m/0bl9f' in parents:
        parents.remove('/m/0bl9f')
    idx_parents = []
    for parent in parents:
        idx_parents.append(unified_space_names.index(oid_cat2labels[parent]))   
    idx = unified_space_names.index(label_src)
    if len(idx_parents) > 0:
        is_parents[idx][idx_parents] = 1
    is_parents[idx][unified_space_names.index(label_target)] = 1

    childs = []
    oid_label = mapping_labels[label_target]['oid_boxable_leaf']
    assert find_all_childs(oid_hierarchy, oid_label, childs), "something wrong on {} when searching childs".format(oid_label)
    idx_childs = []
    for child in childs:
        idx_childs.append(unified_space_names.index(oid_cat2labels[child]))
    if len(idx_childs) > 0:
        idx = unified_space_names.index(label_src)
        is_childs[idx][idx_childs] = 1

    # step 2
    idx = unified_space_names.index(label_target)
    is_parents[idx][unified_space_names.index(label_src)] = 1

    # step 3
    for parent in parents:
        idx_p = unified_space_names.index(oid_cat2labels[parent])
        idx = unified_space_names.index(label_src)
        is_childs[idx_p][idx] = 1

    # step 4
    for child in childs:
        idx_c = unified_space_names.index(oid_cat2labels[child])
        idx = unified_space_names.index(label_src)
        is_parents[idx_c][idx] = 1

# insert category 
for label_src, label_target in inserted_categories.items():

    # step 1
    parents = []
    oid_label = mapping_labels[label_target]['oid_boxable_leaf']
    assert find_all_parents(oid_hierarchy, oid_label, parents), "something wrong on {} when searching parents".format(oid_label)
    while '/m/0bl9f' in parents:
        parents.remove('/m/0bl9f')
    idx_parents = []
    for parent in parents:
        idx_parents.append(unified_space_names.index(oid_cat2labels[parent]))
    idx = unified_space_names.index(label_src)
    if len(idx_parents) > 0:
        is_parents[idx][idx_parents] = 1
    is_parents[idx][unified_space_names.index(label_target)] = 1

    # step 2
    idx = unified_space_names.index(label_target)
    is_childs[idx][unified_space_names.index(label_src)] = 1

    # step 3
    for parent in parents:
        idx_p = unified_space_names.index(oid_cat2labels[parent])
        idx = unified_space_names.index(label_src)
        is_childs[idx_p][idx] = 1

hirarchy_rvc = {'is_parents': is_parents.tolist(), 'is_childs': is_childs.tolist()}
with open(hirarchy_rvc_file, 'w') as f:
    json.dump(hirarchy_rvc, f)


# pdb.set_trace()
print("Succeed!")