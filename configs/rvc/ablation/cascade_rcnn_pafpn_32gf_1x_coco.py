_base_ = ['../_base_/default_runtime.py']

# model settings
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='RegNet32gf',
        freeze_at=5,
        pretrain="checkpoints/mmdet_SEER-regnet32gf.pth"),
    neck=dict(
        type='PAFPN',
        in_channels=[232, 696, 1392, 3712],
        out_channels=256,
        start_level=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        sigmoid_cls=True,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=540,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='HierarchyLoss',
                    pos_parents="oid",
                    ignore="child+",
                    hierarchy_oid_file="./label_spaces/hierarchy_oid.json",
                    hierarchy_rvc_file="./label_spaces/hierarchy_rvc.json",
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=540,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='HierarchyLoss',
                    pos_parents="oid",
                    ignore="child+",
                    hierarchy_oid_file="./label_spaces/hierarchy_oid.json",
                    hierarchy_rvc_file="./label_spaces/hierarchy_rvc.json",
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=540,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='HierarchyLoss',
                    pos_parents="oid",
                    ignore="child+",
                    hierarchy_oid_file="./label_spaces/hierarchy_oid.json",
                    hierarchy_rvc_file="./label_spaces/hierarchy_rvc.json",
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
        img_scale=[(1600, 480), (1600, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

CLASSES = ['accordion', 'adhesive_tape', 'aircraft_super', 'airplane', 'alarm_clock', 'alpaca', 'ambulance', 'american_football', 'animal_super', 'ant', 'antelope', 'apple', 'arm', 'artichoke', 'backpack', 'bagel', 'ball', 'ball_super', 'balloon', 'banana', 'banner', 'barge', 'barrel', 'baseball_bat', 'baseball_glove', 'bat', 'bathroom_cabinet', 'bathtub', 'beaker', 'bear', 'bear_super', 'beard', 'bed', 'bed_super', 'bee', 'beehive', 'beer', 'beetle_super', 'bell_pepper', 'belt', 'bench', 'bicycle', 'bicycle_helmet', 'bicycle_wheel', 'bicyclist', 'bidet', 'bike_rack', 'billboard', 'billiard_table', 'binoculars', 'bird', 'bird_super', 'blender', 'blue_jay', 'boat', 'boat_super', 'book', 'bookcase', 'boot', 'bottle', 'bow_and_arrow', 'bowl', 'box', 'boy', 'brassiere', 'bread', 'briefcase', 'broccoli', 'bronze_sculpture', 'brown_bear', 'building_super', 'bull', 'burrito', 'bus', 'bust', 'butterfly', 'cabbage', 'cabinetry', 'cake', 'cake_stand', 'camel', 'camera', 'canary', 'candle', 'candy', 'cannon', 'canoe', 'car', 'car_super', 'caravan', 'carnivore_super', 'carrot', 'cart', 'castle', 'cat', 'catch_basin', 'caterpillar', 'cattle', 'ceiling_fan', 'cello', 'centipede', 'chair', 'cheetah', 'chest_of_drawers', 'chicken', 'chopstick', 'christmas_tree', 'clock', 'clock_super', 'coat', 'cocktail', 'coconut', 'coffee', 'coffee_cup', 'coffee_table', 'coffeemaker', 'coin', 'computer_keyboard', 'computer_monitor', 'computer_mouse', 'convenience_store', 'cookie', 'corded_phone', 'couch', 'couch_super', 'countertop', 'cow', 'cowboy_hat', 'crab', 'cricket_ball', 'crocodile', 'croissant', 'crown', 'crutch', 'cucumber', 'cup', 'cupboard', 'curtain', 'cutting_board', 'dagger', 'deer', 'desk', 'dessert_super', 'dice', 'digital_clock', 'dining_table', 'dinosaur', 'dog', 'dog_bed', 'doll', 'dolphin', 'donut', 'door', 'door_handle', 'dragonfly', 'drawer', 'dress', 'drink_super', 'drinking_straw', 'drum', 'duck', 'dumbbell', 'eagle', 'ear', 'earring', 'egg', 'electric_fan', 'elephant', 'envelope', 'eye', 'eyeglasses', 'face', 'falcon', 'fedora', 'fig', 'filing_cabinet', 'fire_hydrant', 'fireplace', 'fish_super', 'flag', 'flashlight', 'flower_super', 'flowerpot', 'flute', 'food_processor', 'foot', 'football_helmet', 'footwear_super', 'fork', 'fountain', 'fox', 'french_fries', 'french_horn', 'frisbee', 'frog', 'fruit_super', 'frying_pan', 'furniture_super', 'garden_asparagus', 'gas_stove', 'giraffe', 'girl', 'glove_super', 'goat', 'goggles', 'goldfish', 'golf_ball', 'golf_cart', 'gondola', 'goose', 'grape', 'grapefruit', 'ground_animal', 'guacamole', 'guitar', 'hair', 'hair_dryer', 'hamburger', 'hamster', 'hand', 'handbag', 'handcart', 'handgun', 'harbor_seal', 'harp', 'harpsichord', 'hat_super', 'head', 'headphone', 'helicopter', 'helmet_super', 'high_heels', 'home_appliance_super', 'honeycomb', 'horse', 'hot_dog', 'house', 'ice_cream', 'infant_bed', 'insect_super', 'invertebrate_super', 'jacket', 'jaguar', 'jeans', 'jellyfish', 'jet_ski', 'jug', 'juice', 'junction_box', 'kangaroo', 'kettle', 'kitchen_and_dining_room_table', 'kitchen_appliance_super', 'kitchen_knife', 'kite', 'knife', 'ladder', 'ladybug', 'lamp', 'land_vehicle', 'land_vehicle_super', 'lantern', 'laptop', 'lavender', 'leg', 'lemon', 'leopard', 'license_plate', 'light_bulb', 'light_switch', 'lighthouse', 'lily', 'limousine', 'lion', 'lizard', 'lobster', 'luggage_and_bags_super', 'lynx', 'mailbox', 'man', 'mango', 'manhole', 'maple', 'marine_invertebrates_super', 'marine_mammal_super', 'measuring_cup', 'microphone', 'microwave', 'miniskirt', 'mirror', 'missile', 'mixer', 'mobile_phone', 'monkey', 'moths_and_butterflies_super', 'motorcycle', 'motorcyclist', 'mouse', 'mouth', 'muffin', 'mug', 'mule', 'mushroom', 'musical_instrument_super', 'musical_keyboard', 'nail', 'necklace', 'nightstand', 'nose', 'oboe', 'office_building', 'office_supplies_super', 'orange', 'organ', 'ostrich', 'otter', 'oven', 'owl', 'oyster', 'paddle', 'palm_tree', 'pancake', 'paper_towel', 'parachute', 'parking_meter', 'parrot', 'pasta', 'peach', 'pear', 'pen', 'penguin', 'person', 'person_super', 'personal_care_super', 'personal_flotation_device', 'phone_booth', 'piano', 'picnic_basket', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pitcher', 'pizza', 'plain_crosswalk', 'plastic_bag', 'plate', 'platter', 'plumbing_fixture_super', 'polar_bear', 'pole', 'pomegranate', 'popcorn', 'porch', 'porcupine', 'poster', 'potato', 'potted_plant', 'power_plugs_and_sockets', 'pressure_cooker', 'pretzel', 'printer', 'pumpkin', 'punching_bag', 'rabbit', 'raccoon', 'racket_super', 'radish', 'raven', 'refrigerator', 'remote', 'reptile_super', 'rhinoceros', 'rider', 'rifle', 'ring_binder', 'rocket', 'roller_skates', 'rose', 'rugby_ball', 'ruler', 'salad', 'salt_and_pepper_shakers', 'sandal', 'sandwich', 'sandwich_super', 'saucer', 'saxophone', 'scarf', 'scissors', 'scoreboard', 'screwdriver', 'sculpture_super', 'sea_lion', 'sea_turtle', 'seafood_super', 'seahorse', 'seat_belt', 'segway', 'serving_tray', 'sewing_machine', 'shark', 'sheep', 'shelf', 'shellfish_super', 'shirt', 'shorts', 'shotgun', 'showerhead', 'shrimp', 'sink', 'skateboard', 'ski', 'skirt_super', 'skull', 'skyscraper', 'slow_cooker', 'snail', 'snake', 'snowboard', 'snowman', 'snowmobile', 'snowplow', 'sock', 'sofa_bed', 'sombrero', 'sparrow', 'spatula', 'spider', 'spoon', 'sports_uniform', 'squash_super', 'squirrel', 'stairs', 'starfish', 'stationary_bicycle', 'stool', 'stop_sign', 'strawberry', 'streetlight', 'stretcher', 'striped_crosswalk', 'studio_couch', 'submarine_sandwich', 'suit', 'suitcase', 'sun_hat', 'sunflower', 'sunglasses', 'surfboard', 'surveillance_camera', 'sushi', 'swan', 'swim_cap', 'swimming_pool', 'swimwear', 'sword', 'table_super', 'table_tennis_racket', 'tablet_computer', 'tableware_super', 'taco', 'tank', 'tap', 'tart', 'taxi', 'tea', 'teapot', 'teddy_bear', 'telephone_super', 'television_set', 'tennis_ball', 'tennis_racket', 'tent', 'tiara', 'tick', 'tie', 'tiger', 'tin_can', 'tire', 'toaster', 'toilet', 'toilet_paper', 'tomato', 'toothbrush', 'torch', 'tortoise', 'towel', 'tower', 'toy_super', 'traffic_light', 'traffic_sign_backside', 'traffic_sign_frame', 'traffic_sign_front', 'traffic_sign_super', 'trailer', 'train', 'training_bench', 'treadmill', 'tree_super', 'tripod', 'trombone', 'trousers_super', 'truck', 'trumpet', 'turkey', 'turtle_super', 'umbrella', 'utility_pole', 'van', 'vase', 'vegetable_super', 'vehicle_super', 'violin', 'volleyball', 'waffle', 'wall_clock', 'washing_machine', 'waste_container', 'watch', 'watercraft_super', 'watermelon', 'weapon_super', 'whale', 'wheel', 'wheelchair', 'whiteboard', 'willow', 'window', 'window_blind', 'wine', 'wine_glass', 'winter_melon', 'wok', 'woman', 'wood-burning_stove', 'woodpecker', 'wrench', 'zebra', 'zucchini']

coco = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='CocoDataset',
        ann_file='./data/rvc/annotations/coco_boxable.rvc_train.json',
        img_prefix='./data/rvc/annotations/',
        classes=CLASSES,
        pipeline=train_pipeline))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    # train_dataloader=dict(class_aware_sampler=dict(num_sample_class=1)),
    train=coco,
    val=dict(
        type='CocoDataset',
        ann_file= './data/rvc/annotations/coco_boxable.rvc_val.json',
        img_prefix='./data/rvc/annotations/',
        classes=CLASSES,
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file= './data/rvc/annotations/coco_boxable.rvc_val.json',
        img_prefix='./data/rvc/annotations/',
        classes=CLASSES,
        pipeline=test_pipeline))
evaluation = dict(interval=1, save_best='auto', metric='bbox')

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=0.00005,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, max_keep_ckpts=20, save_last=True)
fp16 = dict(loss_scale=512.)

load_from = None