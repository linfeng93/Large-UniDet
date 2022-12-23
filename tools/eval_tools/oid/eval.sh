#!/usr/bin/env bash

INPUT_PREDICTIONS=$1
OUTPUT_FILE=$2

python tools/eval_tools/oid/oid_eval.py \
    --input_annotations_boxes=./data/oid/annotations/challenge-2019-validation-detection-bbox_expanded.csv \
    --input_annotations_labels=./data/oid/annotations/challenge-2019-validation-detection-human-imagelabels_expanded.csv \
    --input_class_labelmap=tools/eval_tools/oid/oid_object_detection_challenge_500_label_map.pbtxt \
    --input_predictions=$INPUT_PREDICTIONS \
    --output_metrics=$OUTPUT_FILE