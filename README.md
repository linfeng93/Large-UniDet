# Large-UniDet
A practice for million-scale multi-domain universal object detection.   
The 2nd place (**IFFF_RVC**) in ECCV 2022 [Robust Vision Challenge (RVC 2022)](http://www.robustvision.net/leaderboard.php?benchmark=object).

<p align="center"> <img src='docs/visualization.jpg' align="center" height="225px"> </p>

> [**Million-scale Object Detection with Large Vision Model**](https://arxiv.org/abs/2212.09408),   
> Feng Lin, Wenze Hu, Yaowei Wang, Yonghong Tian, Guangming Lu, Fanglin Chen, Yong Xu, Xiaoyu Wang

Contact: [lin1993@mail.ustc.edu.cn](mailto:lin1993@mail.ustc.edu.cn). Feel free to contact me if you have any question!

## Overview
<p align="left"> <img src='docs/top.jpg' align="center" height="600px"> </p>

## Installation

Our project works with [mmdetection](https://github.com/open-mmlab/mmdetection) **v2.25.1** or higher. Please refer to [Installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation) for installation instructions.

We build the experimental environment via [docker](https://docs.docker.com/engine). We also provide [DockerImage]() (fetch code: **1024**).

## Data Preparation

- **Datasets**

  Use the [devkits](https://github.com/ozendelait/rvc_devkit) to prepare three dataset: MS COCO, OpenImages, and Mapillary Vistas.
  
  Link the processed datasets to ```$ROOT/data/rvc```

- **Large Vision Models**

  Employ the [SEER](https://github.com/facebookresearch/vissl/tree/main/projects/SEER) models ([RegNet32gf](https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch) and [RegNet256gf](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_cosine_rg256gf_noBNhead_wd1e5_fairstore_bs16_node64_sinkhorn10_proto16k_apex_syncBN64_warmup8k/model_final_checkpoint_phase0.torch)) as the *frozen* backbones. 
  ```
  # convert the downloaded checkpoints
  seer = torch.load("seer_model_name.torch", map_location="cpu")
  state_dict = {"state_dict": seer["classy_state_dict"]["base_model"]["model"]["trunk"]}
  torch.save(state_dict, FILENAME)
  ```
  Save the processed checkpoints to ```$ROOT/checkpoints```
  
- **Label Hierarchy Files**
  
  Run the scripts to generate the label hierarchy files for *HierarchyLoss*.
  ```
  # generate the label hierarchy files
  cd ./label_spaces
  python gen_hierarchy.py
  ```
## Training

- To train the SEER-RegNet32gf-based Large-UniDet on 8 GPUs, use the following command line

  ```
  bash tool/dist_train.sh configs/rvc/cascade_rcnn_nasfpn_crpn_32gf_1x_rvc.py 8
  ```
- To train the SEER-RegNet256gf-based Large-UniDet on 8 GPUs, use the following command line

  ```
  bash tool/dist_train.sh configs/rvc/cascade_rcnn_nasfpn_crpn_256gf_1x_rvc.py 8
  ```
- To conduct the dataset-specific finetuning on MS COCO, use the following command line

  ```
  bash tool/dist_train.sh configs/rvc/finetune/cascade_rcnn_nasfpn_crpn_32gf_2x_coco.py 8
  ```
  
## Evaluation
- To perform evaluations on MS COCO *val* set, run
  ```
  bash tool/dist_test.sh configs/rvc/finetune/cascade_rcnn_nasfpn_crpn_32gf_2x_coco.py CHECKPOINT 8 --eval bbox
  ```
  
- To perform evaluations on Mapillary Vistas *val* set, run
  ```
  bash tool/dist_test.sh configs/rvc/finetune/cascade_rcnn_nasfpn_crpn_32gf_2x_mvd.py CHECKPOINT 8 --eval bbox
  ```
  
- To perform evaluations on OpenImages *val* set, run
  ```
  bash tool/dist_test.sh configs/rvc/finetune/cascade_rcnn_nasfpn_crpn_32gf_0.5x_oid.py CHECKPOINT 8 --format-only --eval-options jsonfile_prefix=$ROOT/results/oid
  python rvc_devkit/common/map_coco_back.py --predictions ...
  ```
  
## License

See the [LICENSE](LICENSE) file for more details.

## Citation

If you find this project useful in your research, please cite:

    @article{lin2022million,
      title={Million-scale Object Detection with Large Vision Model},
      author={Lin, Feng and Hu, Wenze and Wang, Yaowei and Tian, Yonghong and Lu, Guangming and Chen, Fanglin and Xu, Yong and Wang, Xiaoyu},
      journal={arXiv preprint arXiv:2212.09408},
      year={2022}
    }
