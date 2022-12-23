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
Use the [devkits](https://github.com/ozendelait/rvc_devkit) to prepare three dataset: MS COCO, OpenImages, and Mapillary Vistas.

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
