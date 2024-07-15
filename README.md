# Pytorch-Pointnet-modified-for-4-input-columns 

This repo is modified version of [Pytorch Implementation of PointNet and PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) for 4 input columns in classification training ONLY.

## Update
**2022/09/12:**
Initial upload.

## Install
The code for classification training has been modified and tested on Windows 10, CUDA10.1 (requires NVIDIA GPU and Driver), PyTorch 1.6 and Python 3.7, (Anaconda Navigator 2.2.0).

The original author yanx27 tested the remaining codes on Ubuntu 16.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## Classification
### Data Preparation
The dataset for our use case is already included.

If you wish to use the original dataset:
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Run
You can run different modes with following codes. 
*Note that the smallest point cloud in our dataset consist of 244 points only.*
```shell
## e.g., pointnet_cls
python train_classification.py --model pointnet_cls --log_dir pointnet_cls --num_point 244

## e.g., pointnet2_ssg
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_point 244
```


## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)


## Citation
If you find this repo useful in your research, please consider citing it and our other works:
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
```
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
```
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```
