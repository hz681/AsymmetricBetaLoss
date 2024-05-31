# [KDD-24] Asymmetric Beta Loss for Evidence-Based Safe Semi-Supervised Multi-Label Learning

The implementation for the paper Asymmetric Beta Loss for Evidence-Based Safe Semi-Supervised Multi-Label Learning (KDD 2024).

## Preparing Data

The partition of the OOD dataset in the paper is available at [formatted OOD data](https://drive.google.com/file/d/1ArIEVt-qjr41w5i3FJdjq3ne8S6qI92G/view?usp=sharing). 

The manually selected subset of ImageNet-21K can be downloaded from [selected ImageNet-21K for voc&coco](https://drive.google.com/file/d/13-fVGpQMJsZgo4lK5r3VonEduYSYkdc0/view?usp=sharing) and [selected ImageNet-21K for nus](https://drive.google.com/file/d/18XfXYWvMKLzu0y04kUZEFcycQIgU4SlK/view?usp=sharing).

We can also generate the formatted OOD dataset with different parameters through `./generate_data.py`.

## Training Model

1. Warm up the model with the labeled data.
```
python run_warmup.py --dataset_name coco-imagenet --lb_ratio 0.03 --warmup_epochs 12 --seed 1
```

2. Perform the main training process.
```
python run_abl.py --dataset_name coco-imagenet --lb_ratio 0.03 --epoch 40 --seed 1
```