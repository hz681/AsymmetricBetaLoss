# [KDD-24] Asymmetric Beta Loss for Evidence-Based Safe Semi-Supervised Multi-Label Learning

The implementation for the paper Asymmetric Beta Loss for Evidence-Based Safe Semi-Supervised Multi-Label Learning (KDD 2024).

## Preparing Data

The partition of the OOD dataset in the paper is available at [link]. 

The manually selected subset of ImageNet-21K can be downloaded from [link].

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