# Group: EVA6 - Group 3
1. Muhsin Abdul Mohammed - muhsin@omnilytics.co 
2. Nilanjana Dev Nath - nilanjana.dvnath@gmail.com
3. Pramod Ramachandra Bhagwat - pramod@mistralsolutions.com
4. Udaya Kumar NAndhanuru - udaya.k@mistralsolutions.com
------

## Train / Test model
The model is trained using a training flow we created which can be found in this [repo](https://github.com/askmuhsin/eva_training_flow).   
In order to train locally you can either run the the `train_model.ipynb` notebook or do the following steps.   
1. clone the training flow rep -- 
```bash
git clone https://github.com/askmuhsin/eva_training_flow
```
2. Then run this script -- 
```python
from models import resnet_v2_6ch_ending     ## import model
from main import Trainer
from main import show_misclassification     ## utility to show misclassifications and gradcam

trainer = Trainer(
    resnet_v2_6ch_ending.ResNet18(),
    # model_path='../data/model_state/R18_6_channel_with_augmentation_3_repeat.pt',     ## model path is optional, if required to resume training
)
trainer.train_model(epochs=40)  ## to start training
show_misclassification(trainer) ## to view misclassifications
```

## Gradcam Visuals (Incorrect Classifications)
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_1.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_2.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_3.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_4.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_5.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_6.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_7.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_8.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_mistakes_9.png" width="400"/>


## Gradcam Visuals (Correct Classifications)
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_correct_1.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_correct_2.png" width="400"/>
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/sample_correct_3.png" width="400"/>


## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 28, 28]           1,728
       BatchNorm2d-2           [-1, 64, 28, 28]             128
            Conv2d-3           [-1, 64, 28, 28]          36,864
       BatchNorm2d-4           [-1, 64, 28, 28]             128
            Conv2d-5           [-1, 64, 28, 28]          36,864
       BatchNorm2d-6           [-1, 64, 28, 28]             128
        BasicBlock-7           [-1, 64, 28, 28]               0
            Conv2d-8           [-1, 64, 28, 28]          36,864
       BatchNorm2d-9           [-1, 64, 28, 28]             128
           Conv2d-10           [-1, 64, 28, 28]          36,864
      BatchNorm2d-11           [-1, 64, 28, 28]             128
       BasicBlock-12           [-1, 64, 28, 28]               0
           Conv2d-13          [-1, 128, 15, 15]          73,728
      BatchNorm2d-14          [-1, 128, 15, 15]             256
           Conv2d-15          [-1, 128, 15, 15]         147,456
      BatchNorm2d-16          [-1, 128, 15, 15]             256
           Conv2d-17          [-1, 128, 15, 15]           8,192
      BatchNorm2d-18          [-1, 128, 15, 15]             256
       BasicBlock-19          [-1, 128, 15, 15]               0
           Conv2d-20          [-1, 128, 15, 15]         147,456
      BatchNorm2d-21          [-1, 128, 15, 15]             256
           Conv2d-22          [-1, 128, 15, 15]         147,456
      BatchNorm2d-23          [-1, 128, 15, 15]             256
       BasicBlock-24          [-1, 128, 15, 15]               0
           Conv2d-25            [-1, 256, 9, 9]         294,912
      BatchNorm2d-26            [-1, 256, 9, 9]             512
           Conv2d-27            [-1, 256, 9, 9]         589,824
      BatchNorm2d-28            [-1, 256, 9, 9]             512
           Conv2d-29            [-1, 256, 9, 9]          32,768
      BatchNorm2d-30            [-1, 256, 9, 9]             512
       BasicBlock-31            [-1, 256, 9, 9]               0
           Conv2d-32            [-1, 256, 9, 9]         589,824
      BatchNorm2d-33            [-1, 256, 9, 9]             512
           Conv2d-34            [-1, 256, 9, 9]         589,824
      BatchNorm2d-35            [-1, 256, 9, 9]             512
       BasicBlock-36            [-1, 256, 9, 9]               0
           Conv2d-37            [-1, 512, 6, 6]       1,179,648
      BatchNorm2d-38            [-1, 512, 6, 6]           1,024
           Conv2d-39            [-1, 512, 6, 6]       2,359,296
      BatchNorm2d-40            [-1, 512, 6, 6]           1,024
           Conv2d-41            [-1, 512, 6, 6]         131,072
      BatchNorm2d-42            [-1, 512, 6, 6]           1,024
       BasicBlock-43            [-1, 512, 6, 6]               0
           Conv2d-44            [-1, 512, 6, 6]       2,359,296
      BatchNorm2d-45            [-1, 512, 6, 6]           1,024
           Conv2d-46            [-1, 512, 6, 6]       2,359,296
      BatchNorm2d-47            [-1, 512, 6, 6]           1,024
       BasicBlock-48            [-1, 512, 6, 6]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 10.82
Params size (MB): 42.63
Estimated Total Size (MB): 53.45
----------------------------------------------------------------
```

## Training Log
```bash
TRAIN Epoch:0 Loss:1.3217 Batch:390 Acc:47.13: 100%|██████████| 391/391 [00:34<00:00, 11.27it/s]
TEST         Loss:0.0093         Acc:59.22         [5922 / 10000]
TRAIN Epoch:1 Loss:0.6765 Batch:390 Acc:63.89: 100%|██████████| 391/391 [00:35<00:00, 10.96it/s]
TEST         Loss:0.0068         Acc:70.13         [7013 / 10000]
TRAIN Epoch:2 Loss:0.9805 Batch:390 Acc:70.92: 100%|██████████| 391/391 [00:35<00:00, 10.93it/s]
TEST         Loss:0.0076         Acc:69.15         [6915 / 10000]
TRAIN Epoch:3 Loss:0.8344 Batch:390 Acc:75.32: 100%|██████████| 391/391 [00:36<00:00, 10.84it/s]
TEST         Loss:0.0054         Acc:75.73         [7573 / 10000]
TRAIN Epoch:4 Loss:0.658 Batch:390 Acc:78.30: 100%|██████████| 391/391 [00:35<00:00, 10.89it/s] 
TEST         Loss:0.0102         Acc:62.23         [6223 / 10000]
TRAIN Epoch:5 Loss:0.5953 Batch:390 Acc:81.01: 100%|██████████| 391/391 [00:35<00:00, 10.91it/s]
TEST         Loss:0.0050         Acc:79.70         [7970 / 10000]
TRAIN Epoch:6 Loss:0.491 Batch:390 Acc:82.72: 100%|██████████| 391/391 [00:35<00:00, 10.94it/s] 
TEST         Loss:0.0048         Acc:80.07         [8007 / 10000]
TRAIN Epoch:7 Loss:0.617 Batch:390 Acc:84.36: 100%|██████████| 391/391 [00:36<00:00, 10.69it/s] 
TEST         Loss:0.0062         Acc:77.50         [7750 / 10000]
TRAIN Epoch:8 Loss:0.4774 Batch:390 Acc:86.09: 100%|██████████| 391/391 [00:36<00:00, 10.75it/s]
TEST         Loss:0.0058         Acc:78.66         [7866 / 10000]
TRAIN Epoch:9 Loss:0.2462 Batch:390 Acc:87.02: 100%|██████████| 391/391 [00:35<00:00, 10.88it/s]
TEST         Loss:0.0047         Acc:82.97         [8297 / 10000]
TRAIN Epoch:10 Loss:0.3827 Batch:390 Acc:88.19: 100%|██████████| 391/391 [00:35<00:00, 10.89it/s]
TEST         Loss:0.0054         Acc:81.12         [8112 / 10000]
TRAIN Epoch:11 Loss:0.2179 Batch:390 Acc:88.80: 100%|██████████| 391/391 [00:36<00:00, 10.79it/s]
TEST         Loss:0.0123         Acc:67.71         [6771 / 10000]
TRAIN Epoch:12 Loss:0.2893 Batch:390 Acc:89.46: 100%|██████████| 391/391 [00:35<00:00, 10.87it/s]
TEST         Loss:0.0054         Acc:81.40         [8140 / 10000]
TRAIN Epoch:13 Loss:0.2886 Batch:390 Acc:90.03: 100%|██████████| 391/391 [00:35<00:00, 10.87it/s]
TEST         Loss:0.0055         Acc:82.59         [8259 / 10000]
TRAIN Epoch:14 Loss:0.1843 Batch:390 Acc:90.47: 100%|██████████| 391/391 [00:35<00:00, 11.04it/s]
TEST         Loss:0.0061         Acc:79.81         [7981 / 10000]
TRAIN Epoch:15 Loss:0.2012 Batch:390 Acc:90.92: 100%|██████████| 391/391 [00:36<00:00, 10.82it/s]
TEST         Loss:0.0052         Acc:81.45         [8145 / 10000]
TRAIN Epoch:16 Loss:0.2135 Batch:390 Acc:91.18: 100%|██████████| 391/391 [00:35<00:00, 10.88it/s]
TEST         Loss:0.0050         Acc:83.59         [8359 / 10000]
TRAIN Epoch:17 Loss:0.2901 Batch:390 Acc:91.44: 100%|██████████| 391/391 [00:35<00:00, 11.07it/s]
TEST         Loss:0.0050         Acc:82.83         [8283 / 10000]
TRAIN Epoch:18 Loss:0.3035 Batch:390 Acc:91.48: 100%|██████████| 391/391 [00:35<00:00, 10.98it/s]
TEST         Loss:0.0053         Acc:82.93         [8293 / 10000]
TRAIN Epoch:19 Loss:0.2338 Batch:390 Acc:91.95: 100%|██████████| 391/391 [00:35<00:00, 11.00it/s]
TEST         Loss:0.0043         Acc:85.76         [8576 / 10000]
TRAIN Epoch:20 Loss:0.2566 Batch:390 Acc:91.98: 100%|██████████| 391/391 [00:35<00:00, 10.97it/s]
TEST         Loss:0.0047         Acc:83.72         [8372 / 10000]
TRAIN Epoch:21 Loss:0.1864 Batch:390 Acc:92.23: 100%|██████████| 391/391 [00:36<00:00, 10.82it/s]
TEST         Loss:0.0066         Acc:80.08         [8008 / 10000]
TRAIN Epoch:22 Loss:0.2019 Batch:390 Acc:92.43: 100%|██████████| 391/391 [00:35<00:00, 10.96it/s]
TEST         Loss:0.0052         Acc:83.79         [8379 / 10000]
TRAIN Epoch:23 Loss:0.2879 Batch:390 Acc:92.51: 100%|██████████| 391/391 [00:35<00:00, 11.07it/s]
TEST         Loss:0.0056         Acc:82.99         [8299 / 10000]
TRAIN Epoch:24 Loss:0.1885 Batch:390 Acc:92.62: 100%|██████████| 391/391 [00:35<00:00, 10.95it/s]
TEST         Loss:0.0058         Acc:82.47         [8247 / 10000]
TRAIN Epoch:25 Loss:0.1519 Batch:390 Acc:92.86: 100%|██████████| 391/391 [00:35<00:00, 10.96it/s]
TEST         Loss:0.0064         Acc:81.82         [8182 / 10000]
TRAIN Epoch:26 Loss:0.1788 Batch:390 Acc:92.78: 100%|██████████| 391/391 [00:35<00:00, 10.93it/s]
TEST         Loss:0.0049         Acc:83.96         [8396 / 10000]
TRAIN Epoch:27 Loss:0.1501 Batch:390 Acc:93.29: 100%|██████████| 391/391 [00:35<00:00, 10.98it/s]
TEST         Loss:0.0060         Acc:81.12         [8112 / 10000]
TRAIN Epoch:28 Loss:0.269 Batch:390 Acc:92.83: 100%|██████████| 391/391 [00:35<00:00, 10.96it/s] 
TEST         Loss:0.0056         Acc:82.95         [8295 / 10000]
TRAIN Epoch:29 Loss:0.1782 Batch:390 Acc:93.16: 100%|██████████| 391/391 [00:36<00:00, 10.84it/s]
TEST         Loss:0.0043         Acc:85.27         [8527 / 10000]
TRAIN Epoch:30 Loss:0.159 Batch:390 Acc:93.12: 100%|██████████| 391/391 [00:35<00:00, 11.01it/s] 
TEST         Loss:0.0042         Acc:85.26         [8526 / 10000]
TRAIN Epoch:31 Loss:0.1486 Batch:390 Acc:93.46: 100%|██████████| 391/391 [00:35<00:00, 11.05it/s]
TEST         Loss:0.0048         Acc:84.35         [8435 / 10000]
TRAIN Epoch:32 Loss:0.1802 Batch:390 Acc:93.47: 100%|██████████| 391/391 [00:35<00:00, 10.98it/s]
TEST         Loss:0.0061         Acc:82.18         [8218 / 10000]
TRAIN Epoch:33 Loss:0.1526 Batch:390 Acc:93.59: 100%|██████████| 391/391 [00:35<00:00, 11.03it/s]
TEST         Loss:0.0059         Acc:82.84         [8284 / 10000]
TRAIN Epoch:34 Loss:0.0928 Batch:390 Acc:93.36: 100%|██████████| 391/391 [00:35<00:00, 10.87it/s]
TEST         Loss:0.0054         Acc:83.47         [8347 / 10000]
TRAIN Epoch:35 Loss:0.1565 Batch:390 Acc:93.80: 100%|██████████| 391/391 [00:35<00:00, 10.98it/s]
TEST         Loss:0.0047         Acc:84.67         [8467 / 10000]
TRAIN Epoch:36 Loss:0.2227 Batch:390 Acc:93.70: 100%|██████████| 391/391 [00:35<00:00, 10.87it/s]
TEST         Loss:0.0045         Acc:85.28         [8528 / 10000]
TRAIN Epoch:37 Loss:0.3468 Batch:390 Acc:93.88: 100%|██████████| 391/391 [00:36<00:00, 10.79it/s]
TEST         Loss:0.0052         Acc:84.28         [8428 / 10000]
TRAIN Epoch:38 Loss:0.2565 Batch:390 Acc:93.74: 100%|██████████| 391/391 [00:36<00:00, 10.73it/s]
TEST         Loss:0.0051         Acc:83.79         [8379 / 10000]
TRAIN Epoch:39 Loss:0.2324 Batch:390 Acc:93.95: 100%|██████████| 391/391 [00:36<00:00, 10.59it/s]
TEST         Loss:0.0044         Acc:84.88         [8488 / 10000]
```

## Augmentations
we applied the folowing augmentations --
```python
A.Compose([
    A.Normalize(
        mean=self.mean, 
        std=self.std,
        always_apply=True
    ),
    A.RandomCrop(32, 32, always_apply=False, p=0.5),
    A.CoarseDropout(
        max_holes=3, max_height=16, max_width=16, min_holes=None, min_height=None, min_width=None, 
        fill_value=(0.491, 0.482, 0.447), mask_fill_value=None, always_apply=False, p=0.25
    ),
    A.Rotate(limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    ToTensorV2()
])
```
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/augmentations.png" width="400"/>


## Model Experiments
We tried out several models before finalizing on this one.   <br>
We initially started out with a vanially Resnet which had 3 blocks and ended with a linear layer. Since we wanted to visualize using gradcam the last channel of the vanilla R18 network was not suitable it was 4X4. So we increased padding to obtain a 6X6 channel in the last conv layer. We also tried removing the linear layer and using only GAP. Here are the results for all model runs.

- Final Model log (one of the runs)
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/final_model.png" width="700"/>

- Comparison of all trained models
<img src="https://github.com/askmuhsin/eva_experiments/blob/main/S8_resnet_18_gradcam/resources/model_comparisons.png" width="700"/>

_Note: we used wandb for logging experiments, but because of the credentials and other dependencies it is not included in the nb with the model training here._

