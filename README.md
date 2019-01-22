# pytorch-finetuner

## Setup

```
$ git clone https://github.com/knjcode/pytorch-finetuner
$ cd pytorch-finetuner
$ pip install -r requirements.txt
```

## Example usage

### (Prerequisites) Arrange images into their respective directories

A training data directory (`images/train`), validation data directory (`images/valid`), and test data directory (`images/test`) should containing one subdirectory per image class.

For example, arrange training, validation, and test data as follows.

```
images/
    train/
        airplanes/
            airplane001.jpg
            airplane002.jpg
            ...
        watch/
            watch001.jpg
            watch002.jpg
            ...
    valid/
        airplanes/
            airplane101.jpg
            airplane102.jpg
            ...
        watch/
            watch101.jpg
            watch102.jpg
            ...
    test/
        airplanes/
            airplane201.jpg
            airplane202.jpg
            ...
        watch/
            watch201.jpg
            watch202.jpg
            ...
```


### Prepare example images using bundled shell script

Download [Caltech 101] dataset, and split part of it into the `example_images` directory.

```
$ util/caltech101_prepare.sh
```

- `example_images/train` is train set of 60 images for each classes
- `example_images/valid` is validation set of 20 images for each classes
- `example_imags/test` is test set of 20 images for each classes

```
$ util/counter.sh example_images/train
example_images/train contains 10 directories
Faces       60
Leopards    60
Motorbikes  60
airplanes   60
bonsai      60
car_side    60
chandelier  60
hawksbill   60
ketch       60
watch       60
```

With this dataset you can immediately try fine-tuning with pytorch-finetuner.

```
$ ./train.py example_images --model resnet50 --epochs 30 --lr-step-epochs 10,20
```


### train

```
$ ./train.py example_images --epochs 5
```

```
$ ./train.py example_images --epochs 5
Running script with args: Namespace(base_lr=0.0125, batch_size=128, cosine_annealing_eta_min=1e-05, cosine_annealing_mult=2, cosine_annealing_t_max=None, cuda=True, cutout=False, cutout_holes=None, cutout_length=None, data='example_images', disp_batches=1, epochs=3, image_dump=False, input_size=224, jitter_brightness=0.1, jitter_contrast=0.1, jitter_hue=0.05, jitter_saturation=0.1, log_dir='logs', lr_factor=0.1, lr_patience=None, lr_step_epochs=[30, 60, 90], mixup=False, mixup_alpha=None, model='resnet18', model_dir='model', momentum=0.9, no_cuda=False, optimizer='sgd', prefix='20181226184319', random_erasing=False, random_erasing_p=None, random_erasing_r1=None, random_erasing_r2=None, random_erasing_sh=None, random_erasing_sl=None, random_horizontal_flip=0.5, random_resized_crop_ratio=[0.75, 1.3333333333333333], random_resized_crop_scale=[0.08, 1.0], random_rotate_degree=3.0, random_vertical_flip=0.0, resume=None, rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225], ricap=False, ricap_beta=None, ricap_with_line=False, save_best_and_last=False, save_best_only=False, scale_size=256, scratch=False, seed=None, start_epoch=0, warmup_epochs=5, wd=0.0001, workers=22)
scale_size: 256  input_size: 224
rgb_mean: [0.0, 0.0, 0.0]
rgb_std: [1.0, 1.0, 1.0]
number of train dataset: 600
number of validation dataset: 200
number of classes: 10
=> using pre-trained model 'resnet18'
=> using GPU
=> using optimizer: sgd
=> using MultiStepLR scheduler
=> model and logs prefix: 20181226184319
=> log dir: logs
=> model dir: model
=> tensorboardX log dir: logs/20181226184319-tensorboardX
Epoch[0] Batch[1] [128/600 (20%)]       speed: 15.22 samples/sec        accuracy: 0.0781250000  loss: 2.5856597424
Epoch[0] Batch[2] [256/600 (40%)]       speed: 30.00 samples/sec        accuracy: 0.0820312500  loss: 2.5729613304
Epoch[0] Batch[3] [384/600 (60%)]       speed: 44.36 samples/sec        accuracy: 0.0885416642  loss: 2.5417649746
Epoch[0] Batch[4] [512/600 (80%)]       speed: 58.31 samples/sec        accuracy: 0.0996093750  loss: 2.5194582939
Epoch[0] Batch[5] [600/600 (100%)]      speed: 60.46 samples/sec        accuracy: 0.1183238626  loss: 2.4788246155
Epoch[0] Train-accuracy: 0.11832386255264282
Epoch[0] Train-loss: 2.4788246154785156
Epoch[0] learning-rate: 0.0020833333333333333
Epoch[0] Validation-accuracy: 0.2777777910232544
Epoch[0] Validation-loss: 2.0763139724731445
Epoch[0] Time cost: 13.761763095855713 [sec]
=> Saved checkpoint to "model/20181226184319-resnet18-0001.model"
=> Saved checkpoint to "model/20181226184319-resnet18-best.model"
Epoch[1] Batch[1] [128/600 (20%)]       speed: 42.31 samples/sec        accuracy: 0.2578125000  loss: 2.2109849453
Epoch[1] Batch[2] [256/600 (40%)]       speed: 81.27 samples/sec        accuracy: 0.3007812500  loss: 2.1127164364
...
(snip)
...
Epoch[4] Batch[1] [128/600 (20%)]       speed: 42.00 samples/sec        accuracy: 0.9531250000  loss: 0.2027403414
Epoch[4] Batch[2] [256/600 (40%)]       speed: 80.71 samples/sec        accuracy: 0.9648437500  loss: 0.1552993804
Epoch[4] Batch[3] [384/600 (60%)]       speed: 116.50 samples/sec       accuracy: 0.9661458135  loss: 0.1483899951
Epoch[4] Batch[4] [512/600 (80%)]       speed: 149.72 samples/sec       accuracy: 0.9726562500  loss: 0.1362725645
Epoch[4] Batch[5] [600/600 (100%)]      speed: 170.88 samples/sec       accuracy: 0.9690340757  loss: 0.1381170452
Epoch[4] Train-accuracy: 0.9690340757369995
Epoch[4] Train-loss: 0.13811704516410828
Epoch[4] learning-rate: 0.010416666666666668
Epoch[4] Validation-accuracy: 1.0
Epoch[4] Validation-loss: 0.016005855053663254
Epoch[4] Time cost: 5.863810300827026 [sec]
=> Saved checkpoint to "model/20181226184319-resnet18-0005.model"
=> Saved checkpoint to "model/20181226184319-resnet18-best.model"
```

### test

```
$ ./test.py example_images -m model/20181226184319-resnet18-best.model
```

```
$ ./test.py example_images -m model/20181226184319-resnet18-best.model
Running script with args: Namespace(batch_size=128, cuda=True, data='example_images', input_size=None, log_dir='logs', model='model/20181226184319-resnet18-best.model', no_cuda=False, num_classes=None, prefix='20190117043959', print_cr=False, rgb_mean='0,0,0', rgb_std='1,1,1', scale_size=None, seed=None, topk=3, tta=False, tta_custom_seven_crop=False, tta_custom_six_crop=False, tta_custom_ten_crop=False, tta_custom_twenty_crop=False, tta_ten_crop=False, workers=22)
=> loading saved checkpoint 'model/20181226184319-resnet18-best.model'
scale_size: 256  input_size: 224
rgb_mean: [0.0, 0.0, 0.0]
rgb_std: [1.0, 1.0, 1.0]
number of test dataset: 200
number of classes: 10
Test: 100%|██████████| 2/2 [00:06<00:00,  4.37s/it, loss=0.0231, accuracy=99.6]
=> Saved test results to "logs/20190117043959-test-results.log"
=> Saved classification report to "logs/20190117043959-test-classification_report.log"
model: model/20181226184319-resnet18-best.model
Test-loss: 0.023076653480529785
Test-accuracy: 0.995 (199/200)
=> Saved test log to "logs/20190117043959-test.log"
```

### Calculate RGB mean and std of dataset

```
$ ./train.py example_images/train --calc-rgb-mean-and-std --batch-size 600
=> Calculate rgb mean and std (dir: example_images/train  images: 600  batch-size: 600)
Calc rgb mean/std: 100%|████████████████████████| 1/1 [00:04<00:00,  4.19s/it]
=> processed: 600 images
=> calculated rgb mean: [0.5068701  0.50441307 0.47790593]
=> calculated rgb std: [0.28773245 0.27445307 0.29044855]
Please use following command options when train and test:
 --rgb-mean 0.507,0.504,0.478 --rgb-std 0.288,0.274,0.290
```

## With Data Augmentation

### Random Rotation (Enabled by default)

```
./train.py example_images --random-rotate-degree 3.0
```

### Random Resized Crop (Enabled by default)

```
./train.py example_images \
  --random-resized-crop-scale 0.08,1.0 \
  --random-resized-crop-ratio 0.75,1.3333333333333333
```

### Random Horizontal Flip (Enabled by default)

```
./train.py example_images --random-horizontal-flip 0.5
```

### Random Vertical Flip

```
./train.py example_images --random-vertical-flip 0.5
```

### Color Jitter (Enabled by default)

```
./train.py example_images \
  --jitter-brightness 0.10 \
  --jitter-contrast 0.10 \
  --jitter-saturation 0.10 \
  --jitter-hue 0.05
```

### Normalize

```
./train.py example_images \
  --rgb-mean 0.485,0.456,0.406 \
  --rgb-std 0.229,0.224,0.225
```

### Cutout, Random Erasing, mixup, RICAP

```
# Cutout
$ ./train.py example_images --cutout

# Random Erasing
$ ./train.py example_images --random-erasing

# mixup
$ ./train.py example_images --mixup

# RICAP
$ ./train.py example_images --ricap

# mixup + Cutout
$ ./train.py example_images --mixup --cutout

# mixup + Random Erasing
$ ./train.py example_images --mixup --random-erasing

# RICAP + Cutout
$ ./train.py example_images --ricap --cutout

# RICAP + Random Erasing
$ ./train.py example_images --ricap --random-erasing
```

- Cutout: [Improved Regularization of Convolutional Neural Networks with Cutout]
- Random Erasing: [Random Erasing Data Augmentation]
- mixup: [mixup: Beyond Empirical Risk Minimization]
- RICAP: [Data Augmentation using Random Image Cropping and Patching for Deep CNNs]


## Usage

```
usage: train.py [-h] [--model ARCH] [--from-scratch] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [-j WORKERS] [--prefix PREFIX]
                [--log-dir LOG_DIR] [--model-dir MODEL_DIR] [--resume MODEL]
                [--start-epoch START_EPOCH] [--disp-batches DISP_BATCHES]
                [--save-best-only] [--save-best-and-last] [--base-lr BASE_LR]
                [--lr-factor LR_FACTOR] [--lr-step-epochs LR_STEP_EPOCHS]
                [--lr-patience LR_PATIENCE]
                [--cosine-annealing-t-max COSINE_ANNEALING_T_MAX]
                [--cosine-annealing-mult COSINE_ANNEALING_MULT]
                [--cosine-annealing-eta-min COSINE_ANNEALING_ETA_MIN]
                [--optimizer OPTIMIZER] [--momentum MOMENTUM] [--wd WD]
                [--warmup-epochs WARMUP_EPOCHS] [--scale-size SCALE_SIZE]
                [--input-size INPUT_SIZE] [--rgb-mean RGB_MEAN]
                [--rgb-std RGB_STD]
                [--random-resized-crop-scale RANDOM_RESIZED_CROP_SCALE]
                [--random-resized-crop-ratio RANDOM_RESIZED_CROP_RATIO]
                [--random-horizontal-flip RANDOM_HORIZONTAL_FLIP]
                [--random-vertical-flip RANDOM_VERTICAL_FLIP]
                [--jitter-brightness JITTER_BRIGHTNESS]
                [--jitter-contrast JITTER_CONTRAST]
                [--jitter-saturation JITTER_SATURATION]
                [--jitter-hue JITTER_HUE]
                [--random-rotate-degree RANDOM_ROTATE_DEGREE] [--image-dump]
                [--calc-rgb-mean-and-std] [--cutout]
                [--cutout-holes CUTOUT_HOLES] [--cutout-length CUTOUT_LENGTH]
                [--random-erasing] [--random-erasing-p RANDOM_ERASING_P]
                [--random-erasing-sl RANDOM_ERASING_SL]
                [--random-erasing-sh RANDOM_ERASING_SH]
                [--random-erasing-r1 RANDOM_ERASING_R1]
                [--random-erasing-r2 RANDOM_ERASING_R2] [--mixup]
                [--mixup-alpha MIXUP_ALPHA] [--ricap]
                [--ricap-beta RICAP_BETA] [--ricap-with-line] [--no-cuda]
                [--seed SEED]
                DIR

train

positional arguments:
  DIR                   path to dataset (train and validation)

optional arguments:
  -h, --help            show this help message and exit
  --model ARCH, -m ARCH
                        specify model architecture (default: resnet18)
  --from-scratch        do not use pre-trained weights (default: False)
  --epochs EPOCHS       number of total epochs to run (default: 30)
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        the batch size (default: 128)
  -j WORKERS, --workers WORKERS
                        number of data loading workers (default: 80% of the
                        number of cores)
  --prefix PREFIX       prefix of model and logs (default: auto)
  --log-dir LOG_DIR     log directory (default: logs)
  --model-dir MODEL_DIR
                        model saving dir (default: model)
  --resume MODEL        path to saved model (default: None)
  --start-epoch START_EPOCH
                        manual epoch number (default: 0)
  --disp-batches DISP_BATCHES
                        show progress for every n batches (default: auto)
  --save-best-only      save only the latest best model according to the
                        validation accuracy (default: False)
  --save-best-and-last  save last and latest best model according to the
                        validation accuracy (default: False)
  --base-lr BASE_LR     initial learning rate (default: 0.001)
  --lr-factor LR_FACTOR
                        the ratio to reduce lr on each step (default: 0.1)
  --lr-step-epochs LR_STEP_EPOCHS
                        the epochs to reduce the lr (default: 10,20)
  --lr-patience LR_PATIENCE
                        enable ReduceLROnPlateau lr scheduler with specified
                        patience (default: None)
  --cosine-annealing-t-max COSINE_ANNEALING_T_MAX
                        enable CosineAnnealinigLR scheduler with specified
                        T_max (default: None)
  --cosine-annealing-mult COSINE_ANNEALING_MULT
                        T_mult of CosineAnnealingLR scheduler
  --cosine-annealing-eta-min COSINE_ANNEALING_ETA_MIN
                        Minimum learning rate of CosineannealingLR scheduler
  --optimizer OPTIMIZER
                        the optimizer type (default: sgd)
  --momentum MOMENTUM   momentum (default: 0.9)
  --wd WD               weight decay (default: 1e-04)
  --warmup-epochs WARMUP_EPOCHS
                        number of warmup epochs (default: 5)
  --scale-size SCALE_SIZE
                        scale size (default: auto)
  --input-size INPUT_SIZE
                        input size (default: auto)
  --rgb-mean RGB_MEAN   RGB mean (default: 0,0,0)
  --rgb-std RGB_STD     RGB std (default: 1,1,1)
  --random-resized-crop-scale RANDOM_RESIZED_CROP_SCALE
                        range of size of the origin size cropped (default:
                        0.08,1.0)
  --random-resized-crop-ratio RANDOM_RESIZED_CROP_RATIO
                        range of aspect ratio of the origin aspect ratio
                        cropped (defaullt: 0.75,1.3333333333333333)
  --random-horizontal-flip RANDOM_HORIZONTAL_FLIP
                        probability of the image being flipped (default: 0.5)
  --random-vertical-flip RANDOM_VERTICAL_FLIP
                        probability of the image being flipped (default: 0.0)
  --jitter-brightness JITTER_BRIGHTNESS
                        jitter brightness of data augmentation (default: 0.10)
  --jitter-contrast JITTER_CONTRAST
                        jitter contrast of data augmentation (default: 0.10)
  --jitter-saturation JITTER_SATURATION
                        jitter saturation of data augmentation (default: 0.10)
  --jitter-hue JITTER_HUE
                        jitter hue of data augmentation (default: 0.05)
  --random-rotate-degree RANDOM_ROTATE_DEGREE
                        rotate degree of data augmentation (default: 3.0)
  --image-dump          dump batch images and exit (default: False)
  --calc-rgb-mean-and-std
                        calculate rgb mean and std of train images and exit
                        (default: False)
  --cutout              apply cutout (default: False)
  --cutout-holes CUTOUT_HOLES
                        number of holes to cut out from image (default: 1)
  --cutout-length CUTOUT_LENGTH
                        length of the holes (default: 64)
  --random-erasing      apply random erasing (default: False)
  --random-erasing-p RANDOM_ERASING_P
                        random erasing p (default: 0.5)
  --random-erasing-sl RANDOM_ERASING_SL
                        random erasing sl (default: 0.02)
  --random-erasing-sh RANDOM_ERASING_SH
                        random erasing sh (default: 0.4)
  --random-erasing-r1 RANDOM_ERASING_R1
                        random erasing r1 (default: 0.3)
  --random-erasing-r2 RANDOM_ERASING_R2
                        random erasing r2 (default: 3.3333333333333335)
  --mixup               apply mixup (default: Falsse)
  --mixup-alpha MIXUP_ALPHA
                        mixup alpha (default: 0.2)
  --ricap               apply RICAP (default: False)
  --ricap-beta RICAP_BETA
                        RICAP beta (default: 0.3)
  --ricap-with-line     RICAP with boundary line (default: False)
  --no-cuda             disables CUDA training (default: False)
  --seed SEED           seed for initializing training. (default: None)
```


[Caltech 101]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
[Improved Regularization of Convolutional Neural Networks with Cutout]: https://arxiv.org/abs/1708.04552
[Random Erasing Data Augmentation]: https://arxiv.org/abs/1708.04896
[mixup: Beyond Empirical Risk Minimization]: https://arxiv.org/pdf/1710.09412.pdf
[Data Augmentation using Random Image Cropping and Patching for Deep CNNs]: https://arxiv.org/abs/1811.09030
