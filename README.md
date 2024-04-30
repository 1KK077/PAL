PAL: Prompt Adversarial Learning for Visible-Infrared Person Re-Identification

## Environmental requirements:

PyTorch == 1.10.0

ignite == 0.2.1

torchvision == 0.11.2

apex == 0.1

**Training:**

To train the model, you can use following command:

SYSU-MM01:
```Shell
python train.py --cfg ./configs/SYSU.yml
```

RegDB:
```Shell
python train.py --cfg ./configs/RegDB.yml
```

RegDB:
```Shell
python train.py --cfg ./configs/RegDB.yml
```
