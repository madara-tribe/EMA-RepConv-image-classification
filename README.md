# Abstract

This is a program for image Classification task.

With this program, it is possible to get higher accuracy than simple image Classification ones.
because there are many methods good for image Classification task.

This time, ResNeXt and RepConv are used for the main model.

# model : ResNeXt + RepConv
Classification task sample with ResNeXt and RepConv model 

<b>How to place RepConv in ResNet</b>

<img src="https://github.com/madara-tribe/onnxed-RepConv-ResNeXt/assets/48679574/52a55d59-6108-43ec-aa13-c35f514cd8c8" width="500px" height="400px"/>

# Perfomance

## Dataset
- [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

| Model | Head | Pretrain | class | model param | accuracy |
| :---         |     :---:      |     :---:      |     :---:      |     :---:      |         ---: |
| resnext50d_32x4d(timm) | fc | imageNet |69|25,270,000|74.09%|
| RepConv-ResNeXt | RepConv + fc | None |69|13,895,408|79.55%|

## loss(train/valid)

<img src="https://github.com/madara-tribe/onnxed-RepConv-ResNeXt/assets/48679574/ee856f28-91d1-4320-ba62-77fcfa941aa9" width="300px" height="200px"/>

<img src="https://github.com/madara-tribe/onnxed-RepConv-ResNeXt/assets/48679574/34ae5fe6-f8c6-4e1b-b75a-3f08df1599fa" width="300px" height="200px"/>

## train/eval
```bash
python3 main.py --mode train (-w <weight path>)
python3 main.py --mode eval -w <weight path>
```

## ONNX convert
```bash
python3 onnx_export.py <weight_path>
```

# improvent methods
- ExponentialMovingAverage(EMA)
- resize image size during training
- augumentation that fit dataset
- norm layer
- RexNext + AdamW
- image padding resize

# References
- [kaggle](https://www.kaggle.com/code/nachiket273/pytorch-resnetrs50-ema-wandb)
- [yolov7](https://github.com/WongKinYiu/yolov7)
