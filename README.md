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
| RepConv-ResNeXt | RepConv + fc | None |59|13,895,408|79.55%|

## loss(train/valid)



# ONNX convert
```bash
python3 onnx_export.py <weight_path>
```

# improvent methods
- ExponentialMovingAverage(EMA)
- resize image size during training
- model half
- augumentation that fit dataset
- norm layer
- RexNext + AdamW
- image padding resize

# References
- kaggle
