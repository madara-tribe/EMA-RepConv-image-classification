# ResNeXt-RepConv-Classification
Classification task sample with ResNeXt and RepConv model 


<b>How to place RepConv in ResNet</b>

<img src="https://github.com/madara-tribe/onnxed-RepConv-ResNeXt/assets/48679574/52a55d59-6108-43ec-aa13-c35f514cd8c8" width="500px" height="400px"/>

# Perfomance

## Dataset
- [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)


| Model | Head | Pretrain | class | model param | accuracy |
| :---         |     :---:      |     :---:      |     :---:      |     :---:      |         ---: |
| resnext50d_32x4d(timm) | fc | imageNet |59|25,270,000|74.09%|
| ResNext(custum) | fc | None|59|11,459,824|77.43%|
| ResNext(LLMfc) | LLM-fc | None|59|15,801,584|77.71%|
| RepConv-ResNeXt | RepConv + fc | None |59|13,895,408|78.55%|


# ONNX convert
```bash
python3 onnx_export.py <weight_path>
```


# Features
- RepConv (re-parameterized model)
- Exponential moving average (EMA)
- image padding resize
- ONNX convert
