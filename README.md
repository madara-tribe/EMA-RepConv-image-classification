# ResNeXt-RepConv-Classification
Classification task sample with ResNeXt and RepConv model 


<b>How to place RepConv in ResNet</b>

<img src="https://github.com/madara-tribe/onnxed-RepConv-ResNeXt/assets/48679574/c624c06c-5e2b-42a6-8515-a8f4a4f8eac8" width="500px" height="400px"/>

# Perfomance

| Model | Head | Pretrain | class | model param | accuracy |
| :---         |     :---:      |     :---:      |     :---:      |     :---:      |         ---: |
| resnext50d_32x4d(timm) | fc | imageNet | 16 | | %|
| ResNext(original) | fc | None| 16  | | %|
| ResNext-LLMfc | LLM-fc | None| 16  | | %|
| RepConv-ResNeXt | RepConv + fc | None | 16  | | %|

