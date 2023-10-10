import sys
import time
from PIL import Image
import numpy as np
import onnxruntime as ort
from torchsummary import summary
import torch
import torch.onnx 
from cfg import Cfg
from models.models import call_RepConvResNeXt
from utils.augmentations import transforms_


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def load_img(path, cfg):
    _, val_transform = transforms_(cfg)
    x = pil_loader(path)
    tensor_img = val_transform(image=np.array(x))['image']
    x = tensor_img.to('cpu').detach().numpy().copy()
    img_in = np.expand_dims(x, axis=0)
    print(img_in.shape, img_in.max(), img_in.min())
    label = int(path.split('/')[-1].split('_')[0])
    return img_in, label

def inference(onnx_file_name, img_path, cfg):
    img_in, label = load_img(img_path, cfg)
    print("Shape of the network input: ", img_in.shape, img_in.min(), img_in.max())

    ort_session = ort.InferenceSession(onnx_file_name)
    # IMAGE_HEIGHT = ort_session.get_inputs()[0].shape[2]
    # IMAGE_WIDTH = ort_session.get_inputs()[0].shape[3]
    input_name = ort_session.get_inputs()[0].name
    print("The model expects input shape: ", ort_session.get_inputs()[0].shape)

    print('start calculation')
    start_time = time.time()
    outputs = ort_session.run(None, {input_name: img_in.astype(np.float32)})[0]
    pred = np.argmax(outputs)
    print(label, pred)
    if pred==int(label):
        print("correct")
    else:
        print("wrong")
    latency = time.time()-start_time
    print(f"inference time is {latency}")

def load_trained_model(cfg, device, weight_path):
    model = call_RepConvResNeXt(cfg, device, deploy=True)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    if cfg.half==1:
        model.half().float()
    summary(model, (3, cfg.input_size, cfg.input_size))
    return model

#Function to Convert to ONNX 
def Convert_ONNX(model, device):
    H = W = cfg.input_size
    # set the model to inference mode 
    model.eval() 
    dummy_input = torch.randn(1, 3, H, W, requires_grad=True).to(device)  
    onnx_file_name = "repconv{}_{}.onnx".format(H, W)
    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         onnx_file_name,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}})
                                
    print(f'Model has been converted to ONNX as {onnx_file_name}') 
    return onnx_file_name

def main(cfg, device, weight_path, img_path):
    model = load_trained_model(cfg, device, weight_path)
    onnx_file_name = Convert_ONNX(model, device)
    inference(onnx_file_name, img_path, cfg)


if __name__=="__main__":
    if len(sys.argv)>1:
        weight_path = sys.argv[1]
    else:
        print("weight_path should be difined")
        sys.exit(1)
    cfg = Cfg
    img_path = '0_1_4_5_2023.jpg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(cfg, device, weight_path, img_path)


