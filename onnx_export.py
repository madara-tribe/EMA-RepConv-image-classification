import sys, os
import time
import numpy as np
import onnxruntime as ort
from torchsummary import summary
import torch
import torch.onnx 
from models.BaseModel import RecognitionModel
from cfg import Cfg

def load_img(path='datasets/npy'):
    X = np.load(os.path.join(path, 'X_test.npy'))
    y = np.load(os.path.join(path, 'y_test.npy'))
    label = y[0]
    img_in = np.transpose(X[0], (2, 0, 1))
    img_in = np.expand_dims(img_in, axis=0)/255
    return img_in, label

def inference(onnx_file_name):
    img_in, label = load_img(path='datasets/npy')
    print("Shape of the network input: ", img_in.shape, img_in.min(), img_in.max())

    ort_session = ort.InferenceSession(onnx_file_name)
    IMAGE_HEIGHT = ort_session.get_inputs()[0].shape[2]
    IMAGE_WIDTH = ort_session.get_inputs()[0].shape[3]
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
    model = RecognitionModel(embedding_size=cfg.emmbed_size, deploy=False, bottleneck_width=1.5, cardinality=4).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    #print(model)
    summary(model, (3, cfg.input_size, cfg.input_size))
    return model

#Function to Convert to ONNX 
def Convert_ONNX(model):
    H = W = cfg.input_size
    # set the model to inference mode 
    model.eval() 
    dummy_input = torch.randn(1, 3, H, W, requires_grad=True)  
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
    inference(onnx_file_name)


if __name__=="__main__":
    if len(sys.argv)>1:
        weight_path = sys.argv[1]
    else:
        print("weight_path should be difined")
        sys.exit(1)
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(cfg, device, weight_path)
    Convert_ONNX(model)


