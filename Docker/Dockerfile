ARG CUDA=11.2.0
ARG UBUNTU=20.04
#FROM nvidia/cuda:${CUDA}-devel-ubuntu${UBUNTU}
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /home/app
RUN apt-get update && apt-get upgrade -y

RUN apt install -y \
    sudo tzdata vim git cmake wget unzip 
RUN apt-get update && apt install -y build-essential libssl-dev libffi-dev python3-dev libgl1 
RUN apt install -y python3-pip
# for opencv
RUN apt-get install -y libglib2.0-0

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

RUN pip3 install mxnet-mkl==1.6.0 numpy==1.23.1
RUN pip3 uninstall -y onnx onnxruntime onnxruntime-gpu onnxruntime-silicon
RUN pip3 install onnx==1.14.0 onnxruntime==1.15.0
CMD [ "/bin/bash" ]
