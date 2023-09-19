FROM nvcr.io/nvidia/pytorch:22.05-py3

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ="Europe/Berlin"

# ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# USER appuser
WORKDIR /home/appuser

RUN pip list

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

RUN apt-get update && apt-get install -y redis ffmpeg vim

RUN pip install opencv-python==4.5.5.64

# We used fallowing line to fix the RuntimeError: nms_impl: implementation for device cuda:0 not found
#     MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.5.2
RUN git clone --branch v1.5.2 https://github.com/open-mmlab/mmcv.git &&\
    cd mmcv &&\
    pip install -r requirements/optional.txt &&\
    MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e .
RUN cd ..

RUN git clone --branch v2.25.0 https://github.com/open-mmlab/mmdetection &&\
    cd mmdetection &&\
    pip install -r requirements/build.txt &&\
    pip install -v -e .
RUN cd ..

RUN git clone --branch v0.27.0 https://github.com/open-mmlab/mmpose.git &&\
    cd mmpose &&\
    pip install -r requirements.txt &&\
    pip install -v -e .
RUN cd ..

RUN git clone --branch v0.13.0 https://github.com/open-mmlab/mmtracking.git &&\
    cd mmtracking &&\
    pip install -r requirements.txt &&\
    pip install -v -e .
RUN cd ..

RUN pip list

#RUN git clone https://github.com/open-mmlab/mmtracking.git &&\
#    cd mmtracking &&\
#    pip install -r requirements.txt --ignore-installed llvmlite &&\ 
#    pip install -v -e .

#########
#PyZbar
RUN apt-get install libzbar0 -y
RUN pip install pyzbar

#########
#yolov7-package
RUN pip install yolov7-package

RUN pip install --user filterpy==1.4.5 scikit-image==0.17.2 lap==0.4.0
RUN pip install --user flask loguru rq

### to get rotatelogs
RUN sudo apt-get install --yes apache2-utils

###############
# For perpendicular
RUN pip install -U scikit-learn pytest

RUN apt-get install -y ranger mc sshfs 

RUN pip install gspread
RUN pip install --upgrade google-api-python-client oauth2client
RUN pip install jupyterlab protobuf==3.20 ipywidgets

# for password generation for jupyterlab
RUN apt-get install -y net-tools sharutils

RUN pip list

RUN mkdir -p mnt/pole

############
# MAIN
RUN mkdir logs

# Get models
RUN mkdir -p ./resources/tracker_model_bytetrack
RUN mkdir -p ./.cache/torch/hub/checkpoints/
RUN mkdir -p ./resources/incision_detection_models
RUN mkdir -p ./resources/yolov7_qrdet
RUN mkdir -p ./resources/single_image_detector
RUN mkdir -p ./resources/stitch_detection_models/

# this is for final user appuser
# RUN TORCH_CHECKPOINTS=./.cache/torch/hub/checkpoints/
# this is for final user root
RUN TORCH_CHECKPOINTS=/root/.cache/torch/hub/checkpoints/

COPY ./resources/tracker_model_bytetrack/bytetrack_pigleg.py ./resources/tracker_model_bytetrack
COPY ./resources/tracker_model_bytetrack/epoch.pth ./resources/tracker_model_bytetrack
COPY ./resources/incision_detection_models/220326_234659_mmdet.pth ./resources/incision_detection_models/
# COPY ./resources/incision_detection_models/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth $TORCH_CHECKPOINTS 
COPY ./resources/torch/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth $TORCH_CHECKPOINTS
COPY ./resources/yolov7_qrdet/qrdet-yolov7.pt ./resources/yolov7_qrdet
COPY ./resources/single_image_detector/mdl.pth ./resources/single_image_detector
COPY ./resources/stitch_detection_models/model.pth ./resources/stitch_detection_models/

# Copy scripts
# check permissions of folowing files before this docker file build!!!
COPY ./worker.py .
COPY ./app.py .
COPY ./pigleg_cv.py .
COPY ./cascade_rcnn_x101_64x4d_fpn_1class.py .
COPY ./res50_onehand10k_256x256.py .
COPY ./run_qr.py .
COPY ./qrdet.py .
COPY ./qreader.py .
COPY ./run_report.py .
COPY ./run_perpendicular.py .
COPY ./tools.py .
COPY ./run_services.sh .
COPY ./incision_detection_mmdet.py .
COPY ./run_tracker_bytetrack.py .
COPY ./stitch_detection_mmdet.py .
COPY ./stitch_detection_mmdet_config.py .

# USER root
RUN chmod 777 ./
# USER appuser

# CMD ["./data/images"]
CMD cd /home/appuser/ && \
    bash ./run_services.sh && \
    tail -f /dev/null


