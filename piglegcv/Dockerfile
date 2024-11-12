FROM nvcr.io/nvidia/pytorch:22.05-py3

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ="Europe/Berlin"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget sudo ninja-build

# removed in 24.05-py3
 RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER appuser
WORKDIR /home/appuser

RUN pip list

## TODO check if this is needed
#ENV PATH="/home/appuser/.local/bin:${PATH}"
#RUN wget https://bootstrap.pypa.io/get-pip.py && \
#	python3 get-pip.py --user && \
#	rm get-pip.py


# Update pip to a specific version
RUN python3 -m pip install --upgrade pip==21.1 --user


RUN sudo apt-get update && sudo apt-get install -y redis ffmpeg vim

RUN pip install --user opencv-python==4.5.5.64

# We used fallowing line to fix the RuntimeError: nms_impl: implementation for device cuda:0 not found
#     MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.5.2
RUN git clone --branch v1.5.2 https://github.com/open-mmlab/mmcv.git
RUN cd mmcv && pip install --user -r requirements/optional.txt
RUN cd mmcv && MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e .

RUN git clone --branch v2.25.0 https://github.com/open-mmlab/mmdetection
RUN cd mmdetection && pip install --user -r requirements/build.txt
RUN cd mmdetection && pip install --user -v -e .

RUN git clone --branch v0.27.0 https://github.com/open-mmlab/mmpose.git
RUN cd mmpose && pip install --user -r requirements.txt
RUN cd mmpose && pip install --user -v -e .

RUN git clone --branch v0.13.0 https://github.com/open-mmlab/mmtracking.git
RUN cd mmtracking && pip install --user -r requirements.txt
RUN cd mmtracking && pip install --user -v -e .

RUN pip list

#########
#PyZbar
RUN sudo apt-get install libzbar0 -y
RUN pip install --user pyzbar

#########
#yolov7-package
RUN pip install --user yolov7-package

RUN pip install --user filterpy==1.4.5 scikit-image==0.17.2 lap==0.4.0
RUN pip install --user flask loguru "rq<2.0.0"

RUN sudo apt-get update
### to get rotatelogs
RUN sudo apt-get install --yes apache2-utils

###############
# For perpendicular
RUN pip install --user -U scikit-learn pytest

RUN sudo apt-get install -y ranger mc sshfs

RUN pip install --user gspread
RUN pip install --user --upgrade google-api-python-client oauth2client
RUN pip install --user jupyterlab==3.6 protobuf==3.20
RUN pip install --user rich
# RUN pip install ipywidgets rise 
# RUN pip install jupyterlab-vim

# for password generation for jupyterlab
RUN sudo apt-get install -y net-tools sharutils

RUN pip list

RUN mkdir -p mnt/pole

############
# MAIN
RUN mkdir logs

# Get models

#RUN mkdir -p ./.cache/torch/hub/checkpoints/
#COPY ./resources/torch/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth ./.cache/torch/hub/checkpoints/

# for easier debugg with this environment connected in pycharm
RUN pip install django==3.2.23 django-q==1.3.9 plotly==5.18.0

# it is possible to do it togehter but then everything is in one layer and slow to build
RUN mkdir -p ./resources
#COPY ./resources/ ./resources/

RUN mkdir -p ./resources/incision_detection_models
RUN mkdir -p ./resources/single_image_detector
RUN mkdir -p ./resources/stitch_detection_models/
RUN mkdir -p ./resources/tracker_model_bytetrack
RUN mkdir -p ./resources/tracker_model_bytetrack_hands_tools
RUN mkdir -p ./resources/tracker_model_bytetrack_microsurgery
RUN mkdir -p ./resources/yolov7_qrdet
RUN mkdir -p ./resources/movement_evaluation_models
#
COPY ./resources/incision_detection_models/220326_234659_mmdet.pth ./resources/incision_detection_models/
COPY ./resources/single_image_detector/mdl_sid_2.pth ./resources/single_image_detector
COPY ./resources/stitch_detection_models/model.pth ./resources/stitch_detection_models/
COPY ./resources/tracker_model_bytetrack/bytetrack_pigleg.py ./resources/tracker_model_bytetrack
COPY ./resources/tracker_model_bytetrack/yolox_l_8x8.py ./resources/tracker_model_bytetrack
COPY ./resources/tracker_model_bytetrack/epoch.pth ./resources/tracker_model_bytetrack
COPY ./resources/tracker_model_bytetrack_hands_tools/* ./resources/tracker_model_bytetrack_hands_tools/
COPY ./resources/tracker_model_bytetrack_microsurgery/* ./resources/tracker_model_bytetrack_microsurgery/
COPY ./resources/yolov7_qrdet/qrdet-yolov7.pt ./resources/yolov7_qrdet
COPY ./resources/movement_evaluation_models/* ./resources/movement_evaluation_models
#
#COPY ./resources/tracker_model_bytetrack/bytetrack_pigleg.py ./resources/tracker_model_bytetrack
#COPY ./resources/tracker_model_bytetrack/epoch.pth ./resources/tracker_model_bytetrack
#COPY ./resources/incision_detection_models/220326_234659_mmdet.pth ./resources/incision_detection_models/
#COPY ./resources/yolov7_qrdet/qrdet-yolov7.pt ./resources/yolov7_qrdet
#COPY ./resources/single_image_detector/mdl.pth ./resources/single_image_detector
#COPY ./resources/stitch_detection_models/model.pth ./resources/stitch_detection_models/
RUN sudo chmod a+rw -R ./resources


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
# COPY ./stitch_detection_mmdet.py .
# COPY ./stitch_detection_mmdet_config.py .
# COPY ./media_tools.py .
# COPY ./static_stitch_analysis.py .
# COPY ./dynamic_stitch_analysis.py .
# COPY ./structure_tools.py .
COPY ./*.py .

# USER root
RUN chmod 777 ./
# USER appuser

# CMD ["./data/images"]
CMD cd /home/appuser/ && \
    bash ./run_services.sh && \
    tail -f /dev/null


