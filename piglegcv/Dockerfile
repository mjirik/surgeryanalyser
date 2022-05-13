FROM nvidia/cuda:11.1.1-cudnn8-devel
ENV TZ="Europe/Berlin"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies	
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install --user torch==1.9 torchvision==0.10 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo
RUN pip install --user -U git+https://github.com/albumentations-team/albumentations

# SORT
# RUN git clone https://github.com/abewley/sort.git extern
ADD ./extern ./extern
RUN pip install --user filterpy==1.4.5 scikit-image==0.17.2 lap==0.4.0
RUN pip install --user flask loguru rq


# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
# WORKDIR /home/appuser/data

#ENTRYPOINT ["/home/appuser/run_tracker_lite.sh"]
USER root
RUN apt-get update && apt-get install -y redis ffmpeg vim

#########
#PyZbar
RUN apt-get install libzbar0 -y
USER appuser
RUN pip install pyzbar

###############
# mmpose + mmdet
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
# Install MMDetection
# USER root
RUN git clone https://github.com/open-mmlab/mmdetection.git mmdetection
# RUN chmod -R 777 /mmdetection/
# USER appuser
WORKDIR /home/appuser/mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
# Install MMPose
# USER root
WORKDIR /home/appuser
RUN git clone https://github.com/open-mmlab/mmpose.git mmpose
WORKDIR /home/appuser/mmpose
RUN mkdir -p /home/appuser/mmpose/data
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
WORKDIR /home/appuser

### to get rotatelogs
RUN sudo apt-get install --yes apache2-utils

###############
# For perpendicular
RUN pip install -U scikit-learn pytest

#install mmtrack
RUN git clone https://github.com/open-mmlab/mmtracking.git
WORKDIR /home/appuser/mmtracking
RUN pip install -r requirements/build.txt
RUN pip install -v -e .
WORKDIR /home/appuser


############
# MAIN
RUN mkdir logs

# check permissions of folowing files before this docker file build!!!
COPY ./run_tracker_lite.py .
COPY ./worker.py .
COPY ./app.py .
COPY ./pigleg_cv.py .
COPY ./run_services.sh .
COPY ./cascade_rcnn_x101_64x4d_fpn_1class.py .
COPY ./res50_onehand10k_256x256.py .
COPY ./run_mmpose.py .
COPY ./run_qr.py .
COPY ./run_report.py .
COPY ./run_perpendicular.py .
COPY ./incision_detection_mmdet.py .
RUN mkdir -p ./.cache/torch/hub/checkpoints/
COPY ./resources/mmpose_model/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth ./.cache/torch/hub/checkpoints/
COPY ./resources/mmpose_model/res50_onehand10k_256x256-e67998f6_20200813.pth ./.cache/torch/hub/checkpoints/
RUN mkdir -p ./.cache/tracker_model
RUN mkdir -p ./resources/incision_detection_models
COPY ./resources/incision_detection_models/220326_234659_mmdet.pth ./resources/incision_detection_models/
COPY ./resources/tracker_model ./.cache/tracker_model

USER root
RUN chmod 777 ./
USER appuser

#CMD ["./data/images"]
CMD cd /home/appuser/ && \
    bash ./run_services.sh && \
    tail -f /dev/null

#    sudo service redis-server start && \
#    /webapps/piglegsurgery/docker/bin/qcluster_start  & && \
#    /webapps/piglegsurgery/docker/bin/gunicorn_start & && \

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

