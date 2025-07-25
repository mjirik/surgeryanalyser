FROM nvidia/cuda:11.3.1-cudnn8-devel
ENV TZ="Europe/Berlin"
#FROM nginx
MAINTAINER Miroslav Jiřík <mjirik@kky.zcu.cz>
EXPOSE 8000

ENV DEBIAN_FRONTEND noninteractive
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update --yes
RUN apt-get install --yes \
    python3-opencv ca-certificates python3-dev git wget sudo ninja-build \
    gpg vim redis ffmpeg libsm6 libxext6 curl supervisor multitail \
    apache2-utils
# apache2-utils  # get rotatelogs

RUN ln -sv /usr/bin/python3 /usr/bin/python

## create a non-root user
#ARG USER_ID=1000
#RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER appuser
##WORKDIR /home/appuser
WORKDIR /webapps

#ENV PATH="/home/appuser/.local/bin:${PATH}"
#RUN wget https://bootstrap.pypa.io/get-pip.py && \
#	python3 get-pip.py --user && \
#	rm get-pip.py


#####################

#VOLUME .. /
USER root
SHELL ["/bin/bash", "--login", "-c"]
RUN #apt-get update --yes
# PyQt5 => ffmpeg libsm6 libxext6
#RUN apt-get install --yes gpg vim redis ffmpeg libsm6 libxext6 curl supervisor multitail \
#    # get rotatelogs
#    apache2-utils \

# Install our public GPG key to trusted store
RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg
RUN install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg

# Check whether fingerprint is correct (will output an error message otherwise)
RUN gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806

# Add our Debian repo
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list
RUN apt-get update --yes
RUN apt-get install --yes conda

# make conda visible
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc
RUN /opt/conda/condabin/conda init bash
# RUN source /opt/conda/etc/profi le.d/conda.sh
ENV PATH="/opt/conda/condabin:${PATH}"
ARG PATH="/opt/conda/condabin:${PATH}"

RUN conda update conda
RUN conda update --name base conda
# RUN #conda update conda-build
#RUN conda install -n base conda-libmamba-solver
#RUN conda config --set solver libmamba

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER appuser

# COPY scaffanweb .
COPY environment.yml .
COPY requirements_pip.txt .

RUN conda env create -v -v -f environment.yml
#RUN conda install -n piglegsurgery --yes -c conda-forge redis loguru=0.5 gunicorn=20 opencv=4.8 ffmpeg pillow pytest
#RUN conda install -n piglegsurgery --yes -c conda-forge google-api-python-client=2.88 gspread=5.7
#RUN conda install -n piglegsurgery --yes -c conda-forge google-auth-httplib2=0.1 google-auth=2.19 oauth2client=4.1 google-auth-oauthlib=1.0
#RUN conda install -n piglegsurgery --yes -c conda-forge rich
#RUN conda install -n piglegsurgery --yes -c conda-forge rich django==3.2.23
#RUN conda install -n piglegsurgery --yes -c conda-forge django-widget-tweaks
RUN conda list
#RUN cd /webapps/scaffanweb_django
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "scaffanweb", "/bin/bash", "--login", "-c"]
RUN /opt/conda/condabin/conda init bash

# installed from environment.yml
# RUN conda run -n piglegsurgery --no-capture-output pip install -r requirements_pip.txt

# gunicorn is in conda env
#RUN conda run -n piglegsurgery --no-capture-output pip install gunicorn

#CMD conda run -n piglegsurgery --no-capture-output python manage.py qcluster
#CMD service redis-server start && tail -F /var/log/redis/error.log

# Install Detectron2 dependencies
#################################

# See https://pytorch.org/ for other options if you use a different version of CUDA
#RUN conda run -n piglegsurgery --no-capture-output pip install tensorboard cmake   # cmake from apt-get is too old
#RUN conda run -n piglegsurgery --no-capture-output pip install torch==1.9 torchvision==0.10 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#
#RUN conda run -n piglegsurgery --no-capture-output pip install 'git+https://github.com/facebookresearch/fvcore'
## install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
## set FORCE_CUDA because during `docker build` cuda is not accessible
#ENV FORCE_CUDA="1"
## This will by default build detectron2 for all common cuda architectures and take a lot more time,
## because inside `docker build`, there is no way to tell which architecture will be used.
#ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
#ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
#
#RUN conda run -n piglegsurgery --no-capture-output pip install -e detectron2_repo
#
## Set a fixed model cache directory.
#ENV FVCORE_CACHE="/tmp"

######### - Detectron 2 install end
#WORKDIR /home/appuser/detectron2_repo



WORKDIR /webapps

# RUN conda run -n piglegsurgery --no-capture-output pip install django-q=1.3.9

#RUN apt-get install multitail
RUN mkdir -p ~/piglegcv/logs/

# for dash conversion
# RUN apt-get install --yes gpac

# remove files to fix E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease' is no longer signed.
# https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN sudo rm /etc/apt/sources.list.d/cuda.list
RUN sudo rm /etc/apt/sources.list.d/nvidia-ml.list

# to get rotatelogs
#RUN sudo apt-get update --yes
#RUN sudo apt-get install --yes apache2-utils

# RUN conda install -n piglegsurgery --yes -c conda-forge plotly=5.18.0 statsmodels pandas=1.5.3 openpyxl

RUN sudo mkdir -p /webapps/piglegsurgery/piglegsurgeryweb/static/
RUN sudo chmod a+w -R /webapps/piglegsurgery/piglegsurgeryweb/static/
# RUN sudo chmod a+w /webapps/piglegsurgery/piglegsurgeryweb/private/db.sqlite3
#ls -l /webapps/piglegsurgery/piglegsurgeryweb/private/

#RUN conda install -n piglegsurgery -c conda-forge django-environ
#COPY start.sh
#CMD cd /webapps/piglegsurgery/piglegsurgeryweb && conda run -n piglegsurgery --no-capture-output python manage.py qcluster
#CMD service redis-server start && tail -F /var/log/redis/error.log
#CMD tail -f /dev/null

# install gunicorn
# RUN conda install -n piglegsurgery --yes -c conda-forge gunicorn=20.1.0

ARG ENV=development
ENV ENV=$ENV

#ENV PIGLEGCV_HOSTNAME "PIGLEGCV_HOSTNAME"
CMD cd /webapps/piglegsurgery/docker && \
    bash ./start.sh && \
    tail -f /dev/null
#CMD tail -f /dev/null

#    conda run -n piglegsurgery --no-capture-output python manage.py qcluster && \
#    conda run -n piglegsurgery --no-capture-output python manage.py runserver 0:8000 && \
