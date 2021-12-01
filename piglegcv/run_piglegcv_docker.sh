DATADIR="/webapps/piglegsurgery/piglegsurgeryweb/media"
LOGDIR="/webapps/piglegsurgery/piglegsurgeryweb/log"
MODELDIR="/home/dnn-user/piglegcv/model"
docker run --gpus all -it --rm \
  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $DATADIR:/webapps/piglegsurgery/piglegsurgeryweb/media \
  -v $LOGDIR:/home/appuser/logs \
  -v $MODELDIR:/home/appuser/tracker_model \
  -p 5000:5000 \
  --name=piglegcv piglegcv
