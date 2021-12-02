DATADIR="/webapps/piglegsurgery/piglegsurgeryweb/media"
LOGDIR="/webapps/piglegsurgery/piglegsurgeryweb/log_devel"
MODELDIR="/home/dnn-user/piglegcv/model"
docker run --gpus all -d --rm \
  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $DATADIR:/webapps/piglegsurgery/piglegsurgeryweb/media \
  -v $LOGDIR:/home/appuser/logs \
  -v $MODELDIR:/home/appuser/tracker_model \
  -p 5000:5001 \
  --name=piglegcv piglegcv
