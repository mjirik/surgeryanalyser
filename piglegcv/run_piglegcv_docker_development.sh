DATADIR="/webapps/piglegsurgery/piglegsurgeryweb/media"
#LOGDIR="/webapps/piglegsurgery/piglegsurgeryweb/log_devel"
LOGDIR="~/piglegcv/logs"
mkdir -p $LOGDIR
docker run --gpus all -d --rm \
  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $DATADIR:/webapps/piglegsurgery/piglegsurgeryweb/media \
  -v $LOGDIR:/home/appuser/logs \
  -p 5001:5000 \
  --name=piglegcv_devel piglegcv
