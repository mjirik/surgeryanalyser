#!/bin/bash

echo "DOCKERLOGNAME=${DOCKERLOGNAME}"
# there is primary log in .txt file and 3 rotating backup logs
sudo service redis-server start 2>&1 | \
#   tee >(rotatelogs -n 3 logs/piglegcv_${DOCKERLOGNAME}_redis_log.txt.bck 1M) | \
   rotatelogs -n 1 logs/piglegcv_${DOCKERLOGNAME}_redis_log.txt 1M
python worker.py 2>&1 | \
#   tee >(rotatelogs -n 3 logs/worker_${DOCKERLOGNAME}_log_0.txt.bck 1M) | \
   rotatelogs -n 1 logs/worker_${DOCKERLOGNAME}_log_0.txt 1M &
python worker.py 2>&1 | \
#   tee >(rotatelogs -n 3 logs/worker_${DOCKERLOGNAME}_log_1.txt.bck 1M) | \
   rotatelogs -n 1 logs/worker_${DOCKERLOGNAME}_log_1.txt 1M &
#python worker.py 2>&1 | \
#   tee >(rotatelogs -n 3 logs/worker_${DOCKERLOGNAME}_log_2.txt.bck 1M) | \
#   rotatelogs -n 1 logs/worker_${DOCKERLOGNAME}_log_2.txt 1M &
jupyter lab jupyter lab --ip 147.228.47.69 --no-browser --port 8888 2>&1 | \
#   tee >(rotatelogs -n 3 logs/jupyterlab_${DOCKERLOGNAME}_log_2.txt.bck 1M) | \
   rotatelogs -n 1 logs/jupyterlab_${DOCKERLOGNAME}_log_2.txt 1M &
echo "jupyterlab started"

python app.py 2>&1 | \
   tee >(rotatelogs -n 3 logs/app_${DOCKERLOGNAME}_log.txt.bck 1M) | \
   rotatelogs -n 1 logs/app_${DOCKERLOGNAME}_log.txt 1M &
echo "Services started"
