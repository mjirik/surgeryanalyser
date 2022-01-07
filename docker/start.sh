#!/bin/bash

echo "Starting services..."
echo "PATH=${PATH}"
cd /webapps/piglegsurgery/piglegsurgeryweb
sudo service redis-server start |& \
    tee >(rotatelogs -n 1 ~/piglegcv/logs/redis_log.txt 1M) | \
    tee >(rotatelogs -n 3 ~/piglegcv/logs/redis_log.txt.bck 1M)
echo "  Redis started"
conda run -n piglegsurgery --no-capture-output python manage.py qcluster |& \
    tee >(rotatelogs -n 1 ~/piglegcv/logs/qcluster_log.txt 1M) | \
    tee >(rotatelogs -n 3 ~/piglegcv/logs/qcluster_log.txt.bck 1M) &
echo "  QCluster started"
conda run -n piglegsurgery --no-capture-output python manage.py runserver 0:8000 |& \
    tee >(rotatelogs -n 1 ~/piglegcv/logs/runserver_log.txt 1M) | \
    tee >(rotatelogs -n 3 ~/piglegcv/logs/runserver_log.txt.bck 1M) &
echo "Services started"
