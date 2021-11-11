#!/bin/bash

echo "Starting services..."
echo "PATH=${PATH}"
cd /webapps/piglegsurgery/piglegsurgeryweb
sudo service redis-server start |& tee -a ~/piglegcv/logs/redis_log.txt &
conda run -n piglegsurgery --no-capture-output python manage.py qcluster |& tee -a ~/piglegcv/logs/qcluster_log.txt &
conda run -n piglegsurgery --no-capture-output python manage.py runserver 0:8000 |& tee -a ~/piglegcv/logs/runserver_log.txt &
echo "Services started"
