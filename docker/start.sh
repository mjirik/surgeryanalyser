#!/bin/bash

echo "Starting services..."
echo "PATH=${PATH}"
mkdir -p ~/piglegcv/logs/
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
echo "  Django (debug) webserver started"

# TODO production run
#gunicorn --log-level debug piglegsurgeryweb.wsgi:application --bind 0:8000 --timeout 1800 --workers 5 |& \
#    tee >(rotatelogs -n 3 /webapps/piglegsurgery/piglegsurgeryweb/log/gunicorn_log.txt.bck 1k) | tee  >(rotatelogs -n 1 /webapps/piglegsurgery/piglegsurgeryweb/log/gunicorn_log.txt 1M)
echo "Services started"
