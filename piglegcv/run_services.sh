#!/bin/bash


sudo service redis-server start &>> logs/redis_log.txt
python worker.py &>> logs/worker_log.txt &
python app.py &>> logs/app_log.txt &
echo "Services started"
