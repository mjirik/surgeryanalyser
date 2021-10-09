#!/bin/bash

cd /webapps/piglegsurgery/piglegsurgeryweb

conda run -n piglegsurgery --no-capture-output python manage.py qcluseter >> log_qcluster.txt&

conda run -n piglegsurgery --no-capture-output gunicorn piglegsurgeryweb.wsgi:application --bind 0:8001 >> log_qcluster.txt&
