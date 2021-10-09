#!/bin/bash

cd /webapps/piglegsurgery/piglegsurgeryweb

conda run -n piglegsurgery --no-capture-output python manage.py qcluster &>> log_qcluster.txt&

conda run -n piglegsurgery --no-capture-output gunicorn piglegsurgeryweb.wsgi:application --bind 0:8000 &>> log_gunicorn.txt&
