#!/bin/bash

echo ${PATH}
cd /webapps/piglegsurgery/piglegsurgeryweb
# prepare django
conda run -n piglegsurgery --no-capture-output python manage.py makemigrations --noinput --verbosity 2
conda run -n piglegsurgery --no-capture-output python manage.py migrate --noinput --verbosity 2
conda run -n piglegsurgery --no-capture-output python manage.py collectstatic --noinput --verbosity 2

conda run -n piglegsurgery --no-capture-output python manage.py qcluster &>> log_qcluster.txt&
conda run -n piglegsurgery --no-capture-output gunicorn piglegsurgeryweb.wsgi:application --bind 0:8000 &>> log_gunicorn.txt&
