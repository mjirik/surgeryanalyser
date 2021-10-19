# 

# Pig Leg Surgery - Computer Vision REST API

```commandline
sudo apt-get install redis
conda install -c conda-forge rq flask loguru
```


start redis
```commandline
redis-server
```

start worker
```commandline
python piglegcv/worker.py
```

start rest api
```commandline
python piglegcv/app.py
```


## Test

Start new image processing
```commandline
curl -X POST 127.0.0.1:5000/run?filename=myfile.avi
```







# Pig Leg Surgery - Web App

```commandline
docker build -t piglegsurgery .
docker run -d -v "C:/Users/Jirik/projects/piglegsurgery:/webapps/piglegsurgery" -p 8000:8000 -p 8080:80 --name piglegsurgery piglegsurgery
```


# In docker


## Install

### Ubuntu

Install prerequisites and deploy django

```bash
cd piglegsurgery
conda env create -f docker/environment.yml
conda activate piglegsurgery
pip install -r docker/requirements_pip.txt
cd piglegsurgeryweb/
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic
```

Prepare admin
```commandline
python manage.py createsuperuser
```

Setup email parameters by creating `.env` file. 
You will need to [generate new "application password"](https://support.google.com/accounts/answer/185833?hl=en) for gmail.
See [django tutorial](https://www.sitepoint.com/django-send-email/) for more details.
```ini
EMAIL_HOST=smtp.gmail.com
EMAIL_HOST_USER=YourEmail@address
EMAIL_HOST_PASSWORD=YourAppPassword
```

Start `redis` service
```bash
service redis-server start
```

Run qcluster for comunicating between webapp and `redis`
```bash
python manage.py qcluster

```

Run server for development
```commandline
python manage.py runserver 0:8000
```
or run server for production (multithreaded)
```commandline
gunicorn piglegsurgeryweb.wsgi:application --bind 0:8000 --timeout 150 --workers 6
```


## Run

```bash
cd piglegsurgeryweb/
python manage.py runserver 0:8000
```


In your web browser:

http://127.0.0.1:8000/uploader/



# Contribute CV

All computer vision processing is done in `piglegsurgeryweb/uploader/pigleg_cv.py`.

It can be tested from command line:
```commandline
python piglegsurgeryweb\uploader\pigleg_cv.py "H:\biomedical\orig\pigleg_surgery\first_dataset\b6c6fb92-d8ad-4ccf-994c-5241a89a9274.mp4" "test_outpudir"
```
