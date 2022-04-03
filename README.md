# 

# Pig Leg Surgery - Computer Vision REST API

There are three parts of the application:

* PiglegCV - REST API for computer vision computations
* Piglegsurgery web app - The user interface which calls internally the Rest API
* Moodle - 


# Run in production 

Update repo, Stop, Build and Up again
```shell
cd /webapps/piglegsurgery
git pull
docker-compose down

#docker build -t piglegcv /webapps/piglegsurgery/piglegcv/
docker build -t piglegcv_devel /webapps/piglegsurgery/piglegcv/
docker build -t piglegweb /webapps/piglegsurgery/docker/

docker-compose up

```

## Stop

Stop and remove unused containers

```shell
cd /webapps/piglegsurgery
docker-compose down
```


## Check logs


```shell
multitail ~/pigleg/logs/*.txt
```

## Get into docker

```shell
docker exec -it piglegsurgery_piglegcv_1 bash
```

```shell
docker exec -it piglegsurgery_piglegcv_1 bash
```

```shell
docker exec -it piglegsurgery_piglegcv_devel_1 bash
```

## Make migrations

```shell
docker exec -it piglegsurgery_piglegweb_1 bash
```

```shell
conda run -n piglegsurgery python manage.py makemigrations
conda run -n piglegsurgery python manage.py migrate

```


# Other cases

## Rest API


Run
```shell
bash /webapps/piglegsurgery/piglegcv/run_piglegcv_docker.sh
```

To kill 
```shell
docker stop piglegcv
```


## Web app
Build
```
docker build -t piglegweb /webapps/piglegsurgery/docker/
```

Start
```shell
bash /webapps/piglegsurgery/docker/start.sh
```

To kill the web app get [PGID](https://www.baeldung.com/linux/kill-members-process-group) (fourth column)
```shell
ps -efj | egrep "runserver|qcluster"
```

Use PGID to kill all children processes
```shell
kill -- -$PGID
```



# Special cases


# Run it with docker

```shell
git clone ...
cd piglegsurgery
# download resources
# run docker piglegcv
./piglegcv/run_piglegcv_docker.sh
cd piglegsurgeryweb
python manage.py qcluster |& tee -a /webapps/piglegsurgery/piglegsurgeryweb/log/qcluster_log.txt
```

```shell
conda run -n piglegsurgery --no-capture-output gunicorn --log-level debug piglegsurgeryweb.wsgi:application --bind 0:8000 --timeout 1800 --workers 5 |& tee >(rotatelogs -n 3 /webapps/piglegsurgery/piglegsurgeryweb/log/gunicorn_log.txt.bck 1k) | tee  >(rotatelogs -n 1 /webapps/piglegsurgery/piglegsurgeryweb/log/gunicorn_log.txt 1M)
```


# PiglegCV REST API 

## Run REST API With Docker

Get resources (neural network weights)
```bash
scp -r mjirik@nympha.zcu.cz:/storage/plzen4-ntis/projects/cv/pigleg/git/piglegsurgery/piglegcv/resources /webapps/piglegsurgery/piglegcv/resources
```

Build docker
```bash
cd piglegcv
docker build -t piglegcv .
```

Run from linux (for deployment dnn)
```bash
cd piglegcv
./run_detectron2.sh
```

Run from Windows (for testing)
```bash
docker run -d -v "C:/Users/Jirik/projects/piglegsurgery:/webapps/piglegsurgery" -p 5000:5000 --name piglegcv piglegcv 
```

Run from Linux
```bash
DATADIR="/home/dnn-user/data_detectron2"
docker run --gpus all -it --rm \
  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $DATADIR:/home/appuser/data \
  -p 5000:5000 \
  --name=piglegcv piglegcv
```


### Check function

Start new processing
```bash
curl -X POST 127.0.0.1:5000/run?filename=myfile.avi&outputdir=myoutpudir
```
It will return the hash id of process like `8caa5441-4983-447f-9edc-28fbf4cdf2`

Check if processing is finised
```bash
curl -X GET 127.0.0.1:5000/is_finished/8caa5441-4983-447f-9edc-28fbf4cdf2
```

## Run REST API on Linux (without Docker)
```commandline
sudo apt-get install redis
conda install -c conda-forge rq flask loguru
```


start redis
```commandline
service redis-server start
```

start worker
```commandline
python piglegcv/worker.py
```

start rest api
```commandline
python piglegcv/app.py
```


## Just run the computer vision algorithm

All computer vision processing is done in `piglegcv/pigleg_cv.py`.

It can be tested from command line:
```bash
python piglegcv/pigleg_cv.py "H:\biomedical\orig\pigleg_surgery\first_dataset\b6c6fb92-d8ad-4ccf-994c-5241a89a9274.mp4" "test_outpudir"
```


# Pig Leg Surgery - Web App

## Install with docker

```commandline
docker build -t piglegweb .
docker run -d -v "C:/Users/Jirik/projects/piglegsurgery:/webapps/piglegsurgery" -p 8000:8000 --name piglegweb piglegweb
```

On Linux
```shell
docker run -d -v "/webapps/piglegsurgery:/webapps/piglegsurgery" -p 8000:8000 --name piglegweb piglegweb
```



## Install on Ubuntu

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
python manage.py qcluster |& tee -a ~/piglegcv/logs/qcluster_log.txt

```

Run server for development
```bash
python manage.py runserver 0:8000 |& tee -a ~/piglegcv/logs/runserver_log.txt
```
or run server for production (multithreaded)

### Run in production

Run `qcluster`:
```bash
conda run -n piglegsurgery --no-capture-output python manage.py qcluster |& tee -a /webapps/piglegsurgery/piglegsurgeryweb/log/qcluster_log.txt
```
and in other bash run `qunicorn`:
```bash
conda run -n piglegsurgery --no-capture-output gunicorn --log-level debug piglegsurgeryweb.wsgi:application --bind 0:8000 --timeout 1800 --workers 5 |& tee >(rotatelogs -n 3 /webapps/piglegsurgery/piglegsurgeryweb/log/gunicorn_log.txt.bck 1k) | tee  >(rotatelogs -n 1 /webapps/piglegsurgery/piglegsurgeryweb/log/gunicorn_log.txt 1M)
```


## Run

```bash
cd piglegsurgeryweb/
python manage.py runserver 0:8000
```


In your web browser:

http://127.0.0.1:8000/uploader/


# Moodle





