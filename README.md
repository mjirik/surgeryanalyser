# pythontemplate


```commandline
docker build -t piglegsurgery .
docker run -d -v "C:/Users/Jirik/projects/piglegsurgery:/webapps/piglegsurgery" -p 8000:8000 -p 8080:80 --name piglegsurgery piglegsurgery
```


# In docker


## Install

### Ubuntu

```bash
cd piglegsurgery
conda env create -f docker/environment.yml
conda activate piglegsurgery
pip install -r docker/requirements_pip.txt
cd piglegsurgeryweb/
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py collectstatic
```

Run server for development
```commandline
python manage.py runserver 0:8000
```
or run server for production (multithreaded)
```commandline
gunicorn piglegsurgeryweb.wsgi:application --bind 0:8000
```


## Run

```bash
cd piglegsurgeryweb/
python manage.py runserver 0:8000
```


In your web browser:

http://127.0.0.1:8000/uploader/