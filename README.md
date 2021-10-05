# pythontemplate


```commandline
docker build -t piglegsurgery .
docker run -d -v "C:/Users/Jirik/projects/piglegsurgery:/webapps/piglegsurgery/piglegsurgeryweb" -p 8000:8000 -p 8080:80 --name piglegsurgery piglegsurgery
```


# In docker


## Install

```bash
cd piglegsurgeryweb/
python manage.py runserver 0:8000
```


## Run

```bash
cd piglegsurgeryweb/
python manage.py runserver 0:8000
```


In your web browser:

http://127.0.0.1:8000/uploader/