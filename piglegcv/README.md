


# `mmtrack` training

Stop running docker
```bash
docker stop <docker_id>
```

Build and enter the docker
```bash
docker build -t piglegcv .
bash run_piglegcv_docker_development.py
dokcer exec -it <docker_id> bash
```

In the docker
```bash
git clone https://github.com/mjirik/piglegsurgery.git
cd mmtracking
bash tools/dist_train.sh ~/piglegsurgery/piglegcv/configs/bytetrack_pigleg.py 1

```