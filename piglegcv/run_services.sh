#!/bin/bash

mkdir -p ../logs
echo "DOCKERLOGNAME=${DOCKERLOGNAME}"
# there is primary log in .txt file and 3 rotating backup logs
sudo service redis-server start 2>&1 | \
#   tee >(rotatelogs -n 3 logs/piglegcv_${DOCKERLOGNAME}_redis_log.txt.bck 1M) | \
   rotatelogs -n 1 ../logs/piglegcv_redis_${DOCKERLOGNAME}_log.txt 1M

PIGLEG_WORKER_PIGLEGCV=${PIGLEG_WORKERS_PIGLEGCV:-1}  # výchozí na 1, pokud není definováno

for ((i=0; i<PIGLEG_WORKER_PIGLEGCV; i++)); do
    echo "Starting worker $i"
    PIGLEG_WORKER_ID=$i python worker.py 2>&1 | \
        rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_${i}.txt 1M &
done
# PIGLEG_WORKER_ID=0 python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_0.txt 1M &
# PIGLEG_WORKER_ID=1 python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_1.txt 1M &
# PIGLEG_WORKER_ID=2 python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_2.txt 1M &
# PIGLEG_WORKER_ID=3 python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_3.txt 1M &
# PIGLEG_WORKER_ID=4 python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_4.txt 1M &
# PIGLEG_WORKER_ID=5 python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_5.txt 1M &
# python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_3.txt 1M &
# python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_4.txt 1M &
# python worker.py 2>&1 | \
#    rotatelogs -n 1 ../logs/piglegcv_worker_${DOCKERLOGNAME}_log_5.txt 1M &

python app.py 2>&1 | \
#   tee >(rotatelogs -n 3 logs/piglegcv_app_${DOCKERLOGNAME}_log.txt.bck 1M) | \
   rotatelogs -n 1 ../logs/piglegcv_app_${DOCKERLOGNAME}_log.txt 1M &
echo "Services started"


# Run jupyter lab

# run jupyter if PIGLEG_RUN_JUPYTER is set
if [ -z "$PIGLEG_RUN_JUPYTER" ] && [ "$PIGLEG_RUN_JUPYTER" != "True" ]; then
    echo "PIGLEG_RUN_JUPYTER is not set True, not running jupyterlab."
    exit 0
fi

# HOMEDIR=/storage/plzen1/home/$USER # substitute username and path to to your real username and path
# PRIVATEDIR=/webapps/piglegsurgery/piglegsurgeryweb/private
HOSTNAME=`hostname -f`
JUPYTER_PORT="8888"
HOMEDIR=`eval echo ~$USER`

#find nearest free port to listen
isfree=$(netstat -taln | grep $JUPYTER_PORT)
while [[ -n "$isfree" ]]; do
    JUPYTER_PORT=$[JUPYTER_PORT+1]
    isfree=$(netstat -taln | grep $JUPYTER_PORT)
done

# move into $HOME directory
cd $HOMEDIR
pwd
ls -la
ls -ls .jupyter
NBCONFIGFN=$HOMEDIR/.jupyter/jupyter_notebook_config.json
if [ ! -f ./.jupyter/jupyter_notebook_config.json ]; then
   echo "jupyter passwd reset!"
   mkdir -p .jupyter/
   #here you can commem=nt randomly generated password and set your password
   pass=`dd if=/dev/urandom count=1 2> /dev/null | uuencode -m - | sed -ne 2p | cut -c-12` ; echo $pass
   #pass="SecretPassWord"
   hash=`python -c "from notebook.auth import passwd ; hash = passwd('$pass') ; print(hash)" 2>/dev/null`
   cat > .jupyter/jupyter_notebook_config.json << EOJson
{
  "NotebookApp": {
      "password": "$hash"
    }
}
EOJson
  PASS_MESSAGE="Your password was set to '$pass' (without ticks)."
else
  PASS_MESSAGE="Your password was already set before."
fi

echo $NBCONFIGFN
cat $NBVONFIGFN
echo "hash="
echo $hash
echo "PORT=$JUPYTER_PORT"

# --port $JUPYTER_PORT

cd /webapps/piglegsurgery/ && \
    jupyter lab --no-browser --allow-root  2>&1 | \
   rotatelogs -n 1 /home/appuser/logs/piglegcv_jupyterlab_${DOCKERLOGNAME}_log.txt 1M &
#   tee >(rotatelogs -n 3 ../logs/jupyterlab_${DOCKERLOGNAME}_log_2.txt.bck 1M) | \
#    rotatelogs -n 1 ../logs/piglegcv_jupyterlab_${DOCKERLOGNAME}_log.txt 1M &
echo "jupyterlab started"

printf "Job with JupiterLab was started.\n\
Use URL  http://$HOSTNAME:$JUPYTER_PORT\n\
$PASS_MESSAGE\n\
You can reset password by deleting file $HOMEDIR/.jupyter/jupyter_notebook_config.json and run job again with this script.\n"
