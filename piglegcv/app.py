import os

from rich.traceback import install

install(show_locals=True)

import time
from pathlib import Path

import flask
import loguru
import pigleg_cv

# except ImportError:
#    from .pigleg_cv import do_computer_vision
import requests
import rq.exceptions
from flask import jsonify, render_template, request
from loguru import logger

# from pigleg_cv import run_media_processing
# try:
from pigleg_cv import do_computer_vision
from rq import Queue
from rq.job import Job
from worker import conn
from rq import Worker

PIGLEGCV_TIMEOUT = 10 * 3600
app = flask.Flask(__name__)
q = Queue(connection=conn)


def make_bool_from_string(s: str) -> bool:
    if type(s) == bool:
        return s
    else:
        if s.lower() in ("true", "1"):
            return True
        else:
            return False


@app.route("/run", methods=["GET", "POST"])
def index():
    logger.debug("index in progress")
    # zjisti stav workerů
    workers = Worker.all(connection=conn)
    active_workers = [w for w in workers if w.state == 'busy']
    logger.debug(f"Total workers: {len(workers)}, Active workers: {len(active_workers)}")
    results = {}
    if request.method == "POST":
        # this import solves a rq bug which currently exists

        # get url that the person has entered
        # url = request.form['filename']
        logger.debug(request.form)
        logger.debug(f"{request.args=}")
        filename = request.args.get("filename")

        outputdir = request.args.get("outputdir")
        n_stitches = int(request.args.get("n_stitches"))
        is_microsurgery = make_bool_from_string(request.args.get("is_microsurgery"))
        force_tracking = make_bool_from_string(request.args.get("force_tracking", default=True))

        logger.debug(f"{n_stitches=}")
        logger.debug(f"{is_microsurgery=}, {type(is_microsurgery)=}")

        # if not url[:8].startswith(('https://', 'http://')):
        #     url = 'http://' + url

        # time.sleep(10)
        if not Path(filename).exists():
            logger.debug(f"File does not exist. filename={filename}")
            return jsonify({"error": "File does not exists."})

        meta = {}
        job = q.enqueue_call(
            func=do_computer_vision,
            args=(filename, outputdir, meta, is_microsurgery, n_stitches, force_tracking),
            result_ttl=5000,
            timeout=PIGLEGCV_TIMEOUT,
        )
        job_id = job.get_id()
        logger.debug(f"Job enqueued, job_id={job_id}")
        return jsonify(job_id)
        # return jsonify("Ok")
    return jsonify({})  # "Ok", 100

    # return render_template('index.html', results=results)
    # return
    # yield promise


@app.route("/exists", methods=["GET", "POST"])
def exists():
    if request.method == "POST":
        filename = request.args.get("filename")
        exists = Path(filename).exists()
        logger.debug(f"exists={exists}")
        return jsonify(exists)
        # return jsonify({"exists": exists})
    return jsonify({})


@app.route("/is_finished/<job_key>", methods=["GET"])
def get_results(job_key):
    logger.debug(job_key)

    try:
        job = Job.fetch(job_key, connection=conn)
    except rq.exceptions.NoSuchJobError as e:
        logger.debug(f"Job not found. Job ID={job_key}")
        return jsonify(f"Job not found.")
    logger.debug(
        f"   job.is_finished={job.is_finished}, len(q)={len(q)}, progress={pigleg_cv.PROGRESS}"
    )

    return jsonify(job.is_finished)
    # if job.is_finished:
    #     return str(job.result), 200
    # else:
    #     return "Nay!", 202


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
