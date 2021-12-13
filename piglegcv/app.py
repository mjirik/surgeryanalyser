import os
import rq.exceptions
from rq import Queue
from rq.job import Job

from worker import conn
from pathlib import Path
from loguru import logger
import flask
from flask import request, jsonify, render_template
import time
from pigleg_cv import run_media_processing
from run_tracker_lite import main_tracker
from run_mmpose import main_mmpose
from run_qr import main_qr
from run_report import main_report
from run_perpendicular import main_perpendicular
import requests

PIGLEGCV_TIMEOUT = 5*3600
app = flask.Flask(__name__)
q = Queue(connection=conn)


def do_computer_vision(filename, outputdir):
    logger.debug(f"working on {filename}, outputdir={outputdir}")

    logger.debug("CV processing start ...")

    images_types = [".jpg", ".png", ".bmp", ".jpeg"]
    video_types = [".mp4", ".mov"]
    root, extention = os.path.splitext(sys.argv[1])
    extention = extention.lower()
    #print(extention)

    if extention in video_types:
        main_tracker("./.cache/tracker_model \"{}\" --output_dir {}".format(filename, outputdir))
        #run_media_processing(Path(filename), Path(outputdir))
        logger.debug("Detectron finished.")

        main_mmpose(filename, outputdir)
        logger.debug("MMpose finished.")

        main_qr(filename, outputdir)
        logger.debug("QR finished.")

        main_report(filename, outputdir)

    if extention in images_types:
        main_perpendicular(filename, outputdir)
    
    logger.debug("Work finished")

@app.route("/run", methods=["GET", "POST"])
def index():
    logger.debug("index in progress")
    results = {}
    if request.method == "POST":
        # this import solves a rq bug which currently exists
        from app import do_computer_vision

        # get url that the person has entered
        # url = request.form['filename']
        logger.debug(request.form)
        logger.debug(request.args)
        filename = request.args.get("filename")

        outputdir = request.args.get("outputdir")
        # if not url[:8].startswith(('https://', 'http://')):
        #     url = 'http://' + url

        # time.sleep(10)
        if not Path(filename).exists():
            logger.debug(f"File does not exist. filename={filename}")
            return jsonify({"error": "File does not exists."})

        job = q.enqueue_call(
            func=do_computer_vision, args=(filename, outputdir), result_ttl=5000,
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
    logger.debug(f"   job.is_finished={job.is_finished}, queue length = {len(q)}")

    return jsonify(job.is_finished)
    # if job.is_finished:
    #     return str(job.result), 200
    # else:
    #     return "Nay!", 202


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
