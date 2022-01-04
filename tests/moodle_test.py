import requests
import time
import glob
import json
import traceback
import subprocess
from loguru import logger
from pathlib import Path
import os
MOODLE_HOST = os.getenv('MOODLE_HOST', "localhost")
MOODLE_PORT = int(os.getenv('MOODLE_PORT', 80))
MOODLE_WSTOKEN = os.getenv('MOODLE_WSTOKEN', "b1d806ecc878dc8221191d64500b1064")



def test_moodle_list():

    logger.debug("{}")
    logger.debug(f"{Path(__file__).absolute()}")
    logger.debug(f"{MOODLE_PORT=}")
    logger.debug(f"{MOODLE_HOST=}")
    query = {
        "moodlewsrestformat": "json",
        "wstoken": MOODLE_WSTOKEN,
        # "filename": "/webapps/piglegsurgery/tests/pigleg_test.mp4",
        "wsfunction": "core_course_get_contents",
        "courseid": 2,

    }
    response = requests.post(f'http://{MOODLE_HOST}:{MOODLE_PORT}/webservice/rest/server.php?moodlewsrestformat=json', params=query)
    exists = response.json()
    logger.debug(exists)
    # assert exists, "Test file should exist on web"



