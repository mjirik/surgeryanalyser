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
MOODLE_HOST = os.getenv('MOODLE_HOST', "moodle")
MOODLE_PORT = int(os.getenv('MOODLE_PORT', 80))
MOODLE_PORT = int(os.getenv('MOODLE_PORT', 8080))
MOODLE_WSTOKEN = os.getenv('MOODLE_WSTOKEN', "b1d806ecc878dc8221191d64500b1064")



def test_moodle_list():

    logger.debug("{}")
    logger.debug(f"{Path(__file__).absolute()}")
    logger.debug(f"{MOODLE_PORT=}")
    logger.debug(f"{MOODLE_HOST=}")
    logger.debug(f"{MOODLE_WSTOKEN=}")
    logger.debug(f"{MOODLE_WSTOKEN}")
    query = {
        "username": "user",
        "password": "bitnami",
        "service": "moodle_mobile_app",
    }

    response = requests.post(f'http://{MOODLE_HOST}:{MOODLE_PORT}/login/token.php?moodlewsrestformat=json', params=query)
    token = response.json()['token']
    logger.debug(f"{token=}")


    query = {
        "moodlewsrestformat": "json",
        # "wstoken": MOODLE_WSTOKEN,
        "wstoken": token,
        # "filename": "/webapps/piglegsurgery/tests/pigleg_test.mp4",
        "wsfunction": "core_course_get_contents",
        "courseid": 2,

    }
    response = requests.post(f'http://{MOODLE_HOST}:{MOODLE_PORT}/webservice/rest/server.php?moodlewsrestformat=json', params=query)
    response = response.json()
    logger.debug(response)
    query = {
        "moodlewsrestformat": "json",
        # "wstoken": MOODLE_WSTOKEN,
        "wstoken": token,
        # "filename": "/webapps/piglegsurgery/tests/pigleg_test.mp4",
        # "wsfunction": "mod_assign_get_assignments",
        # "wsfunction": "mod_assign_get_submission_status",
        # "wsfunction": "core_course_get_courses",
        # "wsfunction": "mod_assign_get_assignments",
        "wsfunction": "mod_assign_view_submission_status",

        # "courseid": 2,
        "assignid": 1,

    }
    response = requests.post(f'http://{MOODLE_HOST}:{MOODLE_PORT}/webservice/rest/server.php?moodlewsrestformat=json', params=query)
    response = response.json()
    logger.debug(response)
    # assert exists, "Test file should exist on web"



