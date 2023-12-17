import pytest
import requests
from loguru import logger
import time


def test_pigleg_cv_rest_api_exists():
    query = {"filename": "/webapps/piglegsurgery/tests/pigleg_test.mp4"}
    response = requests.post("http://127.0.0.1:5000/exists", params=query)
    # logger.debug(response)
    exists = response.json()
    assert exists, "Test file should exist on web"


def test_pigleg_cv_rest_api_exists():
    # # query = {'lat': '45', 'lon': '180'}
    # # # response = requests.get('http://api.open-notify.org/iss-pass.json', params=query)
    query = {
        "filename": "/webapps/piglegsurgery/tests/pigleg_test.mp4",
        "outputdir": "/webapps/piglegsurgery/tests/outputdir",
    }
    response = requests.post("http://127.0.0.1:5000/run", params=query)
    print(response)
    hash = response.json()
    print(hash)
    # query = {"job_key": hash}
    response = requests.get(
        f"http://127.0.0.1:5000/is_finished/{hash}",
        # params=query
    )
    finished = response.json()
    logger.debug(finished)
    assert finished == False, "File should be still in processing"

    while not finished:
        response = requests.get(
            f"http://127.0.0.1:5000/is_finished/{hash}",
            # params=query
        )
        finished = response.json()
        time.sleep(3)

    logger.debug("Finished")

    # print(response.json())
    # # response = requests.get("http://api.open-notify.org/astros.json")
