import pytest
import requests
from loguru import logger


def test_pigleg_cv_rest_api():
    # query = {"filename": "/webapps/piglegsurgeryweb/tests/pigleg_test.mp4"}
    # response = requests.post('http://127.0.0.1:5000/exists', params=query)
    # logger.debug(response)

    # query = {'lat': '45', 'lon': '180'}
    # # response = requests.get('http://api.open-notify.org/iss-pass.json', params=query)
    query = {"filename": "/webapps/piglegsurgeryweb/tests/pigleg_test.mp4lk", "outputdir": "/webapps/piglegsurgeryweb/tests/outputdir"}
    response = requests.post('http://127.0.0.1:5000/run', params=query)
    print(response.json())
    # response = requests.get("http://api.open-notify.org/astros.json")
