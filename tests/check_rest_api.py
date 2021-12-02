import requests
import time
import glob
import json
import traceback
import subprocess
from loguru import logger


def check_rest_api(port=5001):
    input_file = '/webapps/piglegsurgery/piglegsurgeryweb/media/upload/20211112-211708_output_b2541fbdd906842b4774db61b095720180813cb1/output.mp4'
    outputdir = '/webapps/piglegsurgery/piglegsurgeryweb/media/test_20211112-211708_output_b2541fbdd906842b4774db61b095720180813cb1'
    query = {
        "filename": str(input_file),
        "outputdir": str(outputdir),
    }
    try:
        response = requests.post(f'http://127.0.0.1:{5001}/run', params=query)
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.debug("REST API processing not finished. Connection refused.")
        return
    logger.debug("Checking if processing is finished...")

    hash = response.json()

    finished = False
    while not finished:
        time.sleep(20)
        response = requests.get(f'http://127.0.0.1:{port}/is_finished/{hash}',
                                # params=query
                                )
        finished = response.json()
        logger.debug(f".    finished={finished}   hash={hash}")


    logger.debug(f"REST API processing finished.")


if __name__ == '__main__':
    check_rest_api()