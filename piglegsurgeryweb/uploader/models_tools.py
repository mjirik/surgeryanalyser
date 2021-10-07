# /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import zipfile
from pathlib import Path
import random
import string
from . import models_tools

from django.conf import settings
from datetime import datetime
import os.path as op

from loguru import logger

import random

try:
    from hashlib import sha1 as sha_constructor
except ImportError:
    from django.utils.hashcompat import sha_constructor


def generate_sha1(string, salt=None):
    """
    Generates a sha1 hash for supplied string.

    :param string:
        The string that needs to be encrypted.

    :param salt:
        Optionally define your own salt. If none is supplied, will use a random
        string of 5 characters.

    :return: Tuple containing the salt and hash.

    """
    string = str(string)
    if not salt:
        salt = str(sha_constructor(str(random.random())).hexdigest()[:5])
    import hashlib

    # >> > sha = hashlib.sha256()
    # >> > sha.update('somestring'.encode())
    # >> > sha.hexdigest()
    hash = sha_constructor((salt + string).encode()).hexdigest()

    return hash


def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))


def get_output_dir():
    #
    # import datetime
    OUTPUT_DIRECTORY_PATH = settings.MEDIA_ROOT
    # datetimestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    datetimestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = op.join(
        op.expanduser(OUTPUT_DIRECTORY_PATH),
        "SA_" + datetimestr + "_" + models_tools.randomString(12),
        "SA_" + datetimestr,
    )
    return filename


def upload_to_unqiue_folder(instance, filename):

    """
    Uploads a file to an unique generated Path to keep the original filename
    """
    logger.debug("upload_to_unique_folder")
    logger.debug(instance)
    logger.debug(filename)
    logger.debug(instance.uploaded_at)
    hash = models_tools.generate_sha1(instance.uploaded_at, "_")

    # instance_filename = Path(instance.imagefile.path).stem # sometimes the instance.imagefile does not exist
    instance_filename = Path(filename).stem

    datetimestr = datetime.now().strftime("%Y%m%d-%H%M%S")

    return op.join(
        settings.UPLOAD_RELATIVE_PATH,
        datetimestr + "_" + instance_filename + "_" + hash,
        filename,
    )
