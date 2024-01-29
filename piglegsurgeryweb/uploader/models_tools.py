# /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import random
import secrets
import string
import zipfile
from datetime import datetime
from pathlib import Path

from django.conf import settings
from loguru import logger

from . import models_tools

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
        salt = str(sha_constructor(str(secrets.random())).hexdigest()[:5])
    import hashlib

    # >> > sha = hashlib.sha256()
    # >> > sha.update('somestring'.encode())
    # >> > sha.hexdigest()
    hash = sha_constructor((salt + string).encode()).hexdigest()

    return hash


def randomString(stringLength=16):
    alphabet = string.ascii_lowercase + string.digits
    # alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for i in range(stringLength))


# def get_outputdir_from_hash(hash:str):
#     OUTPUT_DIRECTORY_PATH = settings.MEDIA_ROOT
#
#     fnid = hash.split('_')
#     datetimestr = fnid[-2]
#     randomstring = fnid[-1]
#
#     filename = op.join(
#         op.expanduser(OUTPUT_DIRECTORY_PATH),
#         "SA_" + datetimestr + "_" + models_tools.randomString(12),
#         "SA_" + datetimestr,
#         )
#     return filename
#
# def get_hash_from_output_dir(filename):
#     hash = Path(filename).parent.name
#     return hash


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


