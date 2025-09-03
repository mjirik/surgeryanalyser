import glob
import json
import os
import os.path as op
import shutil
import subprocess
import time
import traceback
import math
import plotly
import plotly.express as px

# from .pigleg_cv import run_media_processing
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple

import django.utils
import gspread
import loguru
import numpy as np
import pandas as pd
import requests
import django.conf
from django.conf import settings
from django.core.mail import EmailMessage, send_mail
from django.template import defaultfilters
from django.shortcuts import reverse
from django.utils.html import strip_tags
from django_q.models import Schedule
from django.shortcuts import get_object_or_404
from django_q.tasks import async_task, queue_size, schedule
from loguru import logger
from oauth2client.service_account import ServiceAccountCredentials

from .data_tools import (
    flatten_dict,
    google_spreadsheet_append,
    remove_empty_lists,
    remove_iterables_from_dict,
    xlsx_spreadsheet_append,
)
from .media_tools import convert_avi_to_mp4, make_images_from_video, rescale, crop_square
from .models import BitmapImage, UploadedFile, Owner
from . import visualization_tools, models
from .report_tools import set_overall_score


def _run_media_processing_and_wait_for_rest_api(
    input_file: Path,
    outputdir: Path,
    is_microsurgery: bool,
    n_stitches: int,
    hostname="127.0.0.1",
    port=5000,
    force_tracking=False,
):

    # query = {"filename": "/webapps/piglegsurgery/tests/pigleg_test.mp4", "outputdir": "/webapps/piglegsurgery/tests/outputdir"}
    logger.debug("Creating request for processing")
    logger.debug(f"hostname={hostname}, port={port}")
    query = {
        "filename": str(input_file),
        "outputdir": str(outputdir),
        "is_microsurgery": is_microsurgery,
        "n_stitches": n_stitches,
        "force_tracking": force_tracking,  # always force tracking
    }
    url = f"http://{hostname}:{port}/run"
    try:
        response = requests.post(url, params=query)
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.debug(f"REST API processing not finished. Connection refused. url={url}")
        return
    logger.debug("Checking if processing is finished...")

    hash = response.json()
    is_finished = False
    tm = 0
    time_to_sleep = 4
    while not is_finished:
        time_to_sleep = time_to_sleep * 2 if time_to_sleep < 16 else 16
        time_step = 4
        tm += time_to_sleep
        for i in range(int(time_to_sleep / time_step)):
            time.sleep(time_step)
        response = requests.get(
            f"http://{hostname}:{port}/is_finished/{hash}",
            # params=query
        )
        is_finished = response.json()
        logger.debug(
            f".    is_finished={is_finished}  input_file={input_file.name}  time[s]={tm} queue_size={queue_size()}"
        )

    if type(is_finished) == str:
        logger.warning(
            f"REST API processing failed. input_file={input_file.name}  time[s]={tm}"
        )
    else:
        logger.debug(f"REST API processing finished.")

def make_it_run_with_async_task(
        serverfile: UploadedFile,
        absolute_uri: str, hostname: str, port: int,
        send_email:bool=False,
        force_tracking: bool=False,
) -> UploadedFile:
    kwargs = {}
    if send_email:
        # hook="uploader.tasks.email_report_from_task",
        kwargs["hook"] = "uploader.tasks.email_report_from_task"
        logger.debug(f"Sending email to {absolute_uri} added to planned task.")
    serverfile.enqueued_at = django.utils.timezone.now()
    serverfile.finished_at = None
    serverfile.started_at = None
    serverfile.processing_ok = False
    serverfile.processing_message = "Not finished yet."
    serverfile.save(update_fields=["enqueued_at", "finished_at", "processing_ok", "processing_message", "started_at"])
    logger.debug(f"hostname={hostname}, port={port}")
    if serverfile.category and serverfile.category.name == "Other":
        serverfile.finished_at = django.utils.timezone.now()
        serverfile.processing_ok = True
        serverfile.processing_message = "Other category: Processing skipped."
        serverfile.save(update_fields=["finished_at","processing_ok", "processing_message"])
        email_report(serverfile, absolute_uri)

    else:
        async_task(
            "uploader.tasks.run_processing",
            serverfile,
            absolute_uri,
            hostname,
            int(port),
            bool(force_tracking),
            timeout=settings.PIGLEGCV_TIMEOUT,
            **kwargs,
            # hook="uploader.tasks.email_report_from_task",
        )
    return serverfile

def run_processing(
        serverfile: UploadedFile,
        absolute_uri,
        hostname,
        port,
        force_tracking:bool):
    outputdir = Path(serverfile.outputdir)
    logger.debug(f"outputdir={outputdir}")

    # delete outputdir but keep tracks.json
    if outputdir.exists() and outputdir.is_dir():
        # get temp dir
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tempdir = Path(settings.MEDIA_ROOT) / f"temp_{datetime_str}_{str(serverfile.hash[:6])}"
        logger.debug(f"tempdir={tempdir}, {tempdir.exists()}")
        tempdir.mkdir(parents=True, exist_ok=True)

        files_to_keep = list(outputdir.glob("tracks.json")) + list(outputdir.glob("annotation_*.json"))
        logger.debug(f"{files_to_keep=}")
        for fn in files_to_keep:
            shutil.move(fn, tempdir)

        # tracks_json_path = outputdir / "tracks.json"
        # tracks_json_tmp_path = tempdir.parent / "tracks.json.tmp"
        # tracks_json_tmp_path.unlink(missing_ok=True)
        # if tracks_json_path.exists():
        #     shutil.move(tracks_json_path, tracks_json_tmp_path)
        shutil.rmtree(outputdir, ignore_errors=True)
        outputdir.mkdir(parents=True, exist_ok=True)
        # if tracks_json_tmp_path.exists():
        #     outputdir.mkdir(parents=True, exist_ok=True)
        #     shutil.move(tracks_json_tmp_path, tracks_json_path)

        for fn in tempdir.glob("*"):
            shutil.move(fn, outputdir)
        shutil.rmtree(tempdir, ignore_errors=True)
    else:
        if outputdir.exists(): # then it is a file
            outputdir.unlink()
        outputdir.mkdir(parents=True, exist_ok=True)
    log_format = loguru._defaults.LOGURU_FORMAT
    logger.debug(f"outputdir={outputdir}, {outputdir.exists()=}")
    save_annotations_to_json(serverfile)
    logger_id = logger.add(
        str(Path(serverfile.outputdir) / "webapp_log.txt"),
        format=log_format,
        level="DEBUG",
        rotation="1 week",
        backtrace=True,
        diagnose=True,
    )
    logger.debug(f"Image processing of '{serverfile.mediafile}' initiated")
    try:
        make_preview(serverfile)
        if serverfile.zip_file and Path(serverfile.zip_file.path).exists():
            serverfile.zip_file.delete()
        input_file = Path(serverfile.mediafile.path)
        logger.debug(f"input_file={input_file}")
        outputdir = Path(serverfile.outputdir)
        logger.debug(f"outputdir={outputdir}")

        _run_media_processing_and_wait_for_rest_api(
            input_file,
            outputdir,
            serverfile.is_microsurgery,
            int(serverfile.stitch_count),
            hostname,
            port,
            force_tracking=force_tracking,
        )
        logger.debug("Adding records to the database...")

        # (outputdir / "empty.txt").touch(exist_ok=True)

        if input_file.suffix in (".mp4", ".avi", ".mov", ".webm"):
            make_images_from_video(input_file, outputdir=outputdir, n_frames=1)

        # for video_pth in outputdir.glob("*.avi"):
        #     input_video_file = video_pth
        #     output_video_file = video_pth.with_suffix(".mp4")
        #     logger.debug(f"input_video_file={input_video_file}")
        #     logger.debug(f"outout_video_file={output_video_file}")
        #     if output_video_file.exists():
        #         output_video_file.unlink()
        #     _convert_avi_to_mp4(str(input_video_file), str(output_video_file))
        logger.debug("Adding generated images to the database...")
        add_generated_images(serverfile)


        # _add_row_to_spreadsheet(serverfile, absolute_uri)
        _add_rows_to_spreadsheet_for_each_annotation(serverfile, absolute_uri)

        logger.debug("Making graphs...")
        _make_graphs(serverfile)
        set_overall_score(serverfile)

        logger.debug("Updating status...")
        is_ok, status = add_status_to_uploaded_file(serverfile)

        logger.debug("Making zip file...")
        make_zip(serverfile)
        if is_ok:
            logger.debug("Processing finished in API")
        else:
            logger.error("Processing in piglegcv failed: " + status)
        logger.remove(logger_id)
        if not is_ok:
            # error in piglegcv
            # raise exception to be caught by the sentry
            raise Exception("Processing in piglegcv failed: " + status)
    except Exception as err:
        logger.error(err)
        logger.error(traceback.format_exc())
        add_status_to_uploaded_file(serverfile, ok=False, status="Webapp error: " + str(err))
        logger.error("Processing finished in API with error")
        logger.remove(logger_id)
        raise err


def get_graph_path_for_report(serverfile: UploadedFile, stitch_id: Optional[int] = None):
    if stitch_id is not None:
        html_path = Path(serverfile.outputdir) / f"report_graph_{stitch_id}.html"
    else:
        html_path = Path(serverfile.outputdir) / "report_graph.html"
    return html_path

def get_normalization_path():
    normalization_path = settings.MEDIA_ROOT / "generated/normalization.json"
    return normalization_path

def _make_graphs(uploadedfile: UploadedFile):
    # one graph with records of owner
    html_path = make_graph_for_owner(uploadedfile.owner)
    _make_metrics_for_report(uploadedfile)

def update_normalization():
    """
    Update normalization file. The normalization file is used for making graphs.
    """
    logger.debug("Updating normalization...")
    normalization_path = get_normalization_path()
    visualization_tools.calculate_normalization(
        settings.MEDIA_ROOT,
        normalization_path=normalization_path,
    )
    logger.debug("Normalization updated")


def _make_metrics_for_report(uploadedfile: UploadedFile):
    """
    Make metrics for report. The metrics are saved to the database.
    """
    logger.debug("Making metrics for report...")


    # one graph for one report of the owner
    normalization_path = get_normalization_path()
    if not normalization_path.exists():
        update_normalization()

    with open(normalization_path) as f:
        normalization = json.load(f)

    loaded_results = visualization_tools.read_one_result(uploadedfile.outputdir)
    if loaded_results is None:
        logger.warning(f"cannot generate graph for {uploadedfile.outputdir}")
        return

    if "Needle holder stitch 0 visibility [%]" in loaded_results:
        cols = [
            "Needle holder stitch 0 length [m]",
            "Needle holder stitch 0 visibility [s]",
            "Needle holder stitch 0 visibility [%]",
            "Needle holder stitch 0 area presence [%]",
            "Forceps stitch 0 visibility [%]",
            "Left hand bbox stitch 0 visibility [%]",
            "Right hand bbox stitch 0 visibility [%]",
            "stitch 0 duration [s]",
            "Stitches linearity score [%]",
            "Stitches parallelism score [%]",
            "Stitches perpendicular score [%]",
        ]
    else:
        cols = [
            "Needle holder length [m]",
            "Needle holder visibility [s]",
            "Needle holder visibility [%]",
            "Needle holder area presence [%]",
            "Forceps visibility [%]",
            "Left hand bbox visibility [%]",
            "Right hand bbox visibility [%]",
            "all duration [s]",
            "Stitches linearity score [%]",
            "Stitches parallelism score [%]",
            "Stitches perpendicular score [%]",
        ]

    report_graph_html_path = get_graph_path_for_report(uploadedfile)
    report_graph_html_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_tools.make_plot_with_metric(
        loaded_results, normalization, cols=cols, filename=report_graph_html_path)


def make_graph_for_owner(owner:Owner):
    # owner = get_object_or_404(Owner, hash=owner_hash)
    # order_by = request.GET.get("order_by", "-uploaded_at")
    files = UploadedFile.objects.filter(owner=owner)
    return make_graph(files, owner)

    

def get_graph_path_for_owner(owner: Optional[Owner] = None):
    if owner:
        html_path = Path(settings.MEDIA_ROOT) / "generated" / owner.hash / "graph.html"
    else:
        html_path = Path(settings.MEDIA_ROOT) / "generated/graph.html"
    return html_path


def make_graph(
    uploaded_file_set: UploadedFile.objects.all(), owner: Optional[Owner] = None,
):
    import pandas as pd

    html_path = get_graph_path_for_owner(owner)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, uploaded_file in enumerate(uploaded_file_set):

        results_path = Path(uploaded_file.outputdir) / "results.json"
        # read results.json
        if results_path.exists():
            with open(results_path) as f:
                loaded_results = json.load(f)
            # fix typo
            if "Stichtes linearity score" in loaded_results:
                loaded_results["Stitches linearity score"] = loaded_results.pop("Stichtes linearity score")
            loaded_results["Uploaded at"] = uploaded_file.uploaded_at
            loaded_results["i"] = i
            rows.append(loaded_results)

    df = pd.DataFrame(rows)


    if "Stitches linearity score" in df.keys():
        df["Stitches linearity score [%]"] = df["Stitches linearity score"] * 100

    if "Stitches parallelism score" in df.keys():
        df["Stitches parallelism score [%]"] = df["Stitches parallelism score"] * 100

    y = [
        "Needle holder visibility [%]",
        "Needle holder area presence [%]",
        "Forceps visibility [%]",
        "Forceps area presence [%]",
        "Stitches linearity score [%]",
        "Stitches parallelism score [%]",
    ]

    y = [element for element in y if element in df.keys()]
    if len(y) == 0:
        return None
    # x = list(df.keys())
    #
    # x = [el for el in x if el != 'Uploaded at']

    x = "Uploaded at"

    fig = px.scatter(
        df,
        x=x,
        y=y,
        # marginal_x="box",
        # marginal_y="box"
    )
    fig.write_html(html_path, full_html=False)
    return html_path

def add_rows_to_spreadsheet_and_update_zips(uploaded_files, absolute_uri):
    for uploaded_file in uploaded_files:
        add_row_to_spreadsheet_and_update_zip(uploaded_file, absolute_uri, ith_annotation=None)
        # sleep (to avoid too many requests)
        time.sleep(20)


    

def add_row_to_spreadsheet_and_update_zip(serverfile: UploadedFile, absolute_uri, ith_annotation:Optional[int]):
    """Add row to spreadsheet and update zip file. If ith_annotation is not None, add row for each annotation."""
    logger.debug("Updating spreadsheet...")
    if ith_annotation is None:
        _add_rows_to_spreadsheet_for_each_annotation(serverfile, absolute_uri)
    else:
        _add_row_to_spreadsheet(serverfile, absolute_uri, ith_annotation=ith_annotation)
    # logger.debug("Spreadsheet updated")
    # if serverfile.zip_file and Path(serverfile.zip_file.path).exists():
    #     serverfile.zip_file.delete()
    # make_zip(serverfile)
    logger.debug("Zip updated")

def update_started_at_from_log(serverfile: UploadedFile) -> Tuple[Optional[datetime], bool]:
    piglegcv_log_path = Path(serverfile.outputdir) / "piglegcv_log.txt"
    if not serverfile.started_at:
        if piglegcv_log_path.exists():
            with open(piglegcv_log_path, "r") as fr:
                lines = fr.readlines()
                started_at = _get_started_at_from_log_lines(lines)
                if started_at:
                    serverfile.started_at = started_at
                    serverfile.save(update_fields=["started_at"])
                    logger.debug(f"Started at set to {started_at} for {serverfile}")


def _get_started_at_from_log_lines(lines: list[str]) -> Optional[datetime]:
    started_at = None
    for line in lines:
        line = line.strip()
        if line:
            try:
                # vezme prvních 23 znaků: "2025-07-31 13:41:32.787"
                timestamp_str = line[:23]
                started_at = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                started_at = django.utils.timezone.make_aware(started_at)
                break
            except ValueError:
                logger.warning(f"Could not parse time in first log line: {line}")
    return started_at


def add_status_to_uploaded_file(serverfile:UploadedFile, ok:Optional[bool]=None, status:Optional[str]=None):
    """
    Find status in piglegcv log file. Status is the last line of the file.
    If line contain "Work finished", everything is OK. If line contain error,
    the processing failed.
    """
    piglegcv_log_path = Path(serverfile.outputdir) / "piglegcv_log.txt"
    started_at=None
    is_ok = False
    if ok is not None:
        is_ok = ok
        if status is None:
            logger.warning(f"No status provided for {piglegcv_log_path}")
    elif not piglegcv_log_path.exists():
        is_ok = False
        status = "Log file does not exist."
    else:
        with open(piglegcv_log_path, "r") as fr:
            lines = fr.readlines()
            if len(lines) == 0:
                is_ok = False
                status = "Log file is empty."
                return is_ok, status
            # find last not empty line
            for last_line in reversed(lines):
                if last_line.strip():
                    break
            started_at = _get_started_at_from_log_lines(lines)
            # last_line = lines[-1]
            if "Work finished" in last_line:
                is_ok = True
                status = "Work finished."
            else:
                is_ok = False
                status = f"Last log message: {last_line}"
    # read first line of log
    serverfile.processing_ok = is_ok
    serverfile.processing_message = status
    serverfile.finished_at = django.utils.timezone.now()
    serverfile.started_at = started_at
    serverfile.save(update_fields=["processing_ok", "processing_message", "finished_at", "started_at"])


    return is_ok, status

def _add_rows_to_spreadsheet_for_each_annotation(serverfile: UploadedFile, absolute_uri):
    if len(serverfile.mediafileannotation_set.all()) == 0:
        _add_row_to_spreadsheet(serverfile, absolute_uri, ith_annotation=0)
    else:

        for i in range(len(serverfile.mediafileannotation_set.all())):
            _add_row_to_spreadsheet(serverfile, absolute_uri, ith_annotation=i)


def _add_row_to_spreadsheet(serverfile, absolute_uri, ith_annotation=0):

    creds_file = Path(django.conf.settings.CREDS_JSON_FILE)  # 'piglegsurgery-1987db83b363.json'
    if not creds_file.exists():
        logger.error(f"Credetials file does not exist. Expected path: {creds_file}")
        return
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)

    new_data_row = {}

    filename = Path(serverfile.outputdir) / "meta.json"
    if os.path.isfile(filename):
        with open(filename, "r") as fr:
            data = json.load(fr)
            new_data_row.update(data)

    # filename = Path(serverfile.outputdir) / "evaluation.json"
    filename = Path(serverfile.outputdir) / "results.json"
    if os.path.isfile(filename):
        with open(filename, "r") as fr:
            data = json.load(fr)
            new_data_row.update(data)

    new_data_row.update(
        {
            "email": serverfile.email,
            "filename_full": str(Path(serverfile.mediafile.name)),
            # return str(Path(self.mediafile.name).name)
            "filename": str(Path(serverfile.mediafile.name).name),
            # "uploaded_at": None if serverfile.uploaded_at is None else serverfile.uploaded_at.strftime('%Y-%m-%d %H:%M:%S'),
            # "finished_at": None if serverfile.finished_at is None else serverfile.finished_at.strftime('%Y-%m-%d %H:%M:%S'),
            "uploaded_at": None
            if serverfile.uploaded_at is None
            else defaultfilters.date(serverfile.uploaded_at, "Y-m-d H:i"),
            "finished_at": None
            if serverfile.finished_at is None
            else defaultfilters.date(serverfile.finished_at, "Y-m-d H:i"),
            "filename_full": serverfile.mediafile.name,
            "report_url": f"{absolute_uri}/uploader/web_report/{serverfile.hash}",
            "processing_ok": serverfile.processing_ok,
            "processing_message": serverfile.processing_message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "collections": [str(collection) for collection in serverfile.collection_set.all()],
            "category": str(serverfile.category),
        }
    )

    annotation_set = serverfile.mediafileannotation_set.all()
    if len(annotation_set) > 0:
        annotation = annotation_set[ith_annotation]
    else:
        annotation = None


    if annotation:
        # go over all annotation fields and add them to the dictionary
        ann = {}
        for field in annotation._meta.fields:
            value = getattr(annotation, field.name)
            # if type is TimestampField, convert to string
            if isinstance(value, datetime):
                value = defaultfilters.date(value, "Y-m-d H:i")
                ann[field.name] = value
            elif isinstance(value, str):
                ann[field.name] = value
            elif isinstance(value, int):
                ann[field.name] = value
            elif isinstance(value, float):
                ann[field.name] = value
            elif isinstance(value, bool):
                ann[field.name] = value
            elif isinstance(value, UploadedFile):
                continue
            elif isinstance(value, Owner):
                ann[field.name] = str(value)

        ann["i"] = ith_annotation

        new_data_row.update(
            {
                "annotation":{
                    'annotation': ann,
                    #     {
                    #     "id": int(annotation.id),
                    #     "annotation": str(annotation.annotation),
                    #     "stars": int(annotation.stars) if annotation.stars is not None else -1,
                    #     "annotator": str(annotation.annotator) if annotation.annotator is not None else "",
                    #     "updated_at": str(annotation.updated_at),
                    #
                    # }

                }
            }
        )

    pop_from_dict(new_data_row, "incision_bboxes")
    pop_from_dict(new_data_row, "filename_full")
    new_data_row = remove_empty_lists(flatten_dict(new_data_row))
    pop_from_dict(new_data_row, "qr_data_box")

    df_novy = None
    try:
        # remove NaN values from new_data_row, probably this will affect the spreadsheet serialization
        new_data_row = clean_data_for_json(new_data_row)
        serverfile.data_row = new_data_row
        serverfile.save()
        # novy = remove_iterables_from_dict(novy)
        # logger.debug(f"novy={novy}")
        df_novy = pd.DataFrame(new_data_row, index=[0])

    except Exception as e:
        logger.error(f"Error saving data_row preparation: {str(e)}")
        logger.error(traceback.format_exc())
        logger.debug(f"new_data_row={new_data_row}")
    if df_novy is not None:
        try:
            # save to xlsx to media dir
            xlsx_spreadsheet_path = django.conf.settings.XLSX_SPREADSHEET_PATH
            xlsx_spreadsheet_append(df_novy, xlsx_spreadsheet_path)
            # xlsx_spjson_path = Path(serverfile.outputdir) / "report.xlsx"
            # save to local xlsx file
            xlsx_spreadsheet_append(df_novy, Path(serverfile.outputdir) / "report.xlsx")
        except Exception as e:
            logger.error(f"Error saving data_row to XLSX: {str(e)}")
            logger.error(traceback.format_exc())
            logger.debug(f"new_data_row={new_data_row}")

        try:
            google_spreadsheet_append(title="Pigleg Surgery Stats", creds=creds, data=df_novy)
        except Exception as e:
            logger.error(f"Error saving data_row to XLSX: {str(e)}")
            logger.error(traceback.format_exc())
            logger.debug(f"new_data_row={new_data_row}")



def clean_data_for_json(obj):
    """Remove NaN values and clean data for JSON serialization."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: clean_data_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data_for_json(v) for v in obj]
    else:
        return obj


def pop_from_dict(d, key):
    if key in d:
        d.pop(key)

def update_all_uploaded_files():
    files = UploadedFile.objects.all()
    logger.info("update all uploaded files")
    for file in files:
        make_preview(file, force=True)
        update_owner(file)
        add_status_to_uploaded_file(file)

def update_owner(uploadedfile: UploadedFile) -> Owner:
    if not uploadedfile.owner:
        owners = Owner.objects.filter(email=uploadedfile.email)
        if len(owners) == 0:
            owner = Owner(email=uploadedfile.email, hash=models._hash())
            owner.save()
            # create one
        else:
            owner = owners[0]
    else:
        owner = uploadedfile.owner

    # uploadedfiles = UploadedFile.objects.filter(owner=owner)
    if uploadedfile.owner != owner:
        uploadedfile.owner = owner
        uploadedfile.save()

    return owner

def make_preview(
    serverfile: UploadedFile, force: bool = False, height=100, make_square: bool = True
) -> Path:
    if serverfile.mediafile:
        input_file = Path(serverfile.mediafile.path)
        # if not input_file.exists():
        #     return
        filename = input_file.parent / (input_file.stem + ".preview.jpg")
        filename_rel = filename.relative_to(django.conf.settings.MEDIA_ROOT)
        # logger.debug(f"  {input_file=}")
        # logger.debug(f"    {filename=}")
        # logger.debug(f"{filename_rel=}")
        if (not filename.exists()) or force:
            if input_file.suffix.lower() in (".mp4", ".avi", ".mov", ".webm"):
                fn = serverfile.mediafile
                make_images_from_video(
                    input_file,
                    outputdir=input_file.parent,
                    n_frames=1,
                    # filemask="{outputdir}/preview.jpg",
                    filemask=str(filename),
                    # scale=0.125,
                    height=height,
                    make_square=make_square,
                )
            elif input_file.suffix.lower() in (
                ".jpg",
                ".jpeg",
                ".tiff",
                ".tif",
                ".png",
            ):


                import cv2

                # print(input_file)
                frame = cv2.imread(str(input_file))
                scale = height / frame.shape[0]
                frame = rescale(frame, scale)
                if make_square:
                    frame = crop_square(frame)
                cv2.imwrite(str(filename), frame)
            else:
                logger.warning(
                    f"Preview generation skipped. Unknown file type. filename={str(input_file.name)}"
                )
                return

            serverfile.preview.name = str(filename_rel)
            serverfile.save()

# not used any more
def on_finished_run_processing(task):

    email_report_from_task(task)
    serverfile: UploadedFile = task.args[0]



def email_report_from_task(task):

    logger.debug("getting parameters from task for email")
    serverfile: UploadedFile = task.args[0]
    # absolute uri is http://127.0.0.1:8000/. We have to remove last '/' because the url already contains it.
    absolute_uri = task.args[1][:-1]
    # logger.debug(dir(task))
    email_report(serverfile, absolute_uri)


def email_report(serverfile: UploadedFile, absolute_uri: str):
    logger.debug("Sending email report...")
    logger.debug(f"absolute_uri={absolute_uri}, {type(absolute_uri)}")
    logger.debug(f"serverfile={serverfile}, {type(serverfile)}")
    if absolute_uri[-1] == "/":
        pass
    else:
        absolute_uri = absolute_uri + "/"
    html_message = (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        "<head> \n"
        '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />'
        "<title>Order received</title>"
        '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>'
        "</head>"
        f"<body>"
        f"<p>Report processing finished.</p><p>Email: {serverfile.email}</p><p>Filename: {str(Path(str(serverfile.mediafile.name)).name)}</p>"
        f"<p></p>"
        f'<p> <a href="{absolute_uri}uploader/web_report/{serverfile.hash}">Check report here</a> .</p>\n'
        f"<p></p>"
        f"<p></p>"
        f'<p> <a href="{absolute_uri}uploader/owners_reports/{serverfile.owner.hash}">See all your reports here</a> .</p>\n'
        f"<p></p>"
        f"<p></p>"
        f'<p> <a href="{absolute_uri}uploader/assigned_to/{serverfile.owner.hash}">See all your assignments</a> .</p>\n'
        f"<p></p>"
        f"<p></p>"
        f'<p> <a href="{absolute_uri}uploader/go_to_video_for_annotation/{serverfile.owner.hash}">You can also do a review</a> .</p>\n'
        f"<p></p>"
        f"<p>Best regards</p>\n"
        f"<p>Miroslav Jirik</p>\n"
        f"<p></p>"
        "<p>Faculty of Applied Sciences</p\n"
        "<p>University of West Bohemia</p>\n"
        "<p>Pilsen, Czech Republic</p>\n"
        "<p>mjirik@kky.zcu.cz</p>\n"
        f"</body></html>"
    )

    # f'<p> <a href="{absolute_uri}{serverfile.zip_file.url}">Download report here</a> .</p>\n'

    # f'http://127.0.0.1:8000/{request.buld_absolute_uri(serverfile.zip_file.url)}' \
    # logger.debug(f"email_text={html_message}")
    subject = "Pig Leg Surgery Analyser: Report"
    from_email = "mjirik@kky.zcu.cz"
    # to_email = "mjirik@gapps.zcu.cz"
    to_email = serverfile.email

    # async_task('django.core.mail.send_mail',
    # message = EmailMessage(subject, html_message, from_email, [to_email])
    # message.content_subtype = 'html'  # this is required because there is no plain text email message
    # message.send()
    send_mail(
        subject,
        html_message,
        from_email,
        [to_email],
        fail_silently=False,
        html_message=html_message,
    )
    sent_at = django.utils.timezone.now()
    logger.debug(f"{sent_at=}")
    serverfile.email_sent_at = sent_at
    serverfile.save(update_fields=["email_sent_at"])
    logger.debug("Email sent.")
    # send_mail(
    #     "[Pig Leg Surgery]",
    #     html_message,
    #     "mjirik@kky.zcu.cz",
    #     ["miroslav.jirik@gmail.com"],
    #     fail_silently=False,
    # )


def email_media_recived(serverfile: UploadedFile, absolute_uri: str):
    # async_task('django.core.mail.send_mail',
    review_url = reverse("uploader:go_to_video_for_annotation_email", kwargs={
        "annotator_hash": serverfile.owner.hash
    }),
    if absolute_uri[-1] == "/":
        pass
    else:
        absolute_uri = absolute_uri + "/"
    html_message = (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        "<head> \n"
        '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />'
        "<title>Media recived</title>"
        '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>'
        "</head>"
        f"<body>"
        f"<p>Thank you for uploading a file.</p>"
        f"<p>We will let you know when the processing will be finished.</p>"
        f"<p>Meanwhile you can "
        f"<a href='{review_url}'>review other student's video</a> .</p>\n"
        f"<p></p>"
        f"<p></p>"
        f"</p><p>Email: {serverfile.email}</p><p>Filename: {str(Path(str(serverfile.mediafile.name)).name)}</p>"
        f"<p></p>"
        f"<p></p>"
        f'<p> <a href="{absolute_uri}uploader/owners_reports/{serverfile.owner.hash}">See all your reports here</a> .</p>\n'
        f"<p></p>"
        f"<p></p>"
        f'<p> <a href="{absolute_uri}uploader/assigned_to/{serverfile.owner.hash}">See all your assignments</a> .</p>\n'
        f"<p></p>"
        f"<p></p>"
        f'<p> <a href="{absolute_uri}uploader/go_to_video_for_annotation/{serverfile.owner.hash}">You can also do a review</a> .</p>\n'
        f"<p></p>"
        f"<p>Best regards</p>\n"
        f"<p>Miroslav Jirik</p>\n"
        f"<p></p>"
        "<p>Faculty of Applied Sciences</p\n"
        "<p>University of West Bohemia</p>\n"
        "<p>Pilsen, Czech Republic</p>\n"
        "<p>mjirik@kky.zcu.cz</p>\n"
        f"</body></html>"
    )
    send_mail(
        "Pig Leg Surgery Analyser: Media file recived",
        html_message,
        from_email= "mjirik@kky.zcu.cz",
        recipient_list=[serverfile.email],
        fail_silently=False,
        html_message=html_message,
    )


# def run_processing2(serverfile: UploadedFile):
#     log_format = loguru._defaults.LOGURU_FORMAT
#     logger_id = logger.add(
#         str(Path(serverfile.outputdir) / "log.txt"),
#         format=log_format,
#         level="DEBUG",
#         rotation="1 week",
#         backtrace=True,
#         diagnose=True,
#     )
#     # delete_generated_images(
#     #     serverfile
#     # )  # remove images from database and the output directory
#     # mainapp = scaffan.algorithm.Scaffan()
#     # mainapp.set_input_file(serverfile.imagefile.path)
#     # mainapp.set_output_dir(serverfile.outputdir)
#     # fn, _, _ = models.get_common_spreadsheet_file(serverfile.owner)
#     # mainapp.set_common_spreadsheet_file(str(fn).replace("\\", "/"))
#     # settings.SECRET_KEY
#     logger.debug("Scaffan processing run")
#     # if len(centers_mm) > 0:
#     #     mainapp.set_parameter("Input;Lobulus Selection Method", "Manual")
#     # else:
#     #     mainapp.set_parameter("Input;Lobulus Selection Method", "Auto")
#     # mainapp.run_lobuluses(seeds_mm=centers_mm)
#     # serverfile.score = _clamp(
#     #     mainapp.report.df["SNI area prediction"].mean() * 0.5, 0.0, 1.0
#     # )
#     # serverfile.score_skeleton_length = mainapp.report.df["Skeleton length"].mean()
#     # serverfile.score_branch_number = mainapp.report.df["Branch number"].mean()
#     # serverfile.score_dead_ends_number = mainapp.report.df["Dead ends number"].mean()
#     # serverfile.score_area = mainapp.report.df["Area"].mean()
#
#     # add_generated_images(serverfile)  # add generated images to database
#     #
#     # serverfile.processed_in_version = scaffan.__version__
#     # serverfile.process_started = False
#     # serverfile.last_error_message = ""
#     # if serverfile.zip_file and Path(serverfile.zip_file.path).exists():
#     #     serverfile.zip_file.delete()
#     #
#     # views.make_zip(serverfile)
#     # serverfile.save()
#     # logger.remove(logger_id)


def get_zip_fn(serverfile: UploadedFile):
    logger.trace(f"serverfile.imagefile={serverfile.mediafile.name}")
    if not serverfile.mediafile.name:
        logger.debug(f"No file uploaded for {serverfile.mediafile}")
        return None
        # file is not uploaded

    nm = str(Path(serverfile.mediafile.path).name)
    # prepare output zip file path
    pth_zip = serverfile.outputdir + nm + ".zip"
    return pth_zip


def add_generated_images(serverfile: UploadedFile):
    # serverfile.bitmap_image_set.all().delete()
    od = Path(serverfile.outputdir)
    logger.debug(od)
    lst = sorted(glob.glob(str(od / "*.jpg")))
    lst.extend(sorted(glob.glob(str(od / "*.JPG"))))
    lst.extend(glob.glob(str(od / "*.png")))
    # lst.extend(glob.glob(str(od / "slice_label.png")))
    # lst.extend(sorted(glob.glob(str(od / "*.png"))))
    lst.extend(sorted(glob.glob(str(od / "*.PNG"))))
    # lst.extend(glob.glob(str(od / "sinusoidal_tissue_local_centers.png")))
    # lst.extend(sorted(glob.glob(str(od / "lobulus_[0-9]*.png"))))
    logger.debug(lst)
    # remove all older references and objects
    serverfile.bitmapimage_set.all().delete()

    for fn in lst:
        # skip the files with __ in the beginning of the name
        if Path(fn).name.startswith("__"):
            continue
        pth_rel = op.relpath(fn, django.conf.settings.MEDIA_ROOT)
        bi = BitmapImage(server_datafile=serverfile, bitmap_image=pth_rel)
        bi.save()


def make_zip(serverfile: UploadedFile):
    pth_zip = get_zip_fn(serverfile)
    if pth_zip:
        import shutil

        # remove last letters.because of .zip is added by make_archive
        shutil.make_archive(pth_zip[:-4], "zip", serverfile.outputdir)

        serverfile.processed = True
        pth_rel = op.relpath(pth_zip, django.conf.settings.MEDIA_ROOT)
        serverfile.zip_file = pth_rel
        serverfile.save()


def import_files_from_drop_dir(email, absolute_uri):
    """Find files in MEDIA_ROOT / drop_dir and import them to the database
    """
    files = list_files_in_drop_dir()
    for i, file_path in enumerate(files):
        # # Step 1: Read file from the server's hard drive
        # file_path = '/path/to/your/file'  # Replace with the path of your file
        # with open(file_path, 'rb') as f:
        #     file_content = f.read()

        # Step 1: Create a new UploadedFile instance
        new_uploaded_file = UploadedFile()
        new_uploaded_file.email = email
        new_uploaded_file.uploaded_at = datetime.now()
        update_owner(new_uploaded_file)
        # Set other fields as needed
        logger.debug(f"{file_path=}")
        logger.debug(f"{new_uploaded_file=}")

        from . import models_tools
        # Step 2: Save the file to the destination path
        destination_path = models_tools.upload_to_unqiue_folder(new_uploaded_file, os.path.basename(file_path))
        full_destination_path = os.path.join(settings.MEDIA_ROOT, destination_path)
        Path(full_destination_path).parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"{destination_path=}")
        logger.debug(f"{full_destination_path=}")


        # Step 3: Move the file
        shutil.move(file_path, full_destination_path)

        # Step 4: Update the mediafile field
        new_uploaded_file.mediafile = destination_path

        # Step 5: Save the UploadedFile instance
        new_uploaded_file.save()
        call_async_run_processing(new_uploaded_file, absolute_uri)


def list_files_in_drop_dir() -> list[Path]:
    dropdir = settings.DROP_DIR
    dropdir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"{list(dropdir.glob('*'))=}")
    files = (dropdir.glob("**/*"))
    logger.debug(f"{files=}")
    files = [f for f in files if
             f.is_file() and (f.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"))]

    logger.debug(f"{files=}")
    return files


def call_async_run_processing(serverfile, absolute_uri):
    PIGLEGCV_HOSTNAME = os.getenv("PIGLEGCV_HOSTNAME", default="127.0.0.1")
    PIGLEGCV_PORT = os.getenv("PIGLEGCV_PORT", default="5000")
    make_preview(serverfile)
    update_owner(serverfile)

    make_it_run_with_async_task(
        serverfile, absolute_uri, PIGLEGCV_HOSTNAME, int(PIGLEGCV_PORT),
        send_email=True)


def save_annotations_to_json(serverfile: UploadedFile):
    for idx, annotation in enumerate(serverfile.mediafileannotation_set.all()):

        annotation.save()

        annotation_filename = Path(serverfile.outputdir) / f"annotation_{idx}.json"
        logger.debug(f"{annotation_filename=}")
        from django.core.serializers import serialize

        # dump as json file
        with open(annotation_filename, "w") as f:
            # json.dump(json_annotation, f)
            f.write(serialize("json", [annotation]))


