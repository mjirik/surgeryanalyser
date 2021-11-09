from django.core.mail import send_mail
from .models import UploadedFile, BitmapImage
import loguru
from django.conf import settings
from loguru import logger
from pathlib import Path
import os.path as op
import requests
import time
import glob
from django.core.mail import EmailMessage
from django_q.tasks import async_task, schedule
from django_q.models import Schedule
from django.utils.html import strip_tags
# from .pigleg_cv import run_media_processing


def email_media_recived(serverfile: UploadedFile):
    # async_task('django.core.mail.send_mail',
    send_mail(
        "Pig Leg Surgery Analyser: Media file recived",
        "Thank you for uploading a file. \n"
        + "Now we are in an early stage of the project when we plan to collect the data."
        + " The outputs of the analysis will be introduced in few weeks. "
        + "We will let you know when the processing will be finished. \n\n"
        + "Best regards,\n"
        "Miroslav Jirik, Ph.D.\n\n"
        "Faculty of Applied Sciences\n"
        "University of West Bohemia\n"
        "Pilsen, Czech Republic",
        "mjirik@kky.zcu.cz",
        [serverfile.email],
        fail_silently=False,
    )

def _run_media_processing_rest_api(input_file:Path, outputdir:Path):

    # query = {"filename": "/webapps/piglegsurgery/tests/pigleg_test.mp4", "outputdir": "/webapps/piglegsurgery/tests/outputdir"}
    logger.debug("Creating request for processing")
    query = {
        "filename": str(input_file),
        "outputdir": str(outputdir),
    }
    response = requests.post('http://127.0.0.1:5000/run', params=query)
    hash = response.json()
    finished = False
    while not finished:
        time.sleep(30)
        response = requests.get(f'http://127.0.0.1:5000/is_finished/{hash}',
                                # params=query
                                )
        finished = response.json()
        logger.debug(f".    finished={finished}")


    logger.debug(f"Finished. finished={finished}")

def run_processing(serverfile: UploadedFile, absolute_uri):
    outputdir = Path(serverfile.outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    log_format = loguru._defaults.LOGURU_FORMAT
    logger_id = logger.add(
        str(Path(serverfile.outputdir) / "log.txt"),
        format=log_format,
        level="DEBUG",
        rotation="1 week",
        backtrace=True,
        diagnose=True,
    )
    logger.debug(f"Image processing of '{serverfile.mediafile}' initiated")
    if serverfile.zip_file and Path(serverfile.zip_file.path).exists():
        serverfile.zip_file.delete()
    input_file = Path(serverfile.mediafile.path)
    logger.debug(f"input_file={input_file}")
    outputdir = Path(serverfile.outputdir)
    logger.debug(f"outputdir={outputdir}")
    _run_media_processing_rest_api(input_file, outputdir)
    logger.debug(f"")
    (outputdir / "empty.txt").touch(exist_ok=True)

    make_zip(serverfile)
    serverfile.save()
    logger.remove(logger_id)


def email_report(task):
    logger.debug("Sending email report")

    serverfile: UploadedFile = task.args[0]
    # absolute uri is http://127.0.0.1:8000/. We have to remove last '/' because the url already contains it.
    absolute_uri = task.args[1][:-1]
    # logger.debug(dir(task))
    html_message = (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        "<head> \n"
        '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />'
        "<title>Order received</title>"
        '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>'
        "</head>"
        f"<body>"
        f"<p>Finished.</p><p>Email: {serverfile.email}</p><p>Filename: {serverfile.mediafile}</p>"
        f'<p></p>'
        f'<p> <a href="{absolute_uri}/uploader/web_report/{serverfile.hash}">Download report here</a> .</p>\n'
        f'<p></p>'
        f'<p>Best regards</p>\n'
        f'<p>Miroslav Jirik</p>\n'
        f'<p></p>'
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
    to_email = "mjirik@gapps.zcu.cz"

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
    # send_mail(
    #     "[Pig Leg Surgery]",
    #     html_message,
    #     "mjirik@kky.zcu.cz",
    #     ["miroslav.jirik@gmail.com"],
    #     fail_silently=False,
    # )


def run_processing2(serverfile: UploadedFile):
    log_format = loguru._defaults.LOGURU_FORMAT
    logger_id = logger.add(
        str(Path(serverfile.outputdir) / "log.txt"),
        format=log_format,
        level="DEBUG",
        rotation="1 week",
        backtrace=True,
        diagnose=True,
    )
    # delete_generated_images(
    #     serverfile
    # )  # remove images from database and the output directory
    # mainapp = scaffan.algorithm.Scaffan()
    # mainapp.set_input_file(serverfile.imagefile.path)
    # mainapp.set_output_dir(serverfile.outputdir)
    # fn, _, _ = models.get_common_spreadsheet_file(serverfile.owner)
    # mainapp.set_common_spreadsheet_file(str(fn).replace("\\", "/"))
    # settings.SECRET_KEY
    logger.debug("Scaffan processing run")
    # if len(centers_mm) > 0:
    #     mainapp.set_parameter("Input;Lobulus Selection Method", "Manual")
    # else:
    #     mainapp.set_parameter("Input;Lobulus Selection Method", "Auto")
    # mainapp.run_lobuluses(seeds_mm=centers_mm)
    # serverfile.score = _clamp(
    #     mainapp.report.df["SNI area prediction"].mean() * 0.5, 0.0, 1.0
    # )
    # serverfile.score_skeleton_length = mainapp.report.df["Skeleton length"].mean()
    # serverfile.score_branch_number = mainapp.report.df["Branch number"].mean()
    # serverfile.score_dead_ends_number = mainapp.report.df["Dead ends number"].mean()
    # serverfile.score_area = mainapp.report.df["Area"].mean()

    # add_generated_images(serverfile)  # add generated images to database
    #
    # serverfile.processed_in_version = scaffan.__version__
    # serverfile.process_started = False
    # serverfile.last_error_message = ""
    # if serverfile.zip_file and Path(serverfile.zip_file.path).exists():
    #     serverfile.zip_file.delete()
    #
    # views.make_zip(serverfile)
    # serverfile.save()
    # logger.remove(logger_id)


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

def add_generated_images(serverfile:UploadedFile):
    # serverfile.bitmap_image_set.all().delete()
    od = Path(serverfile.outputdir)
    logger.debug(od)
    lst = glob.glob(str(od / "*.png"))
    # lst.extend(glob.glob(str(od / "slice_label.png")))
    # lst.extend(sorted(glob.glob(str(od / "*.png"))))
    lst.extend(sorted(glob.glob(str(od / "*.PNG"))))
    # lst.extend(glob.glob(str(od / "sinusoidal_tissue_local_centers.png")))
    # lst.extend(sorted(glob.glob(str(od / "lobulus_[0-9]*.png"))))
    lst.extend(sorted(glob.glob(str(od / "*.jpg"))))
    lst.extend(sorted(glob.glob(str(od / "*.JPG"))))
    logger.debug(lst)

    for fn in lst:
        pth_rel = op.relpath(fn, settings.MEDIA_ROOT)
        bi = BitmapImage(server_datafile=serverfile, bitmap_image=pth_rel)
        bi.save()

def make_zip(serverfile: UploadedFile):
    pth_zip = get_zip_fn(serverfile)
    if pth_zip:
        import shutil

        # remove last letters.because of .zip is added by make_archive
        shutil.make_archive(pth_zip[:-4], "zip", serverfile.outputdir)

        serverfile.processed = True
        pth_rel = op.relpath(pth_zip, settings.MEDIA_ROOT)
        serverfile.zip_file = pth_rel
        serverfile.save()
