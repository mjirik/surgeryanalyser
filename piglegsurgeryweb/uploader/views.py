from django.shortcuts import render, get_object_or_404, redirect
from django.views import generic
from django.contrib.auth.decorators import login_required
import django.utils
from pathlib import Path
import os

from loguru import logger
# Create your views here.

from django.http import HttpResponse
from .models import UploadedFile, _hash, Owner
from .forms import UploadedFileForm
from .models_tools import randomString
from .tasks import email_media_recived, make_preview
# from .models_tools import get_hash_from_output_dir, get_outputdir_from_hash
from django_q.tasks import async_task, schedule, queue_size
from datetime import datetime
from django.conf import settings
import json
import re
# from piglegsurgeryweb.piglegsurgeryweb.settings import PIGLEGCV_TIMEOUT

def index(request):
    return HttpResponse("Hello, world. You're at piglegsurgeryweb. HAHA")

def message(request, headline=None, text=None, next_text=None, next=None):
    context = {
        'headline': "Thank You" if headline is None else headline,
        'text': "Thank you for uploading media file. We will let you know when the processing will be finished." if text is None else text,
        'next_text': 'Forward' if next_text is None else next_text,
        'next': "uploader/upload" if next is None else next
        # 'next': "uploader:model_form_upload"
        # 'next': "uploader:model_form_upload"
    }
    return render(request, "uploader/message.html", context)

def thanks(request):
    context = {
        'headline': "Thank You",
        'text': "Thank you for uploading media file. We will let you know when the processing will be finished.",
        'next_text': 'Upload next',
        'next': None
        # 'next': "uploader:model_form_upload"
        # 'next': "uploader:model_form_upload"
    }
    return render(request, "uploader/thanks.html", context)

@login_required(login_url='/admin/')
def reset_hashes(request):
    files = UploadedFile.objects.all()
    for file in files:
        file.hash = _hash()
        file.save()
    return redirect("/uploader/thanks/")

@login_required(login_url='/admin/')
def update_all_uploaded_files(request):
    files = UploadedFile.objects.all()
    logger.info("update all uploaded files")
    for file in files:

        make_preview(file, force=True)
        update_owner(file)
    return redirect("/uploader/thanks/")

def resend_report_email(request, filename_id):
    serverfile = get_object_or_404(UploadedFile, pk=filename_id)
    from django_q.tasks import async_task
    async_task(
        "uploader.tasks.email_report",
        serverfile,
        request.build_absolute_uri("/"),
    )
    return redirect("/uploader/thanks/")


# @login_required(login_url='/admin/')
def show_report_list(request):
    files = UploadedFile.objects.all().order_by('-uploaded_at')
    qs_data = {}
    for e in files:
        qs_data[e.id] = str(e.email) + " " + str(e) + " " + str(e.uploaded_at) + " " + str(e.finished_at)

    qs_json = json.dumps(qs_data)
    # logger.debug(qs_data)
    context = {
        "uploadedfiles": files, 'queue_size': queue_size(), 'qs_json': qs_json
    }

    return render(request, "uploader/report_list.html", context)

def owners_reports_list(request, owner_hash:str):
    owner = get_object_or_404(Owner, hash=owner_hash)
    files = UploadedFile.objects.filter(owner=owner).order_by('-uploaded_at')
    qs_data = {}
    for e in files:
        qs_data[e.id] = str(e.email) + " " + str(e) + " " + str(e.uploaded_at) + " " + str(e.finished_at)

    qs_json = json.dumps(qs_data)
    context = {
        "uploadedfiles": files, 'queue_size': queue_size(), "qs_json": qs_json
    }

    return render(request, "uploader/report_list.html", context)


def web_report(request, filename_hash:str):
    # fn = get_outputdir_from_hash(hash)
    serverfile = get_object_or_404(UploadedFile, hash=filename_hash)
    if not bool(serverfile.zip_file.name) or not Path(serverfile.zip_file.path).exists():
        logger.debug("Zip file name does not exist")
        # zip_file does not exists
        context = {
            "headline": "File not exists", "text": "Requested file is probably under processing now.",
            "next": request.GET['next'] if "next" in request.GET else "/uploader/upload/",
            "next_text": "Back"
        }
        logger.debug(context)
        logger.debug(request)
        logger.debug(request.path)
        return render(request, "uploader/message.html", context)
        # return redirect("uploader:message", next=request.path)
    fn = Path(serverfile.zip_file.path)
    logger.debug(fn)
    logger.debug(fn.exists())
    # if not fn.exists():
    #     return render(request, 'uploader/thanks.html', {
    #         "headline": "File not exists", "text": "Requested file is probably under processing now.",
    #         "next": request.GET['next'] if "next" in request.GET else None,
    #         "next_text": "Back"
    #     })
    logger.debug(serverfile.zip_file.url)

    fn_results = Path(serverfile.outputdir) / "results.json"
    results = {}
    if fn_results.exists():
        with open(fn_results) as f:
            loaded_results = json.load(f)
            if "Video duration [s]" in loaded_results:
                video_duration = loaded_results["Video duration [s]"]
            else:
                video_duration = None
            for key in loaded_results:
                new_value = loaded_results[key]
                # backward compatibility
                new_key = key.replace("Tweezes", "Forceps").replace("Tweezers", "Forceps").replace("duration", "visibility")
                new_key = re.sub("visibility$", "visibility [s]", new_key)
                new_key = re.sub("length$", "length [m]", new_key)
                if new_key in (
                        "Needle holder length [pix]",
                        "Needle holder length [m]",
                        "Needle holder visibility [s]",
                        "Needle holder visibility [%]",
                        "Forceps length [pix]",
                        "Forceps length [m]",
                        "Forceps visibility [s]",
                        "Forceps visibility [%]",
                        "Scissors length [pix]",
                        "Scissors length [m]",
                        "Scissors visibility [s]",
                        "Scissors visibility [%]",
                        "Needle holder area presence [%]",
                        # "Tweezes length", "Tweezes duration" # typo in some older processings
                        # "Tweezers length", "Tweezers duration", # backward compatibility
                        # "Scissors length", "Scissors duration", # backward compatibility
                        # "Needle holder length", "Needle holder duration", # backward compatibility
                       ):
                    # new_key = new_key.replace("visibility", "visibility [s]").replace("length", "length [cm]")

                    if new_key.find("[m]") > 0:
                        new_key = re.sub("length \[m\]$", "length [cm]", new_key)
                        new_value = f"{new_value * 100:0.0f}"
                    if new_key.find("[pix]") > 0:
                        # new_key = re.sub("length \[m\]$", "length [cm]", new_key)
                        new_value = f"{new_value:0.0f}"
                    if new_key.find("[s]") > 0:
                        new_value = f"{new_value:0.0f}"
                    if new_key.find("[%]") > 0:
                        new_value = f"{new_value:0.0f}"
                    results[new_key] = new_value

    image_list = serverfile.bitmapimage_set.all()

    videofiles = Path(serverfile.outputdir).glob("*.mp4")
    # videofile = Path(serverfile.outputdir) / "pigleg_results.mp4"
    # if not videofile.exists():
    #     videofile = Path(serverfile.outputdir) / "video.mp4"
    videofiles_url = []
    for videofile in videofiles:
        logger.debug(videofile.name)
        if videofile.exists():
            if videofile.name.startswith("__"):
                logger.debug(f"Skipping video file {videofile.name}")
                continue
            s = str(serverfile.bitmapimage_set.all()[0].bitmap_image.url)[:-4]
            videofile_url = s[:s.rfind("/")] + "/" + videofile.name
            videofiles_url.append(videofile_url)



    context = {
        'serverfile': serverfile,
        'mediafile': Path(serverfile.mediafile.name).name,
        'image_list': image_list,
        "next": request.GET['next'] if "next" in request.GET else None,
        'videofiles_url': videofiles_url,
        "results": results
    }
    return render(request,'uploader/web_report.html', context)


def run(request, filename_id):
    PIGLEGCV_HOSTNAME = os.getenv("PIGLEGCV_HOSTNAME", default="127.0.0.1")
    PIGLEGCV_PORT = os.getenv("PIGLEGCV_PORT", default="5000")
    return _run(request, filename_id, PIGLEGCV_HOSTNAME, port=int(PIGLEGCV_PORT))

# def run_development(request, filename_id):
#     PIGLEGCV_HOSTNAME_DEVEL = os.getenv("PIGLEGCV_HOSTNAME_DEVEL", default="127.0.0.1")
#     PIGLEGCV_PORT_DEVEL = os.getenv("PIGLEGCV_PORT", default="5000")
#     return _run(request, filename_id, PIGLEGCV_HOSTNAME_DEVEL, port=int(PIGLEGCV_PORT_DEVEL))


def _run(request, filename_id, hostname="127.0.0.1", port=5000):
    serverfile = get_object_or_404(UploadedFile, pk=filename_id)

    from django_q.tasks import async_task
    serverfile.started_at = django.utils.timezone.now()
    serverfile.finished_at = None
    serverfile.save()
    logger.debug(f"hostname={hostname}, port={port}")


    async_task(
        "uploader.tasks.run_processing",
        serverfile,
        request.build_absolute_uri("/"),
        hostname,
        port,
        timeout=settings.PIGLEGCV_TIMEOUT,
        # hook="uploader.tasks.email_report_from_task",
    )
    context = {
        'headline': "Processing started",
        'text': f"We are processing file {str(Path(serverfile.mediafile.name).name)}. " +
        "We will let you know by email as soon as it is finished.",  # The output will be stored in {serverfile.outputdir}.",
        "next": request.GET['next'] if "next" in request.GET else None,
        "next_text": "Back"
    }
    return render(request, "uploader/thanks.html", context)
    # return redirect("/uploader/upload/")

def about_ev_cs(request):
    return render(request, "uploader/about_ev_cs.html", {})
def about_ev_en(request):
    return render(request, "uploader/about_ev_en.html", {})

class DetailView(generic.DetailView):
    model = UploadedFile
    template_name = "uploader/model_form_upload.html"


def update_owner(uploadedfile:UploadedFile) -> Owner:
    if not uploadedfile.owner:
        owners = Owner.objects.filter(email=uploadedfile.email)
        if len(owners) == 0:
            owner = Owner(email=uploadedfile.email, hash=_hash())
            owner.save()
            #create one
        else:
            owner = owners[0]
    else:
        owner = uploadedfile.owner

    # uploadedfiles = UploadedFile.objects.filter(owner=owner)
    if uploadedfile.owner != owner:
        uploadedfile.owner = owner
        uploadedfile.save()

    return owner

def model_form_upload(request):
    if request.method == "POST":
        form = UploadedFileForm(
            request.POST,
            request.FILES,
            # owner=request.user
        )
        if form.is_valid():
            from django_q.tasks import async_task

            # logger.debug(f"imagefile.name={dir(form)}")
            # name = form.cleaned_data['imagefile']
            # if name is None or name == '':
            #     return render(request, 'uploader/model_form_upload.html', {
            #         'form': form,
            #         "headline": "Upload",
            #         "button": "Upload",
            #         "error_text": "Image File is mandatory"
            #     })

            serverfile = form.save()
            async_task("uploader.tasks.email_media_recived", serverfile)

            # email_media_recived(serverfile)
            # print(f"user id={request.user.id}")
            # serverfile.owner = request.user
            serverfile.started_at = django.utils.timezone.now()
            serverfile.save()
            PIGLEGCV_HOSTNAME = os.getenv("PIGLEGCV_HOSTNAME", default="127.0.0.1")
            PIGLEGCV_PORT= os.getenv("PIGLEGCV_PORT", default="5000")
            make_preview(serverfile)
            update_owner(serverfile)
            async_task(
                "uploader.tasks.run_processing",
                serverfile,
                request.build_absolute_uri("/"),
                PIGLEGCV_HOSTNAME,
                int(PIGLEGCV_PORT),
                timeout=settings.PIGLEGCV_TIMEOUT,
                hook="uploader.tasks.email_report_from_task",
            )
            return redirect("/uploader/thanks/")
    else:
        form = UploadedFileForm()
    return render(
        request,
        "uploader/model_form_upload.html",
        {"form": form, "headline": "Upload", "button": "Upload"},
    )
