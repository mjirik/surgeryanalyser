import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import django.utils
from django.conf import settings
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render, reverse
from django.views import generic

# from .models_tools import get_hash_from_output_dir, get_outputdir_from_hash
from django_q.tasks import async_task, queue_size, schedule
from loguru import logger

from .forms import AnnotationForm, UploadedFileForm
from .models import MediaFileAnnotation, Owner, UploadedFile, _hash
from .models_tools import randomString
from .tasks import email_media_recived, make_preview
from datetime import timedelta
from django.utils import timezone
from django.db.models import Count, Q
from .models import UploadedFile

# Create your views here.


# from piglegsurgeryweb.piglegsurgeryweb.settings import PIGLEGCV_TIMEOUT


def logout_view(request):
    """Logout from the application."""
    logout(request)
    # Redirect to a success page.
    return redirect("uploader:model_form_upload")


def index(request):
    return HttpResponse("Hello, world. You're at piglegsurgeryweb. HAHA")


def message(request, headline=None, text=None, next_text=None, next=None):
    context = {
        "headline": "Thank You" if headline is None else headline,
        "text": "Thank you for uploading media file. We will let you know when the processing will be finished."
        if text is None
        else text,
        "next_text": "Forward" if next_text is None else next_text,
        "next": reverse("uploader:model_form_upload") if next is None else next
        # 'next': "uploader:model_form_upload"
        # 'next': "uploader:model_form_upload"
    }
    return render(request, "uploader/message.html", context)


def thanks(request):
    context = {
        "headline": "Thank You",
        "text": "Thank you for uploading media file. We will let you know when the processing will be finished.",
        "next_text": "Upload next",
        "next": None
        # 'next': "uploader:model_form_upload"
        # 'next': "uploader:model_form_upload"
    }
    return render(request, "uploader/thanks.html", context)


@login_required(login_url="/admin/")
def reset_hashes(request):
    files = UploadedFile.objects.all()
    for file in files:
        file.hash = _hash()
        file.save()
    return redirect("/uploader/thanks/")


@login_required(login_url="/admin/")
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


# def set_order_by(request, order_by, next_page:str):
#     request.session["order_by"] = order_by
#     request.session.modified = True
#     return redirect(next_page)


@login_required(login_url="/admin/")
def swap_is_microsurgery(request, filename_id: int):
    uploadedfile = get_object_or_404(UploadedFile, pk=filename_id)
    uploadedfile.is_microsurgery = not uploadedfile.is_microsurgery
    uploadedfile.save()
    next_anchor = request.GET.get("next_anchor", None)
    return redirect(reverse("uploader:web_reports", kwargs={}) + f"#{next_anchor}")


# @login_required(login_url='/admin/')
def report_list(request):
    # order_by = request.session.get("order_by", '-uploaded_at')

    if "order_by" in request.GET:
        logger.debug(f"order_by={request.GET['order_by']}")
        request.session["order_by"] = request.GET.get("order_by")
        request.session.modified = True

    order_by = request.session.get("order_by", "-uploaded_at")
    # logger.debug(f"order_by={order_by}")

    # order_by = request.GET.get("order_by", "-uploaded_at")
    if order_by == "filename":
        files = sorted(UploadedFile.objects.all(), key=str)
    else:
        files = UploadedFile.objects.all().order_by(order_by)
    qs_data = {}
    for e in files:
        qs_data[e.id] = (
            str(e.email)
            + " "
            + str(e)
            + " "
            + str(e.uploaded_at)
            + " "
            + str(e.finished_at)
        )

    qs_json = json.dumps(qs_data)
    # logger.debug(qs_data)
    context = {
        "uploadedfiles": files,
        "queue_size": queue_size(),
        "qs_json": qs_json,
        "page_reference": "web_reports",
        "order_by": order_by,
    }

    return render(request, "uploader/report_list.html", context)


def _get_graph_path(owner: Optional[Owner] = None):
    if owner:
        html_path = Path(settings.MEDIA_ROOT) / "generated" / owner.hash / "graph.html"
    else:
        html_path = Path(settings.MEDIA_ROOT) / "generated/graph.html"
    return html_path


def make_graph(
    uploaded_file_set: UploadedFile.objects.all(), owner: Optional[Owner] = None
):
    import pandas as pd
    import plotly.express as px
    from django.utils import timezone

    html_path = _get_graph_path(owner)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, uploaded_file in enumerate(uploaded_file_set):

        results_path = Path(uploaded_file.outputdir) / "results.json"
        # read results.json
        if results_path.exists():
            with open(results_path) as f:
                loaded_results = json.load(f)

            loaded_results["Uploaded at"] = uploaded_file.uploaded_at
            loaded_results["i"] = i
            rows.append(loaded_results)

    df = pd.DataFrame(rows)
    # fix typo
    df.rename(
        columns={"Stichtes linearity score": "Stitches linearity score"}, inplace=True
    )

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
    import plotly.express as px

    fig = px.scatter(
        df,
        x=x,
        y=y,
        # marginal_x="box",
        # marginal_y="box"
    )
    fig.write_html(html_path, full_html=False)
    return html_path


def owners_reports_list(request, owner_hash: str):
    owner = get_object_or_404(Owner, hash=owner_hash)
    order_by = request.GET.get("order_by", "-uploaded_at")
    files = UploadedFile.objects.filter(owner=owner).order_by(order_by)
    qs_data = {}
    for e in files:
        qs_data[e.id] = (
            str(e.email)
            + " "
            + str(e)
            + " "
            + str(e.uploaded_at)
            + " "
            + str(e.finished_at)
        )

    qs_json = json.dumps(qs_data)

    html_path = make_graph(files, owner)

    html = html_path.read_text() if html_path else None
    # logger.debug(html)

    context = {
        "uploadedfiles": files,
        "queue_size": queue_size(),
        "qs_json": qs_json,
        "page_reference": "owners_reports_list",
        "owner": owner,
        "myhtml": html,
    }

    return render(request, "uploader/report_list.html", context)


def _prepare_context_for_web_report(request, serverfile: UploadedFile, review_edit_hash: Optional[str] = None) -> dict:
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
                new_key = (
                    key.replace("Tweezes", "Forceps")
                    .replace("Tweezers", "Forceps")
                    .replace("duration", "visibility")
                )
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
            videofile_url = s[: s.rfind("/")] + "/" + videofile.name
            videofiles_url.append(videofile_url)

    image_list_in, image_list_out = _filter_images(serverfile)
    logger.debug(f"{serverfile.review_edit_hash=}")
    logger.debug(f"{review_edit_hash=}")
    edit_review = serverfile.review_edit_hash == review_edit_hash
    logger.debug(f"{edit_review=}")
    context = {
        "serverfile": serverfile,
        "mediafile": Path(serverfile.mediafile.name).name,
        "image_list": image_list_in,
        "image_list_out": image_list_out,
        "next": request.GET["next"] if "next" in request.GET else None,
        "videofiles_url": videofiles_url,
        "results": results,
        "edit_review": edit_review,
    }

    return context


def _prepare_context_if_web_report_not_exists(request, serverfile: UploadedFile):
    logger.debug("Zip file name does not exist")
    # zip_file does not exists
    context = {
        "headline": "File not exists",
        "text": "Requested file is probably under processing now.",
        "next": request.GET["next"] if "next" in request.GET else "/uploader/upload/",
        "next_text": "Back",
    }
    if request.user.is_authenticated:
        context["key_value"] = _get_logs_as_html(serverfile)
    logger.debug(context)
    logger.debug(request)
    logger.debug(request.path)
    return context


def web_report(request, filename_hash: str, review_edit_hash: Optional[str] = None):
    # fn = get_outputdir_from_hash(hash)
    serverfile = get_object_or_404(UploadedFile, hash=filename_hash)
    if (
        not bool(serverfile.zip_file.name)
        or not Path(serverfile.zip_file.path).exists()
    ):
        context = _prepare_context_if_web_report_not_exists(request, serverfile)
        return render(request, "uploader/message.html", context)
        # return redirect("uploader:message", next=request.path)
    context = _prepare_context_for_web_report(request, serverfile, review_edit_hash)

    # evaluate annotation form
    if request.method == "POST":
        uploaded_file_annotations = serverfile.mediafileannotation_set.first()
        if uploaded_file_annotations:
            logger.debug("annotation loaded from database")
        else:
            logger.debug("saving new form")

        form = AnnotationForm(request.POST, instance=uploaded_file_annotations)
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            if request.user.is_authenticated:
                annotator = _get_owner(request.user.email)
            else:
                annotator = None
            annotation = form.save(commit=False)
            annotation.uploaded_file = serverfile
            annotation.updated_at = django.utils.timezone.now()
            annotation.annotator = annotator
            annotation.save()


            # annotation = MediaFileAnnotation(
            #     uploaded_file=serverfile,
            #     annotation=form.cleaned_data["annotation"],
            #     stars=form.cleaned_data["stars"],
            #     annotator=annotator,
            # )
            annotation.save()
            return redirect(request.path)
    else:

        uploaded_file_annotations_set = serverfile.mediafileannotation_set
        logger.debug(f"{uploaded_file_annotations_set=}")
        uploaded_file_annotations = uploaded_file_annotations_set.first()
        logger.debug(f"{uploaded_file_annotations=}")
        if uploaded_file_annotations:
            form = AnnotationForm(instance=uploaded_file_annotations)
            logger.debug("annotation loaded from database")
        else:
            form = AnnotationForm()
            logger.debug("created empty form")
    # logger.debug(f"form={form}")
    context["form"] = form

    return render(request, "uploader/web_report.html", context)

def _find_video_for_annotation(student_id:Optional[int] = None):

    now = timezone.now()
    thirty_minutes_ago = now - timedelta(minutes=30)
    today = timezone.now().date()


    # Filter videos not uploaded by the requesting student and that have no annotations
    if student_id:
        videos = UploadedFile.objects.exclude(owner__id=student_id)
    else:
        videos = UploadedFile.objects.all()
    videos = videos.annotate(
        num_annotations=Count('mediafileannotation')
    ).filter(
        Q(review_assigned_at__lt=thirty_minutes_ago) | Q(review_assigned_at__isnull=True),
        num_annotations = 0,
    )


    video = None
    # Start with today's videos, then go back each day
    for days_ago in range(0, 7):  # Search up to a week back, adjust as needed
        day = today - timedelta(days=days_ago)

        # Get videos from this day, ordered by time (later times later)
        day_videos = videos.filter(uploaded_at__date=day).order_by('uploaded_at')

        if day_videos.exists():
            video = day_videos.last()  # Return the latest video of the day
            break

    # No videos found within the past week so we are looking for the oldest video

    if video is None:
        video = videos.order_by('uploaded_at').last()

    if video is not None:
        video.review_assigned_at = now
        if student_id:
            video.review_assigned_to = _get_owner(student_id)
        video.save()

    return video

def go_to_video_for_annotation(request, student_email:Optional[str]=None):
    student_id = None
    if student_email is None:
        if request.user.is_authenticated:
            student_email = request.user.email
            if student_email:
                student_id = _get_owner(student_email).id
    else:
        student_id = _get_owner(student_email).id

    video = _find_video_for_annotation(student_id)

    # go to the web_report of the video
    if video:
        return redirect("uploader:web_report", filename_hash=video.hash)
    else:
        # return redirect("uploader:model_form_upload_upload")
        return redirect("uploader:message",
                        headline="No video for review",
                        text="No video for review found. Please try again later.",
                        next=reverse("uploader:model_form_upload"),
                        next_text="Ok"
                        )


def _filter_images(serverfile: UploadedFile):
    allowed_image_patterns = [
        "heatmap",
        "needle_holder_area_presence",
        "stitch_detection_0",
        "jpeg",
        "gif",
    ]
    filtered_in = []
    filtered_out = []

    for image in serverfile.bitmapimage_set.all():
        to_keep = False
        # if image.filename.startswith("_") or image.filename.startswith("__"):
        #     continue
        for allowed_image_pattern in allowed_image_patterns:
            filename = os.path.basename(image.bitmap_image.name)
            if allowed_image_pattern in filename.lower():
                to_keep = True
                filtered_in.append(image)
        if not to_keep:
            filtered_out.append(image)

    return filtered_in, filtered_out


def redirect_to_spreadsheet(request):
    # read env variable PIGLEGCV_SPREADSHEET_URL
    pigleg_spreadsheet_url = os.getenv("PIGLEG_SPREADSHEET_URL", default=None)
    if pigleg_spreadsheet_url is None:
        logger.debug(f"piglegcv_spreadsheet_url={pigleg_spreadsheet_url}")
        return redirect("uploader:web_reports")
    else:
        pigleg_spreadsheet_url = pigleg_spreadsheet_url.replace('"', "")
        logger.debug(f"piglegcv_spreadsheet_url={pigleg_spreadsheet_url}")
        return redirect(pigleg_spreadsheet_url)


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
    next_url = None
    if "next" in request.GET:
        next_url = request.GET["next"] + "#" + request.GET["next_anchor"]
    context = {
        "headline": "Processing started",
        "text": f"We are processing file {str(Path(serverfile.mediafile.name).name)}. "
        + "We will let you know by email as soon as it is finished.",
        # The output will be stored in {serverfile.outputdir}.",
        "next": next_url,
        "next_text": "Back",
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


def update_owner(uploadedfile: UploadedFile) -> Owner:
    if not uploadedfile.owner:
        owners = Owner.objects.filter(email=uploadedfile.email)
        if len(owners) == 0:
            owner = Owner(email=uploadedfile.email, hash=_hash())
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


def _get_owner(owner_email: str):
    owners = Owner.objects.filter(email=owner_email)
    if len(owners) == 0:
        owner = Owner(email=owner_email, hash=_hash())
        owner.save()
        # create one
    else:
        owner = owners[0]
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
            PIGLEGCV_PORT = os.getenv("PIGLEGCV_PORT", default="5000")
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


def test(request):
    return render(request, "uploader/test.html", {})


def show_logs(request, filename_hash: str):
    serverfile = get_object_or_404(UploadedFile, hash=filename_hash)
    key_value = _get_logs_as_html(serverfile)
    return render(
        request,
        "uploader/message.html",
        {
            "headline": "Logs",
            "key_value": key_value,
            "next": request.GET["next"]
            if "next" in request.GET
            else "/uploader/upload/",
        },
    )


def _get_logs_as_html(serverfile: UploadedFile) -> dict:
    outputdir = Path(serverfile.outputdir)
    key_value = {}
    for file in outputdir.glob("*_log.txt"):
        with open(file) as f:
            lines = f.readlines()
        # lines = [_set_loglevel_color(line) for line in lines]
        key_value.update(
            {str(file.stem): '<p class="text-monospace">' + "<br>".join(lines) + "</p>"}
        )
    # logpath = Path(serverfile.outputdir) / "piglegcv_log.txt"
    return key_value
    # return render(_as_htmlrequest,'uploader/show_logs.html', {"key_value": key_value, "logpath": logpath})


def _make_html_from_log(logpath: Path):
    logpath = opath / "log.txt"
    if logpath.exists():
        with open(logpath) as f:
            lines = f.readlines()

        lines = [_set_loglevel_color(line) for line in lines]
        key_value.update(
            {"Log": '<p class="text-monospace">' + "<br>".join(lines) + "</p>"}
        )
