import json
import re
from pathlib import Path
from typing import Optional, Tuple, List
import traceback

import django.utils
from django.conf import settings
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render, reverse
from django.views import generic
from django.template.loader import render_to_string
from django.core.paginator import Page, Paginator
from django.http import StreamingHttpResponse, Http404, FileResponse
from django.shortcuts import get_object_or_404
# from .models import UploadedFile
import os

# from .models_tools import get_hash_from_output_dir, get_outputdir_from_hash
from django_q.tasks import async_task, queue_size
from loguru import logger

from .forms import AnnotationForm, UploadedFileForm
from .models import Owner
from .tasks import get_graph_path_for_owner, call_async_run_processing, make_it_run_with_async_task
from datetime import timedelta
from django.utils import timezone
from django.db.models import Count, Q
from .models import UploadedFile, _hash, Collection
from . import tasks, models, report_tools, forms
import numpy as np
from .report_tools import load_results, load_per_stitch_data

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
        "text": "Thank you for uploading media file. We will let you know when the processing will be finished. "+
                "Meanwhile you can review other student's video.",
        "next_text": "Review other student's video",
        "next": reverse("uploader:go_to_video_for_annotation_random", ),
        "next_text_secondary" : "Upload another video",
        "next_secondary": reverse("uploader:model_form_upload")
        # 'next': "uploader:model_form_upload"
        # 'next': "uploader:model_form_upload"
    }
    return render(request, "uploader/message.html", context)


@login_required(login_url="/admin/")
def reset_hashes(request):
    files = UploadedFile.objects.all()
    for file in files:
        file.hash = _hash()
        file.save()
    return redirect("/uploader/thanks/")


@login_required(login_url="/admin/")
def update_all_uploaded_files(request):
    async_task(
        "uploader.tasks.update_all_uploaded_files"
    )
    message_context = {
        "headline": "Update all uploaded files",
        "text": "We will update all uploaded files. It may take a while.",
        "next": reverse("uploader:web_reports", kwargs={}),
        "next_text": "Back",
    }
    return render(request, "uploader/message.html", message_context)

def add_uploaded_file_to_collection(request, collection_id, filename_id):
    collection = get_object_or_404(models.Collection, id=collection_id)
    uploaded_file = get_object_or_404(models.UploadedFile, id=filename_id)
    collection.uploaded_files.add(uploaded_file)
    return redirect(request.META.get('HTTP_REFERER', '/'))

def remove_uploaded_file_from_collection(request, collection_id, filename_id):
    collection = get_object_or_404(models.Collection, id=collection_id)
    uploaded_file = get_object_or_404(models.UploadedFile, id=filename_id)
    collection.uploaded_files.remove(uploaded_file)
    return redirect(request.META.get('HTTP_REFERER', '/'))

def resend_report_email(request, filename_id):
    serverfile = get_object_or_404(UploadedFile, pk=filename_id)

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

    uploaded_file_set = UploadedFile.objects.all()

    context = _general_report_list(request, uploaded_file_set)

    return render(request, "uploader/report_list.html", context)


def _general_report_list(request, uploaded_file_set):
    query = request.GET.get("q")
    if query:
        words = query.split()
        q_objects = Q()

        for word in words:
            word_filter = (
                  Q(email__icontains=word)
                  | Q(mediafile__icontains=word)
                  | Q(collection__name__icontains=word)  # Pokud máš pole s výsledky
                  | Q(category__name__icontains=word)
            )
            q_objects &= word_filter  # Každé slovo musí být nalezeno někde

        uploaded_file_set = uploaded_file_set.filter(q_objects).distinct()

    if "order_by" in request.GET:
        logger.debug(f"order_by={request.GET['order_by']}")
        request.session["order_by"] = request.GET.get("order_by")
        request.session.modified = True
    order_by = request.session.get("order_by", "-uploaded_at")
    # logger.debug(f"order_by={order_by}")
    # order_by = request.GET.get("order_by", "-uploaded_at")
    if order_by == "filename":
        files = sorted(uploaded_file_set, key=str)
    else:
        files = uploaded_file_set.order_by(order_by)
    records_per_page = 100
    paginator = Paginator(files, per_page=records_per_page)
    page_obj, _, page_context = _prepare_page(paginator, request=request)

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
        # "uploadedfiles": files,
        "uploadedfiles": page_obj,
        "queue_size": queue_size(),
        "qs_json": qs_json,
        "page_reference": "web_reports",
        "order_by": order_by,
        "collections": models.Collection.objects.all(),
        **page_context
    }
    return context


def show_collection_reports_list(request, collection_id:Optional[int]=None, collection_hash:Optional[str]=None):
    if collection_id:
        collection = get_object_or_404(models.Collection, id=collection_id)
    else:
        collection = get_object_or_404(models.Collection, hash=collection_hash)
    upload_files_set = collection.uploaded_files.all()
    context = _general_report_list(request, upload_files_set)
    context["headline"] = collection.name
    context["private_mode"] = True
    return render(request, "uploader/report_list.html", context)


def owners_reports_list(request, owner_hash: str):
    owner = get_object_or_404(Owner, hash=owner_hash)
    order_by = request.GET.get("order_by", "-uploaded_at")
    files = UploadedFile.objects.filter(owner=owner).order_by(order_by)
    context = _general_report_list(request, files)

    # qs_data = {}
    # for e in files:
    #     qs_data[e.id] = (
    #         str(e.email)
    #         + " "
    #         + str(e)
    #         + " "
    #         + str(e.uploaded_at)
    #         + " "
    #         + str(e.finished_at)
    #     )
    #
    # qs_json = json.dumps(qs_data)

    
    html_path = get_graph_path_for_owner(owner)

    html = None
    if html_path:
        if not html_path.exists():
            tasks.make_graph(files, owner)
        if html_path.exists():
            html = html_path.read_text()
    # logger.debug(html)

    # context = {
    #     "uploadedfiles": files,
    #     "queue_size": queue_size(),
    #     "qs_json": qs_json,
    #     "page_reference": "owners_reports_list",
    #     "owner": owner,
    #     "myhtml": html,
    # }
    context["myhtml"] = html

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

    loaded_results = load_results(serverfile)
    results = prepare_results_for_visualization(loaded_results)

    image_list = serverfile.bitmapimage_set.all()

    videofiles = (Path(serverfile.outputdir).glob("*.mp4"))
    videofiles = [vf for vf in videofiles if not vf.name.startswith("__")]
    # show_alternative = len(videofiles) == 0
    show_alternative = True  # after discussion we want to show the original video as default
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

    image_list_in, image_list_out, static_analysis_image = _filter_images(serverfile)
    logger.debug(f"{serverfile.review_edit_hash=}")
    logger.debug(f"{review_edit_hash=}")
    # if no everyone should annotate the video
    # edit_review = serverfile.review_edit_hash == review_edit_hash
    # if everyone can do the review
    edit_review = True
    logger.debug(f"{edit_review=}")

    logger.debug(f"Image list in {len(image_list_in)}")
    for image in image_list_in:
        image: models.BitmapImage
        logger.debug(image.bitmap_image.name)
    logger.debug(f"Image list out {len(image_list_out)}")
    for image in image_list_out:
        logger.debug(image.bitmap_image.name)

    html_path = tasks.get_graph_path_for_report(serverfile)

    html = None
    if html_path:
        # if not html_path.exists():

        tasks._make_metrics_for_report(serverfile)
        if html_path.exists():
            html = html_path.read_text()

    reviews = [review.annotator for review in serverfile.mediafileannotation_set.all()]

    per_stitch_report = load_per_stitch_data(loaded_results, serverfile)

    # logger.debug(f"{per_stitch_report=}")

    # get collections with serverfile
    collections_with = models.Collection.objects.filter(uploaded_files=serverfile)
    # get collections without serverfile
    collections_without = models.Collection.objects.exclude(uploaded_files=serverfile)

    context = {
        "serverfile": serverfile,
        "mediafile": Path(serverfile.mediafile.name).name,
        "image_list": image_list_in,
        "image_list_out": image_list_out,
        "next": request.GET["next"] if "next" in request.GET else None,
        "videofiles_url": videofiles_url,
        "alternative_videofiles_url": [serverfile.mediafile.url],
        "results": results,
        "edit_review": edit_review,
        "per_stitch_report": per_stitch_report,
        "myhtml": html,
        "static_analysis_image": static_analysis_image,
        "review_number": len(serverfile.mediafileannotation_set.all()),
        "reviews": reviews,
        "collections_with": collections_with,
        "collections_without": collections_without,
        "show_alternative": show_alternative
    }

    return context


def prepare_results_for_visualization(loaded_results:Optional[dict]) -> dict:
    results = {}
    loaded_results = None
    if loaded_results:
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
                    "Needle holder length [cm]",
                    "Needle holder visibility [s]",
                    "Needle holder visibility [%]",
                    "Forceps length [pix]",
                    "Forceps length [m]",
                    "Forceps length [cm]",
                    "Forceps visibility [s]",
                    "Forceps visibility [%]",
                    "Scissors length [pix]",
                    "Scissors length [m]",
                    "Scissors length [cm]",
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
    return results



def download_sample_image(request):
    """Download uploaded file."""

    collection = get_object_or_404(models.Collection, name="test_data")
    media_file = collection.uploaded_files.first()

    file_path = Path(settings.MEDIA_ROOT) / media_file.mediafile.name
    if file_path.exists():
        with open(file_path, "rb") as fh:
            response = HttpResponse(fh.read(), content_type="application/zip")
            response["Content-Disposition"] = "inline; filename=" + os.path.basename(file_path)
            return response
    # raise Http404
    else:
        return redirect("uploader:upload_mediafile")


@login_required(login_url="/admin/")
def delete_media_file(request, filename_id):
    serverfile = get_object_or_404(UploadedFile, pk=filename_id)
    serverfile.delete()
    return redirect("uploader:web_reports")

def _prepare_context_if_web_report_not_exists(request, serverfile: UploadedFile):
    edit_review = True

    videofiles = (Path(serverfile.outputdir).glob("*.mp4"))
    videofiles = [vf for vf in videofiles if not vf.name.startswith("__")]
    # show_alternative = len(videofiles) == 0
    show_alternative = True  # after discussion we want to show the original video as default

    reviews = [review.annotator for review in serverfile.mediafileannotation_set.all()]
    # get collections with serverfile
    collections_with = models.Collection.objects.filter(uploaded_files=serverfile)
    # get collections without serverfile
    collections_without = models.Collection.objects.exclude(uploaded_files=serverfile)
    context = {
        "serverfile": serverfile,
        "mediafile": Path(serverfile.mediafile.name).name,
        # "image_list": image_list_in,
        # "image_list_out": image_list_out,
        "next": request.GET["next"] if "next" in request.GET else None,
        "videofiles_url": [serverfile.mediafile.url],
        "alternative_videofiles_url": [serverfile.mediafile.url],
        # "results": results,
        "edit_review": edit_review,
        # "myhtml": "<b>The report is not ready yet.</b>"
        # "static_analysis_image": static_analysis_image,
        "review_number": len(serverfile.mediafileannotation_set.all()),
        "reviews": reviews,
        "collections_with": collections_with,
        "collections_without": collections_without,
        "show_alternative": show_alternative,

    }

    return context


def _prepare_context_for_message_if_web_report_not_exists(request, serverfile: UploadedFile):
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

def common_review(request):
    collection = get_object_or_404(models.Collection, name="common_review")

    url = reverse("uploader:web_report", kwargs={"filename_hash": collection.uploaded_files.first().hash}) + "?review_idx=new"
    return redirect(url)

# @login_required(login_url="/admin/")
def rotate_mediafile_right(request, mediafile_hash: str):
    uploaded_file = get_object_or_404(UploadedFile, hash=mediafile_hash)
    logger.debug(uploaded_file.rotation)
    uploaded_file.rotation = (uploaded_file.rotation + 90) % 360
    logger.debug(uploaded_file.rotation)
    uploaded_file.save()
    return redirect("uploader:web_report", filename_hash=mediafile_hash)


@login_required(login_url="/admin/")
def delete_annotation(request, annotation_id:int):
    annotation = get_object_or_404(models.MediaFileAnnotation, id=annotation_id)
    uploaded_file = annotation.uploaded_file
    annotation.delete()
    return redirect("uploader:web_report", filename_hash=uploaded_file.hash, review_edit_hash=uploaded_file.review_edit_hash)

def web_report(request, filename_hash: str, review_edit_hash: Optional[str] = None, review_annotator_hash: Optional[str] = None):
    # fn = get_outputdir_from_hash(hash)

    if review_annotator_hash:
        annotator = get_object_or_404(Owner, hash=review_annotator_hash)
    else:
        if request.user.is_authenticated:
            annotator = _get_owner(request.user.email)
        else:
            annotator = None

    serverfile = get_object_or_404(UploadedFile, hash=filename_hash)
    if (
        not bool(serverfile.zip_file.name)
        or not Path(serverfile.zip_file.path).exists()
    ):
        # context = _prepare_context_for_message_if_web_report_not_exists(request, serverfile)
        # return render(request, "uploader/message.html", context)
        context = _prepare_context_if_web_report_not_exists(request, serverfile)
    else:
        context = _prepare_context_for_web_report(request, serverfile, review_edit_hash)


    uploaded_file_annotations_set = serverfile.mediafileannotation_set.all()
    logger.debug(f"{uploaded_file_annotations_set=}")

    review_idx = request.GET.get("review_idx", -1)
    review_idx = None if review_idx == "new" else int(review_idx)
    if (review_idx is not None) and (review_idx < 0):
        review_idx = len(uploaded_file_annotations_set) + review_idx
        if review_idx < 0:
            review_idx = None

    if review_idx is not None:
        uploaded_file_annotation = uploaded_file_annotations_set[review_idx]
        logger.debug("annotation loaded from database")
    else:
        uploaded_file_annotation = None
        logger.debug("created empty form")

    annotation = uploaded_file_annotation
    logger.debug(f"{request.method=}")
    logger.debug(f"{annotation=}, {annotation is None=}")
    if annotation is not None:
        logger.debug(f"{annotation.respect_for_tissue=}")
        logger.debug(f"{annotation.time_and_movements=}")
        logger.debug(f"{annotation.instrument_handling=}")
        logger.debug(f"{annotation.procedure_flow=}")
    # evaluate annotation form
    if request.method == "POST":
        # if (review_idx >= 0) and (review_idx < len(uploaded_file_annotations_set)):

        form = AnnotationForm(request.POST, instance=uploaded_file_annotation)
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            logger.debug(f"{len(serverfile.mediafileannotation_set.all())=}")
            annotation = form.save(commit=False)
            annotation.uploaded_file = serverfile
            annotation.updated_at = django.utils.timezone.now()
            annotation.annotator = annotator
            annotation.save()

            logger.debug(f"{len(serverfile.mediafileannotation_set.all())=}")
            new_review_idx = 0
            for idx, ann in enumerate(serverfile.mediafileannotation_set.all()):
                if ann == annotation:
                    new_review_idx = idx

            annotation.save()

            # probably not necessary because all annotations are saved just before the run
            # annotation_filename = Path(serverfile.outputdir) / f"annotation_{review_idx}.json"
            annotation_filename = Path(serverfile.outputdir) / f"annotation_{new_review_idx}.json"
            logger.debug(f"{annotation_filename=}")
            annotation_filename.parent.mkdir(parents=True, exist_ok=True)
            from django.core.serializers import serialize
            # dump as json file
            with open(annotation_filename, "w") as f:
                # json.dump(json_annotation, f)
                f.write(serialize("json", [annotation]))


            logger.debug("preparing async_task for add_row_to_spreadsheet_and_update_zip")
            async_task(
                "uploader.tasks.add_row_to_spreadsheet_and_update_zip",
                serverfile,
                request.build_absolute_uri("/"),
                new_review_idx,
                timeout=settings.PIGLEGCV_TIMEOUT,
            )
            return redirect(request.path + "?review_idx=" + str(new_review_idx))
        else:
            logger.debug("Errors")
            logger.debug(f"{form.errors=}")
            # context["form"] = form
            # return render(request, "uploader/web_report.html", context)
    else:
        form = AnnotationForm(instance=uploaded_file_annotation)
        # check if in request get is review_idx
        # if (review_idx >= 0) and (review_idx < len(uploaded_file_annotations_set)):
        #     uploaded_file_annotation = uploaded_file_annotations_set[review_idx]
        #     form = AnnotationForm(instance=uploaded_file_annotation)
        #     logger.debug("annotation loaded from database")
        # else:
        #     form = AnnotationForm()
        #     logger.debug("created empty form")
            # uploaded_file_annotation = serverfile.mediafileannotation_set.last()
            # here the output might be None if the annotation does not exist

    # logger.debug(f"form={form}")
    context["form"] = form
    context["actual_annotation"] = uploaded_file_annotation


    return render(request, "uploader/web_report.html", context)

def _find_video_for_annotation(student_id:Optional[int] = None):
    """Find a video for annotation.

    If student_id is not None, then the video will not be owned by the student.
    We are looking for videos with finished processing, with no annotations and if we know the email of the user,
    we are looking for videos not uploaded by the user.
    """

    now = timezone.now()
    thirty_minutes_ago = now - timedelta(minutes=20)
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
        ##  comment next line to allow to annotate videos that are not processed yet
        # processing_ok=True,
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
            video.review_assigned_to = Owner.objects.get(id=student_id)
        video.save()

    return video

def students_list_view(request, days:Optional[int]=14):

    uploaded_files = UploadedFile.objects.filter(uploaded_at__gte=django.utils.timezone.now()-timedelta(days=days))
    # get all owners of uploaded_files
    owners = Owner.objects.filter(uploadedfile__in=uploaded_files).distinct()

    logger.debug
    context = {
        "headline": "Students list",
        "owners": owners,
    }
    return render(request, "uploader/owners.html", context)

def assigned_to_student(request, owner_hash: str):
    owner = get_object_or_404(Owner, hash=owner_hash)
    uploaded_files = UploadedFile.objects.filter(review_assigned_to=owner)

    context = _general_report_list(request, uploaded_files)
    return render(request, "uploader/report_list.html", context)

def go_to_video_for_annotation(request, annotator_hash:Optional[str]=None):
    logger.debug(f"{annotator_hash=}")
    if annotator_hash:
        # get the annotator
        annotator = get_object_or_404(Owner, hash=annotator_hash)
        student_id = annotator.id
    else:
        annotator = None
        student_id = None

        # if request.user.is_authenticated:
        #     email = request.user.email
        #     if email:
        #         student_id = _get_owner(email).id

    video = _find_video_for_annotation(student_id)

    # go to the web_report of the video
    if video:
        if annotator_hash:
            return redirect("uploader:web_report", filename_hash=video.hash,
                            review_edit_hash=video.review_edit_hash,
                            review_annotator_hash=annotator.hash
                            )
        else:
            return redirect("uploader:web_report", filename_hash=video.hash)
    else:
        # return redirect("uploader:model_form_upload_upload")
        return redirect("uploader:message_with_next",
                        headline="No video for review",
                        text="No video for review found. Please try again later.",
                        # next=reverse("uploader:model_form_upload"),
                        next_text="Ok"
                        )


def _filter_images(serverfile: UploadedFile):
    if serverfile.stitch_count > 0:
        allowed_image_patterns = [
            "needle_holder_heatmap_0",
            "forceps_heatmap_0",
            "stitch_detection_0",
            "needle_holder_all_area_presence",
            "forceps_all_area_presence",
            "jpeg",
        ]
    else:
        allowed_image_patterns = [
            "needle_holder_heatmap_all",
            "forceps_heatmap_all",
            "stitch_detection_0",
            "needle_holder_all_area_presence",
            "forceps_all_area_presence",
            "jpeg",
            "gif",
        ]
    filtered_in = []
    filtered_out = []
    static_analysis_image = None

    for image in serverfile.bitmapimage_set.all():
        to_keep = False
        # if image.filename.startswith("_") or image.filename.startswith("__"):
        #     continue
        filename = os.path.basename(image.bitmap_image.name)
        logger.debug(f"{filename=}")
        for allowed_image_pattern in allowed_image_patterns:
            if allowed_image_pattern in filename.lower():
                to_keep = True
                filtered_in.append(image)
        if not to_keep:
            filtered_out.append(image)
        if "incision_stitch_0.jpg" in filename.lower():
            logger.debug(f"static_analysis_image={image}")
            static_analysis_image = image

    return filtered_in, filtered_out, static_analysis_image

def redirect_to_spreadsheet_xlsx(request):
    settings.XLSX_SPREADSHEET_URL
    return redirect(settings.XLSX_SPREADSHEET_URL)

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


def categories_view(request):
    collections = models.Category.objects.all()
    context = {
        "categories": collections,
    }
    return render(request, "uploader/categories.html", context)

def category_view(request, category_id):
    category = get_object_or_404(models.Category, id=category_id)
    uploadedfile_set = category.uploadedfile_set.all()
    context = _general_report_list(request, uploadedfile_set)

    return render(request, "uploader/report_list.html", context)

def collections_view(request):
    collections = models.Collection.objects.all()
    context = {
        "collections": collections,
    }
    return render(request, "uploader/collections.html", context)

def collection_update_spreadsheet(request, collection_id):
    collection = get_object_or_404(models.Collection, id=collection_id)
    uploaded_files = collection.uploaded_files.all()
    async_task(
        # "uploader.tasks.add_row_to_spreadsheet_and_update_zip",
        "uploader.tasks.add_rows_to_spreadsheet_and_update_zips",
        uploaded_files,
        request.build_absolute_uri("/"),
        timeout=settings.PIGLEGCV_TIMEOUT,
    )
    return redirect("uploader:collections")

@staff_member_required(login_url="/admin/")
def run_collection_force_tracking(request, collection_id):
    """Run collection with force tracking."""
    return run_collection(request, collection_id, force_tracking=True)

@staff_member_required(login_url="/admin/")
def run_collection(request, collection_id, force_tracking:bool=False):
    collection = get_object_or_404(models.Collection, id=collection_id)
    PIGLEGCV_HOSTNAME = os.getenv("PIGLEGCV_HOSTNAME", default="127.0.0.1")
    PIGLEGCV_PORT = os.getenv("PIGLEGCV_PORT", default="5000")
    collection_len = len(collection.uploaded_files.all())
    for uploaded_file in collection.uploaded_files.all():
        _ = _run(
            request,
            uploaded_file.hash,
            PIGLEGCV_HOSTNAME,
            port=int(PIGLEGCV_PORT),
            force_tracking=force_tracking,
        )
    # next_url = None

    # if "next" in request.GET:
    #     next_url = request.GET["next"] + "#" + request.GET["next_anchor"]
    context = {
        "headline": "Processing started",
        "text": f"Processing of {collection_len} files started. ",
        # The output will be stored in {serverfile.outputdir}.",
        "next": reverse("uploader:web_reports", kwargs={}),
        "next_text": "Back",
    }
    return render(request, "uploader/thanks.html", context)
    # return redirect("uploader:web_reports")

def run_and_send_email(request, filename_hash:str):
    return run(request, filename_hash, send_email=True)

def run_and_force_tracking(request, filename_hash:str):
    return run(request, filename_hash, force_tracking=True)

def run(request, filename_hash:str, send_email:bool=False, force_tracking:bool=False):
    PIGLEGCV_HOSTNAME = os.getenv("PIGLEGCV_HOSTNAME", default="127.0.0.1")
    PIGLEGCV_PORT = os.getenv("PIGLEGCV_PORT", default="5000")
    serverfile = _run(request, filename_hash, PIGLEGCV_HOSTNAME, port=int(PIGLEGCV_PORT), send_email=send_email,
                      force_tracking=bool(force_tracking))
    return _render_run(request, serverfile, )


# def run_development(request, filename_id):
#     PIGLEGCV_HOSTNAME_DEVEL = os.getenv("PIGLEGCV_HOSTNAME_DEVEL", default="127.0.0.1")
#     PIGLEGCV_PORT_DEVEL = os.getenv("PIGLEGCV_PORT", default="5000")
#     return _run(request, filename_id, PIGLEGCV_HOSTNAME_DEVEL, port=int(PIGLEGCV_PORT_DEVEL))


def _run(request, filename_hash:str, hostname="127.0.0.1", port=5000,
         send_email=False,
         force_tracking=False
         ):
    serverfile = get_object_or_404(UploadedFile, hash=filename_hash)
    absolute_uri = request.build_absolute_uri("/")

    return make_it_run_with_async_task(serverfile, absolute_uri, hostname, port, send_email, force_tracking=force_tracking)
    # return redirect("/uploader/upload/")


def _render_run(request, serverfile):
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


def about_ev_cs(request):
    return render(request, "uploader/about_ev_cs.html", {})


def about_ev_en(request):
    return render(request, "uploader/about_ev_en.html", {})


class DetailView(generic.DetailView):
    model = UploadedFile
    template_name = "uploader/model_form_upload.html"




def _get_owner(owner_email: str):
    owners = Owner.objects.filter(email=owner_email)
    if len(owners) == 0:
        owner = Owner(email=owner_email, hash=_hash())
        owner.save()
        # create one
    else:
        owner = owners[0]
    return owner

@login_required(login_url="/admin/")
def import_files_from_drop_dir_view(request):
    """Import files from MEDIA_ROOT/drop_dir"""
    email = request.user.email
    absolute_uri = request.build_absolute_uri("/")
    async_task("uploader.tasks.import_files_from_drop_dir", email, absolute_uri)
    files = tasks.list_files_in_drop_dir()

    files_str= ""
    for file in files:
        files_str += str(file.name) + "<br>"

    key_value = {
        "Drop dir": settings.DROP_DIR,
        "Files": files_str
    }

    context = {
        "headline": "Import files from drop_dir",
        "text": "We will import files from drop_dir. It may take a while.",
        "key_value": key_value,
        "next_text": "Back",
        "next": reverse("uploader:web_reports", kwargs={}),
    }
    return render(request, "uploader/message.html", context )


def update_issue(request, uploadedfile_hash:Optional[str]=None, issue_hash: Optional[str] = None, user_hash: Optional[str] = None):
    if issue_hash:
        issue = get_object_or_404(models.Issue, hash=issue_hash)
        uploadedfile = issue.uploaded_file
    else:
        issue = None
        uploadedfile = get_object_or_404(UploadedFile, hash=uploadedfile_hash)


    text_note = ''

    # Handle POST request
    if request.method == "POST":
        form = forms.IssueForm(request.POST, instance=issue)
        if form.is_valid():
            text_note = "post"
            issue = form.save(commit=False)
            issue.uploaded_file = uploadedfile
            issue.save()
            next_url = reverse("uploader:model_form_upload")
            # encode url to be able to pass it as a parameter
            import urllib
            next_url = urllib.parse.quote(next_url)

            # next = reverse("uploader:model_form_upload"),
            return redirect("uploader:message_with_next",
                            headline="Issue",
                            text="Thank you for letting us know.",
                            next_text="Ok"
                            )
        else:
            return render(request, "uploader/update_form.html", {
                'button': "Ok",
                "form": form, 'related_uploadedfile': uploadedfile, 'text_note': text_note})

    # Handle GET request (form rendering)
    form = forms.IssueForm(instance=issue)
    return render(request, "uploader/update_form.html", {
        'button': "Ok",
        "headline": "New issue", "form": form, 'related_uploadedfile': uploadedfile, 'text_note': text_note})

@login_required(login_url="/admin/")
def issues_view(report):
    all_issues = models.Issue.objects.all()
    return render(report, "uploader/issues.html", {"issues": all_issues})


def upload_mediafile(request):
    if request.method == "POST":
        form = UploadedFileForm(
            request.POST,
            request.FILES,
            # owner=request.user
        )
        if form.is_valid():

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
            owner = _get_owner(serverfile.email)
            from .media_tools import make_images_from_video
            mediafile_path = Path(serverfile.mediafile.path)

            serverfile.owner = owner
            serverfile.mediafile = str(mediafile_path)
            serverfile.save()
            if mediafile_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
                make_images_from_video(
                   mediafile_path , mediafile_path.parent, filemask=str(mediafile_path) + ".jpg", n_frames=1)
            async_task("uploader.tasks.email_media_recived", serverfile, absolute_uri=request.build_absolute_uri("/"))

            # email_media_recived(serverfile)
            # print(f"user id={request.user.id}")
            # serverfile.owner = request.user
            absolute_uri = request.build_absolute_uri("/")
            call_async_run_processing(serverfile, absolute_uri)
            # url = reverse("uploader:go_to_video_for_annotation_email", kwargs={"email":serverfile.email})
            # logger.debug(f"{url=}")
            # return redirect("/uploader/thanks/")

            context = {
                "headline": "Thank You",
                "text": "Thank you for uploading media file. We will let you know when the processing will be finished. " +
                        "Meanwhile you can review other student's video.",
                "next_text": "Review other student's video",
                "next": reverse("uploader:go_to_video_for_annotation_email", kwargs={
                    # "email":serverfile.email
                    "annotator_hash": owner.hash
                }),
                "next_text_secondary": "Upload another video",
                "next_secondary": reverse("uploader:model_form_upload"),
                # 'next': "uploader:model_form_upload"
                # 'next': "uploader:model_form_upload"
                'sample_collection': Collection.objects.filter(name="Sample Reports").first()
            }

            logger.debug("redirecting to thanks")
            html = render_to_string("uploader/partial_message.html", context=context, request=request)
            return JsonResponse({"html": html})
    else:
        form = UploadedFileForm()
    return render(
        request,
        "uploader/model_form_upload.html",
        {
            "form": form,
            "headline": "Upload",
            "button": "Upload",
            'sample_collection': Collection.objects.filter(name="Sample Reports").first()
    },
    )


def test(request):
    return render(request, "uploader/test.html", {})


def show_mediafile_logs(request, filename_hash: str):
    """Show logs for the media file and check started_at."""
    serverfile = get_object_or_404(UploadedFile, hash=filename_hash)
    if serverfile.started_at is None:
        from .tasks import update_started_at_from_log
        update_started_at_from_log(serverfile)
    key_value = _get_logs_as_html(serverfile)
    return render(
        request,
        # "uploader/message.html",
        "uploader/show_logs.html",
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
            lines = [_set_loglevel_color(line) for line in lines]
        key_value.update(
            {str(file.stem): '<p class="text-monospace">' + "<br>".join(lines) + "</p>"}
        )
    # logpath = Path(serverfile.outputdir) / "piglegcv_log.txt"
    return key_value
    # return render(_as_htmlrequest,'uploader/show_logs.html', {"key_value": key_value, "logpath": logpath})

def _set_loglevel_color(line:str) -> str:
    """Set color of the log level in the log line"""
    from django.utils import html
    safe_line = html.escape(line)
    if "DEBUG" in line:
        return f"<span class='text-secondary'>{safe_line}</span>"  # decentní šedá
    elif "INFO" in line:
        return f"<span class='text-primary'>{safe_line}</span>"
    elif "WARNING" in line:
        return f"<span class='text-warning'>{safe_line}</span>"
    elif "ERROR" in line:
        return f"<span class='text-danger'>{safe_line}</span>"
    elif "CRITICAL" in line:
        return f"<span class='bg-danger text-white fw-bold'>{safe_line}</span>"
    else:
        return f"<span class='text-body'>{safe_line}</span>"


@login_required(login_url="/admin/")
def show_logs(request, n_lines: int = 100):
    key_value = {}

    files = settings.LOG_DIR.glob("**/*")

    for file in files:
        if file.is_file():
            with open(file) as f:
                lines = f.readlines()
            lines = [_set_loglevel_color(line) for line in lines]
            # take just first n_lines lines
            lines = lines[-n_lines:] if len(lines) > n_lines else lines
            key_value.update(
                {str(file.stem): '<p class="font-monospace">' + "<br>".join(lines) + "</p>"}
            )
    # logpath = Path(serverfile.outputdir) / "piglegcv_log.txt"
    return render(
        request,
        "uploader/show_logs.html",
        {
            "headline": "Logs",
            "key_value": key_value,
            "next": request.GET["next"]
            if "next" in request.GET
            else "/uploader/upload/",
        },
    )


# def _make_html_from_log(logpath: Path):
#     logpath = opath / "log.txt"
#     if logpath.exists():
#         with open(logpath) as f:
#             lines = f.readlines()
#
#         lines = [_set_loglevel_color(line) for line in lines]
#         key_value.update(
#             {"Log": '<p class="text-monospace">' + "<br>".join(lines) + "</p>"}
#         )


def download_original_video(request, uploadedfile_hash: str, ith_video:Optional[int]=None):
    uploaded_file = get_object_or_404(UploadedFile, hash=uploadedfile_hash)

    logger.debug(f"{uploaded_file=}, {ith_video=}")


    if ith_video is None:
        video_path = uploaded_file.mediafile.path
    else:
        # remove starting "_" from the video name
        video_list = [ element for element in Path(uploaded_file.outputdir).glob("*.mp4") if not element.name.startswith("_")]
        logger.debug(f"{video_list=}")
        video_path = str(video_list[ith_video])

    logger.debug(f"{video_path=}")
    if not os.path.exists(video_path):
        raise Http404()

    # return http response with the video
    return FileResponse(open(video_path, 'rb'), as_attachment=True)


def stream_video(request, uploadedfile_hash:str, ith_video:Optional[int]=None):
    uploaded_file = get_object_or_404(UploadedFile, hash=uploadedfile_hash)

    logger.debug(f"{uploaded_file=}, {ith_video=}")


    if ith_video is None:
        video_list = list(Path(uploaded_file.outputdir).glob("__cropped.mp4"))
        if len(video_list) == 0:
            video_path = uploaded_file.mediafile.path
        else:
            video_path = str(video_list[0])

    else:
        # remove starting "_" from the video name
        video_list = [ element for element in Path(uploaded_file.outputdir).glob("*.mp4") if not element.name.startswith("_")]
        logger.debug(f"{video_list=}")
        video_path = str(video_list[ith_video])

    logger.debug(f"{video_path=}")
    if not os.path.exists(video_path):
        raise Http404()

    def file_iterator(file_name, chunk_size=8192, offset=0, length=None):
        with open(file_name, 'rb') as f:
            f.seek(offset, os.SEEK_SET)
            remaining = length
            while True:
                bytes_length = chunk_size if remaining is None else min(remaining, chunk_size)
                data = f.read(bytes_length)
                if not data:
                    break
                if remaining:
                    remaining -= len(data)
                yield data

    file_size = os.path.getsize(video_path)
    content_type = 'video/mp4'
    range_header = request.META.get('HTTP_RANGE', '').strip()
    range_match = re.match(r'bytes=(\d+)-(\d+)?', range_header)
    if range_match:
        first_byte, last_byte = range_match.groups()
        first_byte = int(first_byte)
        last_byte = int(last_byte) if last_byte else file_size - 1
        length = last_byte - first_byte + 1
        response = StreamingHttpResponse(file_iterator(video_path, offset=first_byte, length=length), status=206,
                                         content_type=content_type)
        response['Content-Range'] = f'bytes {first_byte}-{last_byte}/{file_size}'
    else:
        response = StreamingHttpResponse(file_iterator(video_path), content_type=content_type)
        response['Content-Length'] = str(file_size)
    response['Accept-Ranges'] = 'bytes'

    return response



def _prepare_page(
        paginator: Paginator, request: Optional = None, page_number: Optional[int] = None
) -> Tuple[Page, List, dict]:
    if page_number is None:
        page_number = request.GET.get("page", 1)
    logger.debug(f"{page_number=}")
    elided_page_range = paginator.get_elided_page_range(page_number, on_each_side=4, on_ends=4)
    page_obj = paginator.get_page(page_number)

    context = {
        "page_obj": page_obj,
        "elided_page_range": elided_page_range,
    }

    return page_obj, elided_page_range, context

from django.shortcuts import redirect
from django.http import JsonResponse
from .models import UploadedFile  # Adjust as per your actual model name

from typing import Union
def add_multiple_to_collection(request, collection_id: Optional[int]=None):

    if request.method == "POST":
        selected_reports = request.POST.getlist('selected_reports')
        reports = UploadedFile.objects.filter(id__in=selected_reports)
        collection_id = request.POST.get('collection_id')
        if collection_id== "":
            number_of_collections = models.Collection.objects.count()
            collection = models.Collection(name=f"Collection {number_of_collections + 1}")
            collection.save()
        else:
            collection = get_object_or_404(models.Collection, id=int(collection_id))
        for report in reports:
            collection.uploaded_files.add(report)
        collection.save()

        # Process each selected report here
        # for report in reports:
        #
        #     # Add processing logic here (e.g., mark as reviewed, generate report, etc.)
        #     pass

        # Redirect back or send a success response
        return redirect('uploader:web_reports')  # Adjust as per your view name

