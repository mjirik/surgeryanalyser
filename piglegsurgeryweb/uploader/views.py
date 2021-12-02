from django.shortcuts import render, get_object_or_404, redirect
from django.views import generic
from django.contrib.auth.decorators import login_required
from pathlib import Path

from loguru import logger
# Create your views here.

from django.http import HttpResponse
from .models import UploadedFile, _hash
from .forms import UploadedFileForm
from .models_tools import randomString
from .tasks import email_media_recived
# from .models_tools import get_hash_from_output_dir, get_outputdir_from_hash
from django_q.tasks import async_task, schedule, queue_size

def index(request):
    return HttpResponse("Hello, world. You're at the polls index. HAHA")

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
    context = {
        "uploadedfiles": files, 'queue_size': queue_size()
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

    image_list = serverfile.bitmapimage_set.all()

    videofile = Path(serverfile.outputdir) / "video.mp4"
    logger.debug(videofile)
    videofile_url = None
    if videofile.exists():
        s = str(serverfile.bitmapimage_set.all()[0].bitmap_image.url)[:-4]
        videofile_url = s[:s.rfind("/")] + "/video.mp4"


    context = {
        'serverfile': serverfile,
        'mediafile': Path(serverfile.mediafile.name).name,
        'image_list': image_list,
        "next": request.GET['next'] if "next" in request.GET else None,
        'videofile_url': videofile_url
    }
    return render(request,'uploader/web_report.html', context)

def run(request, filename_id):
    serverfile = get_object_or_404(UploadedFile, pk=filename_id)

    from django_q.tasks import async_task
    async_task(
        "uploader.tasks.run_processing",
        serverfile,
        request.build_absolute_uri("/"),
        timeout=3600*2,
        hook="uploader.tasks.email_report_from_task",
    )
    context = {
        'headline': "Processing started",
        'text': f"Processing file {serverfile.mediafile}. The output will be stored in {serverfile.outputdir}."
    }
    return render(request, "uploader/thanks.html", context)
    # return redirect("/uploader/upload/")

class DetailView(generic.DetailView):
    model = UploadedFile
    template_name = "uploader/model_form_upload.html"


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
            # serverfile.save()
            async_task(
                "uploader.tasks.run_processing",
                serverfile,
                request.build_absolute_uri("/"),
                timeout=2*3600,
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
