from django.shortcuts import render, get_object_or_404, redirect
from django.views import generic
from pathlib import Path

# Create your views here.

from django.http import HttpResponse
from .models import UploadedFile
from .forms import UploadedFileForm
from .models_tools import randomString
from .tasks import email_media_recived
# from .models_tools import get_hash_from_output_dir, get_outputdir_from_hash

def index(request):
    return HttpResponse("Hello, world. You're at the polls index. HAHA")


def thanks(request):
    context = {
        'headline': "Thank You",
        'text': "Thank you for uploading media file. We will let you know when the processing will be finished."
    }
    return render(request, "uploader/thanks.html", context)

def reset_hashes(request):
    files = UploadedFile.objects.all()
    for file in files:
        file.hash = randomString(12)
        file.save()
    return redirect("/uploader/thanks/")


def web_report(request, filename_hash:str):
    # fn = get_outputdir_from_hash(hash)
    serverfile = get_object_or_404(UploadedFile, hash=filename_hash)
    context = {
        'serverfile': serverfile,
        'mediafile': Path(serverfile.mediafile.name).name
    }
    return render(request,'uploader/web_report.html', context)

def run(request, filename_id):
    serverfile = get_object_or_404(UploadedFile, pk=filename_id)

    from django_q.tasks import async_task
    async_task(
        "uploader.tasks.run_processing",
        serverfile,
        request.build_absolute_uri("/"),
        hook="uploader.tasks.email_report",
    )
    context = {}
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
                hook="uploader.tasks.email_report",
            )
            return redirect("/uploader/thanks/")
    else:
        form = UploadedFileForm()
    return render(
        request,
        "uploader/model_form_upload.html",
        {"form": form, "headline": "Upload", "button": "Upload"},
    )
