from django.shortcuts import render, get_object_or_404, redirect
from django.views import generic

# Create your views here.

from django.http import HttpResponse
from .models import UploadedFile
from .forms import UploadedFileForm
from .tasks import email_media_recived


def index(request):
    return HttpResponse("Hello, world. You're at the polls index. HAHA")


def thanks(request):
    context = {}
    return render(request, "uploader/thanks.html", context)


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
