from django.shortcuts import render, get_object_or_404, redirect
from django.views import generic

# Create your views here.

from django.http import HttpResponse
from .models import UploadedFile
from .forms import UploadedFileForm


def index(request):
    return HttpResponse("Hello, world. You're at the polls index. HAHA")


class DetailView(generic.DetailView):
    model = UploadedFile
    template_name = "uploader/model_form_upload.html"


def model_form_upload(request):
    if request.method == 'POST':
        form = UploadedFileForm(request.POST, request.FILES,
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
            # print(f"user id={request.user.id}")
            # serverfile.owner = request.user
            # serverfile.save()
            # async_task('uploader.tasks.make_thumbnail', serverfile,
            #            # hook='tasks.email_report'
            #            )
            return redirect('/uploader/')
    else:
        form = UploadedFileForm()
    return render(request, 'uploader/model_form_upload.html', {
        'form': form,
        "headline": "Upload",
        "button": "Upload"
    })

