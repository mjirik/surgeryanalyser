from django.shortcuts import render
from django.views import generic
# Create your views here.

from django.http import HttpResponse
from .models import UploadedFile


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

class DetailView(generic.DetailView):
    model = UploadedFile
    template_name = 'uploader/model_form_upload.html'