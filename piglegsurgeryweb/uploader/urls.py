from django.urls import path

from . import views

app_name = "uploader"

urlpatterns = [
    path("", views.index, name="index"),
    # path('a/', views.index, name='index2'),
    path("upload/", views.model_form_upload, name="model_form_upload"),
]
