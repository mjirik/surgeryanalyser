from django.urls import path

from . import views

app_name = "uploader"

urlpatterns = [
    path("", views.index, name="index"),
    # path('a/', views.index, name='index2'),
    path("upload/", views.model_form_upload, name="model_form_upload"),
    path("thanks/", views.thanks, name="thanks"),
    path('<int:filename_id>/run/', views.run, name='run'), # used for debugging purposes
    path('web_report/<str:filename_hash>/', views.web_report, name='web_report'),  # used for debugging purposes
    path("reset_hashes/", views.reset_hashes, name="reset_hashes"),
]
