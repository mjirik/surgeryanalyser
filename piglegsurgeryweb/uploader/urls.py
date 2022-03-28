from django.urls import path

from . import views

app_name = "uploader"

urlpatterns = [
    path("", views.index, name="index"),
    # path('a/', views.index, name='index2'),
    path("upload/", views.model_form_upload, name="model_form_upload"),
    path("thanks/", views.thanks, name="thanks"),
    path("message/", views.message, name="message"),
    path('<int:filename_id>/run/', views.run, name='run'), # used for debugging purposes
    path('<int:filename_id>/run_development/', views.run_development, name='run_development'), # used for debugging purposes
    path('<int:filename_id>/resend_report_email/', views.run, name='resend_report_email'), # used for debugging purposes
    path('web_report/<str:filename_hash>/', views.web_report, name='web_report'),  # used for debugging purposes
    path('ths6eei8sadfwebw7s8d6s5e4vs8eqpzmg4710awo/', views.show_report_list, name='web_reports'),  # used for debugging purposes
    path("reset_hashes/", views.reset_hashes, name="reset_hashes"),
]
