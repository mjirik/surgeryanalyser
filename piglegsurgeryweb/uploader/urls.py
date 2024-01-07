from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

app_name = "uploader"

urlpatterns = [
    path("", views.index, name="index"),
    # path('a/', views.index, name='index2'),
    path("upload/", views.model_form_upload, name="model_form_upload"),
    path("thanks/", views.thanks, name="thanks"),
    path("message/", views.message, name="message"),
    path("message/<str:headline>/<str:text>/<str:next>/<str:next_text>", views.message, name="message"),
    path(
        "<int:filename_id>/run/", views.run, name="run"
    ),  # used for debugging purposes
    # path('<int:filename_id>/run_development/', views.run_development, name='run_development'), # used for debugging purposes
    path(
        "<int:filename_id>/resend_report_email/",
        views.resend_report_email,
        name="resend_report_email",
    ),  # used for debugging purposes
    path(
        "<int:filename_id>/swap_is_microsurgery/",
        views.swap_is_microsurgery,
        name="swap_is_microsurgery",
    ),
    path("web_report/<str:filename_hash>/", views.web_report, name="web_report"),
    path("web_report/<str:filename_hash>/<str:review_edit_hash>/", views.web_report, name="web_report"),
    path(
        "owners_reports/<str:owner_hash>/",
        views.owners_reports_list,
        name="owners_reports_list",
    ),  # used for debugging purposes
    path(
        "ths6eei8sadfwebw7s8d6s5e4vs8eqpzmg4710awo/",
        views.report_list,
        name="web_reports",
    ),  # used for debugging purposes
    # path('set_order_by/<str:order_by>/<str:next_page>', views.set_order_by, name='set_order_by'),
    path("reset_hashes/", views.reset_hashes, name="reset_hashes"),
    path(
        "update_all_uploaded_files/",
        views.update_all_uploaded_files,
        name="update_all_uploaded_files",
    ),
    path("logout/", views.logout_view, name="logout_view"),
    path("spreadsheet/", views.redirect_to_spreadsheet, name="spreadsheet"),
    # path("about_ev_cs/", views.about_ev_cs, name="about_ev_cs"),
    path("test/", views.test, name="test"),
    path("show_logs/<str:filename_hash>", views.show_logs, name="show_logs"),
    path("login/", auth_views.LoginView.as_view(), name="login"),
    path("go_to_video_for_annotation/", views.go_to_video_for_annotation, name="go_to_video_for_annotation_random"),
    path("go_to_video_for_annotation/<str:email>/", views.go_to_video_for_annotation, name="go_to_video_for_annotation_email"),
    # path("about_ev_en/", views.about_ev_en, name="about_ev_en"),
]
