from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

app_name = "uploader"

urlpatterns = [
    path("", views.index, name="index"),
    # path('a/', views.index, name='index2'),
    path("upload/", views.upload_mediafile, name="model_form_upload"),
    path("thanks/", views.thanks, name="thanks"),
    path("message/", views.message, name="message"),
    path("message/<str:headline>/<str:text>/<str:next>/<str:next_text>/", views.message, name="message_with_next"),
    path(
        "<str:filename_hash>/run/", views.run, name="run"
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
    path("web_report/<str:filename_hash>/<str:review_edit_hash>/<str:review_annotator_hash>/", views.web_report, name="web_report"),

    path(
        "owners_reports/<str:owner_hash>/",
        views.owners_reports_list,
        name="owners_reports_list",
    ),  # used for debugging purposes
    path(
        "assigned_to/<str:owner_hash>/",
        views.assigned_to_student,
        name="assigned_to",
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
    path("show_logs/<str:filename_hash>", views.show_mediafile_logs, name="show_mediafile_logs"),
    path("login/", auth_views.LoginView.as_view(), name="login"),
    path("go_to_video_for_annotation/", views.go_to_video_for_annotation, name="go_to_video_for_annotation_random"),
    path("go_to_video_for_annotation/<str:annotator_hash>/", views.go_to_video_for_annotation, name="go_to_video_for_annotation_email"),
    path("import_files_from_drop_dir/", views.import_files_from_drop_dir_view, name="import_files_from_drop_dir"),
    # path("about_ev_en/", views.about_ev_en, name="about_ev_en"),
    path("show_logs/", views.show_logs, name="show_logs"),
    path("delete_media_file/<int:filename_id>/", views.delete_media_file, name="delete_media_file"),
    path("download_sample_image", views.download_sample_image, name="download_sample_image"),
    path("common_review", views.common_review, name="common_review"),
    path("collections/", views.collections_view, name="collections"),
    path("run_collection/<int:collection_id>/", views.run_collection, name="run_collection"),
    path("show_collection/<int:collection_id>/", views.show_collection_reports_list, name="show_collection"),
    path("collection_update_spreadsheet/<int:collection_id>/", views.collection_update_spreadsheet, name="collection_update_spreadsheet"),
    path("add_to_collection/<int:collection_id>/<int:filename_id>/", views.add_uploaded_file_to_collection, name="add_to_collection"),
    path("remove_from_collection/<int:collection_id>/<int:filename_id>/", views.remove_uploaded_file_from_collection, name="remove_from_collection"),
    path("categories/", views.categories_view, name="categories"),
    path("category/<int:category_id>/", views.category_view, name="show_category"),
    path("mediafile/rotate_right/<str:mediafile_hash>/", views.rotate_mediafile_right, name="rotate_mediafile_right"),
    path("students_list/<int:days>/", views.students_list_view, name="students_list"),
    path("stream_video/<str:uploadedfile_hash>/", views.stream_video, name="stream_video"),
    path("stream_ith_video/<str:uploadedfile_hash>/<int:i>", views.stream_video, name="stream_ith_video"),
]
