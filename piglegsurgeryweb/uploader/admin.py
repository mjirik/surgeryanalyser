from django.contrib import admin

# Register your models here.

from .models import UploadedFile, BitmapImage, Owner, MediaFileAnnotation
from django.db.migrations.recorder import MigrationRecorder
from django.contrib import admin


admin.site.register(UploadedFile)
admin.site.register(BitmapImage)
admin.site.register(Owner)
admin.site.register(MediaFileAnnotation)
admin.site.register(MigrationRecorder.Migration)


# @admin.register(UploadedFile)
# class UploadedFileeeAdmin(admin.ModelAdmin):
#     fields = ('email', 'mediafile')
#     # date_hierarchy = 'pub_date'
#     pass

# class UploadedFileAdmin(admin.ModelAdmin):
#     fields = ('email', 'mediafile')
#     # date_hierarchy = 'pub_date'
#     pass
#
# admin.site.register(UploadedFile, UploadedFileAdmin)
