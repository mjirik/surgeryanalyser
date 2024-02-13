from django.contrib import admin
from django.db.migrations.recorder import MigrationRecorder

from . import models

# Register your models here.


admin.site.register(MigrationRecorder.Migration)
admin.site.register(models.UploadedFile)
admin.site.register(models.BitmapImage)
admin.site.register(models.Owner)
admin.site.register(models.MediaFileAnnotation)
admin.site.register(models.Collection)


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
