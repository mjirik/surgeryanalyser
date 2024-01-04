from django.contrib import admin
from django.db.migrations.recorder import MigrationRecorder

from .models import BitmapImage, MediaFileAnnotation, Owner, UploadedFile

# Register your models here.


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
