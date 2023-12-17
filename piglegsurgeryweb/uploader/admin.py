from django.contrib import admin

# Register your models here.

from .models import UploadedFile, BitmapImage, Owner

admin.site.register(UploadedFile)
admin.site.register(BitmapImage)
admin.site.register(Owner)


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
