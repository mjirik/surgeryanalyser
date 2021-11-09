from django.contrib import admin

# Register your models here.

from .models import UploadedFile, BitmapImage

admin.site.register(UploadedFile)
admin.site.register(BitmapImage)
