from django.db import models
from .models_tools import upload_to_unqiue_folder, get_output_dir, randomString
from datetime import datetime
import os.path as op

# Create your models here.


class UploadedFile(models.Model):
    email = models.EmailField(max_length=200)
    # hash = scaffanweb_tools.randomString(12)
    uploaded_at = models.DateTimeField("Uploaded at", default=datetime.now)
    mediafile = models.FileField(
        "Media File",
        upload_to=upload_to_unqiue_folder,
        # blank=True,
        # null=True,
        max_length=500,
    )
    outputdir = models.CharField(max_length=255, blank=True, default=get_output_dir)
    zip_file = models.FileField(upload_to="cellimage/", blank=True, null=True)
    hash = models.CharField(max_length=255, blank=True,default=randomString)

class BitmapImage(models.Model):
    server_datafile = models.ForeignKey(UploadedFile, on_delete=models.CASCADE)
    bitmap_image = models.ImageField()

    @property
    def filename(self):
        return op.basename(self.bitmap_image.name)
