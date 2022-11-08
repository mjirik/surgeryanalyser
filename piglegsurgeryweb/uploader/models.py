from django.db import models
from .models_tools import upload_to_unqiue_folder, get_output_dir, randomString, generate_sha1
from datetime import datetime
import os.path as op
from pathlib import Path

# Create your models here.

def _hash():
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    hash = generate_sha1(dt, salt=randomString())
    return hash

class Owner(models.Model):
    email = models.EmailField(max_length=200)
    hash = models.CharField(max_length=255, blank=True, default=_hash)
    def __str__(self):
        return str(self.email)

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
    preview = models.ImageField(blank=True, null=True)
    outputdir = models.CharField(max_length=255, blank=True, default=get_output_dir)
    zip_file = models.FileField(upload_to="cellimage/", blank=True, null=True)
    hash = models.CharField(max_length=255, blank=True, default=_hash)
    started_at = models.DateTimeField("Started at", blank=True, null=True)
    finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    owner = models.ForeignKey(Owner, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return str(Path(self.mediafile.name).name)

class BitmapImage(models.Model):
    server_datafile = models.ForeignKey(UploadedFile, on_delete=models.CASCADE)
    bitmap_image = models.ImageField()

    @property
    def filename(self):
        return op.basename(self.bitmap_image.name)

    def __str__(self):
        return str(op.basename(self.bitmap_image.name))

