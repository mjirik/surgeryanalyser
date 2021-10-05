from django.db import models
from .models_tools import upload_to_unqiue_folder
from datetime import datetime

# Create your models here.


class UploadedFile(models.Model):
    email = models.CharField(max_length=200)
    # hash = scaffanweb_tools.randomString(12)
    uploaded_at = models.DateTimeField(
        "Uploaded at",
        default=datetime.now
    )
    mediafile = models.FileField("Media File", upload_to=upload_to_unqiue_folder, blank=True, null=True, max_length=500)
