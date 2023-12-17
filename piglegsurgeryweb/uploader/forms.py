from django import forms
from .models import UploadedFile


class UploadedFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ("email", "mediafile", "stitch_count", "is_microsurgery")
        help_texts = {
            "stitch_count": "Number of stitches created in uploaded video. If you don't know, leave it zero.",
            "is_microsurgery": "Check if your video is microsurgery.",
        }
