from django import forms
from .models import UploadedFile, Tag


class ImageQuatroForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ('email', 'mediafile')

