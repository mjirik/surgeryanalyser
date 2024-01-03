from django import forms
from .models import UploadedFile, MediaFileAnnotation


class UploadedFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ("email", "mediafile", "stitch_count", "is_microsurgery")
        help_texts = {
            "stitch_count": "Number of stitches created in uploaded video. If you don't know, leave it zero.",
            "is_microsurgery": "Check if your video is microsurgery.",
        }


class AnnotationForm(forms.ModelForm):
    STAR_CHOICES = [(i, str(i)) for i in range(1,6)]  # 0 to 5

    stars = forms.ChoiceField(choices=STAR_CHOICES, widget=forms.RadioSelect)

    def __init__(self, *args, **kwargs):
        super(AnnotationForm, self).__init__(*args, **kwargs)
        self.fields['stars'].initial = 5  # Default to 1 star
    class Meta:
        model = MediaFileAnnotation
        fields = ("annotation", "stars")
        help_texts = {
            "annotation": "Write your annotation here.",
            "stars": "How many stars do you give to this video?",
        }