from django import forms

from .models import MediaFileAnnotation, UploadedFile


class UploadedFileForm(forms.ModelForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'autocomplete': 'email'}))
    class Meta:
        model = UploadedFile
        fields = ("email", "mediafile", "stitch_count", "is_microsurgery")
        help_texts = {
            "stitch_count": "Number of stitches created in uploaded video. If you don't know, leave it zero.",
            "is_microsurgery": "Check if your video is microsurgery.",
        }


class AnnotationForm(forms.ModelForm):
    STAR_CHOICES_RTL = [(i, str(i)) for i in range(5, 0, -1)]  # 0 to 5
    STAR_CHOICES = [(i, str(i)) for i in range(0, 5, 1)]  # 0 to 5

    stars = forms.ChoiceField(choices=STAR_CHOICES_RTL, widget=forms.RadioSelect)
    respect_for_tissue = forms.ChoiceField(choices=STAR_CHOICES_RTL, widget=forms.RadioSelect)
    time_and_movements = forms.ChoiceField(choices=STAR_CHOICES_RTL, widget=forms.RadioSelect)
    instrument_handling = forms.ChoiceField(choices=STAR_CHOICES_RTL, widget=forms.RadioSelect)
    procedure_flow = forms.ChoiceField(choices=STAR_CHOICES_RTL, widget=forms.RadioSelect)

    def __init__(self, *args, **kwargs):
        super(AnnotationForm, self).__init__(*args, **kwargs)
        self.fields["stars"].initial = 1  # Default to 1 star
        self.fields["stars"].label = "Global Assessment"
        # self.fields["stars"].group = 1
        # for field in self.group3():
        #     self.fields[field].initial = 1

        for field in self.group4():
            field.initial = 1


    def group1(self):
        return [self["annotation"], self["stars"]]

    def group2(self):
        return [

            self["needle_grabbed_correctly"],
            self["needle_holder_stabilized"],
            self["needle_pierced_at_first_try"],
            self["needle_pierced_at_right_angle"],
            self["needle_rotated_correctly_on_opposite_side"],
            self["needle_rotated_correctly_on_students_side"],
            self["forceps_grabs_the_edge"],
            self["three_knots_per_stitch"],
            self["knots_are_done_right"],
            self["threads_shortened_appropriately"],
            self["threads_shortened_appropriately"]

        ]

    def group3(self):
        return [
            self["stitch_to_wound_distance_is_correct"],
            self["distance_between_stitches_is_correct"],
            self["stitches_perpendicular_to_wound"],
            self["equal_sized_wound_portions"],
            self["no_excessive_tension"],
        ]

    def group4(self):
        return [
            self["respect_for_tissue"],
            self["time_and_movements"],
            self["instrument_handling"],
            self["procedure_flow"]
        ]



    class Meta:
        model = MediaFileAnnotation
        fields = (
            "annotation", "stars",
            # ("Group 2", (
            "needle_grabbed_correctly",
            "needle_holder_stabilized",
            "needle_pierced_at_first_try",
            "needle_pierced_at_right_angle",
            "needle_rotated_correctly_on_opposite_side",
            "needle_rotated_correctly_on_students_side",
            "forceps_grabs_the_edge",
            "three_knots_per_stitch",
            "knots_are_done_right",
            "threads_shortened_appropriately",
            "threads_shortened_appropriately",
            # )),
            # ("Group 3", (
            "stitch_to_wound_distance_is_correct",
            "distance_between_stitches_is_correct",
            "stitches_perpendicular_to_wound",
            "equal_sized_wound_portions",
            "no_excessive_tension",

            "respect_for_tissue",
            "time_and_movements",
            "instrument_handling",
            "procedure_flow",
            # ))
        )



        # help_texts = {
        #     "annotation": "Write your annotation here.",
        #     "stars": "How many stars do you give to this video?",
        # }
