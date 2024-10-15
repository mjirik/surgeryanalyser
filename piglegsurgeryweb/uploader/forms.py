from django import forms

from .models import MediaFileAnnotation, UploadedFile, Issue
from loguru import logger


class IssueForm(forms.ModelForm):
    class Meta:
        model = Issue
        fields = ("description",)


class UploadedFileForm(forms.ModelForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'autocomplete': 'email'}))
    class Meta:
        model = UploadedFile
        fields = ("email", "mediafile", "category", "stitch_count", "is_microsurgery", "consent",)
        help_texts = {
            "stitch_count": "Number of stitches created in uploaded video. If you don't know, leave it zero.",
            "is_microsurgery": "Check if your video is microsurgery.",
            "consent": "I agree to the use and anonymized sharing of my data for scientific purposes."
        }


class AnnotationForm(forms.ModelForm):
    STAR_CHOICES_RTL = [(i, str(i)) for i in range(5, 0, -1)]  # 0 to 5
    STAR_CHOICES = [(i, str(i)) for i in range(0, 5, 1)]  # 0 to 5

    stars = forms.ChoiceField(
        choices=STAR_CHOICES_RTL, widget=forms.RadioSelect,
        help_text = "1: Beginner Many unnecessary movements, frequent interruptions, uncertainty, no planning ahead, many stitches incorrectly placed; " +
                    "3: Competent some unnecessary moves, effective use of time, planning ahead, most stitches neatly placed; " +
                    "5: Expert economical movements, obviously pre - planned course of action, fluid movements, clean suture "


    )
    respect_for_tissue = forms.ChoiceField(
        choices=STAR_CHOICES_RTL, widget=forms.RadioSelect,
        help_text = "1: Often unnecessary force applied to the tissue or damage caused by improper use of the instruments; " +
                    "3: Careful handling of the fabric, yet sometimes unintentional damage is caused; " +
                    "5: Careful handling of the fabric throughout with minimal damage "
    )
    time_and_movements = forms.ChoiceField(
        choices=STAR_CHOICES_RTL, widget=forms.RadioSelect,
        help_text = "1: Many unnecessary movements; "
                    "3: Effective work but some unnecessary movements; " +
                    "5: economical movements and maximum effectiveness"

    )
    instrument_handling = forms.ChoiceField(
        choices=STAR_CHOICES_RTL, widget=forms.RadioSelect,
        help_text = "1: Frequently timid or awkward movements due to improper use of the instruments; " +
                    "3: Competent use of instruments, but sometimes stiff or awkward movements; " +
                    "5: Fluid movements and no clumsiness"
    )
    procedure_flow = forms.ChoiceField(
        choices=STAR_CHOICES_RTL, widget=forms.RadioSelect,
        help_text = "1: Frequent interruptions and uncertainty about the next step;" +
                    "3: Next steps planned in advance and appropriate course of the procedure;" +
                    "5: Clearly planned procedure with effortless transition to the next step "
    )

    def __init__(self, *args, **kwargs):
        super(AnnotationForm, self).__init__(*args, **kwargs)
        # logger.debug(f"{args=}")
        # logger.debug(f"{kwargs=}")
        self.fields["stars"].initial = 1  # Default to 1 star
        self.fields["stars"].label = "Global Assessment"
        self.fields["respect_for_tissue"].initial = 1  # Default to 1 star
        self.fields["time_and_movements"].initial = 1  # Default to 1 star
        self.fields["instrument_handling"].initial = 1  # Default to 1 star
        self.fields["procedure_flow"].initial = 1  # Default to 1 star
        # logger.debug(f"{dir(self.fields['respect_for_tissue'])}")
        # logger.debug(f"{self.fields['respect_for_tissue'].help_text}")
        # logger.debug(f"{self.fields['time_and_movements'].help_text}")
        # logger.debug(f"{kwargs.get('instance').respect_for_tissue=}")
        # logger.debug(f"{dir(kwargs.get('instance').respect_for_tissue)=}")
        # logger.debug(f"{type(kwargs.get('instance').respect_for_tissue)=}")
        # log help text for respect_for_tissue from instance
        # logger.debug(f"{kwargs.get('instance').respect_for_tissue.help_text=}")

        # self.fields["stars"].group = 1
        # for field in self.group3():
        #     self.fields[field].initial = 1

        # for field in self.group4():
        #     field.initial = 1


    def group1(self):
        return [self["annotation"], self["stars"]]

    def group2(self):
        return [
            # Video
            self["needle_grabbed_correctly"],
            self["needle_holder_stabilized"],
            self["needle_pierced_at_first_try"],
            self["needle_pierced_at_right_angle"],
            self["needle_rotated_correctly_on_opposite_side"],
            self["needle_rotated_correctly_on_students_side"],
            self["forceps_grabs_the_edge"],
            self["three_knots_per_stitch"],
            self["knots_are_done_right"],
            self["threads_shortened_appropriately"]

        ]

    def group3(self):
        # Result
        return [
            self["stitch_to_wound_distance_is_correct"],
            self["distance_between_stitches_is_correct"],
            self["stitches_perpendicular_to_wound"],
            self["equal_sized_wound_portions"],
            self["no_excessive_tension"],
        ]


    def group4(self):
       # Global Rating Scale
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
