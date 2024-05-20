import os.path as op
from datetime import datetime
from pathlib import Path
import numpy as np

from django.db import models

from .models_tools import (
    generate_sha1,
    get_output_dir,
    randomString,
    upload_to_unqiue_folder,
)

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

class Category(models.Model):
    name = models.CharField(max_length=255, blank=True, default="")
    def __str__(self):
        return str(self.name)


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
    stitch_count = models.IntegerField("Stitch count", default=0)
    preview = models.ImageField(blank=True, null=True)
    outputdir = models.CharField(max_length=255, blank=True, default=get_output_dir)
    zip_file = models.FileField(upload_to="cellimage/", blank=True, null=True)
    hash = models.CharField(max_length=255, blank=True, default=_hash)
    started_at = models.DateTimeField("Started at", blank=True, null=True)
    finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    processing_ok = models.BooleanField(default=False)
    processing_message = models.TextField(blank=True, default="")
    owner = models.ForeignKey(Owner, on_delete=models.CASCADE, null=True, blank=True)
    is_microsurgery = models.BooleanField(default=False)
    review_assigned_at = models.DateTimeField("Assigned at", null=True, blank=True, default=None)
    review_assigned_to = models.ForeignKey(Owner, on_delete=models.CASCADE, null=True, blank=True, related_name="review_assigned_to")
    review_edit_hash = models.CharField(max_length=255, blank=True, default=_hash)
    consent = models.BooleanField(default=False)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return str(Path(self.mediafile.name).name)

class Collection(models.Model):
    name = models.CharField(max_length=255, blank=True, default="")
    owner = models.ForeignKey(Owner, on_delete=models.CASCADE, null=True, blank=True)
    # one collection can have many uploaded files
    uploaded_files = models.ManyToManyField(UploadedFile)

    def __str__(self):
        return str(self.name)

class MediaFileAnnotation(models.Model):
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE)
    # annotation text field might be empty
    annotation = models.TextField(blank=True, default="")
    # title = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField("Created at", default=datetime.now)
    updated_at = models.DateTimeField("Updated at", default=datetime.now)
    annotator = models.ForeignKey(
        Owner, on_delete=models.CASCADE, null=True, blank=True
    )
    stars = models.IntegerField(default=-1)

    needle_grabbed_correctly = models.BooleanField(default=False, help_text="Needle grabbed in first or second third of needle holder")
    needle_holder_stabilized = models.BooleanField(default=False, help_text="Needle holder is stabilized with the index finger")
    needle_pierced_at_first_try = models.BooleanField(default=False, help_text="Needle is pierced through skin at the first try")
    needle_pierced_at_right_angle = models.BooleanField(default=False)
    needle_rotated_correctly_on_opposite_side = models.BooleanField(default=False, help_text="Needle rotated according to the curvature (opposite wound edge)")
    needle_rotated_correctly_on_students_side = models.BooleanField(default=False, help_text="Needle rotated according to the curvature (wound edge on students side)")
    forceps_grabs_the_edge = models.BooleanField(default=False, help_text="Forceps only grabs the edge of the wound")
    three_knots_per_stitch = models.BooleanField(default=False, help_text="Minimum of three knots per stitch")
    knots_are_done_right = models.BooleanField(default=False, help_text="Knots are done right (square knots)")
    threads_shortened_appropriately = models.BooleanField(default=False, help_text="Threads are shortened to an appropriate length")

    stitch_to_wound_distance_is_correct = models.BooleanField(default=False, help_text="Distance of stitches to wound is 4-6mm")
    distance_between_stitches_is_correct = models.BooleanField(default=False, help_text="Distance between stitches is 9-11mm")
    stitches_perpendicular_to_wound = models.BooleanField(default=False, help_text="Stitches perpendicular to wound")
    equal_sized_wound_portions = models.BooleanField(default=False, help_text="Equal-sized wound portions")
    no_excessive_tension = models.BooleanField(default=False, help_text="Wound edges are brought together without excessive tension")

    respect_for_tissue = models.IntegerField(default=-1, help_text="1 = Often unnecessary force applied to the tissue or damage caused by improper use of the instruments <br> "+
"3 = Careful handling of the fabric, yet sometimes unintentional damage is caused <br> "+
"5 = Careful handling of the fabric throughout with minimal damage ")
    time_and_movements = models.IntegerField(default=-1, help_text="")
    instrument_handling = models.IntegerField(default=-1, help_text="")
    procedure_flow = models.IntegerField(default=-1, help_text="")


    def __str__(self):
        # return first line of annotation
        return (
            str(self.id) + ", " + str(self.uploaded_file) + ", " + str(self.annotation.split("\n")[0]) + (", " + str(self.annotator))
            if self.annotator
            else str(self.id) + ", " + str(self.uploaded_file)
        )


class BitmapImage(models.Model):
    server_datafile = models.ForeignKey(UploadedFile, on_delete=models.CASCADE)
    bitmap_image = models.ImageField()
    title = models.CharField(max_length=255, blank=True, default="")

    @property
    def filename(self):
        return op.basename(self.bitmap_image.name)

    def __str__(self):
        return str(op.basename(self.bitmap_image.name))
