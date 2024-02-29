# Generated by Django 3.2.23 on 2024-02-17 15:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('uploader', '0019_collection'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedfile',
            name='consent',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='distance_between_stitches_is_correct',
            field=models.BooleanField(default=False, help_text='Distance between stitches is 9-11mm'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='equal_sized_wound_portions',
            field=models.BooleanField(default=False, help_text='Equal-sized wound portions'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='forceps_grabs_the_edge',
            field=models.BooleanField(default=False, help_text='Forceps only grabs the edge of the wound'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='knots_are_done_right',
            field=models.BooleanField(default=False, help_text='Knots are done right (square knots)'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='needle_grabbed_correctly',
            field=models.BooleanField(default=False, help_text='Needle grabbed in first or second third of needle holder'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='needle_holder_stabilized',
            field=models.BooleanField(default=False, help_text='Needle holder is stabilized with the index finger'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='needle_pierced_at_first_try',
            field=models.BooleanField(default=False, help_text='Needle is pierced through skin at the first try'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='needle_rotated_correctly_on_opposite_side',
            field=models.BooleanField(default=False, help_text='Needle rotated according to the curvature (opposite wound edge)'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='needle_rotated_correctly_on_students_side',
            field=models.BooleanField(default=False, help_text='Needle rotated according to the curvature (wound edge on students side)'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='no_excessive_tension',
            field=models.BooleanField(default=False, help_text='Wound edges are brought together without excessive tension'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='respect_for_tissue',
            field=models.IntegerField(default=-1, help_text='1 = Often unnecessary force applied to the tissue or damage caused by improper use of the instruments <br> 3 = Careful handling of the fabric, yet sometimes unintentional damage is caused <br> 5 = Careful handling of the fabric throughout with minimal damage '),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='stitch_to_wound_distance_is_correct',
            field=models.BooleanField(default=False, help_text='Distance of stitches to wound is 4-6mm'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='stitches_perpendicular_to_wound',
            field=models.BooleanField(default=False, help_text='Stitches perpendicular to wound'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='threads_shortened_appropriately',
            field=models.BooleanField(default=False, help_text='Threads are shortened to an appropriate length'),
        ),
        migrations.AlterField(
            model_name='mediafileannotation',
            name='three_knots_per_stitch',
            field=models.BooleanField(default=False, help_text='Minimum of three knots per stitch'),
        ),
    ]