# Generated by Django 3.2.23 on 2024-11-14 22:19

from django.db import migrations, models
import uploader.models


class Migration(migrations.Migration):

    dependencies = [
        ('uploader', '0024_issue'),
    ]

    operations = [
        migrations.AddField(
            model_name='collection',
            name='hash',
            field=models.CharField(blank=True, default=uploader.models._hash, max_length=255),
        ),
    ]
