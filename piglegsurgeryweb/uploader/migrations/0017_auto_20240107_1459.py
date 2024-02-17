# Generated by Django 3.2.23 on 2024-01-07 13:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('uploader', '0016_uploadedfile_review_edit_hash'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedfile',
            name='processing_message',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='uploadedfile',
            name='processing_ok',
            field=models.BooleanField(default=False),
        ),
    ]