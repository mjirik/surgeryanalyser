# Generated by Django 3.2.23 on 2024-10-15 14:03

import datetime
from django.db import migrations, models
import django.db.models.deletion
import uploader.models


class Migration(migrations.Migration):

    dependencies = [
        ('uploader', '0023_uploadedfile_rotation'),
    ]

    operations = [
        migrations.CreateModel(
            name='Issue',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('hash', models.CharField(blank=True, default=uploader.models._hash, max_length=255)),
                ('description', models.TextField(blank=True, default='')),
                ('created_at', models.DateTimeField(default=datetime.datetime.now, verbose_name='Created at')),
                ('uploaded_file', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='uploader.uploadedfile')),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='uploader.owner')),
            ],
        ),
    ]