# Generated by Django 2.2.12 on 2020-06-18 07:20

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_service', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelresult',
            name='timestamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2020, 6, 18, 16, 20, 22, 762065)),
        ),
    ]
