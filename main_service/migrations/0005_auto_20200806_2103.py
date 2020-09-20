# Generated by Django 2.2.12 on 2020-08-06 12:03

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_service', '0004_auto_20200724_1942'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelresult',
            name='timestamp',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2020, 8, 6, 21, 3, 14, 982452)),
        ),
        migrations.CreateModel(
            name='AppUsed',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uid', models.TextField()),
                ('day_num', models.IntegerField(default=0)),
                ('ema_order', models.IntegerField(default=0)),
                ('Entertainment_Music', models.TextField()),
                ('Utilities', models.TextField()),
                ('Shopping', models.TextField()),
                ('Games_Comics', models.TextField()),
                ('Others', models.TextField()),
                ('Health_Wellness', models.TextField()),
                ('Social_Communication', models.TextField()),
                ('Education', models.TextField()),
                ('Travel', models.TextField()),
                ('Art_Photo', models.TextField()),
                ('News_Magazine', models.TextField()),
                ('Food_Drink', models.TextField()),
            ],
            options={
                'unique_together': {('uid', 'day_num', 'ema_order')},
            },
        ),
    ]
