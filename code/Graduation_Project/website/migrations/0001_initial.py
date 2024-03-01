# Generated by Django 4.2 on 2023-04-20 09:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Category',
            fields=[
                ('category', models.PositiveIntegerField(primary_key=True, serialize=False, unique=True)),
                ('name', models.CharField(max_length=100, null=True)),
                ('video_clip_number', models.PositiveIntegerField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Video_Clip',
            fields=[
                ('video_id', models.PositiveIntegerField(primary_key=True, serialize=False, unique=True)),
                ('sentences', models.TextField(null=True)),
                ('category', models.ForeignKey(null=True, on_delete=django.db.models.deletion.PROTECT, to='website.category')),
            ],
        ),
    ]
