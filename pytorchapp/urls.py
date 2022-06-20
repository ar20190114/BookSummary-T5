from django.urls import path
from . import views

app_name = 'pytorchapp'


urlpatterns = [
    path('upload/', views.upload, name='upload'),
    path('result/', views.result, name='result'),
]
