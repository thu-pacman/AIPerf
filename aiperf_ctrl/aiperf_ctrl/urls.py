"""aiperf_ctrl URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
from django.urls import path
import trial.views as trial

urlpatterns = [
    path('api/trial/create', trial.create_trial),
    path('api/trial/finish', trial.finish_trial),
    path('api/trial/heartbeat', trial.heartbeat),
    path('api/trial/stop', trial.stop_all),
    path('api/trial/clear', trial.clear),
]
